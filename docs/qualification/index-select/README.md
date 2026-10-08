# Indexed producer fusion — issue #222

The compiler can fold a CPU row selection and optional cast into a portable
indexed plan. The runtime gathers directly into wire-format chunk staging,
removing the full gathered FP32 tensor. This integrates the compiler/frontend
foundation from [PR #216](https://github.com/JueonPark/sym/pull/216) with main's
retained typed transform/H2D pipeline. It is based on merged main, without a
dependency on the workload branch in PR #162.

On the qualified host, large selections improve over prior Sym, while small
selections remain slower than Torch. Inductor writing into retained pinned
memory is still faster than Sym at every measured size. The results below
separate this producer improvement from the broader end-to-end performance goal.

## Coverage and execution

| Property | Supported contract |
|---|---|
| Operation | `torch.index_select`, tensor method, or `aten.index_select.default` inside the captured H2D region |
| Source | Dense, zero-offset CPU tensor, rank >= 1; FP32, FP16, or int8 |
| Indices | CPU rank-one int32/int64 vector; strided vectors are snapshotted contiguously |
| Axis | 0, including its negative-rank spelling |
| Values | Every index must satisfy `0 <= i < source.shape[0]`; duplicates and arbitrary order are preserved, and the selection may exceed the source's row count |
| Conversion | Identity for the source dtypes above, FP32→FP16 (`ieee_rne`), or FP16→FP32 (`exact`) |
| Result layout | Dense selected rows with the source's inner dimensions preserved |
| Exclusions | Empty tensors, other axes/layout transforms, CUDA indices, `out=`, autograd, nonblocking calls, escaping gathered intermediates, and unsafe mutations remain in Torch |

`features[indices]` is not captured: advanced indexing permits negative values
where `index_select` rejects them. For the PR #162 GNN, whose node IDs are
nonnegative, move selection inside capture and spell it explicitly:

```python
backend = RelocBackend(transfer_options={"pinning": "pinned", "gather_threads": 8})
def load_features(features, nodes):
    return torch.index_select(features, 0, nodes).to("cuda:0", dtype=torch.float16)
load_features = torch.compile(load_features, backend=backend, dynamic=True)
```

The compiler exports wire v2/schema-3 manifests with separate physical source
and runtime index operands. Format-3 portable recipes retain those contracts;
old v0/v1 plans remain compatible. Native binding checks descriptor agreement,
overflow, index byte length, and every index value before any source access.
Unsupported/invalid frontend calls fall back before launch, preserving Torch's
errors. Prepared calls reject index changes, including NumPy writes without a
Tensor version bump. Symbolic metadata proofs are cached; current index values
are copied and validated on every invocation, with fresh output/request state.

Chunks preserve complete selected rows and rebase writes to the local staging
slot. A previous DMA must complete before its slot is overwritten. Indirection
uses the host CPU implementation; GPU-producing typed rows are not eligible.
The implementation inherits bounded resources, current-stream ordering,
drain/quarantine handling, and completed-call semantics from #221. Model-compute
prefetch and asynchronous frontend return are outside this change.

## Measurement protocol

- Implementation: `dd533437f73aa47ccef1f5ab7c3f0c415686b202`, based on merged main
  `afb3622558b4c4bae8de4bb476c01cab693cc9cb`. Baseline binary is the preserved
  #221 implementation `79254f81032600b71b6e3a70aa8340fb82d34c2e`; its `libreloc`
  and `sym` sources are identical to that merged main. Benchmark revisions,
  runtime/frontend/exporter hashes, and configurations are in provenance.
- EPYC 7351, RTX 2080 Ti GPU 0, CPU affinity `4-7,20-23` on GPU-local NUMA node
  1: four physical cores/eight hardware threads. Torch/OMP/MKL/gather use eight
  workers, interop one; one native copy stream and one or two staging slots.
- Release GCC 11.4, CUDA toolkit 12.5.82, native SM 75/89, Python 3.14.7,
  Torch 2.14.0+cu126, driver 595.71.05, Linux 6.8.0-124-generic. Clocks unlocked.
  NVTX is off for timings; a separate NVTX-on build uses Nsight 2024.2.3.38.
- `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, and
  `TORCHINDUCTOR_COMPILE_THREADS=1` apply to every timed path. Preliminary
  interleaved runs with default worker waiting produced inconsistent small-call
  timings; these explicit settings avoid idle OpenMP spinning against the next
  backend. Preliminary runs are excluded from the reported distributions.
- Three independent processes per revision, eight warmups and 100 samples
  per path/size/round; candidate path order rotates by round and sample. Revision
  order alternates. Builds, tests, profiling, and timing jobs run separately.
- The GNN-shaped store is `500000 x 128` FP32. Each sample changes source
  values and generates fresh sorted unique indices, like the workload's node
  set. Counts are 4,096, 16,384, and 65,536. Random/repeated order is covered by
  exactness tests; graph sampling and GNN computation are not timed.
- Completed time includes gathering, validation/binding, output allocation,
  conversion, H2D, and synchronization. Mutation, selection generation, and
  byte-exact CPU oracle checks are outside timing. Every measured output passed.
  Compilation/setup plus first completion is recorded separately from warm
  distributions; initialized CUDA and existing compiler disk caches mean it is
  not a cold-process measurement. There is no end-to-end model claim.
- Sym has a 64 MiB retained/live scratch limit per owner, seven background
  workers plus the caller, and pinned staging. Torch/Inductor pinned controls
  receive a reusable FP16 output buffer too. Resource counters cover native
  scratch, not total RSS, source/output tensors, index snapshots, or Torch's
  allocator cache. All controls fit the scratch envelope at these sizes.
- `unfused` times Torch gathering followed by Sym conversion/upload, including
  the producer. Both the preserved main binary and the candidate execute this
  control. `ring1` and `ring2` capture the full producer region; their identical
  chunk schedules isolate overlap. Small payloads (1 and 4 MiB) execute whole;
  the 16 MiB case uses sixteen 1 MiB chunks.
- `inductor_host` compiles gather+cast on CPU then uploads FP16.
  `inductor_pinned` compiles gather+conversion into a retained pinned FP16
  destination, then uploads it. Both timers include the complete producer.
  `torch_pinned` uses eager gather then copies/converts into its pinned output.
  Full-function `inductor` instead lowers to CPU FP32 gathering, FP32 H2D,
  and a GPU cast; it has twice the wire traffic and is reported separately.

## Completed producer latency

Milliseconds, **p50 / p95**, each the median of three rounds' percentiles
(not pooled percentiles or confidence intervals). Matched FP16-wire paths:

| Path | 4,096 rows | 16,384 rows | 65,536 rows |
|---|---:|---:|---:|
| Prior main: gather + Sym | 1.228 / 1.575 | 2.529 / 3.668 | 6.590 / 14.524 |
| Same-build unfused control | 1.188 / 1.404 | 2.532 / 3.503 | 6.536 / 14.558 |
| Fused, one buffer | 1.238 / 1.555 | 2.593 / 2.821 | 6.392 / 6.649 |
| **Fused, two buffers** | **1.187 / 1.574** | **2.538 / 2.776** | **5.501 / 5.780** |
| Eager Torch | 0.523 / 0.565 | 1.668 / 2.591 | 7.818 / 15.447 |
| Torch, retained pinned output | 0.374 / 0.449 | 1.555 / 2.558 | 6.683 / 14.805 |
| CPU Inductor + FP16 upload | 0.542 / 0.745 | 1.565 / 1.869 | 5.072 / 6.213 |
| **CPU Inductor into pinned output + upload** | **0.432 / 0.656** | **1.448 / 1.689** | **4.547 / 4.733** |

Large fusion reduces median p50 by **16.5% versus prior main**, **15.8% versus
the same-build unfused control**, and **13.9% versus one-buffer fusion**.
Its 29.6% improvement over eager Torch's median does **not** beat the strongest
matched baseline: pinned Inductor is still 17.3% faster (Sym takes 21.0% longer).
Inductor already fuses CPU gathering and casting without the FP32 intermediate.
The research objective of beating an equally optimized Inductor path is not
established by this feature.

Small and medium fusion is effectively neutral against unfused Sym and remains
about 2.3x / 1.5x slower than eager Torch. Index snapshots/bounds validation and
fresh native program/request construction still have costs; only immutable
descriptor proofs are reused. The current policy does not automatically fall
back to Torch based on size. Generic additional layout chains are also outside
this implementation.

Large allocating-gather paths have substantial variability: candidate-round
eager p50 is 7.384, 14.706, 7.818 ms; unfused p50 is 6.536, 14.102, 6.495 ms.
The fused two-buffer p50s are 5.513, 5.497, 5.501 ms; pinned Inductor is
4.541, 4.553, 4.547 ms. The table retains the elevated p95s and every raw sample.
We do not attribute those spikes to a specific allocator/scheduler mechanism
without a separate profile, or generalize percentages beyond this host.

Full-function Inductor, whose actual payload is FP32, measures
0.821 / 0.982, 2.414 / 3.184, and 7.863 / 15.317 ms respectively. These numbers
are useful deployment observations, but a win over them is not evidence of
superior execution at equal wire bytes.

First completed 4,096-row calls, including lazy compilation/setup, range across
rounds: fused two-buffer **74–230 ms**, unfused **118–273 ms**, eager Torch
**1.1–2.0 ms**, full Inductor **90–2,000 ms**, and pinned-output Inductor
**86–1,993 ms**. Which compiled method runs first changes compiler-cache reuse.
Later-size first calls and all first-call observations are retained in CSV.
These startup costs are excluded from the warm table, not erased by folding.

## Removed intermediates and DMA overlap

Separate Torch allocator diagnostics observe one `aten::index_select` FP32
result per eager invocation, versus zero in fused Sym. Source inspection plus
the native direct-to-wire executor establish that this is eliminated data,
not an untracked native FP32 gather allocation. CPU Inductor controls also
eliminate that intermediate; profiler operator absence alone is not a complete
process allocation census.

| Selected rows | Removed FP32 allocation | Avoided logical write + read | FP16 wire |
|---|---:|---:|---:|
| 4,096 | 2 MiB | 4 MiB | 1 MiB |
| 16,384 | 8 MiB | 16 MiB | 4 MiB |
| 65,536 | 32 MiB | 64 MiB | 16 MiB |

The write/read column is derived tensor traffic, not measured DRAM traffic;
caching can change physical memory traffic. Reading selected source features
and validating/copying indices remain necessary. The 65,536 int64 indices alone
occupy 512 KiB; snapshots and native index copies are outside arena counters.

Native retained/peak host scratch after the three-size sequence is **5.75 MiB**
for each Sym control, with two allocations retained from the smaller whole
transfers. In the isolated large trace, a fresh owner retains **1.25 MiB** with
one slot and **2.50 MiB** with two. Native capacities include growth headroom.
Pinned Torch/Inductor receive the full **16 MiB logical pinned output**, so the
comparison gives them the same retention opportunity. GPU logical output is
16 MiB for all paths; full Inductor additionally allocates a 32 MiB FP32 GPU
intermediate.

Five separately profiled large calls per path confirm:

- Eager Torch and both CPU-Inductor controls each transfer **16 MiB**; full
  Inductor transfers **32 MiB**. Both Sym controls transfer sixteen 1 MiB chunks.
- Correlation IDs match native submission ranges to actual GPU memcpy events.
  Unioned CPU-worker intervals for transform(n+1) intersect DMA(n) by
  **0.868–0.937 ms per two-buffer call**, versus **zero in every one-buffer call**.
  This excludes worker barriers and CUDA API duration from the overlap claim.
- Trace timings are not included in headline latency. All request summaries
  and the first request's chunk intervals per control are committed.

## DLRM coverage decision and separate opportunity

**Keep EmbeddingBag lookup/sum-pooling in Torch.** This PR's row selection does
not implement reductions or remove DLRM's pooled/stacked producer tensors.
Existing Sym layout/cast/transfer support can still operate on the completed
pooled tensor. A future pooled producer needs an explicit FP32 accumulation and
ordering contract, including repeated indices and bag offsets. It must narrow
only after the original sum: pooling FP32 values `[2048.5, -2048]` then casting
gives `0.5`, while casting the inputs first gives `0.0`. The exclusion and this
counterexample are tested.

Separate completed-region measurements use PR #162's default dimensions:
26 CPU FP32 tables, 100,000 rows each, width 64, sum bags of 1–4 random rows.
Indices change each sample. These are producer measurements, not complete DLRM
inference or claims about Inductor's reduction performance. Phase diagnostics
run separately from the full-region timer; their medians must not be added to
reconstruct the full time.

Milliseconds, median round p50 / p95:

| Batch | Full producer through GPU | Lookup + pool | Stack | Layout + cast + H2D |
|---|---:|---:|---:|---:|
| 512 | 3.531 / 5.813 | 2.148 / 2.185 | 0.528 / 2.525 | 0.694 / 1.118 |
| 1,536 | 7.124 / 10.542 | 3.522 / 3.609 | 1.517 / 1.675 | 2.121 / 3.025 |
| 2,048 | 8.782 / 13.555 | 4.121 / 4.246 | 1.724 / 4.248 | 3.010 / 4.094 |

At batch 2,048, each FP32 `[26,2048,64]` tensor is **13 MiB**, and the FP16 wire
is **6.5 MiB**. Avoiding the separate stack and dense-transpose tensors could
remove **two allocations totaling 26 MiB** and up to **52 MiB logical write/read
traffic**, while preserving lookup reads and FP32 sum arithmetic. The measured
stack phase (1.724 ms) indicates an opportunity; it is not a promised full-call
saving because direct output writes, scheduling, and memory locality still cost
time. Eliminating the entire downstream layout/cast/H2D phase would be only an
ideal bound and is physically impossible while data must cross PCIe. Pooling
itself accounts for substantial remaining work and requires separate design.

## Reproduction and evidence

Use `bench/issue222/indexed_transfer.py` with the selected build's Python
package and exporter. For example, from the repository root:

```sh
export PYTHONPATH=/tmp/sym-222-build/python
export SYM_RELOC_EXPORT=/tmp/sym-222-build/sym/tools/sym-reloc-export
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0 TORCHINDUCTOR_COMPILE_THREADS=1
taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue222/indexed_transfer.py --samples 100 --output /tmp/candidate \
  --paths torch torch_pinned inductor inductor_host inductor_pinned unfused ring1 ring2
```

Run three fresh processes, rotating paths. For the preserved main package use
`--paths unfused torch inductor_pinned`; preserve the same environment. DLRM is
`--dlrm --selected 512 1536 2048`. Allocation diagnostics use `--memory` in a
separate process. For traces use the NVTX-on runtime, `--trace --selected 65536`,
and Nsight `--trace=cuda,nvtx --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop`. Export SQLite, then:

```sh
python bench/issue221/analyze_trace.py /tmp/indexed.sqlite /tmp/overlap \
  --prefix indexed222/ --first-per-path
python bench/issue222/compact_evidence.py --output /tmp/evidence \
  --trace /tmp/indexed.sqlite /tmp/candidate.json
```

Committed evidence contains lossless timing samples grouped into compact CSV
rows, provenance, allocation/DMA summaries, and first-call interval samples.
Raw profiler databases, generated Inductor kernels, and verbose test logs are
excluded. All trace request summaries are retained; interval selection is
deterministically the first call of each native control.

## Validation

- CUDA build: 1,136 Python tests passed, 4 skipped. Subsequent indexed fault
  coverage and workload integration: 74 passed (21 pipeline, 53 workload tests).
- CPU-only build, CUDA hidden: 803 passed, 20 skipped, 317 GPU tests deselected.
- Six CTest checks passed on each of CUDA and CPU-only builds, including native
  runtime and plan-builder suites. All 41 compiler lit tests passed.
- The existing lit site template assumes an in-tree build; the generated local
  site config was pointed at this worktree to run the external build. No source
  change to the test configuration was needed.
- Exact tests cover both index widths, all five supported dtype pairs, rank-one
  and higher-rank rows, SIMD/scalar tails, unsorted/repeated indices, M > N,
  malformed wire/manifest data, bounds/overflow, fresh values, stale snapshots,
  stream order, bounded one/two/four-slot staging, partial-copy failures, and
  portable compiler-to-runtime execution without importing Torch.
- Changed C++ files pass clang-format 21 checks; `git diff --check` passes.
