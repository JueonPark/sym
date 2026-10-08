# Asynchronous completion and model prefetch (#223)

The implementation adds native completion events, consumer-stream ordering and a
bounded two-slot iterator. GPT prefetches a layer's four projection matrices;
MoE resolves routing and dynamic token selections before prefetching each active
expert's two matrices. The existing blocking APIs and CLI defaults remain.
See the [API/lifetime contract](../../docs/torch-support.md#explicit-asynchronous-groups-and-model-prefetch-223).

## Measurement protocol

- RTX 2080 Ti, EPYC 7351, CUDA toolkit 12.5, driver 595.71.05, Torch
  2.14.0+cu126, CPython 3.14.7. GPU 0 and affinity `4-7,20-23` (four physical
  cores, eight hardware threads). Torch/OMP/MKL threads 8, interop 1,
  `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, Inductor compile threads 1.
  Clocks use normal boost; no locked-clock claim. GPU jobs run serially.
- Three fresh processes per revision, 30 completed samples per path/case, five
  warmups. Main and candidate processes alternate. Three additional candidate
  processes run only the existing blocking API to check that control separately. Candidate path order rotates
  between processes and within each sample. No profiler, builds or tests run
  during headline timings. Lossless samples, p50 and interpolated p95 are retained.
- The baseline build corresponds to merged main `a44985c`; its runtime/frontend
  sources equal `dd53343`. Candidate runtime/frontend implementation is
  `cb835b6`. Full hashes, harness revisions, affinity and binary/frontend SHA-256
  are recorded in `provenance.jsonl`. Both revisions run the same model harness;
  the baseline uses its existing blocking grouped API.
- All paths use the same pinned INT8 checkpoints, per-channel FP32 scales, FP32
  output weights and model arithmetic, with TF32 disabled. Inputs and quantized
  weights/scales change between samples, outside timing. Every candidate call
  compares tokens/routing and byte-exact model outputs with another path on the
  same values. The comparison includes ordinary Torch and **Inductor-compiled
  weight conversion**, both with serial and prefetch scheduling. Model compute
  is common eager Torch, not a whole-model `torch.compile` comparison.
- Each queue permits two output groups within 128 MiB, and 64 MiB combined native
  host/device scratch. Torch gets the same limits: its live temporary upper
  bound is INT8 input + scale + two FP32 conversion intermediates per member;
  conversion temporaries reuse storage on one private stream. Both queues drain
  the preceding producer before reusing scratch and preserve outstanding model
  consumers. Logical reservations exclude pinned checkpoint storage, model
  activations and allocator caches. The diagnostic records native scratch and
  Torch's process-wide allocated/reserved memory separately; Torch allocator
  counters do not include native allocations, and reserved cache is not a live
  temporary-byte count. `sym_blocking` output bytes are a conservative bound.
- The primary measure includes binding, allocations, CPU dispatch, transfers,
  model kernels, consumer retirement and final synchronization. Decode also
  includes KV eviction/restore and token selection. Throughput is input tokens/s
  for prefill/MoE, generated tokens/s for decode. First completed calls and
  artifact setup are separate; first calls can benefit from existing on-disk
  Inductor caches, so they are not cold-cache compiler benchmarks.
- Separate diagnostic calls host-wait the first group to measure startup to
  ready weights and time each window's final consumer drain. These diagnostic
  waits are **not** used in headline timings. Startup/drain are per window
  (four windows for four decode steps, two for two MoE blocks), not independent
  additive costs to subtract from end-to-end latency.

## Results

The median of three round medians, in milliseconds (lower is better):

| Workload | Main blocking | Sym queue, serial | Sym prefetch | Torch prefetch | Inductor conversion + prefetch |
|---|---:|---:|---:|---:|---:|
| GPT, 1,024-token prefill, 4 layers | 21.80 | 23.96 | **18.62** | 19.29 | 17.73 |
| GPT, 256-token prefill, 4 layers | 11.83 | 11.55 | **8.53** | 8.26 | 7.01 |
| GPT, 4 decode steps, 4 layers | 46.92 | 38.63 | **31.95** | 31.18 | 26.24 |
| MoE, 4,096 tokens, 8 experts/block | 25.17 | 29.98 | **19.41** | 18.84 | 17.27 |
| MoE, 512 tokens, 8 experts/block | 19.89 | 19.01 | **16.11** | 12.04 | 12.91 |
| MoE, 1 token, 2 active experts/block | 5.46 | 4.73 | **4.41** | 3.46 | 3.54 |
| GPT, 1 token, 1 layer | 3.32 | 2.50 | **2.50** | 2.11 | 1.80 |

The long prefill is **14.6% lower latency than main**, and large MoE is **22.9%
lower**. Against the matched queue-serial control, the reductions are 22.3% and
35.3%. The other multi-consumer cases improve 19.0–31.9% versus main. This does
**not** close the gap to Inductor: its matched prefetch remains faster in every
case. Sym narrowly beats ordinary Torch prefetch for the long prefill, but trails
it elsewhere.

`Sym queue, serial` uses the same async submission/ownership API and retires each
consumer before submitting the next. It isolates scheduling from the different
allocation-stream and Python reporting paths in `WeightFetcher.fetch_many`.
In particular, the one-layer result has **zero prefetch gain** against this
control; the reduction versus the old blocking delivery must not be attributed
to inter-layer overlap. In the three additional isolated blocking-control
processes, candidate blocking p50 changed −1.9% to +0.1% relative to main across
these cases; the raw controls are retained. The seven-path process has a different
allocator/cache footprint, so its small blocking-path differences alone should
not be interpreted as a runtime regression.

Tail latency and boundary diagnostics (medians across rounds/windows):

| Workload | Sym prefetch p95 ms | Throughput tokens/s | First group ready ms | Final window drain ms |
|---|---:|---:|---:|---:|
| GPT 1,024 prefill | 18.78 | 54,995 | 2.11 | 6.49 |
| GPT 256 prefill | 8.75 | 30,014 | 2.10 | 1.22 |
| GPT decode | 32.62 | 125 generated | 2.11 | 0.63 |
| MoE 4,096 | 19.67 | 211,028 | 0.94 | 1.28 |
| MoE 512 | 16.42 | 31,787 | 0.94 | 0.30 |
| MoE 1 | 4.77 | 227 | 0.93 | 0.21 |
| GPT one layer | 2.72 | 400 | 2.10 | 0.08 |

Artifact setup was 23.3–23.7 ms in the candidate rounds. Per-path first-completed
calls and every raw timing sample are retained in
[`evidence/timings.csv`](evidence/timings.csv). These first calls also share
initialization performed by earlier paths, so they are diagnostic rather than
comparable cold-start benchmarks. Startup/drain and memory counters are in
[`evidence/diagnostics.csv`](evidence/diagnostics.csv). The candidate timing rounds made
4,557 model calls with output assertions, including **3,906 cross-path exact
comparisons** (the first path in each sample supplies the reference).

Observed peak output reservations were 96 MiB for four-layer GPT and 32 MiB for
MoE, equal across Sym/Torch prefetch. Native scratch peaked at 13.82 MiB and
4.52 MiB respectively, below the 64 MiB limit. Checkpoint pinning is shared by all
paths and recorded separately. These existing example models fit on the GPU;
the experiment qualifies scheduling and performance, not a capacity claim.

## Trace interpretation

`analyze_trace.py` joins CUDA API correlation IDs inside each model-call NVTX
range to actual H2D activity and GEMM/GEMV kernels. Model-kernel intervals are the
union of kernels on those GEMM consumer streams; private transfer conversion
kernels are excluded. The report includes both DMA/GEMM overlap and DMA/all-model
kernel overlap. It never treats CPU API duration or an asynchronous return as
proof of DMA overlap. Every serial/blocking control must have zero overlap.

Median milliseconds of actual DMA/model-kernel intersection across three traces:

| Case | Sym serial | Sym prefetch | Torch prefetch | Inductor prefetch |
|---|---:|---:|---:|---:|
| GPT 1,024 prefill | 0 | **2.299** | 2.323 | 2.607 |
| GPT 256 prefill | 0 | **0** | 1.155 | 0.394 |
| MoE 4,096 tokens | 0 | **2.600** | 4.318 | 4.162 |

All blocking/serial paths have zero intersection. Every path moves exactly
50,479,104 H2D bytes for either GPT prefill and 67,305,472 for large MoE, including
scale uploads. Within the Sym totals, DMA/GEMM-only intersection is 0.806 ms for
long prefill and 1.962 ms for large MoE; transfer conversion kernels are excluded.
Thus both workloads demonstrate actual DMA/model-kernel intersection. The 256-token prefill's Sym trace has **zero** such intersection;
its latency change alone must not be described as GPU DMA overlap. These GPU-row
traces also do not demonstrate CPU-transform/DMA overlap from #221: asynchronous
groups use the existing whole-group transform implementation when a CPU row is
selected, not the individual typed transfer's chunk ring.

`intervals.csv` keeps the first complete call for every traced path, compacted by
activity kind. Each tuple is `[start_ns, end_ns, stream, correlation_id, bytes,
kernel]`, relative to that model call's NVTX start; null columns do not apply.
`model_kernel_union` entries are merged device intervals. `overlap.csv` keeps all
three traced calls per path. The databases remain outside the repository;
`trace-sources.json` records their hashes. Profiler timings are never headline
latency measurements.

## Limits and cases without useful lookahead

A single-layer model has no next layer to overlap. Small decode/expert workloads
can finish their GPU kernels before the next CPU preparation/enqueue finishes.
MoE cannot prefetch across an unresolved routing boundary; sparse routing offers
only two experts per block here. CPU preparation is still synchronous, output
allocation and bookkeeping have a cost, and concurrent DMA/conversion/model
kernels can contend for GPU memory bandwidth. Prefetch is explicit rather than
an unconditional replacement for blocking calls. MoE prefetch currently supports
one CUDA device; multi-GPU concurrency remains separate work.

## Reproduce

Use a CUDA build with the qualified interpreter and the compiler exporter:

```sh
export PYTHONPATH=/tmp/sym-223-build/python
export SYM_RELOC_EXPORT=/tmp/sym-223-build/sym/tools/sym-reloc-export
export SYM_RUNTIME_REVISION=cb835b6c5c4f1e8d923f5ef22c7115ffb41c5bdd
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1

taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python bench/issue223/model_prefetch.py \
  --output /tmp/candidate-r1.json --samples 30
# Repeat in two fresh processes, rotating --paths. For main, point PYTHONPATH
# and the exporter at its independent build and use --paths sym_blocking.

/usr/local/cuda-12.5/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  --output=/tmp/long-trace taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue223/model_prefetch.py --output /tmp/long-trace-run.json \
  --cases llm_long_prefill --samples 3 --trace
/usr/local/cuda-12.5/bin/nsys export --type=sqlite --force-overwrite=true \
  --output=/tmp/long-trace.sqlite /tmp/long-trace.nsys-rep
/tmp/sym-210-venv/bin/python bench/issue223/analyze_trace.py \
  /tmp/long-trace.sqlite /tmp/long-overlap --require-overlap
# Also trace moe_large, and llm_prefill without --require-overlap.
```

`compact_evidence.py` combines the timing JSON/CSV pairs and the three trace
summary/interval pairs, with their profiler-run metadata. Run the extractor against profiler databases before
compacting; all summaries and sample arrays remain independently recomputable.

## Validation

- CUDA runtime/frontend: **1,159 passed, 4 skipped**. The explicit fork-guard
  test emits Python's
  expected warning about forking a multithreaded process.
- Workload tests: **56 passed**, including exact GPT generation/KV and routed MoE
  comparisons, changed weights/scales, and one-GPU prefetch enforcement.
- Independent CPU-only runtime build, CUDA hidden: **822 passed, 1 skipped**;
  GPU cases were deselected. Both native CTest suites: **6/6**. Compiler lit:
  **41/41**. Final async/model coverage additionally passed after the last GPU
  consumer-lifetime change; the final full suite includes the consumer-failure
  quarantine test.
- Fault injection covers event record failure, failure after prior copies were
  enqueued, and completion wait failure, each with successful or unknown drain.
  Tests also cover two consumer streams, implicit current-stream use after
  `wait()`, D2H host waits, concurrent submissions, queue limits, fresh values,
  stale metadata, cancellation, early model failure, native destructor/resource
  close and fork rejection. Unknown consumer completion retains output and
  event owners and permanently closes the queue.
