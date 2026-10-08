# Grouped transfer results — issue #220

Explicit consumer-sized groups reduce warm completed-call p50 by **5.9–49.7%**
across seven cases on an EPYC 7351 / RTX 2080 Ti, compared with current main's
individual calls under matched settings. The largest gains are small weights
(**38.5%**) and heterogeneous groups (**49.7%**); large D2H KV gains are only
**5.9%**. Equivalent asynchronously batched eager Torch remains faster in every
warm case. These are direct transfer API measurements, not full-model or
Inductor results. Existing workload call sites remain individual; grouping is
opt-in through the new API and `WeightFetcher.fetch_many()`.

Implementation: `0404006` plus the GPU-only worker fix in `dbb229c`.
Measurement harness: `dbb229c` (including trace ownership fix `b40c3f2`).
Main: `fe092b1898201c55d7ddb9c61ce984b91294e424`, after merged #234. The immutable
main build was compiled from `b0e8e65`; `git diff b0e8e65..fe092b1 -- libreloc sym`
is empty, so its runtime/compiler sources match merged main. Every raw round
records its loaded Python files, extension/runtime/compiler hashes, source and
runtime revisions, environment and clocks. See [API and lifetime contract](../../../docs/grouped-transfers.md).

Full raw rounds, CUDA timestamp exports, test/build logs and checksums are
preserved in the [original qualification snapshot][raw-evidence]. This tree
keeps the report, environment details and reproducible benchmark scripts;
the generated artifacts are excluded from the implementation PR's file diff.

## Completed warm calls

Three independent process rounds per revision, 100 completed samples per case
and eight warmups. Revision order: main/candidate, candidate/main, main/candidate.
Within each iteration, execution order rotates among individual Sym, grouped
Sym and batched Torch (two paths on main). Input values and scales change before
each iteration; all paths receive the same values. All outputs are checked
exactly, including earlier outputs retained across later calls.

Numbers below are **medians of three per-round p50/p95 statistics**, not pooled
percentiles. Units are milliseconds. Allocation, fresh request preparation,
execution and final completion are timed. Recipe compilation, mutations and
oracles are outside the timer. Both Sym controls use retained owners, one CUDA
queue and eight configured gather threads. Individual layout calls use one
buffer. These explicitly matched settings differ from the public API defaults.

| Case | Main individual p50 / p95 | Candidate individual p50 / p95 | Grouped p50 / p95 | p50 reduction vs main | Batched Torch with candidate p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_h2d_small | 1.7035 / 1.7532 | 1.6945 / 1.7501 | 1.1619 / 1.1941 | 31.8% | 0.5110 / 0.5782 |
| kv_d2h_small | 1.8507 / 1.8865 | 1.8421 / 1.8676 | 1.3769 / 1.4319 | 25.6% | 0.4293 / 0.4723 |
| kv_h2d_large | 5.1304 / 5.3149 | 5.1368 / 5.2898 | 4.3267 / 4.6333 | 15.7% | 3.1745 / 3.3411 |
| kv_d2h_large | 5.0439 / 5.2901 | 5.1531 / 5.4269 | 4.7469 / 4.9121 | 5.9% | 1.5458 / 1.5647 |
| weights_small | 2.0415 / 2.0661 | 2.0708 / 2.1008 | 1.2547 / 1.2775 | 38.5% | 0.6666 / 0.6883 |
| weights_large | 3.9592 / 4.0423 | 3.9669 / 4.1126 | 3.3394 / 3.6389 | 15.7% | 2.7748 / 2.8078 |
| heterogeneous | 2.9660 / 3.2580 | 2.9146 / 3.1207 | 1.4914 / 1.6075 | 49.7% | 0.6938 / 0.7965 |

Candidate individual p50 changes range from −1.7% to +2.2% versus main;
that control does not establish an individual-call optimization or a regression
beyond machine/run variability. The grouped gains also appear against the
individual path in the same candidate process. Raw distributions and both
comparisons are in the archived [comparison.json][comparison].

The large-weight group p95 is 3.984 / 3.639 / 3.558 ms across the three rounds;
main individual p95 is 4.018 / 4.042 / 4.609 ms. Tail latency is more variable
than p50. This does not establish a tail-latency bound or portability across
machines. The small large-KV-D2H improvement also needs to be weighed against
its extra 14 MiB of native scratch capacity.

Torch controls recorded alongside main:

| Case | Batched Torch with main p50 / p95 (ms) |
| --- | ---: |
| kv_h2d_small | 0.5086 / 0.5793 |
| kv_d2h_small | 0.4001 / 0.4321 |
| kv_h2d_large | 3.1772 / 3.3117 |
| kv_d2h_large | 1.4914 / 1.5587 |
| weights_small | 0.6529 / 0.6711 |
| weights_large | 2.7566 / 2.8582 |
| heterogeneous | 0.6884 / 0.8083 |

Cases were defined before acceptance runs:

- Small KV: eight float32 `[16,33,64]` tensors, transpose(0,1), H2D or D2H;
  132 KiB each, 1.03125 MiB logical group payload/output.
- Large KV: four float32 `[16,1024,64]` tensors, transpose(0,1), H2D or D2H;
  4 MiB each, 16 MiB per group.
- Small/large weights: eight int8 `[128,128]` or four int8 `[1024,4096]`
  matrices, CPU float32 per-column scales, dequantize + transpose to GPU float32.
  Both Sym paths select `cuda_dequant_relocate` using the same calibration.
- Heterogeneous: H2D KV `[3,5,64]`, weight `[256,512]`, D2H KV `[16,33,64]`,
  weight `[128,512]`, H2D KV `[8,257,32]`, weight `[512,1024]`, in that order.

Within a case, topology/shapes stay fixed; scales alternate between two exact
value sets. Individual and group recipes have separate template caches. This
is a hot-template experiment, not a shape-churn or context-switching stress
benchmark. Grouped CPU transforms are not separately attributed as overlap
here. There is no measured claim of CPU/DMA or transfer/model-compute overlap.

## Requests, CUDA APIs and link traffic

Separate Nsight captures cover **all seven cases**, five completed groups per
path. `I/G/T` means candidate individual / grouped / batched Torch. Counts are
per whole logical group, not per tensor. Native group counters agree with the
correlated CUDA activity. Link bytes below include parameter uploads and are
identical for grouped Sym and batched Torch.

| Case | Native requests I → G | cudaMemcpyAsync I / G / T | cudaEventSynchronize I / G / T | Native cudaEventRecord I → G | Link bytes G = T |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_h2d_small | 8 → 1 | 8 / 8 / 8 | 8 / 1 / 0 | 16 → 2 | 1,081,344 |
| kv_d2h_small | 8 → 1 | 8 / 8 / 8 | 8 / 1 / 0 | 16 → 2 | 1,081,344 |
| kv_h2d_large | 4 → 1 | 4 / 4 / 4 | 4 / 1 / 0 | 8 → 2 | 16,777,216 |
| kv_d2h_large | 4 → 1 | 4 / 4 / 4 | 4 / 1 / 0 | 8 → 2 | 16,777,216 |
| weights_small | 8 → 1 | 16 / 9 / 9 | 16 / 1 / 0 | 24 → 2 | 131,584 |
| weights_large | 4 → 1 | 8 / 5 / 5 | 8 / 1 / 0 | 12 → 2 | 16,793,600 |
| heterogeneous | 6 → 1 | 9 / 8 / 8 | 9 / 1 / 0 | 15 → 2 | 1,129,216 |

Every path additionally performs **one final `cudaDeviceSynchronize`** inside
the completed-call endpoint. Grouped Sym has one producer `cudaStreamWaitEvent`
and two event records: producer ordering plus completion. Its backend report's
`event_records=1` excludes the producer event managed inside `waitStream`;
`caller_waits=1` reports that ordering separately. These are not two completion
barriers. Event creation/destruction and every other CUDA runtime API are in
the archived `*-cuda.json` files, including Torch allocator events and kernel
activity.

Small weights reduce scale uploads from eight to one (512 bytes); large weights
from four to one (16 KiB); heterogeneous weights from three to two (6 KiB total).
The native cache compares exact uploaded float bytes within this group only.
Torch is given the same sharing opportunity and uploads each shared scale tensor
once. No values are retained between invocations. Kernels remain one per Sym
weight; Torch runs three per weight for cast, multiplication and transpose.

Payloads are not concatenated: **packing bytes are zero for all paths**.
Required CPU transform writes are separately reported, e.g. 1.03125 / 16 MiB for
small/large KV and 402,176 bytes in the heterogeneous group. Those writes are
not packing overhead. Per-member binding and source/parameter checks remain;
one native group preparation/execution boundary amortizes submission and
completion without caching mutable request state.

Trace runs are attribution diagnostics, not extra headline timing rounds.
Output retirement occurs outside the next path's NVTX range, so releasing a
previous Torch pinned output cannot be misattributed to a Sym call. Raw
request-correlated API/memcpy/kernel timestamps and SQLite source hashes are
retained in the [qualification snapshot][raw-evidence] as `*-cuda.json`.
GPU durations overlap host work and are not added to completed latency.
The archived `*-trace-run.json` files preserve the separate profiled runs.

## Memory and bounds

Each Sym owner has 64 MiB retained/live staging limits and 64 MiB retained/live
typed scratch limits. Each group also has an explicit 64 MiB scratch cap; the
stricter owner/group limit wins. Output/source allocations and template/key
metadata are outside those limits. Groups with CPU transformations use up to
seven background workers; direct GPU-only weight groups use zero. All paths
have an eight-thread CPU budget and one execution queue per active owner.

The table reports native retained host/device capacities after the warm
sequence, additional native capacity vs individual, logical output bytes on
both devices, and the Torch allocator's extra **GPU** allocation peak during
one completed call. Values are MiB; all three rounds have identical counters.
For groups, native capacity also equals the reported scratch high-water mark.

| Case | Individual native host / device | Group native host / device | Native difference | Logical outputs, all devices | Torch GPU peak: group / Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_h2d_small | 0.250 / 0.000 | 1.031 / 0.000 | +0.781 | 1.031 | 1.031 / 1.031 |
| kv_d2h_small | 0.250 / 0.000 | 1.031 / 0.000 | +0.781 | 1.031 | 0.000 / 0.129 |
| kv_h2d_large | 4.000 / 0.000 | 18.000 / 0.000 | +14.000 | 16.000 | 16.000 / 16.000 |
| kv_d2h_large | 4.000 / 0.000 | 18.000 / 0.000 | +14.000 | 16.000 | 0.000 / 4.000 |
| weights_small | 0.000 / 0.016 | 0.000 / 0.125 | +0.109 | 0.500 | 0.500 / 0.563 |
| weights_large | 0.016 / 4.516 | 0.016 / 18.016 | +13.500 | 64.000 | 64.000 / 80.016 |
| heterogeneous | 0.506 / 0.627 | 0.389 / 0.693 | -0.050 | 3.134 | 3.005 / 5.011 |

The 0.000 MiB small-weight host cell is 512 bytes, not zero. All prior outputs
remain independent live allocations in every path. Native `cudaMalloc` memory
is **not** included in Torch allocator peaks; do not infer total GPU memory
from the last column alone. CPU output allocations, Torch's pinned-host caching
allocator, thread stacks, metadata and process RSS are not measured by that
column. Native capacity includes idle retained blocks and allocation headroom;
that explains why heterogeneous grouping can retain slightly less memory.

Scratch cannot be recycled until the group completes. This deliberately trades
simultaneous scratch for fewer waits. Oversized groups fail under the cap;
consumers must choose smaller completion boundaries. Tests cover a failure
after the first payload/kernel was submitted, reduced caps with old retained
capacity, and ownership quarantine when completion cannot be proven.

## First completed calls

All three independent observations are preserved below (milliseconds). CUDA is
initialized before timing, and compilation of the symbolic recipes is excluded.
Within each case the order is individual, Torch, group. Template caches are
independent, but CUDA lazy kernel/allocator initialization is shared within the
process. These are first API-call observations, **not fair process-cold startup
comparisons**. In particular, first D2H Torch includes pinned allocator startup;
first large H2D groups allocate more staging than individual calls.
Cases run in the table's order within each process.

| Case | Main individual, rounds 1 / 2 / 3 | Grouped, rounds 1 / 2 / 3 | Torch with candidate, rounds 1 / 2 / 3 |
| --- | ---: | ---: | ---: |
| kv_h2d_small | 4.693 / 4.672 / 4.686 | 4.461 / 4.512 / 4.437 | 1.296 / 1.311 / 1.279 |
| kv_d2h_small | 3.166 / 3.202 / 3.355 | 2.778 / 3.201 / 2.681 | 2.831 / 2.831 / 2.913 |
| kv_h2d_large | 7.354 / 6.877 / 6.967 | 8.192 / 7.536 / 8.439 | 4.071 / 4.002 / 4.638 |
| kv_d2h_large | 5.578 / 5.320 / 5.546 | 6.333 / 5.514 / 5.062 | 17.212 / 17.126 / 17.324 |
| weights_small | 3.632 / 3.585 / 3.658 | 1.817 / 1.840 / 1.833 | 6.671 / 6.567 / 6.569 |
| weights_large | 4.895 / 4.919 / 5.031 | 5.158 / 5.072 / 4.966 | 3.355 / 3.504 / 3.395 |
| heterogeneous | 5.645 / 5.591 / 5.559 | 3.784 / 3.865 / 3.794 | 0.687 / 0.680 / 0.692 |

The raw JSON also contains candidate individual and Torch-with-main first calls.
No cold-start win is inferred from these asymmetric first observations.

## Environment and reproduction

- AMD EPYC 7351; RTX 2080 Ti GPU 0; driver 595.71.05. Affinity `4-7,20-23`,
  NUMA node 1 local to GPU 0. Eight Torch/OMP/MKL threads, one interop thread.
  GPU clocks unlocked; snapshots and topology are recorded.
- CPython 3.14.7; Torch 2.14.0+cu126; native CUDA 12.5.82; GCC 11.4.0;
  Release, native NVTX disabled; Nsight Systems 2024.2.3.38.
- The global CMake CUDA architecture setting is 52, but `reloc_runtime`
  overrides it to **75;89** in both builds. The target setting determines the
  actual runtime fatbin. Full configuration details: [environment.json](environment.json).
- Both paths use `calibration/epyc7351-2080ti.cal`, `pinning="auto"` with no
  configured threshold (pageable staging), identical source precision/layout
  semantics, fresh output allocations and one queue. GPU-only rows upload
  original int8 payloads directly. No source prepacking or value precomputation.
- Timing processes were serialized. No build, test or other benchmark job ran
  concurrently with a headline timing process. Tracing ran after timing.

Candidate build (use the main source/build directories for the main control):

```sh
cmake -S . -B /tmp/sym-220-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/llvm \
  -DSYM_BUILD_TESTS=ON -DSYM_BUILD_BENCHMARKS=OFF -DSYM_BUILD_PYTHON=ON \
  -DRELOC_ENABLE_CUDA=ON -DRELOC_ENABLE_NVTX=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=52 \
  -DPython_EXECUTABLE=/tmp/sym-210-venv/bin/python \
  -Dpybind11_DIR=/tmp/sym-210-venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11
cmake --build /tmp/sym-220-build --target pyreloc_ext pyreloc_transfer_test \
  typed_dispatch_faults reloc-run-artifact libreloc-test sym-reloc-export sym-opt \
  reloc-plan-builder-test -j 8
```

For each build/revision, from this source tree (substitute build and revision):

```sh
export PYTHONPATH=/tmp/sym-220-build/python
export SYM_BUILD=/tmp/sym-220-build
export SYM_RELOC_EXPORT=$SYM_BUILD/sym/tools/sym-reloc-export
export SYM_OPT=$SYM_BUILD/sym/tools/sym-opt
export SYM_RUNTIME_REVISION=dbb229c
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p /tmp/sym220-evidence
taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue220/grouped_transfers.py --samples 100 \
  --output /tmp/sym220-evidence/candidate-round1.json
# Main: use the main build/revision and add --individual-only.
# Repeat three fresh processes each in the revision order above.
python3 bench/issue220/summarize.py /tmp/sym220-evidence
```

CUDA attribution, repeated separately for each of the seven case names:

```sh
/usr/local/cuda-12.5/bin/nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o /tmp/group220-weights \
  taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue220/grouped_transfers.py --case weights_small --samples 20 \
  --trace --output /tmp/sym220-evidence/weights_small-trace-run.json
/usr/local/cuda-12.5/bin/nsys export --type=sqlite --force-overwrite=true \
  --output /tmp/group220-weights.sqlite /tmp/group220-weights.nsys-rep
python3 bench/issue220/analyze_trace.py /tmp/group220-weights.sqlite \
  /tmp/sym220-evidence/weights_small-cuda.json
```

The raw Nsight containers/SQLite files remain outside git. Their correlated
records and hashes are in the [qualification snapshot][raw-evidence], whose
SHA256SUMS covers the original evidence and harnesses.

## Validation

- CUDA Python runtime/frontend suite: **1,025 passed, four expected skips**.
- Workload suite, including real grouped WeightFetcher controls: **53 passed**.
- CUDA native CTest: **6/6 passed**; CPU-only native CTest: **3/3 passed**.
- CPU-only frontend checks: **456 passed, one expected skip, 278 GPU tests
  deselected**; raw native typed Python checks: **10 passed**. These ran before
  the final GPU-only worker change; the CPU build/CTest were rerun after it.
- All benchmark outputs and retained previous outputs passed exact equality
  checks. Tests cover mixed shapes/dtypes/directions, typed partitions and
  special values, nondefault stream ordering, separate and shared-owner
  concurrency, changing/stale parameters/storage, no-worker GPU groups,
  resource caps and partial-copy/event failures with drain and quarantine.
- `clang-format` 21.1.8 and `git diff --check` pass.

Python suites were run separately with `PYTHONPATH` set to the corresponding
staged build, and `SYM_RELOC_EXPORT` / `SYM_OPT` set to the built compiler tools.
Logs, raw rounds, all CUDA attribution records and the machine-readable
comparison remain available in the [qualification snapshot][raw-evidence].

[raw-evidence]: https://github.com/JueonPark/sym/tree/00817b63a139346294a3d9960800fef8008d6b9b/bench/results/grouped-transfers-220
[comparison]: https://github.com/JueonPark/sym/blob/00817b63a139346294a3d9960800fef8008d6b9b/bench/results/grouped-transfers-220/comparison.json
