# #230: explicit startup preparation

This qualification measures completed transfers and startup, not model throughput.
The new `PreparedTransfers` API compiles expected recipes, explicitly initializes
CUDA, executes one caller-requested warm transfer per example, and retains bounded
native resources. Serving still binds live inputs and allocates independent outputs.
An opt-in disk cache reuses admitted compiler artifacts across processes.

## Measured outcome

Preparation lowers **first serving latency**, while adding a real warm transfer to
deployment work. In initialized processes, uncached Sym first calls changed from
10.730→0.393 ms (small), 13.063→2.376 ms (DLRM), and 18.722→2.619 ms (KV).
Complete Sym preparation cost 11.069/15.548/21.927 ms respectively. The 100-call
totals increased 4.5%/1.6%/2.7%; steady-state medians increased about 1–4%.
The wrapper adds checks/accounting and a serial lock; independent-round ranges
are retained in the CSV. This is a readiness/SLO facility, not a throughput win.

Torch remains faster on these transfers. Prepared Sym warmed medians were
0.234/2.442/2.484 ms versus eager Torch 0.041/1.540/1.665 ms and CPU Inductor
0.158/1.768/1.768 ms. These controls use the same payload, output ownership and
CPU/staging budgets. No model performance improvement is inferred; the broader
[four-model comparison from #229](../issue229/README.md) remains relevant.

### Initialized-process calls and total horizons

All values are milliseconds. `cached*` excludes the separately measured offline
cache-seeding process (about **112 ms** for one recipe); add that cost when it
is incurred by this deployment. Lazy paths put all setup in the first call.

| Case | Path | Prep | First | Warm p50 / p95 | Total 1 | Total 10 | Total 100 |
|---|---|---:|---:|---:|---:|---:|---:|
| small | sym lazy | 0.000 | 10.730 | 0.226 / 0.243 | 11.975 | 14.284 | 34.778 |
| small | sym prepared | 11.069 | 0.393 | 0.234 / 0.251 | 12.796 | 15.026 | 36.328 |
| small | sym cached* | 16.048 | 0.398 | 0.234 / 0.250 | 17.785 | 20.019 | 41.222 |
| small | torch lazy | 0.000 | 2.561 | 0.041 / 0.048 | 2.614 | 3.035 | 6.787 |
| small | torch prepared | 2.593 | 0.057 | 0.041 / 0.048 | 2.699 | 3.109 | 6.854 |
| small | inductor lazy | 0.000 | 8898.907 | 0.163 / 0.174 | 8898.982 | 8900.599 | 8914.950 |
| small | inductor prepared | 8901.703 | 0.267 | 0.158 / 0.173 | 8902.036 | 8903.549 | 8917.832 |
| dlrm | sym lazy | 0.000 | 13.063 | 2.412 / 2.672 | 14.355 | 36.210 | 262.853 |
| dlrm | sym prepared | 15.548 | 2.376 | 2.442 / 2.669 | 19.575 | 42.631 | 267.092 |
| dlrm | sym cached* | 20.652 | 2.634 | 2.449 / 2.621 | 24.578 | 47.214 | 268.885 |
| dlrm | torch lazy | 0.000 | 10.496 | 1.540 / 1.558 | 10.582 | 24.448 | 163.209 |
| dlrm | torch prepared | 10.982 | 1.525 | 1.540 / 1.563 | 12.605 | 26.445 | 165.343 |
| dlrm | inductor lazy | 0.000 | 9272.526 | 1.775 / 1.811 | 9272.650 | 9288.657 | 9449.090 |
| dlrm | inductor prepared | 9260.911 | 1.964 | 1.768 / 1.803 | 9263.133 | 9279.479 | 9438.547 |
| kv | sym lazy | 0.000 | 18.722 | 2.447 / 2.564 | 23.194 | 45.599 | 265.394 |
| kv | sym prepared | 21.927 | 2.619 | 2.484 / 2.583 | 28.498 | 51.155 | 272.661 |
| kv | sym cached* | 26.731 | 2.561 | 2.472 / 2.589 | 33.348 | 56.238 | 281.393 |
| kv | torch lazy | 0.000 | 10.399 | 1.647 / 1.659 | 10.485 | 25.270 | 173.846 |
| kv | torch prepared | 14.312 | 1.756 | 1.665 / 1.678 | 16.155 | 31.123 | 180.960 |
| kv | inductor lazy | 0.000 | 9268.137 | 1.765 / 1.790 | 9268.265 | 9284.140 | 9444.424 |
| kv | inductor prepared | 9261.137 | 1.959 | 1.768 / 1.802 | 9263.211 | 9279.165 | 9439.138 |

### Process-cold startup

CUDA initialization itself took roughly 700 ms under this affinity; warm-call
latency does not include that cost. The following totals include process launch,
imports, fixture creation, preparation, completed transfers and retirement,
excluding inter-call mutation/checking. The full CSV includes all lazy/cached
controls as well. Prepared first-serving calls alone therefore substantially
understate process startup.

| Case | Prepared path | Process total 1 | Process total 10 | Process total 100 |
|---|---|---:|---:|---:|
| small | sym | 2444.481 | 2446.770 | 2467.990 |
| small | torch | 2391.127 | 2391.549 | 2395.304 |
| small | inductor | 11320.043 | 11321.519 | 11335.777 |
| dlrm | sym | 2485.425 | 2507.477 | 2735.896 |
| dlrm | torch | 2445.883 | 2459.722 | 2598.067 |
| dlrm | inductor | 11711.904 | 11727.846 | 11887.405 |
| kv | sym | 2480.835 | 2503.249 | 2730.753 |
| kv | torch | 2426.166 | 2441.118 | 2590.954 |
| kv | inductor | 11687.846 | 11703.636 | 11864.490 |

### Compiler cache: benefit depends on recipe count

These are fresh compiler-only processes, without Torch/CUDA initialization.
Each uses one client for 1, 8, or 32 distinct static transpose/cast recipes.
The client hashes the exporter once and revalidates each cached artifact.
`prepare` includes client construction, hashing, native plan loading/admission
and all compilation/cache I/O. Process totals also include launch/imports.

| Recipes | Uncached prepare | Seed prepare | Hit prepare | Uncached process | Seed process | Hit process |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.330 | 20.332 | 15.678 | 98.761 | 109.014 | 104.714 |
| 8 | 37.565 | 54.959 | 21.305 | 126.038 | 152.741 | 110.105 |
| 32 | 131.789 | 173.775 | 48.878 | 221.059 | 262.744 | 138.065 |

For 32 recipes, hits reduced compiler preparation by 63% and process startup by
38%, but seeding cost 263 ms as a separate process. Including that offline seed,
four future processes are needed to amortize it against the measured 83 ms
per-process saving. If the seeding process itself serves as the first useful
preparation, the second process already amortizes the extra seeding overhead.
For one recipe, hashing/admission costs more than recompiling; persistence stays
opt-in. These are distinct recipes, not a requirement to compile every concrete
shape: a symbolic recipe can already cover multiple shapes.

### Preparation phases and memory

Uncached Sym, initialized process, medians in milliseconds:

| Case | Compiler | CUDA initialization | Example/resource warming | Retirement |
|---|---:|---:|---:|---:|
| small | 6.123 | 0.085 | 4.422 | 1.308 |
| dlrm | 6.214 | 0.073 | 8.786 | 1.288 |
| kv | 6.127 | 0.069 | 15.208 | 4.072 |

Owner creation cost 0.087–0.118 ms. The complete preparation envelope above also
includes client construction, diagnostic snapshots and Python orchestration.
First real and subsequent call times are recorded separately. Warming includes
one real transfer; these phase timers are not a GPU/CPU critical-path breakdown.

| Case | Sym retained native scratch | Torch/Inductor retained wire | Torch GPU reservation after warm | Sym / Torch / Inductor peak RSS after prep |
|---|---:|---:|---:|---:|
| small | 0.031 MiB | 0.031 MiB | 2 MiB | 611.2 / 608.1 / 879.8 MiB |
| dlrm | 2.000 MiB | 6.500 MiB | 22 MiB | 631.8 / 634.3 / 908.2 MiB |
| kv | 8.000 MiB | 8.000 MiB | 22 MiB | 634.1 / 630.8 / 904.2 MiB |

All native scratch was released on close; no quarantine occurred. In initialized
processes, Torch retained 2/22/22 MiB of GPU allocator reservations after close, although live GPU output
allocations were released. RSS is a process high-water mark including imports,
CUDA and compiler state, **not** an estimate of live native scratch or current
RSS after retirement. Caller-held outputs and framework/driver caches are outside
native resource quotas. The raw snapshots preserve before/after-close counters.

Validation: 1,235 Python tests passed (four skips), 58 workload tests, all six
CTest targets and 41 compiler tests. The final focused 21-test rerun also passed.
Across 126 transfer processes, all 12,600 completed calls and 25,074 result/storage
checks passed, with no foreign build/GPU activity detected. Cache tests cover
corruption, recipe/direction/schema/namespace changes, rebuilt binaries including
same-size/restored-mtime changes, bounds, failed compiles, concurrent processes,
and Torch-free construction. GPU tests cover changing sources/scales/indices,
independent outputs, D2H, shape changes, quotas, retirement and fork rejection.

## Protocol and provenance

- Baseline main: `14cd89f284973ed5852c3b3ecd0edd70a4d6e27b` (#244).
  Measured implementation: `fe50c569c59440ee2085a393dc5139f008295229`.
  Native/exporter sources are unchanged; all paths use the same frozen #229
  Release CUDA build (`/tmp/sym-229-build`, NVTX off) and candidate Python package
  copied to `/tmp/sym-230-python`. The raw samples record the loaded runtime,
  exporter and every frontend file's SHA-256. Later edits add tests/evidence only.
- EPYC 7351, RTX 2080 Ti GPU 0, driver 595.71.05, CPU affinity
  `4-7,20-23` (GPU-local NUMA node 1; four physical cores plus SMT).
  CPython 3.14.7, Torch 2.14.0+cu126, CUDA toolkit 12.5, GNU C++ 11.4,
  Release `-O3`, CUDA architectures 75/89. GPU clocks are unlocked.
- Three independent rounds, rotating seven paths. Each case/path has its own
  fresh process and fresh Inductor cache. A process-tree-aware monitor rejects
  unrelated compiler/packaging and GPU jobs. Cases and paths run serially.
- Two startup regimes: **cold** starts without initializing CUDA; **initialized**
  first initializes CUDA with one unrelated one-element allocation, measured
  separately. Neither means a cold filesystem cache or machine reboot.
- Cases: small `(128,128)` FP32→FP16 (32 KiB wire); DLRM-like
  `(26,2048,64)`→`(2048,26,64)` FP32→FP16 (6.5 MiB wire); KV restore
  `(2048,16,64)`→`(16,2048,64)` FP32 (8 MiB wire). CPU inputs are pageable.
- Eight Torch/OMP threads, one interop/compiler thread, passive OMP waits.
  Sym has two buffers, one stream, seven background workers, pinned staging,
  64 MiB layout and typed live/retained budgets, 64 MiB maximum fresh output.
  Torch and Inductor use a retained pinned destination-format host buffer and
  fresh GPU output; Inductor compiles the CPU transform directly into that buffer.
  All fit the same 64 MiB staging budget, though actual staging sizes differ.
- `lazy` defers compiler/resource/CUDA setup until the first timed call.
  `prepared` does one explicit completed warm call before readiness.
  `sym_cached` is prepared with a separately primed compiler cache. Seeding is
  recorded separately and must be amortized. Torch/Inductor get the same single
  warm-call opportunity; no path receives immutable input prepacking.
- Each process measures 100 completed calls. Timers include binding, fresh output,
  transformation, DMA and CUDA synchronization. Inputs change before every call;
  exact results and preserved first-output storage are checked outside timers.
  Every warmed distribution excludes the first real call, leaving 99 samples.
  Tables use medians of three per-process statistics, not pooled percentiles.
- `total N` = complete preparation + first N completed calls + retirement.
  It excludes fixture creation, common imports, input mutation and correctness
  checks. Process totals add launch-to-script time and readiness setup, including
  imports, fixtures and any initialized-process prelude. They still exclude
  inter-call benchmark checks, and are not wall-clock model latency.
- Retirement closes Sym resources or releases the Torch wire/compiler references
  and synchronizes. Framework/driver caches can outlive those objects; their GPU
  reservation and process RSS high-water marks are recorded separately. Native
  stats give actual retained scratch, not an inferred RSS difference.

## Reproduce

Use a Release CUDA build of this commit and the repository's qualified Python/Torch
installation. No log files, compiler caches or profiler databases belong in the PR.

```sh
export PYTHONPATH=/path/to/build/python
export LD_LIBRARY_PATH=/path/to/build/libreloc
export SYM_RELOC_EXPORT=/path/to/build/sym/tools/sym-reloc-export
export SYM_OPT=/path/to/build/sym/tools/sym-opt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1
python bench/issue230/run.py --output /tmp/issue230-runs
python bench/issue230/compiler_cache.py --output /tmp/issue230-runs/compilers
python bench/issue230/summarize.py --input /tmp/issue230-runs --output bench/issue230/results
```

`samples.json.gz` preserves all scalar samples, memory/capacity snapshots,
phase timers, isolation checks and compiler-cache experiments; `summary.csv`
has each median and independent-round min/max. `manifest.json` hashes both files.
The runtime/exporter are unchanged from #229; this is a frontend-only change.
