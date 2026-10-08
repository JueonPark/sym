# Request overhead results — issue #219

The bounded-template change reduces warmed completed-call p50 by **6.8–13.7%**
and p95 by **6.5–13.2%** across these four cases. Instrumented host preparation
falls **8.6–25.0%**. Equivalent eager Torch remains faster in every warmed case.
First-call latency is essentially unchanged; this is not a compilation-speed or
complete-model inference result, and there is no claim of beating Inductor.

The [current-main diagnosis](baseline-diagnosis.md) was committed as `d1db65f`
before the optimization. The implementation and measurement scripts are at
`b0e8e65`; main is `98f5fa1fb0dba82c1339ab8fc23e48e946b122f6` (merged #231).
See [cache keys and validity](../../../docs/request-overhead.md).

## Completed warmed calls

Three fresh process rounds per revision, 100 completed samples per case/round,
eight warmups. Revision order: main/candidate, candidate/main, main/candidate.
Torch and Sym alternate within each round; each pair gets identical changed
inputs/scales. The table reports the median of the three per-round p50/p95
statistics, not a percentile of pooled samples. Units: milliseconds.

| Case | Main Sym p50 / p95 | Candidate Sym p50 / p95 | p50 reduction | Torch with main p50 / p95 | Torch with candidate p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_evict | 0.5503 / 0.5823 | 0.5129 / 0.5447 | 6.8% | 0.1106 / 0.1194 | 0.1098 / 0.1208 |
| kv_restore | 0.5393 / 0.5533 | 0.4843 / 0.4979 | 10.2% | 0.1235 / 0.1913 | 0.1243 / 0.1889 |
| weight_1m | 0.5353 / 0.5487 | 0.4617 / 0.4765 | 13.7% | 0.2626 / 0.2749 | 0.2643 / 0.2760 |
| weight_4m | 1.0073 / 1.0525 | 0.9011 / 0.9372 | 10.5% | 0.6862 / 0.7134 | 0.6782 / 0.7084 |

KV evict: float32 `[16,33,64]` GPU→CPU transpose(0,1); restore:
`[33,16,64]` CPU→GPU transpose(0,1), both 132 KiB. Weight cases: int8
`[1024,1024]` / `[1024,4096]` CPU inputs, float32 per-output-channel scales,
dequantize + transpose to GPU float32 `[out,in]`. All outputs are compared
against the equivalent Torch operation. Previous outputs are retained and
checked after subsequent calls. Mutations/oracles are outside the timer.

Weight shapes are fixed within a case and scales alternate between two exact
value sets. Candidate caches contain two entries (8 KiB / 32 KiB parameter-key
bytes) and report 237 hits / two misses, including warmup, correctness and
profiling calls. This is a hot-cache experiment. Higher shape/parameter churn
can reduce gains, and parameter keys over 256 KiB bypass retention.

## First completed calls

All three independent first-call observations are listed, in round order (ms).
CUDA is initialized before timing; Torch always precedes Sym. Lazy thread-pool,
allocator, compiler and kernel initialization can be asymmetric. These numbers
are not process-cold startup comparisons or evidence of Sym outperforming Torch.
The first KV call includes Dynamo/frontend compilation. Weight recipe compilation
happens before its timer. All cases run in the order shown within each process.

| Case | Main Sym | Candidate Sym | Main Torch | Candidate Torch |
| --- | ---: | ---: | ---: | ---: |
| kv_evict | 265.190 / 263.558 / 263.582 | 262.872 / 264.438 / 262.667 | 20.527 / 20.628 / 20.683 | 21.078 / 20.768 / 20.768 |
| kv_restore | 87.064 / 87.557 / 87.414 | 87.349 / 87.607 / 86.865 | 0.172 / 0.160 / 0.789 | 0.155 / 0.159 / 0.163 |
| weight_1m | 2.327 / 2.423 / 2.301 | 2.363 / 2.401 / 2.338 | 10.133 / 9.824 / 9.838 | 10.188 / 10.136 / 9.785 |
| weight_4m | 1.855 / 1.853 / 1.845 | 1.874 / 1.867 / 1.873 | 1.206 / 1.225 / 1.222 | 1.212 / 1.223 / 1.199 |

## Preparation and exclusive host phases

Separate instrumented passes: 30 calls per case/round. Every call's exclusive
phase sum is asserted equal to its completed host wall time. Preparation is the
per-sample sum of frontend/other validation, binding, descriptor construction,
fresh validation and native preparation. It excludes output allocation, native
execution/completion and the explicit end sync. The following are medians of
per-round means; independent column medians need not sum to the total median.
Do not add these instrumented numbers to headline timings or GPU durations.

| Case | Main preparation | Candidate preparation | Reduction |
| --- | ---: | ---: | ---: |
| kv_evict | 0.4715 | 0.4308 | 8.6% |
| kv_restore | 0.4817 | 0.4191 | 13.0% |
| weight_1m | 0.3258 | 0.2444 | 25.0% |
| weight_4m | 0.4215 | 0.3198 | 24.1% |

| Case / revision | Frontend/other | Binding | Descriptors | Fresh validation | Native preparation | Output allocation | Native incl. completion | Explicit end sync |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kv_evict / main | 0.3255 | 0.0220 | 0.0127 | 0.1010 | 0.0093 | 0.0099 | 0.1215 | 0.0240 |
| kv_evict / candidate | 0.3292 | 0.0101 | 0.0067 | 0.0740 | 0.0104 | 0.0099 | 0.1219 | 0.0240 |
| kv_restore / main | 0.3184 | 0.0223 | 0.0132 | 0.1182 | 0.0097 | 0.0154 | 0.0808 | 0.0232 |
| kv_restore / candidate | 0.3182 | 0.0102 | 0.0066 | 0.0740 | 0.0105 | 0.0153 | 0.0812 | 0.0236 |
| weight_1m / main | 0.1192 | 0.0376 | 0.0072 | 0.1100 | 0.0530 | 0.0132 | 0.2132 | 0.0209 |
| weight_1m / candidate | 0.1262 | 0.0169 | 0.0063 | 0.0866 | 0.0084 | 0.0129 | 0.2110 | 0.0205 |
| weight_4m / main | 0.1532 | 0.0538 | 0.0085 | 0.1353 | 0.0701 | 0.0147 | 0.5665 | 0.0280 |
| weight_4m / candidate | 0.1689 | 0.0222 | 0.0075 | 0.1053 | 0.0148 | 0.0142 | 0.5647 | 0.0284 |

The phase/call-count evidence matches the intended mechanism. Warm KV calls
make no native `bind` call, and source admission/destination description each
run once rather than twice. Warm typed calls make no `bind_typed`,
`prepare_typed_program`, `query_capability` or `select_dispatch` call; one
`prepare_dispatch_from_template` validates fresh views and materializes the
request. Both parameter snapshots/rechecks remain. Native execution and
allocation costs are essentially unchanged in the matched phase measurements.

The remaining frontend, fresh validation, transfer execution and completion
costs explain why this bounded change does not close the gap with Torch.
Layout-native preparation was already small; caching it alone is insufficient.

Separate Nsight captures split synchronization API time out of native execution
without double-counting it:

| Trace (10 calls each) | Native host interval | Of which synchronization APIs |
| --- | ---: | ---: |
| baseline / kv | 0.1130 | 0.0082 |
| baseline / weight | 0.7647 | 0.1942 |
| candidate / kv | 0.1011 | 0.0095 |
| candidate / weight | 0.6539 | 0.1924 |

Trace cases are KV restore and weight_4m. These are separate instrumented
attribution runs, not additional speedup trials. Correlated CUDA API, memcpy
and kernel timestamps are in `*-cuda-phases.json`; GPU work overlaps host
intervals and is **not additive**. SQLite source hashes are recorded. Initial
baseline trace/profile harness revisions have their own recorded hashes.

## Environment, resources and provenance

- AMD EPYC 7351; RTX 2080 Ti GPU 0; driver 595.71.05. CPU affinity 4–7,20–23;
  eight Torch/OMP/MKL threads, one interop thread. GPU clocks unlocked.
- CPython 3.14.7; Torch 2.14.0+cu126. Native toolkit CUDA 12.5.82. Release build,
  CUDA architecture setting 52 in both native builds, native NVTX off. Nsight
  uses Python NVTX ranges and CUDA API tracing.
- Both Sym revisions retain the same `TransferResources` owners and use the
  same execution configuration/budgets. Weights use the unmodified merged
  `WeightFetcher`, one owner per GPU, 64 MiB retained-byte limit. No prepacking,
  output reuse, asynchronous API or reduced correctness checks are introduced.
- Main uses `/home/jueonpark/sym/build/issue210`; candidate uses
  `/tmp/sym-219-build`. Main's runtime/compiler sources are identical between
  `45d6495` (that qualified build) and `98f5fa1`; examples use the merged source.
  Each matched JSON records explicit runtime revision, Python/native/compiler/
  calibration/helper hashes, affinity, software versions and working-tree state.
  The benchmark script's Git HEAD is distinct from the loaded runtime revision.
- `comparison.json` is generated from the six `*-round*.json` files using
  `bench/issue219/summarize.py`. Raw samples, correctness, call counts and resource
  stats remain in those files. `SHA256SUMS` covers committed evidence files.

## Reproduction and validation

Build each revision into a separate directory using the same toolchain and
configuration. Candidate configuration used:

```sh
cmake -S /tmp/sym-219-worktree -B /tmp/sym-219-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/llvm \
  -DSYM_BUILD_TESTS=ON -DSYM_BUILD_BENCHMARKS=OFF -DSYM_BUILD_PYTHON=ON \
  -DRELOC_ENABLE_CUDA=ON -DRELOC_ENABLE_NVTX=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=52 \
  -DPython_EXECUTABLE=/tmp/sym-210-venv/bin/python \
  -Dpybind11_DIR=/tmp/sym-210-venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11
cmake --build /tmp/sym-219-build --target pyreloc_ext pyreloc_transfer_test \
  typed_dispatch_faults reloc-run-artifact libreloc-test reloc-plan-builder-test \
  sym-reloc-export sym-opt -j4
```

From the candidate checkout, use the following in a fresh process for each
variant/round (substitute the build path, revision and output filename for main).
Do not run tests, builds or other GPU jobs concurrently with timings.

```sh
env PYTHONPATH=/tmp/sym-219-build/python SYM_BUILD=/tmp/sym-219-build \
  SYM_RELOC_EXPORT=/tmp/sym-219-build/sym/tools/sym-reloc-export \
  SYM_OPT=/tmp/sym-219-build/sym/tools/sym-opt \
  SYM_RUNTIME_REVISION=b0e8e65 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue219/request_overhead.py \
  --output bench/results/request-overhead-219/candidate-round1.json
/tmp/sym-210-venv/bin/python bench/issue219/summarize.py bench/results/request-overhead-219
```

For CUDA attribution, under the same environment run one case at a time:

```sh
/usr/local/cuda-12.5/bin/nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o /tmp/request219-trace \
  taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue219/request_overhead.py --output /tmp/request219-trace.json \
  --case weight_4m --samples 20 --profile-samples 10 --trace
/usr/local/cuda-12.5/bin/nsys export --type=sqlite --force-overwrite=true \
  --output /tmp/request219-trace.sqlite /tmp/request219-trace.nsys-rep
/tmp/sym-210-venv/bin/python bench/issue219/analyze_trace.py \
  /tmp/request219-trace.sqlite /tmp/request219-cuda-phases.json
```

Validation at implementation revision `b0e8e65`:

- **991 Python tests passed, four expected skips**, including real CUDA,
  changed shapes/parameter bytes, invalid warm-cache inputs, independent outputs,
  concurrent streams/owners, cache eviction/bypass/clear and CPU-only fork-cache
  reset. Run `python -m pytest -q -rs --tb=short libreloc/python/tests` with the
  candidate environment above. See `pytest.log`.
- **Six CTest checks passed**: native runtime (318 passed, two hardware SIMD
  skips), plan builder (61 passed), runtime/tool dependency boundaries and the
  Python transfer harness import. Run `ctest --test-dir /tmp/sym-219-build
  --output-on-failure`. See `ctest.log`.
- Clang-format 21.1.8 and `git diff --check` passed. The test environment required
  `setuptools==84.0.0` for the existing Torch Inductor smoke test; its first run
  failed because this dependency was absent, then the full suite passed after
  installation. No product dependency change was needed.
