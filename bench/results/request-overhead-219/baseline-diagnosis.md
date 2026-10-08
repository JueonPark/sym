# Current-main diagnosis before optimization (#219)

Source baseline: `98f5fa1fb0dba82c1339ab8fc23e48e946b122f6` (after #231). Compiler/runtime files match the existing main-qualified Release/CUDA, NVTX-off build at `build/issue210`; exact loaded Python files and native extension/compiler hashes are in the reports.

EPYC 7351, RTX 2080 Ti GPU 0, affinity 4–7,20–23, eight Torch threads and one interop thread, unlocked clocks. The script checks fresh changing inputs/scales and independent outputs. These fixed-shape completed-call diagnostics exclude model compute, oracle checks and final owner close.

## Uninstrumented discovery run

| Case | Sym p50 / p95 ms | Torch p50 / p95 ms |
| --- | ---: | ---: |
| kv_evict | 0.5495 / 0.6467 | 0.1104 / 0.1210 |
| kv_restore | 0.5397 / 0.5545 | 0.1187 / 0.1904 |
| weight_1m | 0.5350 / 0.5502 | 0.2636 / 0.2746 |
| weight_4m | 0.9929 / 1.0257 | 0.6832 / 0.7197 |

First-call and all raw samples are in `baseline-profile.json`. First frontend KV calls include graph compilation; weights compile their recipe before timing. Torch precedes Sym for first calls; these are initialized-process measurements, not a process-cold CUDA comparison.

## Exclusive host phase means (separate instrumented pass)

| Case | Frontend/other | Binding | Descriptors | Fresh validation | Native preparation | Output allocation | Native incl. completion | Explicit end sync |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kv_evict | 0.3271 | 0.0221 | 0.0129 | 0.1026 | 0.0095 | 0.0102 | 0.1225 | 0.0239 |
| kv_restore | 0.3192 | 0.0228 | 0.0133 | 0.1206 | 0.0093 | 0.0151 | 0.0818 | 0.0229 |
| weight_1m | 0.1188 | 0.0377 | 0.0070 | 0.1087 | 0.0547 | 0.0137 | 0.2117 | 0.0204 |
| weight_4m | 0.1519 | 0.0546 | 0.0091 | 0.1349 | 0.0706 | 0.0149 | 0.5629 | 0.0280 |

Units are milliseconds. Wrappers exclusively partition each host call and assert that their sum equals its completed wall time. Profiling adds overhead; these means are not the headline timing samples. cProfile text files provide function-level attribution; cumulative rows overlap and must not be summed.

## Completion within native execution

Separate Nsight CUDA/NVTX captures retain correlated per-request CUDA API and GPU timestamps in `baseline-*-cuda-phases.json`. Mean native host interval: KV restore 0.1130 ms, including 0.0082 ms in synchronization APIs; 4 MiB weight 0.7647 ms, including 0.1942 ms in synchronization APIs. GPU copies and kernels overlap those host intervals. Do not add these traced numbers to the instrumented or uninstrumented run above.

## Optimization selected from this evidence

- Reuse immutable typed bindings, checked programs and capability/selection templates with bounded, per-artifact keys covering symbols, exact current parameter bytes/descriptors, direction, device, policy, calibration identity and selection threads. Create a fresh executable request and output every call.
- Consolidate capability/selection and avoid reselecting an already qualified immutable template after output allocation; still prove both fresh buffer views.
- Cache immutable layout bindings and parameter extents by shape. Native layout preparation is small; do not promise large gains from this cache alone.
- Reduce frontend admission overhead and repeated metadata construction while retaining fresh source metadata, allocation span, gradient-state, parameter values and caller-stream checks.

Compare the result against this current-main runtime and equivalent eager Torch in fresh matched rounds, reporting first call, warm p50/p95 and preparation phases separately. Runtime resource retention stays enabled in both Sym revisions. This issue does not alter surrounding model compute, prepacking, batching or asynchronous execution.
