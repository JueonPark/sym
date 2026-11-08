# Issue 268 qualification tables

Values are medians of three independent round-level statistics unless stated otherwise. No rounded value is used for qualification. The public Wiki package retains every per-round result and raw sample.

E2 latency is a completed request. E5 p50/p95 is completed eight-request burst time divided by eight; it is an amortized burst cost, not per-request service latency or sustained server throughput. E5 worker submission, queue, service and sojourn are reported separately.

`plain`: original callable; `off`: identical static-slot input/output copies with ordinary kernels; `on`: replay enabled, with explicit fallback when a shape or budget is not admitted.

## Warm latency and replay activity

| kind | case | compute | callers | runtime | mode | success rounds | p50 ms | p95 ms | warm graph launches/round | warm fallbacks/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2 | llm_decode | eager | 1 | sym | off | 3/3 | 26.904 | 27.871 | 0 | 0 |
| e2 | llm_decode | eager | 1 | sym | on | 3/3 | 19.124 | 19.373 | 1440 | 0 |
| e2 | llm_decode | eager | 1 | sym | plain | 3/3 | 19.099 | 19.330 | 0 | 0 |
| e2 | llm_decode | eager | 1 | torch | off | 3/3 | 27.589 | 28.177 | 0 | 0 |
| e2 | llm_decode | eager | 1 | torch | on | 3/3 | 17.687 | 18.034 | 1440 | 0 |
| e2 | llm_decode | eager | 1 | torch | plain | 3/3 | 19.680 | 19.791 | 0 | 0 |
| e2 | llm_decode | inductor | 1 | sym | off | 3/3 | 26.036 | 26.805 | 0 | 0 |
| e2 | llm_decode | inductor | 1 | sym | on | 3/3 | 19.171 | 19.441 | 1440 | 0 |
| e2 | llm_decode | inductor | 1 | sym | plain | 3/3 | 18.289 | 18.551 | 0 | 0 |
| e2 | llm_decode | inductor | 1 | torch | off | 3/3 | 26.368 | 27.091 | 0 | 0 |
| e2 | llm_decode | inductor | 1 | torch | on | 3/3 | 17.686 | 18.254 | 1440 | 0 |
| e2 | llm_decode | inductor | 1 | torch | plain | 3/3 | 18.038 | 18.359 | 0 | 0 |
| e2 | llm_prefill | eager | 1 | sym | off | 3/3 | 6.901 | 7.530 | 0 | 0 |
| e2 | llm_prefill | eager | 1 | sym | on | 3/3 | 5.085 | 5.221 | 360 | 0 |
| e2 | llm_prefill | eager | 1 | sym | plain | 3/3 | 5.196 | 5.442 | 0 | 0 |
| e2 | llm_prefill | eager | 1 | torch | off | 3/3 | 7.216 | 7.948 | 0 | 0 |
| e2 | llm_prefill | eager | 1 | torch | on | 3/3 | 5.020 | 5.113 | 360 | 0 |
| e2 | llm_prefill | eager | 1 | torch | plain | 3/3 | 5.165 | 5.673 | 0 | 0 |
| e2 | llm_prefill | inductor | 1 | sym | off | 3/3 | 6.448 | 6.779 | 0 | 0 |
| e2 | llm_prefill | inductor | 1 | sym | on | 3/3 | 5.161 | 5.286 | 360 | 0 |
| e2 | llm_prefill | inductor | 1 | sym | plain | 3/3 | 5.013 | 5.134 | 0 | 0 |
| e2 | llm_prefill | inductor | 1 | torch | off | 3/3 | 6.786 | 7.540 | 0 | 0 |
| e2 | llm_prefill | inductor | 1 | torch | on | 3/3 | 5.080 | 5.222 | 360 | 0 |
| e2 | llm_prefill | inductor | 1 | torch | plain | 3/3 | 4.948 | 5.169 | 0 | 0 |
| e2 | moe_dense | eager | 1 | sym | off | 3/3 | 14.415 | 15.876 | 0 | 420 |
| e2 | moe_dense | eager | 1 | sym | on | 3/3 | 12.860 | 14.217 | 660 | 420 |
| e2 | moe_dense | eager | 1 | sym | plain | 3/3 | 9.785 | 10.804 | 0 | 0 |
| e2 | moe_dense | eager | 1 | torch | off | 3/3 | 14.478 | 15.401 | 0 | 420 |
| e2 | moe_dense | eager | 1 | torch | on | 3/3 | 12.984 | 14.151 | 660 | 420 |
| e2 | moe_dense | eager | 1 | torch | plain | 3/3 | 10.659 | 11.013 | 0 | 0 |
| e2 | moe_dense | inductor | 1 | sym | off | 3/3 | 18.086 | 18.717 | 0 | 420 |
| e2 | moe_dense | inductor | 1 | sym | on | 3/3 | 15.274 | 15.911 | 660 | 420 |
| e2 | moe_dense | inductor | 1 | sym | plain | 3/3 | 13.132 | 13.322 | 0 | 0 |
| e2 | moe_dense | inductor | 1 | torch | off | 3/3 | 18.085 | 20.032 | 0 | 420 |
| e2 | moe_dense | inductor | 1 | torch | on | 3/3 | 15.231 | 15.979 | 660 | 420 |
| e2 | moe_dense | inductor | 1 | torch | plain | 3/3 | 13.269 | 14.754 | 0 | 0 |
| e2 | moe_sparse | eager | 1 | sym | off | 3/3 | 4.560 | 4.964 | 0 | 0 |
| e2 | moe_sparse | eager | 1 | sym | on | 3/3 | 3.695 | 3.803 | 360 | 0 |
| e2 | moe_sparse | eager | 1 | sym | plain | 3/3 | 2.904 | 2.986 | 0 | 0 |
| e2 | moe_sparse | eager | 1 | torch | off | 3/3 | 4.470 | 4.550 | 0 | 0 |
| e2 | moe_sparse | eager | 1 | torch | on | 3/3 | 3.709 | 3.819 | 360 | 0 |
| e2 | moe_sparse | eager | 1 | torch | plain | 3/3 | 2.775 | 3.021 | 0 | 0 |
| e2 | moe_sparse | inductor | 1 | sym | off | 3/3 | 5.670 | 5.771 | 0 | 0 |
| e2 | moe_sparse | inductor | 1 | sym | on | 3/3 | 4.143 | 4.238 | 360 | 0 |
| e2 | moe_sparse | inductor | 1 | sym | plain | 3/3 | 3.856 | 3.937 | 0 | 0 |
| e2 | moe_sparse | inductor | 1 | torch | off | 3/3 | 5.673 | 5.824 | 0 | 0 |
| e2 | moe_sparse | inductor | 1 | torch | on | 3/3 | 4.084 | 4.196 | 360 | 0 |
| e2 | moe_sparse | inductor | 1 | torch | plain | 3/3 | 3.779 | 3.889 | 0 | 0 |
| e5 | llm_decode | eager | 1 | sym | off | 3/3 | 26.361 | 26.982 | 0 | 1200 |
| e5 | llm_decode | eager | 1 | sym | on | 3/3 | 20.682 | 21.216 | 4320 | 1440 |
| e5 | llm_decode | eager | 1 | sym | plain | 3/3 | 19.295 | 19.786 | 0 | 0 |
| e5 | llm_decode | eager | 1 | torch | off | 3/3 | 27.016 | 27.566 | 0 | 1200 |
| e5 | llm_decode | eager | 1 | torch | on | 3/3 | 19.477 | 20.009 | 4320 | 1440 |
| e5 | llm_decode | eager | 1 | torch | plain | 3/3 | 19.535 | 21.003 | 0 | 0 |
| e5 | llm_decode | eager | 2 | sym | off | 3/3 | 23.715 | 25.135 | 0 | 4320 |
| e5 | llm_decode | eager | 2 | sym | on | 3/3 | 22.217 | 23.348 | 1440 | 4320 |
| e5 | llm_decode | eager | 2 | sym | plain | 3/3 | 17.691 | 18.447 | 0 | 0 |
| e5 | llm_decode | eager | 2 | torch | off | 3/3 | 25.142 | 26.466 | 0 | 4320 |
| e5 | llm_decode | eager | 2 | torch | on | 3/3 | 22.150 | 24.499 | 1440 | 4320 |
| e5 | llm_decode | eager | 2 | torch | plain | 3/3 | 17.694 | 19.065 | 0 | 0 |
| e5 | llm_decode | eager | 4 | sym | off | 0/3 | — | — | — | — |
| e5 | llm_decode | eager | 4 | sym | on | 3/3 | 27.053 | 28.121 | 0 | 5760 |
| e5 | llm_decode | eager | 4 | sym | plain | 3/3 | 19.637 | 20.513 | 0 | 0 |
| e5 | llm_decode | eager | 4 | torch | off | 0/3 | — | — | — | — |
| e5 | llm_decode | eager | 4 | torch | on | 3/3 | 27.993 | 29.635 | 0 | 5760 |
| e5 | llm_decode | eager | 4 | torch | plain | 3/3 | 20.464 | 21.494 | 0 | 0 |
| e5 | llm_decode | inductor | 1 | sym | off | 3/3 | 25.432 | 25.710 | 0 | 1200 |
| e5 | llm_decode | inductor | 1 | sym | on | 3/3 | 20.554 | 20.985 | 4320 | 1440 |
| e5 | llm_decode | inductor | 1 | sym | plain | 3/3 | 18.475 | 19.721 | 0 | 0 |
| e5 | llm_decode | inductor | 1 | torch | off | 3/3 | 25.884 | 28.842 | 0 | 1200 |
| e5 | llm_decode | inductor | 1 | torch | on | 3/3 | 19.126 | 19.467 | 4320 | 1440 |
| e5 | llm_decode | inductor | 1 | torch | plain | 3/3 | 18.134 | 18.357 | 0 | 0 |
| e5 | moe_sparse | eager | 1 | sym | off | 3/3 | 4.525 | 4.910 | 0 | 0 |
| e5 | moe_sparse | eager | 1 | sym | on | 3/3 | 3.791 | 4.131 | 1440 | 0 |
| e5 | moe_sparse | eager | 1 | sym | plain | 3/3 | 2.956 | 2.994 | 0 | 0 |
| e5 | moe_sparse | eager | 1 | torch | off | 3/3 | 4.598 | 4.928 | 0 | 0 |
| e5 | moe_sparse | eager | 1 | torch | on | 3/3 | 3.754 | 3.813 | 1440 | 0 |
| e5 | moe_sparse | eager | 1 | torch | plain | 3/3 | 2.840 | 2.871 | 0 | 0 |
| e5 | moe_sparse | eager | 2 | sym | off | 3/3 | 4.091 | 4.376 | 0 | 0 |
| e5 | moe_sparse | eager | 2 | sym | on | 3/3 | 3.589 | 4.115 | 1440 | 0 |
| e5 | moe_sparse | eager | 2 | sym | plain | 3/3 | 2.460 | 2.588 | 0 | 0 |
| e5 | moe_sparse | eager | 2 | torch | off | 3/3 | 4.154 | 4.387 | 0 | 0 |
| e5 | moe_sparse | eager | 2 | torch | on | 3/3 | 3.701 | 3.986 | 1440 | 0 |
| e5 | moe_sparse | eager | 2 | torch | plain | 3/3 | 2.397 | 2.446 | 0 | 0 |
| e5 | moe_sparse | eager | 4 | sym | off | 3/3 | 4.402 | 4.759 | 0 | 0 |
| e5 | moe_sparse | eager | 4 | sym | on | 3/3 | 3.927 | 4.100 | 1440 | 0 |
| e5 | moe_sparse | eager | 4 | sym | plain | 3/3 | 2.310 | 2.366 | 0 | 0 |
| e5 | moe_sparse | eager | 4 | torch | off | 3/3 | 4.548 | 4.879 | 0 | 0 |
| e5 | moe_sparse | eager | 4 | torch | on | 3/3 | 4.060 | 4.363 | 1440 | 0 |
| e5 | moe_sparse | eager | 4 | torch | plain | 3/3 | 2.431 | 2.498 | 0 | 0 |
| e5 | moe_sparse | inductor | 1 | sym | off | 3/3 | 5.805 | 5.928 | 0 | 0 |
| e5 | moe_sparse | inductor | 1 | sym | on | 3/3 | 4.162 | 4.308 | 1440 | 0 |
| e5 | moe_sparse | inductor | 1 | sym | plain | 3/3 | 3.876 | 3.915 | 0 | 0 |
| e5 | moe_sparse | inductor | 1 | torch | off | 3/3 | 5.620 | 5.723 | 0 | 0 |
| e5 | moe_sparse | inductor | 1 | torch | on | 3/3 | 4.181 | 4.494 | 1440 | 0 |
| e5 | moe_sparse | inductor | 1 | torch | plain | 3/3 | 3.841 | 3.917 | 0 | 0 |

## Submission and concurrency

E2 submit includes host routing synchronizations inside a request but excludes its final completion fence. E5 enqueue is only the host queue submission cost divided by eight. Worker submit measures GPU submission inside one request. These distinct timings are not additive phases.

| kind | case | compute | callers | runtime | mode | E2 submit / E5 enqueue ms | worker submit p50 ms | worker submit p95 ms | queue p95 ms | service p95 ms | sojourn p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2 | llm_decode | eager | 1 | sym | off | 26.893 | — | — | — | — | — |
| e2 | llm_decode | eager | 1 | sym | on | 19.114 | — | — | — | — | — |
| e2 | llm_decode | eager | 1 | sym | plain | 19.087 | — | — | — | — | — |
| e2 | llm_decode | eager | 1 | torch | off | 27.578 | — | — | — | — | — |
| e2 | llm_decode | eager | 1 | torch | on | 17.677 | — | — | — | — | — |
| e2 | llm_decode | eager | 1 | torch | plain | 19.663 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | sym | off | 26.022 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | sym | on | 19.161 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | sym | plain | 18.278 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | torch | off | 26.356 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | torch | on | 17.674 | — | — | — | — | — |
| e2 | llm_decode | inductor | 1 | torch | plain | 18.026 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | sym | off | 6.890 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | sym | on | 5.075 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | sym | plain | 5.185 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | torch | off | 7.205 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | torch | on | 5.010 | — | — | — | — | — |
| e2 | llm_prefill | eager | 1 | torch | plain | 5.154 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | sym | off | 6.433 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | sym | on | 5.150 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | sym | plain | 5.002 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | torch | off | 6.770 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | torch | on | 5.068 | — | — | — | — | — |
| e2 | llm_prefill | inductor | 1 | torch | plain | 4.937 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | sym | off | 14.403 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | sym | on | 12.848 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | sym | plain | 9.773 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | torch | off | 14.459 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | torch | on | 12.972 | — | — | — | — | — |
| e2 | moe_dense | eager | 1 | torch | plain | 10.645 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | sym | off | 18.059 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | sym | on | 15.260 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | sym | plain | 13.119 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | torch | off | 18.072 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | torch | on | 15.218 | — | — | — | — | — |
| e2 | moe_dense | inductor | 1 | torch | plain | 13.257 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | sym | off | 4.550 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | sym | on | 3.685 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | sym | plain | 2.894 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | torch | off | 4.459 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | torch | on | 3.699 | — | — | — | — | — |
| e2 | moe_sparse | eager | 1 | torch | plain | 2.765 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | sym | off | 5.651 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | sym | on | 4.132 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | sym | plain | 3.845 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | torch | off | 5.662 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | torch | on | 4.073 | — | — | — | — | — |
| e2 | moe_sparse | inductor | 1 | torch | plain | 3.768 | — | — | — | — | — |
| e5 | llm_decode | eager | 1 | sym | off | 0.010 | 26.167 | 26.901 | 184.659 | 26.912 | 210.908 |
| e5 | llm_decode | eager | 1 | sym | on | 0.010 | 20.547 | 21.015 | 144.812 | 21.026 | 165.435 |
| e5 | llm_decode | eager | 1 | sym | plain | 0.010 | 19.149 | 19.425 | 135.061 | 19.440 | 154.240 |
| e5 | llm_decode | eager | 1 | torch | off | 0.010 | 26.872 | 27.484 | 189.213 | 27.495 | 216.176 |
| e5 | llm_decode | eager | 1 | torch | on | 0.010 | 19.359 | 19.715 | 136.182 | 19.725 | 155.725 |
| e5 | llm_decode | eager | 1 | torch | plain | 0.010 | 19.443 | 21.346 | 137.560 | 21.356 | 157.227 |
| e5 | llm_decode | eager | 2 | sym | off | 0.011 | 46.476 | 52.273 | 148.703 | 52.290 | 196.547 |
| e5 | llm_decode | eager | 2 | sym | on | 0.012 | 43.182 | 49.303 | 143.081 | 49.323 | 180.596 |
| e5 | llm_decode | eager | 2 | sym | plain | 0.011 | 34.366 | 38.453 | 112.039 | 38.470 | 143.527 |
| e5 | llm_decode | eager | 2 | torch | off | 0.012 | 48.770 | 55.226 | 159.787 | 55.242 | 204.971 |
| e5 | llm_decode | eager | 2 | torch | on | 0.011 | 44.015 | 50.622 | 135.795 | 50.639 | 179.601 |
| e5 | llm_decode | eager | 2 | torch | plain | 0.012 | 35.167 | 40.097 | 110.382 | 40.114 | 146.896 |
| e5 | llm_decode | eager | 4 | sym | off | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | sym | on | 0.013 | 103.856 | 122.873 | 123.373 | 122.944 | 216.784 |
| e5 | llm_decode | eager | 4 | sym | plain | 0.012 | 76.030 | 87.390 | 87.850 | 87.450 | 159.453 |
| e5 | llm_decode | eager | 4 | torch | off | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | torch | on | 0.013 | 107.900 | 124.047 | 124.516 | 124.144 | 224.565 |
| e5 | llm_decode | eager | 4 | torch | plain | 0.012 | 79.484 | 90.251 | 90.684 | 90.316 | 165.707 |
| e5 | llm_decode | inductor | 1 | sym | off | 0.011 | 25.247 | 26.135 | 178.096 | 26.147 | 203.448 |
| e5 | llm_decode | inductor | 1 | sym | on | 0.011 | 20.466 | 21.174 | 143.918 | 21.186 | 164.379 |
| e5 | llm_decode | inductor | 1 | sym | plain | 0.011 | 18.372 | 19.576 | 129.395 | 19.588 | 147.976 |
| e5 | llm_decode | inductor | 1 | torch | off | 0.011 | 25.700 | 28.757 | 181.322 | 28.769 | 207.084 |
| e5 | llm_decode | inductor | 1 | torch | on | 0.011 | 18.918 | 19.648 | 133.845 | 19.660 | 152.828 |
| e5 | llm_decode | inductor | 1 | torch | plain | 0.011 | 18.058 | 18.201 | 126.926 | 18.211 | 145.085 |
| e5 | moe_sparse | eager | 1 | sym | off | 0.010 | 4.434 | 4.512 | 31.573 | 4.523 | 36.079 |
| e5 | moe_sparse | eager | 1 | sym | on | 0.010 | 3.710 | 3.797 | 26.461 | 3.808 | 30.221 |
| e5 | moe_sparse | eager | 1 | sym | plain | 0.010 | 2.875 | 2.982 | 20.596 | 2.993 | 23.518 |
| e5 | moe_sparse | eager | 1 | torch | off | 0.010 | 4.498 | 4.604 | 32.074 | 4.615 | 36.626 |
| e5 | moe_sparse | eager | 1 | torch | on | 0.011 | 3.691 | 3.793 | 26.270 | 3.804 | 29.959 |
| e5 | moe_sparse | eager | 1 | torch | plain | 0.009 | 2.757 | 2.827 | 19.840 | 2.837 | 22.625 |
| e5 | moe_sparse | eager | 2 | sym | off | 0.011 | 7.933 | 8.588 | 24.959 | 8.603 | 32.787 |
| e5 | moe_sparse | eager | 2 | sym | on | 0.010 | 7.012 | 7.908 | 22.056 | 7.950 | 29.050 |
| e5 | moe_sparse | eager | 2 | sym | plain | 0.010 | 4.660 | 5.154 | 15.371 | 5.167 | 19.720 |
| e5 | moe_sparse | eager | 2 | torch | off | 0.010 | 8.109 | 8.646 | 25.310 | 8.660 | 33.388 |
| e5 | moe_sparse | eager | 2 | torch | on | 0.011 | 7.149 | 7.746 | 22.540 | 7.760 | 29.707 |
| e5 | moe_sparse | eager | 2 | torch | plain | 0.010 | 4.638 | 4.882 | 14.447 | 4.896 | 19.149 |
| e5 | moe_sparse | eager | 4 | sym | off | 0.012 | 16.721 | 19.065 | 19.649 | 19.091 | 35.545 |
| e5 | moe_sparse | eager | 4 | sym | on | 0.012 | 15.005 | 16.970 | 17.544 | 16.987 | 31.736 |
| e5 | moe_sparse | eager | 4 | sym | plain | 0.012 | 8.708 | 9.916 | 10.165 | 9.973 | 18.439 |
| e5 | moe_sparse | eager | 4 | torch | off | 0.012 | 17.363 | 19.459 | 20.127 | 19.588 | 36.540 |
| e5 | moe_sparse | eager | 4 | torch | on | 0.012 | 15.482 | 17.521 | 18.342 | 17.544 | 33.006 |
| e5 | moe_sparse | eager | 4 | torch | plain | 0.011 | 9.295 | 10.143 | 10.491 | 10.176 | 19.532 |
| e5 | moe_sparse | inductor | 1 | sym | off | 0.010 | 5.691 | 5.906 | 40.527 | 5.919 | 46.242 |
| e5 | moe_sparse | inductor | 1 | sym | on | 0.010 | 4.066 | 4.211 | 29.034 | 4.223 | 33.135 |
| e5 | moe_sparse | inductor | 1 | sym | plain | 0.010 | 3.773 | 3.899 | 27.011 | 3.917 | 30.833 |
| e5 | moe_sparse | inductor | 1 | torch | off | 0.010 | 5.512 | 5.643 | 39.220 | 5.655 | 44.807 |
| e5 | moe_sparse | inductor | 1 | torch | on | 0.010 | 4.098 | 4.418 | 29.521 | 4.429 | 33.620 |
| e5 | moe_sparse | inductor | 1 | torch | plain | 0.010 | 3.746 | 3.886 | 26.757 | 3.897 | 30.547 |

## Cold costs and reuse totals

All costs are milliseconds. `setup` is preparation + sum of first calls + sum of warmup costs; `Total(N) = setup + N × warm p50` after that preparation. First/warmup costs sum callers. Capture and region warmup are owner counters contained inside model first/warmup; never add them again. Common fixture/oracle creation precedes deployment timing.

| kind | case | compute | callers | runtime | mode | prepare | first | warmup | capture | region warmup | retire | Total(100) | Total(1000) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2 | llm_decode | eager | 1 | sym | off | 130.141 | 118.207 | 234.937 | 0.000 | 20.186 | 2.978 | 3174.315 | 27387.057 |
| e2 | llm_decode | eager | 1 | sym | on | 134.075 | 135.180 | 174.495 | 19.563 | 20.613 | 3.718 | 2358.632 | 19599.667 |
| e2 | llm_decode | eager | 1 | sym | plain | 131.659 | 55.777 | 98.926 | 0.000 | 0.000 | 2.410 | 2196.238 | 19385.121 |
| e2 | llm_decode | eager | 1 | torch | off | 119.738 | 86.355 | 216.217 | 0.000 | 21.161 | 0.620 | 3184.625 | 28014.817 |
| e2 | llm_decode | eager | 1 | torch | on | 123.988 | 102.470 | 158.462 | 21.629 | 20.154 | 1.455 | 2153.795 | 18073.857 |
| e2 | llm_decode | eager | 1 | torch | plain | 120.541 | 28.011 | 100.235 | 0.000 | 0.000 | 0.327 | 2226.975 | 19939.042 |
| e2 | llm_decode | inductor | 1 | sym | off | 1652.167 | 17484.423 | 228.208 | 0.000 | 17346.868 | 3.688 | 21955.885 | 45379.878 |
| e2 | llm_decode | inductor | 1 | sym | on | 1621.297 | 17468.536 | 173.021 | 19.076 | 17317.251 | 5.113 | 21178.566 | 38432.556 |
| e2 | llm_decode | inductor | 1 | sym | plain | 1615.190 | 17517.733 | 94.118 | 0.000 | 0.000 | 3.058 | 21051.345 | 37516.716 |
| e2 | llm_decode | inductor | 1 | torch | off | 1602.780 | 17309.545 | 209.237 | 0.000 | 17241.262 | 1.070 | 21783.499 | 45514.574 |
| e2 | llm_decode | inductor | 1 | torch | on | 1610.861 | 17486.226 | 160.162 | 22.393 | 17398.446 | 3.085 | 21033.725 | 36947.916 |
| e2 | llm_decode | inductor | 1 | torch | plain | 1603.811 | 17507.120 | 93.328 | 0.000 | 0.000 | 0.457 | 21030.306 | 37284.705 |
| e2 | llm_prefill | eager | 1 | sym | off | 147.838 | 48.347 | 72.606 | 0.000 | 8.667 | 2.501 | 955.870 | 7161.402 |
| e2 | llm_prefill | eager | 1 | sym | on | 142.102 | 60.443 | 62.046 | 13.635 | 9.229 | 2.839 | 775.244 | 5361.723 |
| e2 | llm_prefill | eager | 1 | sym | plain | 143.804 | 30.754 | 41.370 | 0.000 | 0.000 | 2.406 | 742.599 | 5423.253 |
| e2 | llm_prefill | eager | 1 | torch | off | 137.942 | 31.080 | 71.625 | 0.000 | 10.124 | 0.430 | 958.655 | 7451.217 |
| e2 | llm_prefill | eager | 1 | torch | on | 133.082 | 41.472 | 58.364 | 13.311 | 8.817 | 0.892 | 734.279 | 5254.586 |
| e2 | llm_prefill | eager | 1 | torch | plain | 135.518 | 13.080 | 43.154 | 0.000 | 0.000 | 0.397 | 711.259 | 5350.197 |
| e2 | llm_prefill | inductor | 1 | sym | off | 1632.076 | 11404.621 | 76.604 | 0.000 | 11331.667 | 3.397 | 13750.084 | 19510.688 |
| e2 | llm_prefill | inductor | 1 | sym | on | 1636.564 | 11417.857 | 63.303 | 12.613 | 11333.116 | 3.626 | 13633.126 | 18278.402 |
| e2 | llm_prefill | inductor | 1 | sym | plain | 1628.553 | 11423.190 | 41.033 | 0.000 | 0.000 | 3.180 | 13590.178 | 18102.007 |
| e2 | llm_prefill | inductor | 1 | torch | off | 1615.279 | 11270.583 | 69.067 | 0.000 | 11249.429 | 0.768 | 13634.266 | 19741.751 |
| e2 | llm_prefill | inductor | 1 | torch | on | 1636.396 | 11292.260 | 60.323 | 12.422 | 11260.116 | 1.161 | 13495.115 | 18068.702 |
| e2 | llm_prefill | inductor | 1 | torch | plain | 1626.330 | 11406.281 | 41.790 | 0.000 | 0.000 | 0.506 | 13583.722 | 18024.351 |
| e2 | moe_dense | eager | 1 | sym | off | 176.057 | 69.129 | 119.255 | 0.000 | 12.364 | 2.483 | 1806.167 | 14781.831 |
| e2 | moe_dense | eager | 1 | sym | on | 175.372 | 99.947 | 111.863 | 18.310 | 13.362 | 3.131 | 1672.577 | 13242.621 |
| e2 | moe_dense | eager | 1 | sym | plain | 174.870 | 33.462 | 67.047 | 0.000 | 0.000 | 2.005 | 1250.890 | 10051.619 |
| e2 | moe_dense | eager | 1 | torch | off | 154.953 | 48.628 | 114.122 | 0.000 | 12.487 | 0.836 | 1764.477 | 14794.342 |
| e2 | moe_dense | eager | 1 | torch | on | 158.941 | 71.932 | 104.637 | 17.272 | 12.567 | 1.471 | 1633.258 | 13315.273 |
| e2 | moe_dense | eager | 1 | torch | plain | 152.888 | 15.338 | 67.670 | 0.000 | 0.000 | 0.517 | 1309.438 | 10902.187 |
| e2 | moe_dense | inductor | 1 | sym | off | 1678.031 | 5633.523 | 140.860 | 0.000 | 4650.373 | 2.945 | 9271.968 | 25517.336 |
| e2 | moe_dense | inductor | 1 | sym | on | 1679.915 | 5626.196 | 126.843 | 19.098 | 4621.621 | 3.583 | 9078.480 | 22935.176 |
| e2 | moe_dense | inductor | 1 | sym | plain | 1649.866 | 5696.218 | 83.322 | 0.000 | 0.000 | 2.595 | 8755.042 | 20573.604 |
| e2 | moe_dense | inductor | 1 | torch | off | 1654.025 | 5550.810 | 133.701 | 0.000 | 4620.690 | 1.003 | 9148.016 | 25400.222 |
| e2 | moe_dense | inductor | 1 | torch | on | 1636.326 | 5595.067 | 120.476 | 18.455 | 4642.455 | 1.826 | 8875.149 | 22558.961 |
| e2 | moe_dense | inductor | 1 | torch | plain | 1636.555 | 5604.049 | 83.916 | 0.000 | 0.000 | 0.551 | 8777.199 | 20631.421 |
| e2 | moe_sparse | eager | 1 | sym | off | 179.127 | 31.713 | 33.138 | 0.000 | 2.645 | 1.811 | 695.105 | 4799.307 |
| e2 | moe_sparse | eager | 1 | sym | on | 189.065 | 38.332 | 28.809 | 4.362 | 2.656 | 1.933 | 625.725 | 3951.394 |
| e2 | moe_sparse | eager | 1 | sym | plain | 181.826 | 23.590 | 16.559 | 0.000 | 0.000 | 1.706 | 514.394 | 3127.773 |
| e2 | moe_sparse | eager | 1 | torch | off | 165.135 | 15.847 | 31.014 | 0.000 | 2.685 | 0.271 | 660.558 | 4684.683 |
| e2 | moe_sparse | eager | 1 | torch | on | 165.654 | 19.661 | 26.866 | 3.011 | 2.796 | 0.612 | 582.802 | 3922.078 |
| e2 | moe_sparse | eager | 1 | torch | plain | 167.114 | 7.954 | 16.049 | 0.000 | 0.000 | 0.303 | 468.607 | 2966.171 |
| e2 | moe_sparse | inductor | 1 | sym | off | 1666.768 | 4486.286 | 38.378 | 0.000 | 3865.285 | 2.243 | 6756.711 | 11845.635 |
| e2 | moe_sparse | inductor | 1 | sym | on | 1694.166 | 4498.729 | 30.349 | 5.061 | 3876.653 | 2.368 | 6621.864 | 10350.923 |
| e2 | moe_sparse | inductor | 1 | sym | plain | 1668.022 | 4516.160 | 21.187 | 0.000 | 0.000 | 2.135 | 6608.661 | 10061.892 |
| e2 | moe_sparse | inductor | 1 | torch | off | 1673.061 | 4427.860 | 36.602 | 0.000 | 3857.111 | 0.388 | 6704.603 | 11810.443 |
| e2 | moe_sparse | inductor | 1 | torch | on | 1654.638 | 4446.552 | 28.766 | 4.368 | 3870.153 | 0.734 | 6535.337 | 10237.864 |
| e2 | moe_sparse | inductor | 1 | torch | plain | 1649.502 | 4470.762 | 20.807 | 0.000 | 0.000 | 0.384 | 6516.581 | 9909.662 |
| e5 | llm_decode | eager | 1 | sym | off | 128.950 | 113.745 | 1103.099 | 0.000 | 17.984 | 3.441 | 3981.429 | 27706.226 |
| e5 | llm_decode | eager | 1 | sym | on | 129.197 | 130.603 | 887.165 | 18.837 | 18.269 | 4.375 | 3216.407 | 21830.155 |
| e5 | llm_decode | eager | 1 | sym | plain | 129.490 | 55.588 | 475.536 | 0.000 | 0.000 | 3.165 | 2592.658 | 19958.182 |
| e5 | llm_decode | eager | 1 | torch | off | 119.532 | 80.135 | 1030.356 | 0.000 | 18.271 | 0.679 | 3927.311 | 28242.041 |
| e5 | llm_decode | eager | 1 | torch | on | 119.094 | 98.435 | 829.236 | 20.347 | 17.956 | 1.738 | 2995.140 | 20523.762 |
| e5 | llm_decode | eager | 1 | torch | plain | 119.114 | 26.342 | 481.305 | 0.000 | 0.000 | 0.388 | 2592.693 | 20161.723 |
| e5 | llm_decode | eager | 2 | sym | off | 129.385 | 199.341 | 1069.178 | 0.000 | 26.789 | 6.034 | 3800.547 | 25144.245 |
| e5 | llm_decode | eager | 2 | sym | on | 129.803 | 221.114 | 1024.242 | 27.705 | 26.140 | 6.607 | 3594.090 | 23589.789 |
| e5 | llm_decode | eager | 2 | sym | plain | 130.042 | 98.727 | 475.486 | 0.000 | 0.000 | 5.245 | 2487.294 | 18414.494 |
| e5 | llm_decode | eager | 2 | torch | off | 119.951 | 149.303 | 1001.799 | 0.000 | 27.179 | 0.988 | 3807.656 | 26435.863 |
| e5 | llm_decode | eager | 2 | torch | on | 119.626 | 167.493 | 949.283 | 32.501 | 25.127 | 1.979 | 3447.944 | 23354.073 |
| e5 | llm_decode | eager | 2 | torch | plain | 119.576 | 51.048 | 504.785 | 0.000 | 0.000 | 0.648 | 2442.156 | 18368.155 |
| e5 | llm_decode | eager | 4 | sym | off | — | — | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | sym | on | 129.668 | 445.640 | 1032.669 | 81.027 | 43.496 | 10.414 | 4317.869 | 28665.243 |
| e5 | llm_decode | eager | 4 | sym | plain | 130.673 | 184.191 | 468.390 | 0.000 | 0.000 | 10.204 | 2746.185 | 20418.563 |
| e5 | llm_decode | eager | 4 | torch | off | — | — | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | torch | on | 120.055 | 348.371 | 961.198 | 90.681 | 41.726 | 1.355 | 4232.715 | 29421.672 |
| e5 | llm_decode | eager | 4 | torch | plain | 119.666 | 95.800 | 487.895 | 0.000 | 0.000 | 0.918 | 2749.117 | 21163.677 |
| e5 | llm_decode | inductor | 1 | sym | off | 1613.055 | 17382.353 | 1079.929 | 0.000 | 17247.777 | 5.024 | 22641.065 | 45645.442 |
| e5 | llm_decode | inductor | 1 | sym | on | 1612.149 | 17411.929 | 881.097 | 19.370 | 17261.516 | 6.614 | 21962.490 | 40475.136 |
| e5 | llm_decode | inductor | 1 | sym | plain | 1658.658 | 17563.662 | 447.330 | 0.000 | 0.000 | 4.313 | 21464.468 | 38126.914 |
| e5 | llm_decode | inductor | 1 | torch | off | 1637.705 | 17363.575 | 994.480 | 0.000 | 17299.532 | 1.652 | 22587.845 | 45882.574 |
| e5 | llm_decode | inductor | 1 | torch | on | 1611.493 | 17379.609 | 808.575 | 22.435 | 17293.792 | 3.655 | 21718.090 | 38921.667 |
| e5 | llm_decode | inductor | 1 | torch | plain | 1626.287 | 17551.374 | 449.829 | 0.000 | 0.000 | 0.496 | 21599.298 | 38106.600 |
| e5 | moe_sparse | eager | 1 | sym | off | 171.121 | 31.827 | 155.131 | 0.000 | 2.678 | 1.983 | 811.339 | 4883.491 |
| e5 | moe_sparse | eager | 1 | sym | on | 167.793 | 38.562 | 135.589 | 4.340 | 2.566 | 2.257 | 720.583 | 4140.710 |
| e5 | moe_sparse | eager | 1 | sym | plain | 172.935 | 23.525 | 79.631 | 0.000 | 0.000 | 1.796 | 571.413 | 3231.982 |
| e5 | moe_sparse | eager | 1 | torch | off | 155.907 | 15.847 | 146.674 | 0.000 | 2.636 | 0.519 | 784.481 | 4922.643 |
| e5 | moe_sparse | eager | 1 | torch | on | 154.968 | 19.534 | 127.008 | 3.025 | 2.700 | 0.785 | 676.717 | 4055.021 |
| e5 | moe_sparse | eager | 1 | torch | plain | 158.265 | 7.905 | 74.385 | 0.000 | 0.000 | 0.343 | 524.746 | 3080.791 |
| e5 | moe_sparse | eager | 2 | sym | off | 168.265 | 45.816 | 157.005 | 0.000 | 5.290 | 3.624 | 784.769 | 4462.305 |
| e5 | moe_sparse | eager | 2 | sym | on | 173.779 | 55.876 | 141.156 | 7.098 | 5.340 | 3.765 | 734.870 | 3965.399 |
| e5 | moe_sparse | eager | 2 | sym | plain | 170.730 | 30.563 | 83.240 | 0.000 | 0.000 | 3.225 | 530.921 | 2745.338 |
| e5 | moe_sparse | eager | 2 | torch | off | 154.429 | 28.930 | 146.561 | 0.000 | 5.335 | 0.741 | 744.657 | 4483.637 |
| e5 | moe_sparse | eager | 2 | torch | on | 154.745 | 37.067 | 126.569 | 7.342 | 5.259 | 1.163 | 694.815 | 4020.174 |
| e5 | moe_sparse | eager | 2 | torch | plain | 154.738 | 13.151 | 74.114 | 0.000 | 0.000 | 0.467 | 485.211 | 2642.943 |
| e5 | moe_sparse | eager | 4 | sym | off | 170.181 | 74.109 | 157.550 | 0.000 | 10.662 | 6.061 | 845.200 | 4800.615 |
| e5 | moe_sparse | eager | 4 | sym | on | 168.257 | 90.949 | 137.790 | 12.516 | 10.333 | 6.752 | 789.506 | 4325.761 |
| e5 | moe_sparse | eager | 4 | sym | plain | 168.829 | 45.333 | 82.202 | 0.000 | 0.000 | 5.274 | 527.261 | 2604.609 |
| e5 | moe_sparse | eager | 4 | torch | off | 154.892 | 51.915 | 146.284 | 0.000 | 10.246 | 1.192 | 807.426 | 4900.260 |
| e5 | moe_sparse | eager | 4 | torch | on | 154.061 | 71.459 | 128.850 | 16.308 | 10.471 | 1.875 | 760.692 | 4416.088 |
| e5 | moe_sparse | eager | 4 | torch | plain | 154.723 | 22.950 | 74.783 | 0.000 | 0.000 | 0.687 | 495.433 | 2683.300 |
| e5 | moe_sparse | inductor | 1 | sym | off | 1650.820 | 4488.131 | 186.539 | 0.000 | 3873.002 | 2.497 | 6909.148 | 12129.231 |
| e5 | moe_sparse | inductor | 1 | sym | on | 1672.093 | 4484.964 | 145.742 | 4.975 | 3862.643 | 2.929 | 6730.250 | 10507.848 |
| e5 | moe_sparse | inductor | 1 | sym | plain | 1652.192 | 4526.448 | 101.426 | 0.000 | 0.000 | 2.341 | 6671.252 | 10160.017 |
| e5 | moe_sparse | inductor | 1 | torch | off | 1640.575 | 4432.455 | 174.597 | 0.000 | 3863.062 | 0.740 | 6807.845 | 11864.362 |
| e5 | moe_sparse | inductor | 1 | torch | on | 1648.345 | 4459.074 | 137.283 | 4.364 | 3880.416 | 1.101 | 6647.049 | 10409.613 |
| e5 | moe_sparse | inductor | 1 | torch | plain | 1640.050 | 4470.542 | 98.246 | 0.000 | 0.000 | 0.519 | 6593.065 | 10050.506 |

## Memory

All values are MiB. Owner input/private-pool retention, Torch reservation and NVML are overlapping scopes; do not sum them. Warm increment subtracts NVML after the E5 reserve (or initial context baseline for E2). Whole-process NVML is absolute device usage, including the E5 reservation. Sampling at 2 ms gives lower bounds; an unsampled phase is missing, never zero. The JSON also retains every phase boundary, observation count and peak.

| kind | case | compute | callers | runtime | mode | owner retained | private pool | Torch peak reserved | warm NVML increment | whole-process NVML peak | first-phase NVML peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2 | llm_decode | eager | 1 | sym | off | 240.477 | 0.000 | 834.000 | 932.000 | 1550.438 | 1530.438 |
| e2 | llm_decode | eager | 1 | sym | on | 280.477 | 40.000 | 778.000 | 878.000 | 1496.438 | 1476.438 |
| e2 | llm_decode | eager | 1 | sym | plain | 0.000 | 0.000 | 234.000 | 332.000 | 950.438 | 930.438 |
| e2 | llm_decode | eager | 1 | torch | off | 240.477 | 0.000 | 834.000 | 898.000 | 1516.438 | 1516.438 |
| e2 | llm_decode | eager | 1 | torch | on | 280.477 | 40.000 | 778.000 | 844.000 | 1462.438 | 1442.438 |
| e2 | llm_decode | eager | 1 | torch | plain | 0.000 | 0.000 | 234.000 | 298.000 | 916.438 | 916.438 |
| e2 | llm_decode | inductor | 1 | sym | off | 240.477 | 0.000 | 874.000 | 972.000 | 1590.438 | 1570.438 |
| e2 | llm_decode | inductor | 1 | sym | on | 280.477 | 40.000 | 778.000 | 878.000 | 1496.438 | 1476.438 |
| e2 | llm_decode | inductor | 1 | sym | plain | 0.000 | 0.000 | 214.000 | 312.000 | 930.438 | 930.438 |
| e2 | llm_decode | inductor | 1 | torch | off | 240.477 | 0.000 | 874.000 | 938.000 | 1556.438 | 1556.438 |
| e2 | llm_decode | inductor | 1 | torch | on | 280.477 | 40.000 | 778.000 | 844.000 | 1462.438 | 1442.438 |
| e2 | llm_decode | inductor | 1 | torch | plain | 0.000 | 0.000 | 234.000 | 298.000 | 916.438 | 916.438 |
| e2 | llm_prefill | eager | 1 | sym | off | 67.008 | 0.000 | 466.000 | 564.000 | 1182.438 | 1156.438 |
| e2 | llm_prefill | eager | 1 | sym | on | 123.008 | 56.000 | 394.000 | 492.000 | 1110.438 | 1044.438 |
| e2 | llm_prefill | eager | 1 | sym | plain | 0.000 | 0.000 | 236.000 | 334.000 | 952.438 | 952.438 |
| e2 | llm_prefill | eager | 1 | torch | off | 67.008 | 0.000 | 488.000 | 552.000 | 1170.438 | 1166.438 |
| e2 | llm_prefill | eager | 1 | torch | on | 123.008 | 56.000 | 376.000 | 440.000 | 1058.438 | 1032.438 |
| e2 | llm_prefill | eager | 1 | torch | plain | 0.000 | 0.000 | 258.000 | 322.000 | 940.438 | 938.438 |
| e2 | llm_prefill | inductor | 1 | sym | off | 67.008 | 0.000 | 458.000 | 556.000 | 1174.438 | 1172.438 |
| e2 | llm_prefill | inductor | 1 | sym | on | 119.008 | 52.000 | 390.000 | 488.000 | 1106.438 | 1038.438 |
| e2 | llm_prefill | inductor | 1 | sym | plain | 0.000 | 0.000 | 236.000 | 334.000 | 952.438 | 950.438 |
| e2 | llm_prefill | inductor | 1 | torch | off | 67.008 | 0.000 | 480.000 | 544.000 | 1162.438 | 1160.438 |
| e2 | llm_prefill | inductor | 1 | torch | on | 119.008 | 52.000 | 392.000 | 456.000 | 1074.438 | 1030.438 |
| e2 | llm_prefill | inductor | 1 | torch | plain | 0.000 | 0.000 | 258.000 | 322.000 | 940.438 | 938.438 |
| e2 | moe_dense | eager | 1 | sym | off | 146.047 | 0.000 | 568.000 | 648.000 | 1266.438 | 1264.438 |
| e2 | moe_dense | eager | 1 | sym | on | 182.047 | 36.000 | 520.000 | 602.000 | 1220.438 | 1220.438 |
| e2 | moe_dense | eager | 1 | sym | plain | 0.000 | 0.000 | 134.000 | 214.000 | 832.438 | 832.438 |
| e2 | moe_dense | eager | 1 | torch | off | 146.047 | 0.000 | 570.000 | 634.000 | 1252.438 | 1250.438 |
| e2 | moe_dense | eager | 1 | torch | on | 182.047 | 36.000 | 522.000 | 586.000 | 1204.438 | 1204.438 |
| e2 | moe_dense | eager | 1 | torch | plain | 0.000 | 0.000 | 136.000 | 200.000 | 818.438 | 818.438 |
| e2 | moe_dense | inductor | 1 | sym | off | 146.047 | 0.000 | 572.000 | 652.000 | 1270.438 | 1268.438 |
| e2 | moe_dense | inductor | 1 | sym | on | 182.047 | 36.000 | 520.000 | 602.000 | 1220.438 | 1220.438 |
| e2 | moe_dense | inductor | 1 | sym | plain | 0.000 | 0.000 | 134.000 | 214.000 | 832.438 | 832.438 |
| e2 | moe_dense | inductor | 1 | torch | off | 146.047 | 0.000 | 574.000 | 638.000 | 1256.438 | 1254.438 |
| e2 | moe_dense | inductor | 1 | torch | on | 182.047 | 36.000 | 522.000 | 586.000 | 1204.438 | 1204.438 |
| e2 | moe_dense | inductor | 1 | torch | plain | 0.000 | 0.000 | 136.000 | 200.000 | 818.438 | 818.438 |
| e2 | moe_sparse | eager | 1 | sym | off | 16.039 | 0.000 | 192.000 | 272.000 | 890.438 | 890.438 |
| e2 | moe_sparse | eager | 1 | sym | on | 24.039 | 8.000 | 160.000 | 240.000 | 858.438 | 838.438 |
| e2 | moe_sparse | eager | 1 | sym | plain | 0.000 | 0.000 | 104.000 | 184.000 | 802.438 | 782.438 |
| e2 | moe_sparse | eager | 1 | torch | off | 16.039 | 0.000 | 194.000 | 258.000 | 876.438 | 876.438 |
| e2 | moe_sparse | eager | 1 | torch | on | 24.039 | 8.000 | 162.000 | 226.000 | 844.438 | 844.438 |
| e2 | moe_sparse | eager | 1 | torch | plain | 0.000 | 0.000 | 106.000 | 170.000 | 788.438 | 768.438 |
| e2 | moe_sparse | inductor | 1 | sym | off | 16.039 | 0.000 | 212.000 | 292.000 | 910.438 | 910.438 |
| e2 | moe_sparse | inductor | 1 | sym | on | 24.039 | 8.000 | 180.000 | 260.000 | 878.438 | 878.438 |
| e2 | moe_sparse | inductor | 1 | sym | plain | 0.000 | 0.000 | 104.000 | 184.000 | 802.438 | 782.438 |
| e2 | moe_sparse | inductor | 1 | torch | off | 16.039 | 0.000 | 214.000 | 278.000 | 896.438 | 896.438 |
| e2 | moe_sparse | inductor | 1 | torch | on | 24.039 | 8.000 | 182.000 | 246.000 | 864.438 | 844.438 |
| e2 | moe_sparse | inductor | 1 | torch | plain | 0.000 | 0.000 | 106.000 | 170.000 | 788.438 | 788.438 |
| e5 | llm_decode | eager | 1 | sym | off | 185.016 | 0.000 | 770.000 | 868.000 | 11108.438 | 11088.438 |
| e5 | llm_decode | eager | 1 | sym | on | 204.746 | 28.000 | 658.000 | 758.000 | 10998.438 | 10980.438 |
| e5 | llm_decode | eager | 1 | sym | plain | 0.000 | 0.000 | 234.000 | 332.000 | 10572.438 | 10552.438 |
| e5 | llm_decode | eager | 1 | torch | off | 185.016 | 0.000 | 770.000 | 834.000 | 11074.438 | 11074.438 |
| e5 | llm_decode | eager | 1 | torch | on | 204.746 | 28.000 | 658.000 | 722.000 | 10962.438 | 10944.438 |
| e5 | llm_decode | eager | 1 | torch | plain | 0.000 | 0.000 | 234.000 | 298.000 | 10538.438 | 10538.438 |
| e5 | llm_decode | eager | 2 | sym | off | 128.664 | 0.000 | 880.000 | 1022.000 | 11262.438 | 11258.438 |
| e5 | llm_decode | eager | 2 | sym | on | 152.664 | 24.000 | 734.000 | 878.000 | 11118.438 | 11038.438 |
| e5 | llm_decode | eager | 2 | sym | plain | 0.000 | 0.000 | 398.000 | 540.000 | 10780.438 | 10720.438 |
| e5 | llm_decode | eager | 2 | torch | off | 128.664 | 0.000 | 898.000 | 970.000 | 11252.438 | 11252.438 |
| e5 | llm_decode | eager | 2 | torch | on | 152.664 | 24.000 | 694.000 | 766.000 | 11006.438 | 10946.438 |
| e5 | llm_decode | eager | 2 | torch | plain | 0.000 | 0.000 | 358.000 | 430.000 | 10670.438 | 10670.438 |
| e5 | llm_decode | eager | 4 | sym | off | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | sym | on | 0.000 | 0.000 | 594.000 | 824.000 | 11064.438 | 10840.438 |
| e5 | llm_decode | eager | 4 | sym | plain | 0.000 | 0.000 | 706.000 | 936.000 | 11176.438 | 11056.438 |
| e5 | llm_decode | eager | 4 | torch | off | — | — | — | — | — | — |
| e5 | llm_decode | eager | 4 | torch | on | 0.000 | 0.000 | 574.000 | 662.000 | 10902.438 | 10600.438 |
| e5 | llm_decode | eager | 4 | torch | plain | 0.000 | 0.000 | 606.000 | 694.000 | 10934.438 | 10934.438 |
| e5 | llm_decode | inductor | 1 | sym | off | 185.016 | 0.000 | 810.000 | 908.000 | 11148.438 | 11128.438 |
| e5 | llm_decode | inductor | 1 | sym | on | 204.746 | 28.000 | 658.000 | 758.000 | 10998.438 | 10980.438 |
| e5 | llm_decode | inductor | 1 | sym | plain | 0.000 | 0.000 | 234.000 | 332.000 | 10572.438 | 10552.438 |
| e5 | llm_decode | inductor | 1 | torch | off | 185.016 | 0.000 | 810.000 | 874.000 | 11114.438 | 11114.438 |
| e5 | llm_decode | inductor | 1 | torch | on | 204.746 | 28.000 | 658.000 | 722.000 | 10962.438 | 10942.438 |
| e5 | llm_decode | inductor | 1 | torch | plain | 0.000 | 0.000 | 234.000 | 298.000 | 10538.438 | 10538.438 |
| e5 | moe_sparse | eager | 1 | sym | off | 16.039 | 0.000 | 192.000 | 272.000 | 10512.438 | 10512.438 |
| e5 | moe_sparse | eager | 1 | sym | on | 24.039 | 8.000 | 160.000 | 240.000 | 10480.438 | 10460.438 |
| e5 | moe_sparse | eager | 1 | sym | plain | 0.000 | 0.000 | 104.000 | 184.000 | 10424.438 | 10404.438 |
| e5 | moe_sparse | eager | 1 | torch | off | 16.039 | 0.000 | 194.000 | 258.000 | 10498.438 | 10498.438 |
| e5 | moe_sparse | eager | 1 | torch | on | 24.039 | 8.000 | 162.000 | 226.000 | 10466.438 | 10466.438 |
| e5 | moe_sparse | eager | 1 | torch | plain | 0.000 | 0.000 | 106.000 | 170.000 | 10410.438 | 10390.438 |
| e5 | moe_sparse | eager | 2 | sym | off | 32.078 | 0.000 | 342.000 | 446.000 | 10686.438 | 10686.438 |
| e5 | moe_sparse | eager | 2 | sym | on | 48.078 | 16.000 | 298.000 | 402.000 | 10642.438 | 10640.438 |
| e5 | moe_sparse | eager | 2 | sym | plain | 0.000 | 0.000 | 166.000 | 270.000 | 10510.438 | 10490.438 |
| e5 | moe_sparse | eager | 2 | torch | off | 32.078 | 0.000 | 346.000 | 418.000 | 10658.438 | 10658.438 |
| e5 | moe_sparse | eager | 2 | torch | on | 48.078 | 16.000 | 302.000 | 374.000 | 10614.438 | 10570.438 |
| e5 | moe_sparse | eager | 2 | torch | plain | 0.000 | 0.000 | 170.000 | 242.000 | 10482.438 | 10482.438 |
| e5 | moe_sparse | eager | 4 | sym | off | 64.156 | 0.000 | 642.000 | 794.000 | 11034.438 | 11034.438 |
| e5 | moe_sparse | eager | 4 | sym | on | 96.156 | 32.000 | 574.000 | 728.000 | 10968.438 | 10962.438 |
| e5 | moe_sparse | eager | 4 | sym | plain | 0.000 | 0.000 | 290.000 | 442.000 | 10682.438 | 10662.438 |
| e5 | moe_sparse | eager | 4 | torch | off | 64.156 | 0.000 | 650.000 | 738.000 | 10978.438 | 10978.438 |
| e5 | moe_sparse | eager | 4 | torch | on | 96.156 | 32.000 | 582.000 | 670.000 | 10910.438 | 10778.438 |
| e5 | moe_sparse | eager | 4 | torch | plain | 0.000 | 0.000 | 298.000 | 386.000 | 10626.438 | 10606.438 |
| e5 | moe_sparse | inductor | 1 | sym | off | 16.039 | 0.000 | 212.000 | 292.000 | 10532.438 | 10532.438 |
| e5 | moe_sparse | inductor | 1 | sym | on | 24.039 | 8.000 | 180.000 | 260.000 | 10500.438 | 10500.438 |
| e5 | moe_sparse | inductor | 1 | sym | plain | 0.000 | 0.000 | 104.000 | 184.000 | 10424.438 | 10404.438 |
| e5 | moe_sparse | inductor | 1 | torch | off | 16.039 | 0.000 | 214.000 | 278.000 | 10518.438 | 10518.438 |
| e5 | moe_sparse | inductor | 1 | torch | on | 24.039 | 8.000 | 182.000 | 246.000 | 10486.438 | 10466.438 |
| e5 | moe_sparse | inductor | 1 | torch | plain | 0.000 | 0.000 | 106.000 | 170.000 | 10410.438 | 10410.438 |

## Relative gains and break-even

A speedup above 1 favors replay or Sym. Break-even is based on the median warm saving and median setup difference; it is a linear estimate, not a separately observed long run. `—` means no positive warm saving. A graph-enabled arm with zero warm graph launches is a fallback result, not evidence of a replay mechanism gain.

| kind | case | compute | callers | Sym on/off | Sym on/plain | Torch on/off | Torch on/plain | Sym/Torch on | Sym off BE | Sym plain BE | Torch off BE | Torch plain BE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2 | llm_decode | eager | 1 | 1.407 | 0.999 | 1.560 | 1.113 | 0.925 | 0 | — | 0 | 69 |
| e2 | llm_decode | inductor | 1 | 1.358 | 0.954 | 1.491 | 1.020 | 0.923 | 0 | — | 14 | 130 |
| e2 | llm_prefill | eager | 1 | 1.357 | 1.022 | 1.437 | 1.029 | 0.987 | 0 | 435 | 0 | 286 |
| e2 | llm_prefill | inductor | 1 | 1.249 | 0.971 | 1.336 | 0.974 | 0.984 | 6 | — | 15 | — |
| e2 | moe_dense | eager | 1 | 1.121 | 0.761 | 1.115 | 0.821 | 1.010 | 12 | — | 12 | — |
| e2 | moe_dense | inductor | 1 | 1.184 | 0.860 | 1.187 | 0.871 | 0.997 | 0 | — | 7 | — |
| e2 | moe_sparse | eager | 1 | 1.234 | 0.786 | 1.205 | 0.748 | 1.004 | 15 | — | 1 | — |
| e2 | moe_sparse | inductor | 1 | 1.368 | 0.931 | 1.389 | 0.925 | 0.986 | 11 | — | 0 | — |
| e5 | llm_decode | eager | 1 | 1.275 | 0.933 | 1.387 | 1.003 | 0.942 | 0 | — | 0 | 7215 |
| e5 | llm_decode | eager | 2 | 1.067 | 0.796 | 1.135 | 0.799 | 0.997 | 0 | — | 0 | — |
| e5 | llm_decode | eager | 4 | — | 0.726 | — | 0.731 | 1.035 | — | — | — | — |
| e5 | llm_decode | inductor | 1 | 1.237 | 0.899 | 1.353 | 0.948 | 0.931 | 0 | — | 0 | — |
| e5 | moe_sparse | eager | 1 | 1.193 | 0.780 | 1.225 | 0.757 | 0.990 | 0 | — | 0 | — |
| e5 | moe_sparse | eager | 2 | 1.140 | 0.685 | 1.123 | 0.648 | 1.031 | 0 | — | 0 | — |
| e5 | moe_sparse | eager | 4 | 1.121 | 0.588 | 1.120 | 0.599 | 1.034 | 0 | — | 5 | — |
| e5 | moe_sparse | inductor | 1 | 1.395 | 0.931 | 1.344 | 0.919 | 1.005 | 0 | — | 0 | — |

## Parent #247 targets

Pass requires Sym p50 at least 5% below both fresh eager Torch and Inductor controls in the same mode in every round. Tests use unrounded per-round values, not the displayed cross-round median.

| case | Sym compute | mode | round 1 | round 2 | round 3 | qualified |
| --- | --- | --- | --- | --- | --- | --- |
| llm_decode | eager | off | fail | fail | fail | no |
| llm_decode | eager | on | fail | fail | fail | no |
| llm_decode | eager | plain | fail | fail | fail | no |
| llm_decode | inductor | off | fail | fail | fail | no |
| llm_decode | inductor | on | fail | fail | fail | no |
| llm_decode | inductor | plain | fail | fail | fail | no |
| llm_prefill | eager | off | fail | fail | fail | no |
| llm_prefill | eager | on | fail | fail | fail | no |
| llm_prefill | eager | plain | fail | fail | fail | no |
| llm_prefill | inductor | off | fail | pass | fail | no |
| llm_prefill | inductor | on | fail | fail | fail | no |
| llm_prefill | inductor | plain | fail | fail | fail | no |
| moe_dense | eager | off | pass | fail | fail | no |
| moe_dense | eager | on | fail | fail | fail | no |
| moe_dense | eager | plain | fail | fail | pass | no |
| moe_dense | inductor | off | fail | fail | fail | no |
| moe_dense | inductor | on | fail | fail | fail | no |
| moe_dense | inductor | plain | fail | fail | fail | no |
| moe_sparse | eager | off | fail | fail | fail | no |
| moe_sparse | eager | on | fail | fail | fail | no |
| moe_sparse | eager | plain | fail | fail | fail | no |
| moe_sparse | inductor | off | fail | fail | fail | no |
| moe_sparse | inductor | on | fail | fail | fail | no |
| moe_sparse | inductor | plain | fail | fail | fail | no |

## Shape and budget supplements

Three fresh processes per compiler, 100 checked calls per row. `varying` cycles four exact shapes after preparing one; `pressure` limits owner retention to 1 MiB. Fallback counters include first/warmup calls (pressure: 106 total, 100 timed). The maximum and total warm time expose shape-triggered compilation hidden by medians. Maximum below is the worst across all three rounds.

| compute | scenario | mode | p50 ms | p95 ms | submit p50 ms | prepare ms | first ms | worst max ms | warm total ms | retained MiB | fallbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | pressure | off | 0.547 | 0.669 | 0.526 | 3.801 | 0.684 | 0.802 | 56.987 | 0.016 | 0 |
| eager | pressure | on | 0.479 | 0.591 | 0.458 | 3.795 | 0.513 | 0.722 | 49.801 | 0.000 | 106 |
| eager | pressure | plain | 0.304 | 0.437 | 0.286 | 0.023 | 0.323 | 0.558 | 32.587 | 0.000 | 0 |
| eager | stable | off | 0.541 | 0.657 | 0.520 | 6.692 | 0.694 | 0.939 | 56.164 | 0.016 | 0 |
| eager | stable | on | 0.246 | 0.310 | 0.226 | 7.215 | 0.382 | 0.622 | 25.525 | 4.016 | 0 |
| eager | stable | plain | 0.308 | 0.432 | 0.285 | 0.025 | 0.476 | 0.557 | 32.500 | 0.000 | 0 |
| eager | varying | off | 0.479 | 0.622 | 0.456 | 2.925 | 0.703 | 0.934 | 51.195 | 0.016 | 75 |
| eager | varying | on | 0.467 | 0.584 | 0.446 | 5.913 | 0.317 | 0.760 | 43.084 | 4.016 | 75 |
| eager | varying | plain | 0.310 | 0.435 | 0.292 | 0.024 | 0.333 | 0.568 | 33.049 | 0.000 | 0 |
| inductor | pressure | off | 0.352 | 0.401 | 0.331 | 74.648 | 0.427 | 4.071 | 35.865 | 0.016 | 0 |
| inductor | pressure | on | 0.276 | 0.306 | 0.253 | 77.553 | 0.444 | 0.362 | 27.796 | 0.000 | 106 |
| inductor | pressure | plain | 0.123 | 0.144 | 0.102 | 0.029 | 72.756 | 0.174 | 12.891 | 0.000 | 0 |
| inductor | stable | off | 0.359 | 0.399 | 0.337 | 83.973 | 0.524 | 0.468 | 36.239 | 0.016 | 0 |
| inductor | stable | on | 0.246 | 0.293 | 0.225 | 87.674 | 0.482 | 0.683 | 25.868 | 4.016 | 0 |
| inductor | stable | plain | 0.120 | 0.147 | 0.101 | 0.032 | 76.417 | 0.185 | 12.483 | 0.000 | 0 |
| inductor | varying | off | 0.280 | 0.413 | 0.257 | 75.652 | 0.433 | 1092.266 | 124.760 | 0.016 | 75 |
| inductor | varying | on | 0.276 | 0.313 | 0.253 | 77.363 | 0.344 | 1041.326 | 124.376 | 4.016 | 75 |
| inductor | varying | plain | 0.132 | 0.161 | 0.111 | 0.030 | 73.158 | 1052.317 | 107.263 | 0.000 | 0 |
