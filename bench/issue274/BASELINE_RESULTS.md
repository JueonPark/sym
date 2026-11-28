# Replaying the matched E1–E5 baseline

The original campaign isolates a benefit from folding and selected reuse.
The matched model, capacity and lifecycle comparisons retain adverse outcomes.

- E1 folding beats true materialization in **35/36** cells by paired median reduction.
  The manual loop has a lower p50 median of process medians in **31/36**.
- E2 Sym takes **4.6–10.9% longer** in the eight principal FP16 comparisons
  against Torch with the same native kernels, grouping and prefetch. The
  derived p95 comparison takes **4.8–11.0% longer** across those cells.
- E3 has zero additional Dynamo callbacks after the first cycle. The explicit cache and
  captured execution use different paths. Automatic execution trails the archived eager control.
- E4 retains **15** OOMs in the fixed placement grid. Both all-offload arms fit.
  At 32 layers, context 512, eight resident layers and a 512 MiB budget,
  Torch succeeds and Sym OOMs in all three rounds.
- E5 increases full-lifecycle cost in **12/12** paired rounds and reduces
  within-burst throughput in **18/18**. Held-output and native-close checks pass.

## Reproduction and estimators

The replay verifies all **17,264** published inventory
files and audits **163,080** recorded output checks. It reads the raw latency
vectors, process phase counters and dispositions on CPU with Python 3.11+.
[Versioned archive, original analyses and benefit assessment](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

~~~sh
python -m bench.issue274.baseline_analysis /path/to/sym_experiments_261104 \
  --output /tmp/BASELINE_RESULTS.md --json-output /tmp/baseline-analysis.json
~~~

Sym revision `2617ba166f98a7476a5f472d829a573317fc0cfb`. The immutable manifest SHA-256 is
`d23e45465fbf7d5e188a4fdc0e2b5320d815a777d07c4eaa054361be9a81c741`. Common build and environment are pinned by
`00_protocol/provenance.json` (SHA-256 `a4b937b7a8f00d4367c0af3a751f1123096ad03ad6d19033085701435063cf5a`).
Original runs use the EPYC 7351, RTX 2080 Ti GPU 0, CUDA 12.6, Torch 2.14.0+cu126,
Python 3.14.7 and CPU affinity 4–7,20–23. The archive records commands, driver
and clocks, native/compiler hashes, stream settings and per-arm budgets.
The output JSON gives the raw path and phase measurements for every arm/round.

Each p50/p95 is computed inside its process first. The displayed arm
summaries are medians of those process statistics. E1 uses its original
linear p95. E5 bursts use the original nearest-rank p95, also used for the
newly derived E2, E3, E4 and E5 lifecycle tails. At 20 samples this is the
nineteenth order statistic.

Each paired percentage is the median of same-round effects. For latency
the effect is 100 × (1 − candidate/control). For throughput it is
100 × (candidate/control − 1). Positive favors the candidate. The JSON
separately reports candidate/control as a ratio of arm medians and as a
median of same-round ratios. Table brackets give exploratory 95% percentile
intervals from exhaustive resampling of three or five paired processes,
with requests kept inside their processes. These unadjusted intervals have
coarse resolution. A band that touches zero is inconclusive. The JSON retains
individual pairs and ties.
An OOM yields no latency comparison for that pair.

## E1 · True materialization

All cells use one pinned wire and one completed H2D copy. The genuine
materialized control writes each operation's output into separate storage
inside one native invocation. Disabling canonicalization retains folding.
The manual producer uses F16C and can exploit the roundtrip idempotence
allowed by the value contract. Shared warm buffers, one or eight workers,
five independent rounds, five warmups and 60 samples per arm define the
boundary. The JSON also contains producer-only comparisons and preparation
including compile, binding and intermediate storage.

| Chain / N / workers | Unfused p50/p95 ms | Folded p50/p95 ms | No canonicalize p50/p95 ms | Manual p50/p95 ms | Eager p50/p95 ms | Inductor p50/p95 ms | Fold vs unfused p50 | Fold vs manual p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roundtrip1 / 33 / 1 | 0.061 / 0.066 | 0.039 / 0.046 | 0.039 / 0.046 | 0.040 / 0.043 | 0.084 / 0.092 | 0.131 / 0.141 | 0.061 → 0.039 (+35.46% [+35.20, +35.89]) | 0.040 → 0.039 (+2.44% [+1.95, +3.48]) |
| roundtrip1 / 33 / 8 | 0.071 / 0.075 | 0.045 / 0.048 | 0.045 / 0.047 | 0.045 / 0.049 | 0.095 / 0.098 | 0.188 / 0.200 | 0.071 → 0.045 (+35.70% [+35.29, +38.29]) | 0.045 → 0.045 (+0.36% [-1.56, +1.43]) |
| roundtrip1 / 512 / 1 | 0.441 / 0.481 | 0.112 / 0.120 | 0.111 / 0.118 | 0.083 / 0.089 | 0.265 / 0.275 | 0.233 / 0.248 | 0.441 → 0.112 (+75.28% [+74.57, +76.04]) | 0.083 → 0.112 (-31.34% [-36.21, -29.73]) |
| roundtrip1 / 512 / 8 | 0.651 / 0.714 | 0.100 / 0.116 | 0.104 / 0.111 | 0.099 / 0.114 | 0.414 / 0.485 | 0.269 / 0.297 | 0.651 → 0.100 (+83.08% [+80.51, +86.57]) | 0.099 → 0.100 (+3.73% [-23.45, +10.61]) |
| roundtrip1 / 8192 / 1 | 9.724 / 10.136 | 1.563 / 1.625 | 1.554 / 1.620 | 0.860 / 1.067 | 4.498 / 4.799 | 1.065 / 1.296 | 9.724 → 1.563 (+83.93% [+83.16, +85.12]) | 0.860 → 1.563 (-80.30% [-81.90, -77.28]) |
| roundtrip1 / 8192 / 8 | 5.387 / 5.744 | 0.836 / 0.857 | 0.835 / 0.854 | 0.747 / 0.776 | 4.176 / 4.623 | 0.989 / 1.036 | 5.387 → 0.836 (+84.38% [+82.09, +85.70]) | 0.747 → 0.836 (-12.50% [-12.70, -11.77]) |
| roundtrip2 / 33 / 1 | 0.085 / 0.095 | 0.042 / 0.048 | 0.042 / 0.044 | 0.040 / 0.047 | 0.116 / 0.124 | 0.133 / 0.142 | 0.085 → 0.042 (+50.72% [+50.43, +50.97]) | 0.040 → 0.042 (-4.45% [-4.99, -2.25]) |
| roundtrip2 / 33 / 8 | 0.101 / 0.109 | 0.048 / 0.052 | 0.048 / 0.050 | 0.045 / 0.049 | 0.133 / 0.142 | 0.186 / 0.201 | 0.101 → 0.048 (+52.83% [+52.45, +53.30]) | 0.045 → 0.048 (-5.56% [-7.38, -4.67]) |
| roundtrip2 / 512 / 1 | 0.853 / 0.990 | 0.157 / 0.164 | 0.153 / 0.159 | 0.085 / 0.093 | 0.446 / 0.465 | 0.262 / 0.274 | 0.853 → 0.157 (+81.93% [+80.74, +83.17]) | 0.085 → 0.157 (-85.14% [-89.06, -80.20]) |
| roundtrip2 / 512 / 8 | 1.183 / 1.275 | 0.104 / 0.114 | 0.103 / 0.116 | 0.105 / 0.117 | 0.706 / 0.866 | 0.292 / 0.322 | 1.183 → 0.104 (+91.18% [+90.80, +91.84]) | 0.105 → 0.104 (+2.20% [-2.26, +19.45]) |
| roundtrip2 / 8192 / 1 | 20.368 / 20.759 | 2.295 / 2.380 | 2.342 / 2.432 | 0.872 / 1.089 | 8.155 / 8.534 | 1.147 / 1.342 | 20.368 → 2.295 (+88.65% [+87.96, +89.21]) | 0.872 → 2.295 (-162.67% [-177.10, -140.75]) |
| roundtrip2 / 8192 / 8 | 10.058 / 10.865 | 0.897 / 0.933 | 0.890 / 0.920 | 0.744 / 0.791 | 8.007 / 8.203 | 0.988 / 1.038 | 10.058 → 0.897 (+91.15% [+90.57, +91.79]) | 0.744 → 0.897 (-20.51% [-21.56, -19.41]) |
| roundtrip4 / 33 / 1 | 0.132 / 0.144 | 0.046 / 0.050 | 0.046 / 0.053 | 0.040 / 0.048 | 0.179 / 0.190 | 0.134 / 0.144 | 0.132 → 0.046 (+64.90% [+64.88, +65.06]) | 0.040 → 0.046 (-15.56% [-16.06, -12.38]) |
| roundtrip4 / 33 / 8 | 0.157 / 0.167 | 0.053 / 0.057 | 0.053 / 0.055 | 0.045 / 0.053 | 0.213 / 0.224 | 0.189 / 0.199 | 0.157 → 0.053 (+66.42% [+64.52, +66.61]) | 0.045 → 0.053 (-17.35% [-17.59, -17.30]) |
| roundtrip4 / 512 / 1 | 1.624 / 1.884 | 0.237 / 0.245 | 0.228 / 0.239 | 0.089 / 0.095 | 0.820 / 0.848 | 0.319 / 0.329 | 1.624 → 0.237 (+85.63% [+85.14, +87.00]) | 0.089 → 0.237 (-160.26% [-171.24, -139.82]) |
| roundtrip4 / 512 / 8 | 2.454 / 2.745 | 0.152 / 0.171 | 0.146 / 0.162 | 0.104 / 0.115 | 1.292 / 1.477 | 0.304 / 0.325 | 2.454 → 0.152 (+94.07% [+91.25, +94.42]) | 0.104 → 0.152 (-45.99% [-69.56, -19.43]) |
| roundtrip4 / 8192 / 1 | 42.219 / 42.686 | 3.247 / 3.325 | 3.214 / 3.239 | 0.884 / 1.103 | 15.595 / 15.927 | 1.565 / 1.615 | 42.219 → 3.247 (+92.31% [+91.76, +92.51]) | 0.884 → 3.247 (-268.12% [-289.04, -261.10]) |
| roundtrip4 / 8192 / 8 | 19.774 / 20.509 | 1.148 / 1.201 | 1.148 / 1.184 | 0.744 / 0.797 | 15.127 / 15.420 | 0.985 / 1.046 | 19.774 → 1.148 (+94.18% [+93.91, +94.33]) | 0.744 → 1.148 (-55.23% [-58.75, -50.87]) |
| rank3_chain / 33 / 1 | 0.324 / 0.351 | 0.337 / 0.348 | 0.334 / 0.344 | 0.107 / 0.116 | 0.119 / 0.131 | 0.197 / 0.210 | 0.324 → 0.337 (-3.86% [-7.90, +2.23]) | 0.107 → 0.337 (-215.27% [-227.93, -212.14]) |
| rank3_chain / 33 / 8 | 0.488 / 0.500 | 0.182 / 0.199 | 0.190 / 0.199 | 0.105 / 0.111 | 0.257 / 0.287 | 0.206 / 0.227 | 0.488 → 0.182 (+64.09% [+36.10, +69.79]) | 0.105 → 0.182 (-73.90% [-168.40, -22.73]) |
| rank3_chain / 512 / 1 | 5.211 / 5.513 | 4.877 / 4.891 | 4.874 / 4.890 | 1.179 / 1.189 | 1.250 / 1.367 | 0.837 / 0.884 | 5.211 → 4.877 (+6.41% [-4.93, +16.03]) | 1.179 → 4.877 (-313.58% [-314.37, -312.27]) |
| rank3_chain / 512 / 8 | 5.208 / 5.573 | 1.166 / 1.180 | 1.164 / 1.173 | 0.537 / 0.553 | 0.880 / 1.099 | 0.698 / 0.709 | 5.208 → 1.166 (+77.64% [+76.68, +80.44]) | 0.537 → 1.166 (-116.90% [-118.38, -73.90]) |
| rank3_chain / 8192 / 1 | 100.686 / 104.549 | 76.773 / 76.956 | 76.784 / 76.962 | 17.466 / 17.512 | 86.575 / 95.953 | 9.250 / 9.300 | 100.686 → 76.773 (+23.77% [+21.35, +26.12]) | 17.466 → 76.773 (-339.44% [-340.06, -338.83]) |
| rank3_chain / 8192 / 8 | 21.943 / 22.236 | 16.725 / 16.839 | 16.726 / 16.844 | 6.695 / 6.796 | 34.944 / 36.664 | 5.852 / 5.985 | 21.943 → 16.725 (+23.80% [+23.56, +23.94]) | 6.695 → 16.725 (-149.81% [-150.19, -149.45]) |
| dequant_transpose / 33 / 1 | 0.134 / 0.144 | 0.059 / 0.067 | 0.059 / 0.068 | 0.049 / 0.056 | 0.075 / 0.080 | 0.143 / 0.152 | 0.134 → 0.059 (+55.70% [+54.80, +55.84]) | 0.049 → 0.059 (-21.31% [-23.32, -20.92]) |
| dequant_transpose / 33 / 8 | 0.131 / 0.149 | 0.066 / 0.073 | 0.066 / 0.073 | 0.081 / 0.090 | 0.086 / 0.094 | 0.213 / 0.226 | 0.131 → 0.066 (+49.17% [+32.73, +61.72]) | 0.081 → 0.066 (+18.62% [+17.74, +21.30]) |
| dequant_transpose / 512 / 1 | 1.589 / 1.672 | 0.273 / 0.290 | 0.271 / 0.284 | 0.213 / 0.224 | 0.202 / 0.224 | 0.333 / 0.349 | 1.589 → 0.273 (+82.78% [+81.70, +82.82]) | 0.213 → 0.273 (-28.98% [-32.37, -28.04]) |
| dequant_transpose / 512 / 8 | 0.560 / 0.598 | 0.232 / 0.254 | 0.199 / 0.252 | 0.161 / 0.186 | 0.316 / 0.406 | 0.304 / 0.329 | 0.560 → 0.232 (+62.38% [+52.48, +68.50]) | 0.161 → 0.232 (-43.59% [-55.67, +6.52]) |
| dequant_transpose / 8192 / 1 | 25.642 / 26.607 | 4.577 / 4.673 | 4.563 / 4.664 | 4.649 / 4.983 | 3.294 / 3.474 | 2.335 / 2.357 | 25.642 → 4.577 (+82.21% [+81.97, +83.15]) | 4.649 → 4.577 (+1.56% [-6.55, +8.39]) |
| dequant_transpose / 8192 / 8 | 6.106 / 6.205 | 1.876 / 1.905 | 1.629 / 1.903 | 1.615 / 1.638 | 2.919 / 3.086 | 0.925 / 0.941 | 6.106 → 1.876 (+69.37% [+69.03, +74.97]) | 1.615 → 1.876 (-16.52% [-17.14, +7.54]) |
| padded_roundtrip / 33 / 1 | 0.097 / 0.106 | 0.091 / 0.101 | 0.091 / 0.099 | 0.059 / 0.068 | 0.089 / 0.098 | 0.150 / 0.160 | 0.097 → 0.091 (+6.01% [+4.44, +6.63]) | 0.059 → 0.091 (-54.54% [-56.72, -52.76]) |
| padded_roundtrip / 33 / 8 | 0.109 / 0.118 | 0.096 / 0.102 | 0.093 / 0.104 | 0.083 / 0.091 | 0.101 / 0.110 | 0.197 / 0.209 | 0.109 → 0.096 (+11.29% [+10.81, +16.20]) | 0.083 → 0.096 (-15.29% [-20.10, +4.83]) |
| padded_roundtrip / 512 / 1 | 0.783 / 0.886 | 0.670 / 0.743 | 0.664 / 0.736 | 0.362 / 0.373 | 0.268 / 0.295 | 0.433 / 0.459 | 0.783 → 0.670 (+13.62% [+13.55, +18.25]) | 0.362 → 0.670 (-87.46% [-92.30, -72.98]) |
| padded_roundtrip / 512 / 8 | 0.959 / 1.072 | 0.593 / 0.669 | 0.589 / 0.643 | 0.219 / 0.230 | 0.400 / 0.447 | 0.332 / 0.362 | 0.959 → 0.593 (+37.52% [+32.58, +41.40]) | 0.219 → 0.593 (-170.34% [-193.96, -162.95]) |
| padded_roundtrip / 8192 / 1 | 11.975 / 13.810 | 9.198 / 10.343 | 8.818 / 10.087 | 5.436 / 5.523 | 3.746 / 4.041 | 3.745 / 3.818 | 11.975 → 9.198 (+23.19% [+21.98, +24.72]) | 5.436 → 9.198 (-71.25% [-71.65, -65.57]) |
| padded_roundtrip / 8192 / 8 | 11.208 / 11.816 | 7.339 / 8.526 | 7.199 / 8.452 | 2.072 / 2.611 | 3.130 / 3.490 | 1.873 / 2.037 | 11.208 → 7.339 (+34.28% [+30.36, +36.30]) | 2.072 → 7.339 (-253.35% [-295.44, -187.42]) |

## E2 · Matched request attribution

The timer covers the full completed model request. The nominal producer
envelope is 128 MiB with two slots. Inductor arms share compiled consumer
functions. Source snapshot, first calls and diagnostic counters sit outside
the warm 60 samples. These use five process rounds and three warmups with
no warm callbacks. Conversion, grouping and window effects interact.
The archive's factorial interaction table retains those contrasts.

| Cell | Contrast | Control → candidate | p50_ms | p95_ms |
| --- | --- | --- | --- | --- |
| float16 / llm_decode | runtime | torch_eager_native_grouped_step → sym_eager_native_grouped_step | 19.585 → 22.363 (-13.99% [-14.41, -10.73]) | 19.672 → 22.538 (-14.57% [-14.83, -11.05]) |
| float16 / llm_decode | runtime | torch_eager_native_grouped_whole → sym_eager_native_grouped_whole | 19.008 → 21.066 (-10.60% [-11.26, -10.53]) | 19.146 → 21.277 (-11.03% [-11.87, -10.31]) |
| float16 / llm_decode | runtime | torch_eager_native_split_step → sym_eager_native_split_step | 24.376 → 29.551 (-21.05% [-21.49, -18.33]) | 24.520 → 29.656 (-21.20% [-21.53, -17.07]) |
| float16 / llm_decode | runtime | torch_eager_native_split_whole → sym_eager_native_split_whole | 23.615 → 28.214 (-19.21% [-19.94, -18.87]) | 23.796 → 28.354 (-19.11% [-20.21, -15.50]) |
| float16 / llm_decode | runtime | torch_inductor_native_grouped_step → sym_inductor_native_grouped_step | 20.857 → 22.376 (-7.54% [-9.82, -7.19]) | 20.935 → 22.460 (-7.74% [-9.36, -7.01]) |
| float16 / llm_decode | runtime | torch_inductor_native_grouped_whole → sym_inductor_native_grouped_whole | 17.658 → 19.561 (-10.89% [-11.83, -10.40]) | 17.754 → 19.664 (-10.46% [-11.70, -9.93]) |
| float16 / llm_decode | runtime | torch_inductor_native_split_step → sym_inductor_native_split_step | 25.634 → 29.564 (-15.44% [-17.58, -15.16]) | 25.677 → 29.628 (-15.55% [-17.24, -15.16]) |
| float16 / llm_decode | runtime | torch_inductor_native_split_whole → sym_inductor_native_split_whole | 22.148 → 26.758 (-20.75% [-21.39, -20.31]) | 22.186 → 26.819 (-21.40% [-21.56, -20.26]) |
| float16 / llm_decode | native conversion path | torch_eager_arithmetic_grouped_whole → torch_eager_native_grouped_whole | 23.917 → 19.008 (+20.64% [+20.34, +24.65]) | 23.964 → 19.146 (+20.51% [+19.90, +24.26]) |
| float16 / llm_decode | native conversion path | torch_inductor_arithmetic_grouped_whole → torch_inductor_native_grouped_whole | 22.714 → 17.658 (+22.47% [+21.84, +22.92]) | 22.856 → 17.754 (+22.38% [+21.74, +22.78]) |
| float16 / llm_prefill | runtime | torch_eager_native_grouped_whole → sym_eager_native_grouped_whole | 4.768 → 5.038 (-5.70% [-6.17, -5.43]) | 4.805 → 5.095 (-5.87% [-6.93, -5.43]) |
| float16 / llm_prefill | runtime | torch_inductor_native_grouped_whole → sym_inductor_native_grouped_whole | 4.824 → 5.076 (-5.22% [-6.87, -4.00]) | 4.862 → 5.187 (-6.68% [-7.59, -4.05]) |
| float16 / llm_prefill | native conversion path | torch_eager_arithmetic_grouped_whole → torch_eager_native_grouped_whole | 6.377 → 4.768 (+25.19% [+25.16, +25.30]) | 6.403 → 4.805 (+25.06% [+24.84, +26.57]) |
| float16 / llm_prefill | native conversion path | torch_inductor_arithmetic_grouped_whole → torch_inductor_native_grouped_whole | 5.477 → 4.824 (+12.20% [+11.37, +12.35]) | 5.532 → 4.862 (+12.47% [+10.26, +12.86]) |
| float16 / moe_sparse | runtime | torch_eager_native_grouped_whole → sym_eager_native_grouped_whole | 2.730 → 3.018 (-10.58% [-11.01, -10.30]) | 2.768 → 3.064 (-10.54% [-11.10, -10.53]) |
| float16 / moe_sparse | runtime | torch_inductor_native_grouped_whole → sym_inductor_native_grouped_whole | 3.698 → 3.877 (-4.89% [-5.08, -4.53]) | 3.755 → 3.958 (-5.21% [-6.26, -4.84]) |
| float16 / moe_sparse | native conversion path | torch_eager_arithmetic_grouped_whole → torch_eager_native_grouped_whole | 3.093 → 2.730 (+11.80% [+11.63, +11.99]) | 3.152 → 2.768 (+12.07% [+12.04, +12.49]) |
| float16 / moe_sparse | native conversion path | torch_inductor_arithmetic_grouped_whole → torch_inductor_native_grouped_whole | 4.326 → 3.698 (+14.39% [+14.26, +14.67]) | 4.368 → 3.755 (+13.89% [+13.40, +14.64]) |
| float16 / moe_dense | runtime | torch_eager_native_grouped_whole → sym_eager_native_grouped_whole | 9.271 → 9.984 (-8.07% [-8.33, -7.56]) | 9.324 → 10.092 (-8.24% [-8.55, -7.86]) |
| float16 / moe_dense | runtime | torch_inductor_native_grouped_whole → sym_inductor_native_grouped_whole | 12.351 → 12.922 (-4.62% [-4.85, -4.17]) | 12.430 → 13.021 (-4.76% [-5.07, -4.55]) |
| float16 / moe_dense | native conversion path | torch_eager_arithmetic_grouped_whole → torch_eager_native_grouped_whole | 10.330 → 9.271 (+10.32% [+10.23, +10.75]) | 10.391 → 9.324 (+10.14% [+10.05, +10.60]) |
| float16 / moe_dense | native conversion path | torch_inductor_arithmetic_grouped_whole → torch_inductor_native_grouped_whole | 14.872 → 12.351 (+16.95% [+16.77, +17.10]) | 14.957 → 12.430 (+16.81% [+16.60, +17.44]) |
| float32 / llm_decode | runtime | torch_eager_native_grouped_whole → sym_eager_native_grouped_whole | 19.067 → 20.631 (-8.20% [-9.01, -7.24]) | 19.164 → 20.829 (-8.16% [-9.08, -7.27]) |
| float32 / llm_decode | runtime | torch_inductor_native_grouped_whole → sym_inductor_native_grouped_whole | 18.443 → 19.759 (-6.91% [-8.44, -4.55]) | 18.503 → 19.826 (-7.12% [-8.28, -4.66]) |
| float32 / llm_decode | native conversion path | torch_eager_arithmetic_grouped_whole → torch_eager_native_grouped_whole | 22.841 → 19.067 (+16.49% [+12.23, +17.08]) | 22.924 → 19.164 (+15.99% [+11.97, +17.09]) |
| float32 / llm_decode | native conversion path | torch_inductor_arithmetic_grouped_whole → torch_inductor_native_grouped_whole | 22.753 → 18.443 (+18.60% [+17.08, +19.23]) | 22.862 → 18.503 (+18.82% [+17.17, +19.33]) |
| float16 / llm_decode | KV grouping | torch_eager_native_split_whole → torch_eager_native_grouped_whole | 23.615 → 19.008 (+19.54% [+19.51, +19.55]) | 23.796 → 19.146 (+19.54% [+19.29, +22.24]) |
| float16 / llm_decode | KV grouping | torch_eager_native_split_step → torch_eager_native_grouped_step | 24.376 → 19.585 (+19.58% [+19.21, +19.65]) | 24.520 → 19.672 (+19.57% [+19.19, +20.03]) |
| float16 / llm_decode | KV grouping | torch_inductor_native_split_whole → torch_inductor_native_grouped_whole | 22.148 → 17.658 (+20.54% [+20.15, +20.66]) | 22.186 → 17.754 (+20.07% [+19.70, +20.54]) |
| float16 / llm_decode | KV grouping | torch_inductor_native_split_step → torch_inductor_native_grouped_step | 25.634 → 20.857 (+18.71% [+18.63, +19.21]) | 25.677 → 20.935 (+18.57% [+18.33, +19.12]) |
| float16 / llm_decode | KV grouping | sym_eager_native_split_whole → sym_eager_native_grouped_whole | 28.214 → 21.066 (+25.32% [+25.14, +25.40]) | 28.354 → 21.277 (+25.04% [+24.81, +25.38]) |
| float16 / llm_decode | KV grouping | sym_eager_native_split_step → sym_eager_native_grouped_step | 29.551 → 22.363 (+24.33% [+24.18, +24.40]) | 29.656 → 22.538 (+24.00% [+23.90, +24.50]) |
| float16 / llm_decode | KV grouping | sym_inductor_native_split_whole → sym_inductor_native_grouped_whole | 26.758 → 19.561 (+26.90% [+26.80, +26.99]) | 26.819 → 19.664 (+26.88% [+26.45, +27.30]) |
| float16 / llm_decode | KV grouping | sym_inductor_native_split_step → sym_inductor_native_grouped_step | 29.564 → 22.376 (+24.33% [+24.27, +24.48]) | 29.628 → 22.460 (+24.20% [+24.08, +24.35]) |
| float16 / llm_decode | prefetch window | torch_eager_native_grouped_step → torch_eager_native_grouped_whole | 19.585 → 19.008 (+3.12% [+2.76, +5.59]) | 19.672 → 19.146 (+3.12% [+2.50, +5.12]) |
| float16 / llm_decode | prefetch window | torch_eager_native_split_step → torch_eager_native_split_whole | 24.376 → 23.615 (+3.20% [+2.90, +5.23]) | 24.520 → 23.796 (+3.06% [+0.09, +4.71]) |
| float16 / llm_decode | prefetch window | torch_inductor_native_grouped_step → torch_inductor_native_grouped_whole | 20.857 → 17.658 (+15.21% [+13.57, +15.34]) | 20.935 → 17.754 (+14.56% [+13.65, +15.34]) |
| float16 / llm_decode | prefetch window | torch_inductor_native_split_step → torch_inductor_native_split_whole | 25.634 → 22.148 (+13.37% [+12.04, +13.60]) | 25.677 → 22.186 (+13.32% [+12.02, +13.60]) |
| float16 / llm_decode | prefetch window | sym_eager_native_grouped_step → sym_eager_native_grouped_whole | 22.363 → 21.066 (+5.71% [+5.59, +5.84]) | 22.538 → 21.277 (+5.42% [+5.34, +6.13]) |
| float16 / llm_decode | prefetch window | sym_eager_native_split_step → sym_eager_native_split_whole | 29.551 → 28.214 (+4.40% [+4.37, +4.59]) | 29.656 → 28.354 (+4.26% [+4.00, +4.90]) |
| float16 / llm_decode | prefetch window | sym_inductor_native_grouped_step → sym_inductor_native_grouped_whole | 22.376 → 19.561 (+12.65% [+12.03, +12.79]) | 22.460 → 19.664 (+12.69% [+11.73, +12.82]) |
| float16 / llm_decode | prefetch window | sym_inductor_native_split_step → sym_inductor_native_split_whole | 29.564 → 26.758 (+9.52% [+9.09, +9.57]) | 29.628 → 26.819 (+9.21% [+8.77, +9.49]) |

## E3 · Revisit trace and fixed paths

All six arms execute the original 12 requests five times, with empty and
singleton cases, changed stride, live parameters and live indices. Arms
share a process and compiler cache. The first cycle includes compile and
preparation. The repeated distribution uses the final 48 completed calls,
including any fallback. Native execution counts and capture exclusions
describe different event populations. Live dequantization changes its
parameter revision on every eligible explicit call. Both explicit index
arms retain their native descriptor cache, limiting that contrast to the
timing of those specific paths.

| Case | Arm | First cycle ms | All 60 ms | Repeated 48 ms | Repeated p50/p95 ms | Dynamo callbacks | Native calls | Explicit prepares / hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layout_cast | torch | 4.529 | 21.106 | 17.260 | 0.272 / 1.049 | 0 | — | — |
| layout_cast | inductor | 8094.195 | 8118.903 | 24.708 | 0.433 / 1.372 | 4 | — | — |
| layout_cast | sym_auto_eager | 348.244 | 383.440 | 36.294 | 0.815 / 1.206 | 4 | 45 | — |
| layout_cast | sym_auto_inductor | 1679.770 | 1711.942 | 32.680 | 0.736 / 1.119 | 4 | 45 | — |
| layout_cast | sym_fast_reprepare | 21.810 | 39.055 | 18.673 | 0.390 / 0.628 | 0 | — | 50/0 |
| layout_cast | sym_fast_cache | 15.736 | 26.586 | 11.144 | 0.225 / 0.433 | 0 | — | 6/44 |
| kv_restore | torch | 3.292 | 16.370 | 13.318 | 0.251 / 0.547 | 0 | — | — |
| kv_restore | inductor | 1536.663 | 1554.552 | 17.889 | 0.330 / 0.678 | 4 | — | — |
| kv_restore | sym_auto_eager | 309.524 | 340.511 | 31.395 | 0.678 / 1.134 | 4 | 45 | — |
| kv_restore | sym_auto_inductor | 595.220 | 624.624 | 28.056 | 0.604 / 1.078 | 4 | 45 | — |
| kv_restore | sym_fast_reprepare | 12.383 | 27.790 | 15.250 | 0.320 / 0.549 | 0 | — | 50/0 |
| kv_restore | sym_fast_cache | 9.987 | 21.874 | 11.905 | 0.239 / 0.414 | 0 | — | 6/44 |
| dequant_live | torch | 3.333 | 15.910 | 12.517 | 0.202 / 0.554 | 0 | — | — |
| dequant_live | inductor | 2493.830 | 2510.786 | 16.946 | 0.364 / 0.473 | 3 | — | — |
| dequant_live | sym_auto_eager | 235.432 | 272.559 | 37.127 | 0.820 / 1.045 | 3 | 50 | — |
| dequant_live | sym_auto_inductor | 476.972 | 511.592 | 34.952 | 0.786 / 1.001 | 3 | 50 | — |
| dequant_live | sym_fast_reprepare | 11.261 | 27.737 | 16.525 | 0.343 / 0.481 | 0 | — | 55/0 |
| dequant_live | sym_fast_cache | 10.339 | 26.530 | 16.191 | 0.326 / 0.490 | 0 | — | 55/0 |
| index_live | torch | 1.252 | 5.919 | 4.694 | 0.093 / 0.215 | 0 | — | — |
| index_live | inductor | 3178.243 | 3196.492 | 18.430 | 0.406 / 0.488 | 4 | — | — |
| index_live | sym_auto_eager | 276.950 | 316.858 | 39.908 | 0.842 / 1.095 | 4 | 55 | — |
| index_live | sym_auto_inductor | 569.607 | 604.634 | 35.416 | 0.733 / 1.024 | 4 | 55 | — |
| index_live | sym_fast_reprepare | 8.713 | 17.210 | 8.497 | 0.168 / 0.249 | 0 | — | 1/54 |
| index_live | sym_fast_cache | 8.288 | 16.729 | 8.175 | 0.156 / 0.264 | 0 | — | 1/54 |

| Cell | Contrast | Control → candidate | repeated48_ms | p95_ms |
| --- | --- | --- | --- | --- |
| layout_cast | automatic eager | torch → sym_auto_eager | 17.260 → 36.294 (-106.12% [-140.83, -64.31]) | 1.049 → 1.206 (-15.75% [-71.57, +41.46]) |
| layout_cast | automatic Inductor versus eager, original reference | torch → sym_auto_inductor | 17.260 → 32.680 (-89.75% [-119.23, -47.95]) | 1.049 → 1.119 (+6.02% [-65.90, +45.49]) |
| layout_cast | automatic Inductor versus matched compute | inductor → sym_auto_inductor | 24.708 → 32.680 (-30.21% [-38.20, -20.29]) | 1.372 → 1.119 (+18.17% [+10.14, +31.79]) |
| layout_cast | explicit cache | sym_fast_reprepare → sym_fast_cache | 18.673 → 11.144 (+37.66% [+35.50, +45.63]) | 0.628 → 0.433 (+32.09% [+14.42, +34.23]) |
| kv_restore | automatic eager | torch → sym_auto_eager | 13.318 → 31.395 (-144.69% [-157.39, -127.34]) | 0.547 → 1.134 (-111.20% [-127.29, -29.00]) |
| kv_restore | automatic Inductor versus eager, original reference | torch → sym_auto_inductor | 13.318 → 28.056 (-120.78% [-133.72, -105.35]) | 0.547 → 1.078 (-82.84% [-99.03, -27.75]) |
| kv_restore | automatic Inductor versus matched compute | inductor → sym_auto_inductor | 17.889 → 28.056 (-59.77% [-70.45, -37.61]) | 0.678 → 1.078 (-40.22% [-64.65, -2.59]) |
| kv_restore | explicit cache | sym_fast_reprepare → sym_fast_cache | 15.250 → 11.905 (+22.42% [-10.39, +23.53]) | 0.549 → 0.414 (+24.11% [+7.89, +25.02]) |
| dequant_live | automatic eager | torch → sym_auto_eager | 12.517 → 37.127 (-196.23% [-218.99, -192.41]) | 0.554 → 1.045 (-88.46% [-121.13, -80.44]) |
| dequant_live | automatic Inductor versus eager, original reference | torch → sym_auto_inductor | 12.517 → 34.952 (-182.78% [-193.28, -177.75]) | 0.554 → 1.001 (-94.72% [-105.74, -75.88]) |
| dequant_live | automatic Inductor versus matched compute | inductor → sym_auto_inductor | 16.946 → 34.952 (-106.25% [-118.29, -104.18]) | 0.473 → 1.001 (-122.37% [-170.56, -87.67]) |
| dequant_live | explicit cache | sym_fast_reprepare → sym_fast_cache | 16.525 → 16.191 (+3.59% [+0.20, +6.67]) | 0.481 → 0.490 (+6.88% [-16.27, +9.61]) |
| index_live | automatic eager | torch → sym_auto_eager | 4.694 → 39.908 (-728.64% [-760.41, -713.30]) | 0.215 → 1.095 (-392.85% [-547.57, -354.58]) |
| index_live | automatic Inductor versus eager, original reference | torch → sym_auto_inductor | 4.694 → 35.416 (-641.54% [-656.42, -621.76]) | 0.215 → 1.024 (-356.70% [-505.40, -310.42]) |
| index_live | automatic Inductor versus matched compute | inductor → sym_auto_inductor | 18.430 → 35.416 (-92.67% [-96.75, -81.69]) | 0.488 → 1.024 (-88.36% [-134.57, -45.73]) |
| index_live | explicit cache | sym_fast_reprepare → sym_fast_cache | 8.497 → 8.175 (+4.45% [-2.31, +15.21]) | 0.249 → 0.264 (+3.51% [-49.88, +36.96]) |

## E4 · Physical capacity and OOMs

A direct CUDA reservation sets actual free VRAM after context initialization.
Warm time excludes setup, residency preload, first request, three warmups
and oracle work. Three fresh processes per cell record 20 samples on success.
The packed resident arm is a separate supplement that changes preload and
storage allocation. Success counts and OOM phases stay with each policy.

| Layers / context / budget MiB | Policy | Success | OOM phase (rounds) | Achieved free MiB | Preload median ms | First ms | Warm p50/p95 ms | Max observed app increment MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 / 16 / 512 | offload_sym | 3/3 | — | 511.56–511.56 | 0.066 | 35.821 | 18.423 / 18.570 | 244.0 |
| 16 / 16 / 512 | offload_torch | 3/3 | — | 511.56–511.56 | 0.071 | 21.000 | 18.117 / 18.399 | 232.0 |
| 16 / 16 / 512 | partial_sym | 3/3 | — | 511.56–511.56 | 22.890 | 17.066 | 14.647 / 14.752 | 444.0 |
| 16 / 16 / 512 | partial_torch | 3/3 | — | 511.56–511.56 | 9.463 | 16.363 | 14.519 / 14.677 | 412.0 |
| 16 / 16 / 512 | resident | 0/3 | residency_start (3) | 511.56–511.56 | — | — | — / — | 492.0 |
| 16 / 16 / 1024 | offload_sym | 3/3 | — | 1023.56–1023.56 | 0.077 | 35.679 | 18.392 / 18.532 | 244.0 |
| 16 / 16 / 1024 | offload_torch | 3/3 | — | 1023.56–1023.56 | 0.072 | 21.166 | 18.249 / 18.394 | 212.0 |
| 16 / 16 / 1024 | partial_sym | 3/3 | — | 1023.56–1023.56 | 22.776 | 17.006 | 14.643 / 14.803 | 444.0 |
| 16 / 16 / 1024 | partial_torch | 3/3 | — | 1023.56–1023.56 | 9.685 | 16.646 | 14.499 / 14.644 | 412.0 |
| 16 / 16 / 1024 | resident | 3/3 | — | 1023.56–1023.56 | 18.076 | 11.126 | 10.725 / 10.813 | 512.0 |
| 16 / 512 / 512 | offload_sym | 3/3 | — | 511.56–511.56 | 0.067 | 40.890 | 19.100 / 20.661 | 304.0 |
| 16 / 512 / 512 | offload_torch | 3/3 | — | 511.56–511.56 | 0.070 | 25.575 | 19.070 / 20.999 | 272.0 |
| 16 / 512 / 512 | partial_sym | 3/3 | — | 511.56–511.56 | 23.023 | 21.384 | 15.426 / 15.846 | 504.0 |
| 16 / 512 / 512 | partial_torch | 3/3 | — | 511.56–511.56 | 9.433 | 20.442 | 15.500 / 16.820 | 472.0 |
| 16 / 512 / 512 | resident | 0/3 | residency_start (3) | 511.56–511.56 | — | — | — / — | 496.0 |
| 16 / 512 / 1024 | offload_sym | 3/3 | — | 1023.56–1023.56 | 0.070 | 40.788 | 19.028 / 19.367 | 304.0 |
| 16 / 512 / 1024 | offload_torch | 3/3 | — | 1023.56–1023.56 | 0.071 | 25.436 | 19.027 / 20.396 | 292.0 |
| 16 / 512 / 1024 | partial_sym | 3/3 | — | 1023.56–1023.56 | 22.694 | 21.555 | 15.410 / 15.951 | 504.0 |
| 16 / 512 / 1024 | partial_torch | 3/3 | — | 1023.56–1023.56 | 9.457 | 20.696 | 15.486 / 16.175 | 472.0 |
| 16 / 512 / 1024 | resident | 3/3 | — | 1023.56–1023.56 | 18.173 | 15.378 | 11.912 / 12.937 | 572.0 |
| 32 / 16 / 512 | offload_sym | 3/3 | — | 511.56–511.56 | 0.069 | 58.353 | 36.472 / 36.855 | 246.0 |
| 32 / 16 / 512 | offload_torch | 3/3 | — | 511.56–511.56 | 0.080 | 40.163 | 36.329 / 36.453 | 234.0 |
| 32 / 16 / 512 | partial_sym | 3/3 | — | 511.56–511.56 | 25.133 | 37.498 | 32.693 / 33.061 | 446.0 |
| 32 / 16 / 512 | partial_torch | 3/3 | — | 511.56–511.56 | 9.615 | 34.842 | 32.521 / 32.845 | 414.0 |
| 32 / 16 / 512 | resident | 0/3 | residency_start (3) | 511.56–511.56 | — | — | — / — | 492.0 |
| 32 / 16 / 1024 | offload_sym | 3/3 | — | 1023.56–1023.56 | 0.080 | 58.847 | 36.338 / 36.639 | 246.0 |
| 32 / 16 / 1024 | offload_torch | 3/3 | — | 1023.56–1023.56 | 0.120 | 39.883 | 36.002 / 36.399 | 234.0 |
| 32 / 16 / 1024 | partial_sym | 3/3 | — | 1023.56–1023.56 | 26.358 | 36.942 | 32.301 / 32.558 | 446.0 |
| 32 / 16 / 1024 | partial_torch | 3/3 | — | 1023.56–1023.56 | 9.606 | 35.308 | 32.659 / 32.975 | 414.0 |
| 32 / 16 / 1024 | resident | 3/3 | — | 1023.56–1023.56 | 35.497 | 22.376 | 21.500 / 21.585 | 914.0 |
| 32 / 512 / 512 | offload_sym | 3/3 | — | 511.56–511.56 | 0.075 | 67.698 | 36.948 / 40.665 | 336.0 |
| 32 / 512 / 512 | offload_torch | 3/3 | — | 511.56–511.56 | 0.115 | 47.931 | 37.111 / 40.883 | 324.0 |
| 32 / 512 / 512 | partial_sym | 0/3 | first_request (3) | 511.56–511.56 | 26.339 | — | — / — | 510.0 |
| 32 / 512 / 512 | partial_torch | 3/3 | — | 511.56–511.56 | 9.655 | 42.640 | 33.482 / 36.566 | 504.0 |
| 32 / 512 / 512 | resident | 0/3 | residency_start (3) | 511.56–511.56 | — | — | — / — | 496.0 |
| 32 / 512 / 1024 | offload_sym | 3/3 | — | 1023.56–1023.56 | 0.091 | 68.114 | 37.252 / 38.816 | 336.0 |
| 32 / 512 / 1024 | offload_torch | 3/3 | — | 1023.56–1023.56 | 0.114 | 48.002 | 37.117 / 39.667 | 324.0 |
| 32 / 512 / 1024 | partial_sym | 3/3 | — | 1023.56–1023.56 | 26.328 | 45.681 | 33.391 / 36.769 | 536.0 |
| 32 / 512 / 1024 | partial_torch | 3/3 | — | 1023.56–1023.56 | 9.710 | 42.853 | 33.309 / 34.160 | 504.0 |
| 32 / 512 / 1024 | resident | 3/3 | — | 1023.56–1023.56 | 35.528 | 29.984 | 22.287 / 24.105 | 1004.0 |

| Layers / context / budget MiB | Policy | Success | OOM phase (rounds) | Achieved free MiB | Preload median ms | First ms | Warm p50/p95 ms | Max observed app increment MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 / 16 / 512 | resident_packed | 3/3 | — | 511.56–511.56 | 78.361 | 11.253 | 10.775 / 10.840 | 496.0 |
| 16 / 512 / 512 | resident_packed | 0/3 | first_request (3) | 511.56–511.56 | 77.739 | — | — / — | 510.0 |

| Cell | Contrast | Control → candidate | p50_ms | p95_ms |
| --- | --- | --- | --- | --- |
| 16 / 16 / 512 | partial | partial_torch → partial_sym | 14.519 → 14.647 (-1.37% [-1.78, +0.54]) | 14.677 → 14.752 (-1.25% [-1.47, +0.15]) |
| 16 / 16 / 512 | offload | offload_torch → offload_sym | 18.117 → 18.423 (-1.88% [-9.58, +0.41]) | 18.399 → 18.570 (+0.06% [-10.56, +4.04]) |
| 16 / 16 / 1024 | partial | partial_torch → partial_sym | 14.499 → 14.643 (-0.27% [-1.50, -0.06]) | 14.644 → 14.803 (-1.18% [-1.48, -0.15]) |
| 16 / 16 / 1024 | offload | offload_torch → offload_sym | 18.249 → 18.392 (-0.96% [-1.10, -0.15]) | 18.394 → 18.532 (-1.18% [-1.58, -0.10]) |
| 16 / 512 / 512 | partial | partial_torch → partial_sym | 15.500 → 15.426 (+0.74% [+0.48, +1.10]) | 16.820 → 15.846 (+6.80% [-0.37, +7.76]) |
| 16 / 512 / 512 | offload | offload_torch → offload_sym | 19.070 → 19.100 (+0.26% [-0.51, +1.77]) | 20.999 → 20.661 (+1.61% [+1.17, +2.04]) |
| 16 / 512 / 1024 | partial | partial_torch → partial_sym | 15.486 → 15.410 (+0.49% [-0.46, +1.09]) | 16.175 → 15.951 (+3.11% [-1.32, +6.51]) |
| 16 / 512 / 1024 | offload | offload_torch → offload_sym | 19.027 → 19.028 (-0.46% [-1.12, +1.22]) | 20.396 → 19.367 (+5.02% [+0.26, +7.99]) |
| 32 / 16 / 512 | partial | partial_torch → partial_sym | 32.521 → 32.693 (-0.38% [-0.97, -0.03]) | 32.845 → 33.061 (-0.69% [-1.35, -0.20]) |
| 32 / 16 / 512 | offload | offload_torch → offload_sym | 36.329 → 36.472 (-0.39% [-1.19, -0.32]) | 36.453 → 36.855 (-0.80% [-1.10, -0.56]) |
| 32 / 16 / 1024 | partial | partial_torch → partial_sym | 32.659 → 32.301 (+0.76% [-0.04, +1.38]) | 32.975 → 32.558 (+0.49% [+0.06, +1.68]) |
| 32 / 16 / 1024 | offload | offload_torch → offload_sym | 36.002 → 36.338 (-0.93% [-1.36, +0.26]) | 36.399 → 36.639 (-0.66% [-2.31, +0.32]) |
| 32 / 512 / 512 | partial | partial_torch → partial_sym | OOM | OOM |
| 32 / 512 / 512 | offload | offload_torch → offload_sym | 37.111 → 36.948 (-0.25% [-0.60, +0.84]) | 40.883 → 40.665 (+0.64% [-8.80, +8.28]) |
| 32 / 512 / 1024 | partial | partial_torch → partial_sym | 33.309 → 33.391 (-0.24% [-2.50, +1.95]) | 34.160 → 36.769 (-7.49% [-9.69, -1.06]) |
| 32 / 512 / 1024 | offload | offload_torch → offload_sym | 37.117 → 37.252 (-0.92% [-5.33, +0.72]) | 39.667 → 38.816 (+2.09% [-0.54, +2.14]) |

## E5 · Lifecycle, finite bursts and memory

Full120 charges launch, context, fixtures, snapshots and owners, 100
initial requests, ten after a shape change, ten after a scale revision,
and retirement including held-output release. Inductor's first requests
include lazy compile where observed. Shape changes also change model
work. Warm distributions use initial calls 2–100.

E5's separate concurrent trial applies the 1024 MiB physical reservation
with eight aggregate threads, 128 MiB scratch/staging, and one, two or
four callers. Process statistics cover 30 batches of eight fixed inputs.
Burst throughput excludes validation and gaps between batches. Sojourn
includes queue and execution. Three process pairs per cell describe
finite bursts.

| Workload / compute | Arm | Setup ms | First ms | Prefix 10 / 100 ms | Shape first / scale first ms | Warm p50/p95 ms | Retirement ms | Full120 ms | Observed deployment peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_decode / eager | sym | 3919.274 | 331.819 | 4435.397 / 6333.213 | 32.741 / 21.817 | 21.078 / 21.366 | 144.593 | 6937.096 | 944.4 |
| llm_decode / eager | torch | 3894.462 | 300.218 | 4396.586 / 6148.065 | 25.146 / 20.048 | 19.414 / 19.758 | 142.857 | 6703.896 | 912.4 |
| llm_decode / inductor | sym | 4318.037 | 17709.584 | 22223.280 / 24015.868 | 28.289 / 20.383 | 19.749 / 20.024 | 285.444 | 24679.952 | 944.4 |
| llm_decode / inductor | torch | 4319.258 | 17777.614 | 22245.294 / 23851.239 | 24.114 / 18.774 | 17.843 / 18.024 | 240.448 | 24506.906 | 912.4 |
| moe_sparse / eager | sym | 4073.732 | 286.228 | 4380.843 / 4646.160 | 25.252 / 10.313 | 2.951 / 3.045 | 124.920 | 4999.035 | 814.4 |
| moe_sparse / eager | torch | 4065.344 | 250.039 | 4353.052 / 4597.052 | 24.086 / 9.594 | 2.708 / 2.800 | 124.209 | 4931.153 | 800.4 |
| moe_sparse / inductor | sym | 4503.010 | 4703.722 | 9247.759 / 9598.343 | 2743.035 / 13.940 | 3.883 / 4.019 | 201.077 | 12807.443 | 814.4 |
| moe_sparse / inductor | torch | 4475.247 | 4625.446 | 9113.633 / 9454.666 | 2729.727 / 13.173 | 3.697 / 4.016 | 207.330 | 12655.833 | 800.4 |

| Cell | Contrast | Control → candidate | full120_ms | p50_ms | p95_ms |
| --- | --- | --- | --- | --- | --- |
| llm_decode / eager | matched lifecycle | torch → sym | 6703.896 → 6937.096 (-3.51% [-4.55, -2.68]) | 19.414 → 21.078 (-8.54% [-8.98, -7.94]) | 19.758 → 21.366 (-6.91% [-9.79, -0.43]) |
| llm_decode / inductor | matched lifecycle | torch → sym | 24506.906 → 24679.952 (-0.71% [-1.95, -0.38]) | 17.843 → 19.749 (-11.13% [-11.16, -9.67]) | 18.024 → 20.024 (-11.22% [-11.26, -9.56]) |
| moe_sparse / eager | matched lifecycle | torch → sym | 4931.153 → 4999.035 (-1.38% [-1.77, -0.18]) | 2.708 → 2.951 (-9.02% [-9.28, -7.29]) | 2.800 → 3.045 (-8.75% [-9.80, +16.11]) |
| moe_sparse / inductor | matched lifecycle | torch → sym | 12655.833 → 12807.443 (-1.26% [-1.32, -0.78]) | 3.697 → 3.883 (-5.24% [-5.54, -5.02]) | 4.016 → 4.019 (-0.09% [-5.36, +1.17]) |

| Cell | Contrast | Control → candidate | within_burst_rps | service_p50_ms | service_p95_ms | sojourn_p95_ms | queue_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_decode / 1 | matched finite bursts | torch → sym | 50.977 → 47.530 (-6.80% [-6.93, -6.76]) | 19.475 → 20.896 (-7.35% [-7.47, -7.30]) | 19.666 → 21.138 (-7.48% [-7.55, -7.36]) | 156.477 → 167.914 (-7.45% [-7.57, -7.31]) | 136.880 → 147.033 (-7.42% [-7.61, -7.39]) |
| llm_decode / 2 | matched finite bursts | torch → sym | 54.916 → 52.244 (-4.87% [-6.39, -2.96]) | 35.131 → 37.439 (-6.72% [-7.07, -6.51]) | 39.945 → 40.188 (-0.61% [-6.00, +2.13]) | 145.823 → 152.953 (-4.94% [-6.20, -0.36]) | 108.683 → 115.761 (-6.74% [-7.63, +4.11]) |
| llm_decode / 4 | matched finite bursts | torch → sym | 48.546 → 45.487 (-6.49% [-6.73, -5.72]) | 79.370 → 83.936 (-5.61% [-6.05, -5.27]) | 90.757 → 98.161 (-8.12% [-10.25, -8.09]) | 164.675 → 176.271 (-7.46% [-7.89, -6.30]) | 91.206 → 98.655 (-8.16% [-10.23, -8.06]) |
| moe_sparse / 1 | matched finite bursts | torch → sym | 358.409 → 305.013 (-7.91% [-15.05, -6.34]) | 2.707 → 3.173 (-8.69% [-17.35, -7.08]) | 2.775 → 3.285 (-9.75% [-18.67, -6.30]) | 22.089 → 26.009 (-8.55% [-17.94, -6.31]) | 19.345 → 22.805 (-8.60% [-18.07, -6.55]) |
| moe_sparse / 2 | matched finite bursts | torch → sym | 424.051 → 377.018 (-10.84% [-12.21, -3.69]) | 4.558 → 4.980 (-9.67% [-10.04, -3.35]) | 4.823 → 5.562 (-15.32% [-17.87, -2.26]) | 18.776 → 20.962 (-11.55% [-13.77, -3.31]) | 14.198 → 16.405 (-15.25% [-18.16, -3.71]) |
| moe_sparse / 4 | matched finite bursts | torch → sym | 419.346 → 406.785 (-3.33% [-3.84, -3.00]) | 9.122 → 9.373 (-3.01% [-3.28, -1.87]) | 9.846 → 10.271 (-4.32% [-4.32, -3.41]) | 19.095 → 19.649 (-3.16% [-3.59, -2.72]) | 10.195 → 10.521 (-3.81% [-5.20, -3.20]) |

| Workload / callers | Arm | Observed warm app increment median MiB |
| --- | --- | --- |
| llm_decode / 1 | sym | 332.0 |
| llm_decode / 1 | torch | 298.0 |
| llm_decode / 2 | sym | 500.0 |
| llm_decode / 2 | torch | 430.0 |
| llm_decode / 4 | sym | 916.0 |
| llm_decode / 4 | torch | 694.0 |
| moe_sparse / 1 | sym | 184.0 |
| moe_sparse / 1 | torch | 170.0 |
| moe_sparse / 2 | sym | 270.0 |
| moe_sparse / 2 | torch | 242.0 |
| moe_sparse / 4 | sym | 442.0 |
| moe_sparse / 4 | torch | 386.0 |

E4 and E5 burst application increments use their post-reservation
baselines. E5 lifecycle peak is an absolute NVML value in its own phase
window. Sampled peaks are lower bounds on transient peak. OOM peaks
describe observation through failure. Torch allocator, native retention,
pinned-host, RSS and physical-device counters have overlapping scopes.
Keep these values separate.

## Follow-up evidence and remaining scope

The frozen campaign uses synthetic workloads on one GPU architecture.
#267 qualifies native kernel quality. #272 links actual capture to the
same isolated kernel comparison and independently varies cache, groups
and the typed-kernel build on the E3 trace. #271 scores a supplied
executable catalog against a measured oracle and retains calibration
costs, arena refusals, regret and finite-arrival deadline misses. Those
studies use their own machines, phase boundaries and process pairs.
[#267 native results](../issue267/RESULTS.md) · [#272 E3 controls](../issue272/RESULTS.md) ·
[#272 captured quality](../issue272/CAPTURED_QUALITY.md) · [#271 planner](../issue271/README.md)

The separate #274 application evaluates pretrained SqueezeNet 1.1 on one
RTX 4070 Ti SUPER with FP32, batch one and a fixed input shape. That campaign
provides a real application boundary outside the original synthetic workloads.
Portability of the same application across hardware, BF16 model behavior,
productivity and sustained open-loop serving still require evidence.
[Application protocol](APPLICATION_PROTOCOL.md) · [Issue 274 synthesis](README.md)
