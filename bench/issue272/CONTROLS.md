# Matched controls and cost accounting

The archived campaigns isolate three effects at different boundaries. #267 measures the benefit of selecting improved native kernels from identical compiler plans, plus the separate effect of avoiding intermediate materialization. #264 holds reporting policy fixed while toggling reusable native group structure. Original E3 and #265 both toggle an explicit application's cache of handwritten programs on the same changing-shape trace. #265 also compares two automatic frontends whose admission, dispatch and reuse differ. The magnitudes come from separate experiments and cannot be added or used as shares of one model's latency.

## Specialization and materialization on E1

#267 pins baseline native source `c5085cd9`, candidate `3c008646`, harness `18e12e3` and one unchanged compiler binary. Both runtimes load the same folded plan bytes per case. In the candidate, the folded and materialized paths run against the same runtime, but the materialized path invokes one plan per source operation and allocates intermediate buffers. Simply disabling canonicalization would still lower a folded expression. The manual kernels use fixed-family address arithmetic and F16C, with one roundtrip on every roundtrip case because that conversion pair is idempotent under the checked value contract. The generated plans preserve the requested stage count. The E1 harness constructs recipes explicitly to control source-operation boundaries. Evidence for import of ordinary Torch programs comes from E3 and its held-out campaign below. [Source and protocol](../issue267/PROTOCOL.md) · [Full tables](../issue267/TABLES.md) · [Raw evidence and reanalysis](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-267)

All E1 arms write the same preallocated pinned wire and perform the same single H2D copy to the same preallocated device output. Precision, worker count and inputs are held fixed. Torch eager and Inductor produce into that wire, and Inductor emulates intermediate casts. Every timed result is checked for exact non-NaN bits and NaN classification. Three rounds cover each runtime/worker/suite combination, for 36 accepted processes, 93,600 timed producer/transfer checks and 3,600 additional consumer checks on one EPYC 7351 / RTX 2080 Ti system. The eight-worker mask covers four physical cores and their SMT threads.

The table shows medians of three process medians for completed producer + H2D, in ms. Large E1 uses N=8,192, M=128 and rank-three outer extent 8. Materialized, manual and Torch controls are from the candidate-runtime processes. Smaller N=33/512 and p95s are in the full tables.

| Family | Workers | Materialized | Old generated | Improved generated | Manual | Torch eager | Inductor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dequantize + transpose | 1 | 25.517 | 4.586 | 3.147 | 3.653 | 3.279 | 2.332 |
| Dequantize + transpose | 8 | 6.125 | 1.811 | 1.330 | 1.727 | 2.906 | 0.941 |
| Padded roundtrip | 1 | 10.235 | 8.866 | 2.358 | 5.124 | 3.835 | 3.776 |
| Padded roundtrip | 8 | 9.884 | 7.129 | 1.809 | 2.461 | 3.249 | 1.874 |
| Rank-three chain | 1 | 102.270 | 79.065 | 9.089 | 17.462 | 91.898 | 9.267 |
| Rank-three chain | 8 | 21.778 | 16.745 | 5.590 | 6.684 | 29.061 | 5.856 |
| One roundtrip | 1 | 9.230 | 1.620 | 0.816 | 0.862 | 4.509 | 1.073 |
| One roundtrip | 8 | 7.167 | 0.838 | 0.749 | 0.773 | 4.189 | 0.996 |
| Two roundtrips | 1 | 20.531 | 2.379 | 0.894 | 0.878 | 8.131 | 1.154 |
| Two roundtrips | 8 | 10.529 | 0.903 | 0.744 | 0.746 | 8.022 | 0.978 |
| Four roundtrips | 1 | 41.520 | 3.435 | 1.419 | 0.874 | 15.371 | 1.579 |
| Four roundtrips | 8 | 20.416 | 1.142 | 0.748 | 0.749 | 15.305 | 0.983 |

Across the 36 original E1 case/worker combinations, improved folding beats old folding and true materialization in 36/36, manual fusion in 30/36, and the fastest manual/eager/Inductor control in 28/36 for completed H2D. The equal-case geometric means against old folding are 1.92× completed and 2.59× producer-only. The eight losses to the fastest control are the two large dequantization cases (35.0%/41.3% slower than Inductor at one/eight workers), one-roundtrip N=512 at eight workers (+4.5%), two-roundtrip N=8192 at one worker (+1.9%), and four-roundtrip N=33 at eight workers (+0.2%), N=512 at one/eight workers (+30.7%/+13.0%) and N=8192 at one worker (+62.4%). All six roundtrip losses are to manual fusion. Three rounds provide descriptive medians, with no significance claim for near-ties. [Accepted results and losses](../issue267/RESULTS.md)

### Held-out paths and consumer checks

The eight held-out cases were frozen before kernel tuning. The candidate beats old folding on 14/16 completed comparisons, materialization on 15/16 and the fastest available manual/Torch alternative on 12/16. BF16 at one worker is the materialization loss. Unsupported quantize + dequantize regresses 1.4%/1.9% against old folding with one/eight workers. BF16 fallback is 1.89×/1.36× eager. Seven roundtrips take 1.48×/1.84× manual, which uses one roundtrip. The candidate records the contiguous-stage layout with a seven-roundtrip register descriptor, tiled transpose runs with a register-cast descriptor for each rank-three extent (3 and 5), and tiled transpose with register stages for padding and both dequantization chains. BF16 and quantize + dequantize record tiled transpose with buffered stages. Five cases lack an archived handwritten implementation for the new chain or extent and retain eager and Inductor controls. [Held-out contract](../issue267/contract.py) · [Full tables](../issue267/TABLES.md)

The same batch-32 linear + GELU after H2D retains the padded-chain gain at both worker counts (3.31×/3.23× versus old folding and 0.68×/0.78× the fastest control). The dequantized-weight consumer improves 1.53× at one worker and regresses 7.5% at eight workers against old folding. At eight workers, the old process medians are 1.102/0.724/0.726 ms and the candidate's are 0.772/0.782/0.781 ms. Candidate dequant remains 1.18×/1.13× Inductor under consumption. This experiment measures one weight consumer. [Consumer results](../issue267/RESULTS.md#completed-gpu-consumers)

### Attribution, preparation and generated code

The compiler derives the folded layout and ordered value stages. The native runtime recognizes legal patterns and selects from six finite AVX2/F16C vector variants with scalar and buffered fallbacks. The new register path removes intermediate stage loads/stores for eligible chains while keeping each conversion. Rank-three E1 already coalesced to a tiled transpose in the old runtime. The new path can invoke stages once for a contiguous 32-by-eight tile in place of 32 per-row entries. Padding still writes the fill region and then valid elements. The old/new comparison includes register staging, transpose batching and padding fills together. It gives their aggregate effect for those binaries. Debugger samples and static pass counts support the explanation. Direct DRAM traffic and hardware-counter measurements are absent. [Kernel changes and attribution](../issue267/RESULTS.md#attribution-and-bounded-costs)

Across 168 candidate folded records, compiler invocation takes 4.125–5.593 ms (median 4.378), binding 0.016–0.050 ms (median 0.024) and native preparation 0.005–0.408 ms (median 0.138). The old-runtime compiler median is also 4.378 ms. Folded plans are 418–1,078 B versus 994–8,834 B summed materialized-plan bytes. All 2,520 saved plan files match their hashes and operation boundaries. Production ELF `.text` grows from 666,448 to 669,776 B (+3,328 B, +0.50%). The benchmark helper's separate `.text` grows from 189,946 to 190,890 B, including diagnostics. Shape-specific executable cache bytes created by the native runtime are zero. Fresh-process Inductor disk inventory is 4.51–4.58 MB (main), 2.10–2.12 MB (held-out) and 0.574–0.580 MB (consumer), including respectively 1.44–1.50 MB, 0.714–0.724 MB and 0.225–0.229 MB of generated `.py`/`.cpp`/`.so` files. These inventories are per suite and include multiple cases. [Preparation and cache totals](../issue267/RESULTS.md#compilation-preparation-and-cache-measurements)

The candidate's first completed folded calls span 0.042–139.576 ms (median 0.233), and Inductor first calls with lazy compilation span 854–7,124 ms (median 991). Arms share a process and can inherit startup work from earlier arms. For large one-worker rank-three, the improved/materialized/manual/eager/Inductor elapsed-cost estimates at 100 calls are 944.850/10,309.328/1,778.433/9,213.529/1,858.552 ms. The corresponding one-call estimates are 45.041/179.190/49.159/114.981/941.135 ms. Per-process estimates add common allocation, arm-specific setup and the measured first call to `(N−1) × warm median`. Estimates exclude CPU fixture generation, CUDA initialization, warmups and consumer activation/output setup, along with AOT and manual extension builds. [Setup boundary and full tables](../issue267/RESULTS.md#setup-and-total-cost-interpretation)

## Native group preparation with reporting fixed

#264 supplies a same-build two-by-two grouped control. Each of the five process rounds holds kernels, precision, logical transfers and scratch budget fixed, toggling reusable group structure independently of native completion reporting. The 25 processes include the historical baseline and cover 22,500 completed samples plus 7,500 separate CPU submission samples. Reductions are means of round-paired percentages against the indicated control. Brackets give the archived analyzer's five-process bootstrap (10,000 resamples, seed 264) applied to the two matched preparation contrasts. [Grouped ablation](../issue264/GROUPS.md) · [Raw evidence and reanalysis](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264)

| Case | Prepared only, eager reporting, completed p50 | Prepared only, native reporting, completed p50 | Native reporting only, groups off, completed p50 | Native reporting only, groups on, completed p50 |
| --- | ---: | ---: | ---: | ---: |
| Small weights | +16.60% [16.15, 17.16] | +17.36% [16.92, 17.85] | +6.01% | +6.87% |
| Decode weights | +6.99% [6.69, 7.29] | +7.23% [7.09, 7.42] | +1.54% | +1.79% |
| KV restore | +11.11% [10.83, 11.42] | +11.78% [11.40, 12.16] | +5.72% | +6.44% |

Preparation alone reduces p95 by 7.02–16.87% and CPU submission p50 by 14.18–29.67%, with first completed calls 1.38–6.95% slower across the matched contrasts. Ten warmed legacy calls build ten groups and 40/120 view descriptors for two/six-member cases. The corresponding prepared calls build zero groups/descriptors and perform ten live rebinds. Native reporting independently removes ten eager report materializations. Explicit handle metadata preparation takes 0.066–0.120 ms, apart from first native slot creation. Sampled native retained scratch is identical across the matched policies, 17,408 B for small weights, 14,721,024 B for decode weights and 262,144 B for KV restore. Scratch reaches zero on close, with held outputs preserved. The general implicit structural cache defaults to 64 handles and at most `contexts` native slots per handle. Whole-process RSS of roughly 826–925 MiB includes allocators and harness state. The reported scratch counters do not measure host metadata bytes. [Groups and resource accounting](../issue264/GROUPS.md) · [FastTransfers contract](../../docs/fast-transfers.md#reusable-groups-and-reports)

The 135-process/37,440-request #264 E2/E5 validation changes preparation and reporting jointly. E2 p50 improves 2.24–10.28% against old Sym, with four of eight cases beating matched Torch and four slower by 1.16–3.79%. E2/E5 demonstrates the combined runtime behavior with matched workloads, including lifecycle and burst losses. The prepared-only result above applies to the grouped qualification. [Model, lifecycle and concurrency results](../issue264/RESULTS.md)

## Cache reuse and the original E3 request trace

Original E3 records `128,256,128,33,512,33,1,128,0,65,256,65` repeated five times per case. Every request uses new values. The final layout/KV request in each cycle changes stride, dequantization replaces live scale/zero-point values, and the indexed case changes index values. Both arms of the explicit program-cache contrast use the same handwritten `FastPath`, native owner settings, program implementation and fallback policy. The off arm keeps the current signature. The on arm retains prior eligible programs by signature. For both arms, live dequantization forces a fresh immutable snapshot and the indexed workload uses the same one long-lived program with a native descriptor cache. [Original archived `03_dynamic_shapes` protocol and raw records](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

Original E3 measured 7,200 exact timed results and retained-output checks across five processes. The six arms shared one process and Inductor cache per round, with rotated order. The table reports medians across the five rounds in ms. First call is step 0 from `acceptance/requests.csv`, and the other columns are sums of completed calls in each window. The first call and first cycle can include compile, preparation and lazy runtime initialization. Process launch/import and owner construction were outside this envelope, so independent per-arm process-cold results come from #265 below.

| Workload | Arm | First call | First 12 | Total 60 | Repeated 48 |
| --- | --- | ---: | ---: | ---: | ---: |
| Layout + cast | Torch eager | 0.689 | 4.529 | 21.106 | 17.260 |
| Layout + cast | Inductor | 7480.412 | 8094.195 | 8118.903 | 24.708 |
| Layout + cast | Auto eager | 98.035 | 348.244 | 383.440 | 36.294 |
| Layout + cast | Auto Inductor | 191.292 | 1679.770 | 1711.942 | 32.680 |
| Layout + cast | Fast reprepare | 11.198 | 21.810 | 39.055 | 18.673 |
| Layout + cast | Fast cache | 6.379 | 15.736 | 26.586 | 11.144 |
| KV restore | Torch eager | 0.335 | 3.292 | 16.370 | 13.318 |
| KV restore | Inductor | 197.827 | 1536.663 | 1554.552 | 17.889 |
| KV restore | Auto eager | 85.871 | 309.524 | 340.511 | 31.395 |
| KV restore | Auto Inductor | 167.896 | 595.220 | 624.624 | 28.056 |
| KV restore | Fast reprepare | 6.460 | 12.383 | 27.790 | 15.250 |
| KV restore | Fast cache | 5.226 | 9.987 | 21.874 | 11.905 |
| Live dequantization | Torch eager | 0.305 | 3.333 | 15.910 | 12.517 |
| Live dequantization | Inductor | 1402.102 | 2493.830 | 2510.786 | 16.946 |
| Live dequantization | Auto eager | 90.652 | 235.432 | 272.559 | 37.127 |
| Live dequantization | Auto Inductor | 172.902 | 476.972 | 511.592 | 34.952 |
| Live dequantization | Fast reprepare | 5.776 | 11.261 | 27.737 | 16.525 |
| Live dequantization | Fast cache | 5.444 | 10.339 | 26.530 | 16.191 |
| Live indices | Torch eager | 0.128 | 1.252 | 5.919 | 4.694 |
| Live indices | Inductor | 1126.800 | 3178.243 | 3196.492 | 18.430 |
| Live indices | Auto eager | 69.948 | 276.950 | 316.858 | 39.908 |
| Live indices | Auto Inductor | 148.931 | 569.607 | 604.634 | 35.416 |
| Live indices | Fast reprepare | 5.362 | 8.713 | 17.210 | 8.497 |
| Live indices | Fast cache | 5.166 | 8.288 | 16.729 | 8.175 |

In original E3, the layout/KV cache cuts 50 preparations to six and reaches 44 hits. The paired median repeated reduction is 37.66% for layout (35.50–45.63% across rounds) and 22.42% for KV (−10.39% to +23.53%). Both live-dequantization arms reprepare 55 times with zero hits. Both indexed arms prepare once, hit 54 times and finish with six internal descriptor entries. Their measured differences lack a change in application preparation/reuse. Eager Torch wins all four full 60-request comparisons. Original Sym automatic paths execute 45/45/50/55 requests natively for layout/KV/dequant/index on the recorded traces. They compile 1/1/1/3 plans with 4/4/3/4 Dynamo callbacks and no callbacks after the first cycle. Typed automatic paths use the CPU-reference row. Escaping-intermediate and mutation diagnostics execute zero replacements. [Original evidence](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

### Fresh-process E3 replay and automatic qualification

#265 replays the original trace and input/oracle modules, using separate processes and fresh Inductor caches for every arm/case/round. It retains OS and driver caches. All arms import the same E3 module and incur its Sym import. Cold-first sums launch/import/configuration, CUDA initialization, owner setup and the first completed call. Accounted full adds the setup phases and retirement to all 60 timed calls. Fixture generation, oracles and diagnostics are tracked outside these boundaries. Mean milliseconds across three independent processes for all eight arms and four cases are in the report. [Protocol](../issue265/PROTOCOL.md) · [Cold, full and repeated trace table](../issue265/RESULTS.md#cold-and-complete-stream-costs) · [Raw data](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-265)

Re-analysis of the explicit arms below reports cold-first, completed-call total and repeated-window means (ms), plus the mean process-paired reduction for repeated calls. For layout and KV, both cold-first means are slightly higher with the cache, while accounted full changes from 3071.095 to 3070.844 ms (layout) and 3048.108 to 3047.333 ms (KV). The main warm-cache effect survives the isolated replay. Its practical size depends on how startup amortizes.

| Workload | Cold-first off → on | Total 60 off → on | Repeated 48 off → on | Repeated reduction [95% process interval] | Preparations off → on / hits on |
| --- | ---: | ---: | ---: | ---: | ---: |
| Layout + cast | 3042.638 → 3047.313 | 55.322 → 51.148 | 18.424 → 13.874 | +24.71% [22.65, 26.07] | 50 → 6 / 44 |
| KV restore | 3026.873 → 3030.685 | 29.044 → 24.297 | 14.818 → 11.175 | +24.58% [22.43, 25.92] | 50 → 6 / 44 |
| Live dequantization | 3039.444 → 3045.521 | 50.243 → 49.065 | 15.412 → 15.538 | −0.83% [−4.26, 1.59] | 55 → 55 / 0 |
| Live indices | 3052.443 → 3057.875 | 43.285 → 41.653 | 9.622 → 9.381 | +2.36% [−2.55, 8.01] | 1 → 1 / 54 |

#265 also compares the old automatic frontend with opt-in generated prepared execution, using identical native and compiler binaries. This contrast includes preparation, cache reuse, admission/fallback, changed dispatch and frontend diagnostics. The selected policy is 64 KiB of source data and one previous occurrence of the exact specialization key. Candidate repeated-call changes versus old automatic range from −8.44% (KV/eager) to +18.70% (live dequant/Inductor), where all live-dequant requests use the saved original region. The candidate remains 1.52–7.04× the matched eager/Inductor repeated-window costs. Raw data retain 96 E3 processes (5,760 exact outputs), 48 held-out processes (2,880 exact outputs), exact held outputs after retirement, and zero live/retained native bytes after close. There are zero later Dynamo callbacks in both campaigns. [Qualification and results](../issue265/RESULTS.md)

| E3 case | Selected-policy native row and calls / 60 | Preparations / hits / misses | Retained entries / evictions | Plans / Dynamo callbacks |
| --- | --- | ---: | ---: | ---: |
| Layout + cast | `cuda_relocate_f32`, 40 | 5 / 35 / 5 | 5 / 0 | 1 / 4 |
| KV restore | `layout_cuda`, 40 | 5 / 35 / 5 | 5 / 0 | 1 / 4 |
| Live dequantization | Saved original region, 0 native | 0 / 0 / 5 | 5 / 0 | 1 / 3 |
| Live indices | `cpu_reference`, 48 | 6 / 42 / 7 | 7 / 0 | 3 / 4 |

The held-out cases change rank/permutation, cast order, parameter axis/values and indexed rank. Native calls per 60 requests in selected / always-device arms are 17/45 on `cuda_relocate_f32` (`held_cast_chain`), 0/50 on `cpu_stages_cuda_stages@1` (`held_dequant_axis0`), 49/55 on `cpu_reference` (`held_index_rank3`) and 13/45 on `layout_cuda` (`held_layout`). Selected held-out eager runs cost 2.83–9.18× Torch. The always-device live-dequantization arm observes 50 exact live snapshots and preparations, zero hits, 18 evictions and 32 retained entries with 5,888 B of copied parameter keys. That arm is 12.33% slower than legacy automatic and 5.57× eager Torch. This supplies actual generated native execution through live parameter revisions that the selected policy declines. Indexed programs still bind live indices and rebuild the native indexed program on each execution. Graph capture exclusions, runtime fallback reasons and native executions count different events. [Held-out results and small-request costs](../issue265/RESULTS.md#held-out-chains-and-small-requests)

The automatic cache defaults to 32 entries (including admissions and negative entries) and 256 KiB of copied parameter key bytes. E3 retains 5–7 entries and zero evictions. The selected live-dequant trace retains 5,160 B of copied keys for five admissions. The #265 recorded preparation/bookkeeping sums across 60 selected requests are approximately 8.5 ms (layout), 8.1–8.3 ms (KV), 2.4 ms (live dequant) and 11.3–11.4 ms (indices) in round one. These counters include lookup/admission and snapshots, exclude compiler and frontend costs, and overlap timed calls. Live parameter rechecks remain charged in execution. The optional persistent compiler artifact store has its own defaults of 128 entries and 64 MiB when configured. Native scratch, additional parameter copies and preflighted caller-held handles have different lifetimes and may outlive a cache entry. The reported entry/key bounds therefore do not bound total process memory. [Automatic contract](../../docs/automatic-prepared-execution.md#lifetimes-and-bounds) · [Compiler artifact store options](../../libreloc/python/reloc_torch/compiler.py)

## Programmer-supplied source

The automatic #265 path passes the original Torch function to `torch.compile` with `RelocBackend` and an opt-in `PreparedExecutionPolicy`, and closes the backend when finished. It uses **zero per-case explicit recipes, native producers or handwritten cache lookups**. Source inspection of archived `03_dynamic_shapes/benchmark.py` gives 5 physical lines (414 B, lines 20–24) across the four original Torch workload functions, excluding imports and the common compile wrapper. The explicit E3 path adds 19 physical lines (1,033 B, lines 27–45) for per-case source/destination, shape, dtype, operations and parameter coordinates, plus a 30-line (1,958 B, lines 65–94) `FastPath` owner, shape/parameter key, fallback and execution wrapper. Stats are excluded from the latter count. The #267 fixed-family C++ manual control is 64 physical lines (2,925 B, lines 115–178 of `native.cpp`), excluding the extension build, bindings, shared runner and tests. These are reproducible measurements of checked-in or archived source slices. No user study, development-time measurement, or deployment maintenance comparison was performed. The DLRM pooled producer and opaque fast custom ops belong to the explicit runtime API. The claims about generated artifact selection do not credit them. [Original benchmark source](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04) · [Manual C++ control](../issue267/native.cpp#L115-L178) · [Automatic API](../../docs/automatic-prepared-execution.md) · [Explicit API](../../docs/fast-transfers.md)

## Reproducing the archival comparisons

Each Wiki page supplies the exact package manifest, SHA-256, extraction commands and reproduction instructions. Download and verify the package and its file inventory first.

| Campaign | Inputs and CPU-only replay |
| --- | --- |
| Original E3 | Run `03_dynamic_shapes/analyze.py` against a working copy of `03_dynamic_shapes/acceptance` inside the extracted tree, retaining `00_protocol/provenance.json`. CPython 3.14.7 reproduces the summary CSV bytes. For the first-call column, group `requests.csv` by case/path/round, take step 0, then take the median of the five rounds. For cache-only reuse, compute `100 × (1 − cache/reprepare)` on each round's last 48 summed calls and take the median. [Original Wiki package](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04) |
| #264 | Run `source/bench/issue264/analyze.py evidence/groups-v1 --output groups-replay.json`. Select each case's five `records` for off/on policies, hold report mode fixed, and call `paired(off, on, 'p50_ms', 5)` or the same helper with the other metrics. CPython 3.10.12 reproduced the grouped analysis byte for byte. CPython 3.14.7 reproduced the full model analysis when run against `evidence/acceptance-v1`. [#264 Wiki package](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264) |
| #265 | Run `source/bench/issue265/analyze.py evidence/e3-v1 --output e3-replay.json` and similarly for `heldout-v1`. CPython 3.14.7 reproduces both analysis JSON files byte for byte. For explicit-cache reuse, extract arm/round `summaries` and pass the paired `repeated_ms` vectors to the analyzer's `paired(reprepare, cache)`. [#265 Wiki package](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-265) |
| #267 | Run `source/bench/issue267/analyze.py evidence/acceptance` on a working copy using CPython 3.10.12 for byte-identical output. Main and held-out timings, all p95s and the setup model are regenerated into the report tables. [#267 Wiki package](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-267) |

The archives alone leave two independent experimental gaps for stronger attribution. One is an automatic-path cache toggle on the original trace that holds admission, runtime row, source artifacts and report mode fixed. The other is a same-binary specialization toggle that keeps the plan and preparation fixed. Such tests can measure costs independently of switching frontend paths or native revisions. Full model/server latency attribution from an orthogonal factorial design, portability to other hardware, and measured engineering effort require evidence outside these archived campaigns.
