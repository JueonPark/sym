# Issue 264: prepared groups and native completion accounting

The selected default is `FastTransfers(prepared_groups=True, report_mode="native")`. It reuses validated native group structure and aggregates completion counters in C++, while preserving current tensor values, fresh outputs, immutable reports, error handling, and parameter revisions. Explicit handles are available through `prepare_group`; `prepared_groups=False, report_mode="eager"` retains a control/fallback path. See the [API guide](../../docs/fast-transfers.md).

Across the registered E2 matrix, completed p50 falls **2.24–10.28% versus frozen Sym**, with positive process intervals in all eight conditions. Four conditions beat matched Torch p50; four remain **1.16–3.79% slower**. This resolves the targeted repeated-work problem; it does not establish a general Sym advantage.

## Evidence and scope

The Wiki artifact is public and can be downloaded without authentication. The [versioned Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264) contains every raw result, interference audit, memory trace, input/runtime hash, verification log, and CPU-only analysis instructions. [INDEX.json](INDEX.json) pins the two accepted analyses and campaign contracts; [GROUPS.md](GROUPS.md) gives the five-policy ablation and [PROTOCOL.md](PROTOCOL.md) specifies the frozen workloads. Pilots and interim analyses are not acceptance evidence.

There are 25 qualified grouped processes and 135 qualified model processes, with no missing jobs or observed interference. Grouped measurements comprise 22,500 completed samples, 7,500 separate CPU submissions, and 750 separate diagnostics. E2/E5 comprise 37,440 primary model requests: 14,400 E2, 5,760 lifecycle, and 17,280 burst requests. Numerical and held-output checks pass. This is one EPYC 7351 / RTX 2080 Ti system, Python 3.14.7 / Torch 2.14.0+cu126, under fixed synthetic offload conditions.

Effects below are mean process-paired percentage reductions; **positive means faster**, negative means slower. Brackets are 95% bootstrap intervals across five independent process pairs for E2 and three for E5 (10,000 resamples, seed 264). They are not equivalence tests, and intervals crossing zero are inconclusive. Absolute tables use medians of process summaries, so their ratios need not equal the paired effects.

## E2: completed requests with matched eager and Inductor consumers

Kernels, precision, grouping, whole-layer prefetch, depth two, and 128 MiB aggregate scratch are unchanged. Every process includes its own Torch controls. Twenty case/round accounting checks confirm identical group counts, logical transfers, payload bytes, and dispatch implementation counts across frozen Sym, new-build legacy, and the candidate.

| Case | Compute | p50 vs frozen Sym | p95 vs frozen Sym | p50 vs Torch | p95 vs Torch |
| --- | --- | --- | --- | --- | --- |
| llm_decode | eager | 10.28% [9.73, 10.82] | 9.94% [9.20, 10.58] | 2.63% [2.41, 2.83] | 2.89% [2.01, 4.20] |
| llm_decode | inductor | 6.19% [5.70, 6.71] | 6.29% [5.81, 6.85] | -3.79% [-4.31, -3.26] | -3.74% [-4.19, -3.19] |
| llm_prefill | eager | 3.01% [2.49, 3.51] | 2.75% [-5.28, 10.78] | -1.16% [-2.01, -0.32] | 1.01% [-1.88, 5.74] |
| llm_prefill | inductor | 2.24% [1.57, 2.73] | 1.85% [0.85, 2.91] | -2.57% [-3.80, -1.33] | -0.27% [-3.48, 4.04] |
| moe_dense | eager | 7.98% [7.68, 8.33] | 5.42% [0.12, 8.23] | 2.79% [2.67, 2.94] | 2.89% [1.69, 4.37] |
| moe_dense | inductor | 7.31% [6.90, 7.77] | 4.62% [-1.02, 7.85] | 3.01% [2.85, 3.17] | 2.88% [1.84, 3.78] |
| moe_sparse | eager | 4.19% [3.48, 4.89] | 6.70% [-3.19, 16.61] | -3.23% [-3.59, -2.76] | -1.05% [-3.54, 3.62] |
| moe_sparse | inductor | 6.80% [6.41, 7.25] | 8.15% [-3.32, 18.68] | 1.47% [1.25, 1.65] | -0.09% [-1.90, 1.66] |

Decode/eager, dense/eager, dense/Inductor, and sparse/Inductor have positive p50 intervals against Torch. Decode/Inductor, prefill/eager, prefill/Inductor, and sparse/eager remain slower. Sparse/Inductor p95 does not show the same win as its p50. The full analysis retains new-build legacy and every unfavorable control comparison.

## E5 lifecycle: preparation, revisions, and retirement

Each process runs 100 initial requests, ten after a shape change, and ten after scale replacement. Warm percentiles exclude the first initial request. Full accounted time includes launch/setup, all 120 requests, shape update, scale preparation, and retirement; it excludes oracle gaps and is not total process wall time.

| Case | Compute | Warm p50 vs frozen Sym | Warm p50 vs Torch | Warm p95 vs Torch | Full accounted vs frozen Sym | Estimated break-even vs frozen Sym (requests) |
| --- | --- | --- | --- | --- | --- | --- |
| llm_decode | eager | 9.91% [9.48, 10.70] | 3.19% [2.70, 3.67] | 2.87% [2.66, 3.19] | 3.41% [2.90, 4.20] | 22, 1, 13 |
| llm_decode | inductor | 9.02% [8.62, 9.30] | -1.18% [-1.62, -0.95] | -2.15% [-2.35, -1.95] | 0.83% [0.55, 1.18] | 1, 1, 34 |
| moe_sparse | eager | 1.76% [1.08, 2.95] | -5.88% [-6.60, -5.44] | -6.61% [-7.70, -5.68] | 0.90% [-0.18, 1.89] | 1, 1, 544 |
| moe_sparse | inductor | 7.44% [5.82, 9.57] | 0.67% [-0.56, 1.46] | 1.01% [-0.13, 1.69] | 1.33% [0.17, 2.20] | 1, 1, 1 |

Absolute preparation/revision costs in milliseconds (median of three processes):

| Case | Compute | Policy | Setup | First | First shape revision | Scale preparation | First scale revision | Retirement | Full accounted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_decode | eager | baseline | 4256.90 | 352.95 | 28.63 | 8.16 | 21.70 | 138.51 | 7268.35 |
| llm_decode | eager | legacy-eager | 4246.23 | 329.08 | 29.30 | 8.18 | 22.13 | 138.20 | 7299.25 |
| llm_decode | eager | prepared-native | 4277.96 | 344.64 | 27.88 | 8.24 | 19.79 | 138.24 | 7011.96 |
| llm_decode | inductor | baseline | 4547.15 | 17829.36 | 28.51 | 8.30 | 20.42 | 284.85 | 25006.45 |
| llm_decode | inductor | legacy-eager | 4536.31 | 17723.61 | 28.25 | 8.31 | 20.82 | 283.69 | 24936.10 |
| llm_decode | inductor | prepared-native | 4522.41 | 17794.94 | 26.76 | 8.54 | 19.04 | 295.00 | 24767.66 |
| moe_sparse | eager | baseline | 4456.97 | 309.67 | 24.95 | 5.05 | 10.26 | 109.40 | 5388.41 |
| moe_sparse | eager | legacy-eager | 4397.67 | 353.47 | 25.36 | 5.01 | 10.62 | 109.98 | 5391.46 |
| moe_sparse | eager | prepared-native | 4415.35 | 321.14 | 25.88 | 4.99 | 10.05 | 109.80 | 5335.09 |
| moe_sparse | inductor | baseline | 4896.67 | 4783.38 | 2756.69 | 5.09 | 14.03 | 197.46 | 13293.69 |
| moe_sparse | inductor | legacy-eager | 4707.04 | 4670.88 | 2731.64 | 5.13 | 14.43 | 195.73 | 12989.98 |
| moe_sparse | inductor | prepared-native | 4702.27 | 4746.82 | 2762.71 | 5.09 | 13.28 | 199.87 | 13000.77 |

Cold costs do not uniformly improve. Sparse/eager's first shape-revision request regresses 3.68% [−3.95, −3.50] against frozen Sym. Decode/Inductor scale preparation has a large outlying cost (mean paired reduction −35.66%, interval [−107.15, 3.06]); this is retained. Sparse/Inductor full accounted time is inconclusive against new-build legacy (−0.65% [−2.35, 0.68]), despite its warm improvement.

Break-even is an extrapolation from setup plus first-request premium and initial warm savings, not a measurement at every reuse count or a model of repeated revisions. Sparse/eager takes up to 544 requests versus frozen Sym in one process pair; sparse/Inductor takes 243, 876, and 1 versus new-build legacy. Against Torch, decode/eager estimates 127, 94, and 32 requests; decode/Inductor and sparse/eager have no positive warm saving. Sparse/Inductor is inconclusive (one pair has no saving; the other estimates are 8,794 and 1). Compilation variability matters; no blanket amortization claim is supported.

## E5 bursts: 1, 2, and 4 callers

Each process measures 30 eight-request bursts with the previous burst's outputs held. The table separates request service latency from queue-inclusive sojourn latency. This protocol does not measure a sustained arrival-rate service or an SLO capacity curve.

| Case | Callers | Service p50 vs frozen Sym | Service p50 vs Torch | Service p95 vs Torch | Sojourn p95 vs frozen Sym | Mean burst time vs frozen Sym |
| --- | --- | --- | --- | --- | --- | --- |
| llm_decode | 1 | 9.25% [8.66, 9.84] | 2.63% [2.04, 3.52] | 0.40% [-4.83, 3.06] | 7.84% [5.61, 9.87] | 8.05% [6.94, 9.84] |
| llm_decode | 2 | 8.59% [8.23, 8.91] | 2.32% [1.90, 2.97] | 6.03% [3.95, 7.99] | 10.43% [6.40, 12.89] | 9.28% [7.08, 10.92] |
| llm_decode | 4 | 10.98% [10.67, 11.45] | 6.05% [3.88, 7.45] | 5.78% [-0.30, 13.71] | 10.38% [8.22, 11.49] | 11.22% [10.37, 11.66] |
| moe_sparse | 1 | 1.64% [-1.58, 4.40] | -5.17% [-10.35, -0.14] | -6.20% [-12.34, -0.22] | 1.51% [-3.00, 5.13] | 2.13% [-0.67, 4.39] |
| moe_sparse | 2 | 7.43% [6.51, 8.09] | 4.13% [0.23, 8.92] | 9.20% [-3.42, 24.43] | 8.24% [7.06, 9.14] | 7.75% [6.28, 8.97] |
| moe_sparse | 4 | 10.88% [10.64, 11.12] | 9.14% [8.68, 9.46] | 3.92% [3.63, 4.10] | 9.32% [9.04, 9.47] | 9.39% [9.30, 9.54] |

Decode improves over frozen Sym at every caller count. Single-caller sparse MoE is inconclusive against frozen Sym and slower than Torch; two and four callers show positive service-p50 intervals against Torch. Several p95 intervals cross zero, so their point estimates are not established tail wins.

## Retained memory and construction work

In the grouped ablation, ten warm prepared requests create **zero new groups/view descriptors**, perform ten rebinds, and materialize zero detailed reports. Legacy creates ten groups and 40 descriptors for two members (120 for six); eager reporting creates ten detailed reports. CPU submission falls 11.46–28.68% against frozen Sym. Preparation, completed latency, exact native scratch, whole-process RSS, and break-even are reported together in [GROUPS.md](GROUPS.md).

E5 sampled device memory remains a separate scope from exact native scratch. The following are median sampled warm application increments in MiB, relative to the post-reservation measurement. Samples are lower bounds on transient maxima; allocator reservations, held outputs, and overlapping execution contribute. They are not prepared metadata byte counts.

| Case | Callers | Torch | Frozen Sym | New legacy | Prepared/native |
| --- | --- | --- | --- | --- | --- |
| llm_decode | 1 | 298.0 | 332.0 | 332.0 | 332.0 |
| llm_decode | 2 | 430.0 | 500.0 | 500.0 | 540.0 |
| llm_decode | 4 | 694.0 | 916.0 | 876.0 | 956.0 |
| moe_sparse | 1 | 170.0 | 184.0 | 184.0 | 184.0 |
| moe_sparse | 2 | 242.0 | 270.0 | 270.0 | 270.0 |
| moe_sparse | 4 | 386.0 | 442.0 | 442.0 | 442.0 |

Decode's sampled device increment grows with caller count, and the candidate exceeds the new legacy control at two and four callers. Thus unchanged native scratch does **not** imply unchanged whole-application memory. All 54 Sym burst processes keep native retained scratch stable after warmup; all 18 candidate burst processes also keep group/descriptor/live-slot/cache counts stable with no detailed report conversion. Every successful E2/E5 close reports zero native retained/live bytes and preserved held outputs. Lifecycle raw traces include whole-process RSS peaks and deployment device peaks.

The implicit cache is bounded to 64 handles by default, and each handle to the owner's context count; explicit handles remain caller-owned. There is no separate host metadata byte budget. Shape specializations, immutable parameter snapshots, and group structure have separate lifetimes. Scale replacement invalidates affected implicit handles, while explicit handles retain their chosen revision. Failed or quarantined slots are not reused.

## Correctness and verification

The runtime implementation measured here is `a351b0f7f2ce0673145374a7a734217c1d7045bd`; the reported campaign uses that immutable build. A later review fix restores the legacy `RuntimeError` contract when native binding rejects an invalid supplied output. It changes exception handling only; its first/warm binding and retry tests are separate from the performance campaign. The final reviewed Python package is supplied alongside the measured package, and the reported timings are not relabeled as a measurement of that later revision. Baseline is `2617ba166f98a7476a5f472d829a573317fc0cfb`; the acceptance harness was frozen at `416c43b4888789ac2ddc58b25f2d606fad20a300`. Exact runtime and measurement-source hashes are in the contracts. See the Wiki artifact guide and `evidence/review-resolution.md` for the exact measured/reviewed package distinction.

- Final review validation passed all 12 CTest checks and 1,357 Python tests (13 skips, one existing warning); see [review disposition](REVIEW.md). The original measured-runtime validation passed all 12 CTest checks; the full Python suite passed 1,349 tests with 13 capability/environment skips and one existing threaded-fork warning.
- The expanded prepared-group lifecycle/failure suite subsequently passed 29 tests, including injected submission, event-wait, and close failures. Four initial failures in the added cases were a test exception-type mismatch; catching the registered `pyreloc.TransferError` fixed the harness without a runtime change.
- Eight CPU analysis tests passed, covering missing/duplicate rounds, provenance, policy selection, incomplete workloads, and resource-growth rejection. Numerical, live-value, shape/stride/offset/dtype, revision, in-flight replacement, concurrent caller, old-report, held-output, close/fork, and quarantine behavior have regression coverage.
- The optional all-target build encounters the existing `bench/rtrack/dma_engine.cu:57` double-atomicAdd incompatibility with configured sm_52. All runtime/compiler/test targets built and all CTest checks passed. Both measured runtimes retain the same CUDA architecture; this unrelated benchmark is not silently changed.

## Acceptance and related work

| Issue 264 acceptance item | Evidence |
| --- | --- |
| Repeated group/descriptor/report work and CPU submission | Five-policy [grouped ablation](GROUPS.md), native counters, raw diagnostic deltas |
| Correctness and ownership across changes/concurrency/failure | Core/native/frontend tests, all E2/E5 oracles, held-output and zero-after-close gates |
| Preparation, amortization, retention, and completed latency together | Grouped costs and the lifecycle/memory tables above; raw preparation and physical traces |
| Matched E2/E5 versus Torch/Inductor, including regressions | Frozen protocol, 135 clean processes, all paired comparisons and controls |

This work is part of issue #261, part of issue #272, and part of issue #274. It shares the existing dispatch/resources engine with issue #248; it does not close all of #248 or claim the older binding/grouping primitives as new contributions. The direct `TypedProgram.execute` / `PreparedWireWeights` submission paths retain their previous behavior. Additional hardware, workloads, sustained serving, exact host metadata accounting, and broader automatic specialization remain outside this qualification.
