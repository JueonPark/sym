# Issue #263: completed-request attribution

Report materialization/accounting is a measurable Sym cost on the consumer-launch path. Bypassing it reduces E2 decode/MoE median latency by **2.77–4.55%**, and prefill by **0.56–0.98%**. Signature-check bypass is smaller and sometimes indistinguishable from noise. Their combination brings dense eager MoE close to parity, while decode and prefill retain substantial gaps.

The result is workload dependent. In E5, report bypass improves warm medians, but its total-lifecycle benefit is small or uncertain. Four-caller sparse MoE obtains a modest advantage over Torch in this fixed-burst test; the other conditions do not establish a general advantage. Production accounting still needs a cheaper implementation that preserves its semantics.

Evidence: [versioned release](https://github.com/JueonPark/sym/releases/tag/profile-issue263-2026-11-07-v1), [full bundle](https://github.com/JueonPark/sym/releases/download/profile-issue263-2026-11-07-v1/sym-issue263-2026-11-07-v1.tar.gz), [checksums](https://github.com/JueonPark/sym/releases/download/profile-issue263-2026-11-07-v1/SHA256SUMS), [access/reproduction guide](https://github.com/JueonPark/sym/releases/download/profile-issue263-2026-11-07-v1/ARTIFACT_GUIDE.md). [Protocol](PROTOCOL.md), [reproduction instructions](README.md), and [compact result index](result-index.json) accompany the harness.

## Scope and qualification

- **150 qualified processes:** 110 unprofiled comparisons and 40 diagnostic processes. Accepted timings contain **36,480 completed requests**: 19,200 E2, 4,320 lifecycle, and 12,960 burst requests. First calls, warmups, allocator probes, and diagnostics are separate.
- E2 uses all four FP16 workloads and both eager/Inductor consumers, the same deployed native conversion/permutation kernels, matched grouping, and a two-slot whole-request window. E5 retains the original 100+10+10 lifecycle sequence or 30 eight-request bursts, including distinct offered inputs, scale revisions, and held outputs.
- All **23 CPU/CUDA traces** have zero unassigned CUDA activity overlapping annotated requests. All **24 E2 trace comparisons** match representative kernel/copy signatures, copy bytes, and group counts. Group counts are 22 / 4 / 4 / 16 for decode / prefill / sparse / dense MoE.
- The unchanged runtime is `2617ba166f98a7476a5f472d829a573317fc0cfb`, on an RTX 2080 Ti and AMD EPYC 7351 host (eight fixed logical CPUs), with Python 3.14.7, Torch 2.14.0+cu126, and Nsight Systems 2024.2. The original 17,264 inventoried artifact files and 41 runtime provenance files still match their published hashes.
- The fresh baseline is unfavorable to Sym in **11/12 total-lifecycle**, **18/18 throughput**, and **17/18 burst-sojourn p95** process pairs. This preserves the broad direction of the historical regression, without reproducing every historical pair. E2 selects the principal matched paths from the larger historical diagnostic matrix.

## E2: causal bypasses versus parity

`no_report` omits full report extraction and Python accounting after native completion; waits, output stream recording, references, and retirement remain active. `no_check` omits Python signature checks for the qualified metadata while preserving native validation. `both` is a separate measured contrast. These are diagnostic interventions; disabled accounting/validation is not an acceptable production replacement.

The historical 4.6–10.9% slowdown implies a **4.4–9.8% reduction of Sym's own latency** for parity. The current requirement is **3.69–9.23%**. It is an arithmetic target, not a predicted optimization gain.

| Workload | Compute | Torch p50 ms | Sym p50 ms | Sym reduction needed for parity | `no_report` reduction | `no_check` reduction | Combined reduction |
|---|---|---|---|---|---|---|---|
| llm_decode | eager | 19.200 | 20.739 | 7.75% [7.19, 8.17] | 4.36% [4.05, 4.64] | 1.39% [0.86, 2.00] | 5.39% [4.90, 5.80] |
| llm_decode | inductor | 17.539 | 19.436 | 9.23% [8.70, 9.78] | 2.77% [2.60, 3.05] | 1.16% [0.83, 1.47] | 3.55% [3.22, 3.88] |
| llm_prefill | eager | 4.792 | 4.982 | 3.69% [3.33, 4.13] | 0.56% [0.09, 1.03] | 0.34% [-0.11, 0.85] | 0.75% [0.34, 1.17] |
| llm_prefill | inductor | 4.812 | 5.033 | 4.93% [4.12, 5.85] | 0.98% [0.71, 1.38] | 1.14% [0.65, 1.65] | 1.32% [0.85, 1.74] |
| moe_sparse | eager | 2.790 | 2.999 | 7.10% [6.98, 7.23] | 4.25% [3.97, 4.65] | 0.64% [-0.34, 1.58] | 4.72% [4.01, 5.50] |
| moe_sparse | inductor | 3.725 | 3.909 | 4.75% [4.43, 5.01] | 3.45% [3.12, 3.73] | 0.18% [-0.73, 1.09] | 3.88% [3.23, 4.56] |
| moe_dense | eager | 9.360 | 9.910 | 5.56% [5.39, 5.74] | 4.55% [4.27, 4.97] | 1.23% [0.10, 2.35] | 5.42% [4.71, 6.15] |
| moe_dense | inductor | 12.340 | 12.904 | 4.39% [4.12, 4.66] | 3.79% [3.59, 4.00] | 0.92% [0.12, 1.60] | 4.34% [3.70, 4.81] |

Displayed latencies are medians of process p50s. Effects are means of paired process percentage reductions, so they need not equal a ratio of the displayed medians. Brackets are exploratory 95% percentile-bootstrap intervals over **five processes**, with 10,000 resamples and seed 263. Each process has 60 warm samples per path after three warmups. Request samples are not bootstrap replicates.

The combined intervention leaves decode Inductor needing another **5.78% [5.37, 6.27]** reduction to match its contemporaneous Torch control. Dense eager MoE's remaining requirement is **0.09% [−0.02, 0.21]**, consistent with parity rather than a demonstrated win. Effects are not added across interventions. Eager prefill's `no_report` p95 effect is **−0.21% [−1.71, 0.67]**; a general tail-latency improvement is not established. Every per-process p50/p95 and its raw samples remain in the bundle.

## Why report processing reaches the critical path

The source-level chain is concrete: [`FastPrefetch.__next__`](../../libreloc/python/reloc_torch/fast.py#L856) fills/retires groups before returning weights to the consumer. A completion query or host wait can call [`FastCompletion._finish`](../../libreloc/python/reloc_torch/fast.py#L570), which eagerly materializes [`groupReport`](../../libreloc/python/PyDispatch.cpp#L245) and records it in Python. That work can delay the next consumer launch even when a prior transfer has completed.

CUDA launch correlation associates each copy/kernel with its submitting thread and `group263/<request>/gN` marker. Event IDs connect producer records to consumer stream waits. For example, `e2-r1-nvtx`, decode/Sym/eager/warm/1, event **5264** connects group **g1** on stream **141** to consumer stream **7**: the producer's final `dequantS8F16Quad` ends before the stream-wait API, and the next consumer is enqueued after that wait. The raw trace retains the intervening host phases and exact timestamps.

A stream wait constrains work enqueued after the wait call; its API return timestamp is not GPU event completion. Witness selection therefore uses enqueue order, and synchronization API durations are never treated as GPU stall durations. See NVIDIA's [CUDA stream-management reference](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-runtime-api/group__CUDART__STREAM.html).

The following **instrumented** round-1 warm medians are diagnostic. The report column is exclusive `_finish` time; nested accounting has its own phase. Host and GPU intervals overlap, and phase medians do not form an additive latency decomposition.

| Sym eager workload | Groups | Traced request ms | Assigned GPU union ms | Exclusive report ms | Report coincident with GPU idle ms |
|---|---|---|---|---|---|
| llm_decode | 22 | 35.088 | 17.293 | 0.968 | 0.891 |
| llm_prefill | 4 | 7.856 | 4.810 | 0.184 | 0.073 |
| moe_sparse | 4 | 5.402 | 1.690 | 0.138 | 0.087 |
| moe_dense | 16 | 17.706 | 6.501 | 0.484 | 0.438 |

Report processing is more exposed during decode and MoE, while more of prefill's report work overlaps GPU activity. This supports the hypothesis; the unprofiled bypass supplies the causal estimate. We do not subtract profiled GPU durations from unprofiled model latency or claim all GPU-idle host time is removable.

Diagnostic timelines are retained in the published artifact snapshot: [decode](https://github.com/JueonPark/sym/blob/profile-issue263-2026-11-07-v1/bench/issue263/figures/decode.svg) and [prefill](https://github.com/JueonPark/sym/blob/profile-issue263-2026-11-07-v1/bench/issue263/figures/prefill.svg).

Figures select the request nearest the within-path median in trace round 1. Purple is host consumer wall time, including launch overhead/waits; it is not an on-CPU utilization measurement. All streams are retained in the raw Nsight/SQLite files.

## E5: lifecycle and revisions

These are three process pairs per condition. Total time follows the original accounted envelope: launch/setup, 120 completed requests, shape/scale updates, and retirement, excluding oracle-validation intervals. The adapter's common imports/provenance verification are included in launch/setup.

| Workload | Compute | Torch total s | Sym baseline total s | Warm p50 reduction with `no_report` | Total reduction with `no_report` |
|---|---|---|---|---|---|
| llm_decode | eager | 7.062 | 7.292 | 5.11% [4.71, 5.76] | 1.93% [1.56, 2.19] |
| llm_decode | inductor | 24.556 | 24.844 | 4.62% [4.08, 4.99] | 0.61% [0.55, 0.67] |
| moe_sparse | eager | 5.453 | 5.428 | 3.54% [3.37, 3.74] | 2.61% [-0.90, 6.78] |
| moe_sparse | inductor | 12.905 | 13.020 | 2.96% [1.93, 4.98] | -0.20% [-2.35, 1.77] |

Sparse MoE's total-lifecycle intervals include zero; Inductor's point estimate is slightly negative. Warm savings do not establish a cold-lifecycle win. For context, the following are median **Torch / Sym baseline** durations in milliseconds; revision-first requests and explicit scale preparation are distinct:

| Workload | Compute | Initial request ms | First shape-revision request ms | First scale-revision request ms | Scale preparation ms |
|---|---|---|---|---|---|
| llm_decode | eager | 298.4 / 360.9 | 25.6 / 28.7 | 20.2 / 21.7 | 8.0 / 8.2 |
| llm_decode | inductor | 17588.0 / 17657.3 | 24.0 / 28.5 | 18.6 / 20.5 | 8.1 / 8.3 |
| moe_sparse | eager | 397.3 / 357.6 | 23.6 / 25.0 | 9.6 / 10.4 | 4.9 / 4.9 |
| moe_sparse | inductor | 4663.1 / 4748.9 | 2741.1 / 2720.3 | 13.2 / 15.5 | 5.0 / 6.0 |

The trace files separately mark owner/snapshot setup, initial/warm requests, snapshot refresh, shape/scale revisions, and retirement. Cold compilation remains visible inside initial or revision consumer phases. No claim is made that report bypass removes those costs.

## E5: fixed bursts and caller count

All arms use eight requests per burst, fixed eight-logical-CPU affinity, 128 MiB aggregate native scratch, two contexts per caller, and a 1024 MiB physical free-VRAM budget. Throughput is requests divided by summed burst duration; it does not model sustained arrival traffic. Sojourn includes executor queueing. The same eight distinct input tensors are reused across bursts, as in the frozen E5 protocol.

| Workload | Callers | `no_report` throughput gain versus Sym baseline | `no_report` sojourn-p95 reduction versus Sym baseline |
|---|---|---|---|
| llm_decode | 1 | 3.57% [-0.36, 6.30] | 1.49% [-4.64, 4.72] |
| llm_decode | 2 | 4.12% [3.84, 4.41] | 3.86% [2.68, 4.79] |
| llm_decode | 4 | 3.11% [0.63, 4.44] | -0.01% [-8.25, 4.20] |
| moe_sparse | 1 | 3.49% [-2.56, 8.21] | 2.06% [-6.36, 9.18] |
| moe_sparse | 2 | 5.82% [0.26, 12.96] | 4.71% [-1.45, 11.12] |
| moe_sparse | 4 | 4.91% [3.62, 6.14] | 4.91% [3.40, 6.06] |

The direct comparison with the matched Torch control is separate:

| Workload | Callers | `no_report` Sym throughput gain over Torch | `no_report` Sym sojourn-p95 reduction versus Torch |
|---|---|---|---|
| llm_decode | 1 | -5.04% [-9.10, -2.89] | -6.69% [-13.79, -3.02] |
| llm_decode | 2 | -1.53% [-3.48, 0.28] | -1.71% [-3.43, -0.56] |
| llm_decode | 4 | -3.96% [-6.54, -2.59] | -7.29% [-16.51, -2.49] |
| moe_sparse | 1 | -3.87% [-7.63, 0.40] | -4.88% [-12.17, 2.06] |
| moe_sparse | 2 | 0.98% [-1.65, 2.89] | 1.00% [-3.41, 4.72] |
| moe_sparse | 4 | 1.62% [0.89, 2.30] | 1.94% [1.03, 2.59] |

Four-caller sparse MoE shows a small advantage in both metrics in these three process pairs. The broad confidence intervals elsewhere, including decode/four-caller p95, preclude a blanket serving claim. The original observed regression remains the default outcome across most matched conditions.

## Instrumentation and memory

Added host counters inflate E2 latency by **0.89–21.85%** across paths; NVTX plus Nsight inflates it by **29.62–79.20%**. Decode lifecycle counters inflate warm p50 by **9.98–14.87%**, and decode/four-caller burst counters inflate mean burst duration by **11.80–21.65%**. These are ranges of process-paired mean effects, not confidence bounds. All intervals are in `summary.json`. NVTX E2 runs have only three warm samples per path and three process rounds; they remain diagnostics.

These contrasts measure added instrumentation within the original timing brackets. The runtime's normal accounting and E5's original physical-memory sampler stay active. Host ranges measure wall time, including waits and scheduling delays.

Every accepted Sym burst has **zero warm native host-allocation, device-allocation, and context-creation delta**. That does not imply allocation-free requests: decode's separate E2 probe counts **505 Sym versus 639 Torch** CUDA allocator requests with eager compute, and **289 versus 423** with Inductor. Sym's lower count does not establish that its allocation path is faster. Python's counter measures net live blocks, Torch's count includes cache reuse, native counters describe their own retained pool, and NVML/RSS peaks are sampled lower bounds. These scopes are retained separately and never summed into an exact memory decomposition. Held outputs survive owner retirement; native retained/live bytes, streams, workers, and active contexts return to zero.

## Prioritized follow-ups and residuals

1. **[#264](https://github.com/JueonPark/sym/issues/264): make report accounting cheaper, then reduce repeated group preparation/binding.** Preserve compact accounting and error propagation while materializing detailed Python reports on demand; retire leases/references promptly. The measured report bypass supplies a concrete target. Prepared group structure and immutable metadata can reduce repeated work, but reusable native request state needs a separate lifetime-safe design and its own controlled measurement.
2. **[#264](https://github.com/JueonPark/sym/issues/264), [#265](https://github.com/JueonPark/sym/issues/265): handle preparation/compilation and revision costs explicitly.** Warm-request gains alone do not justify a short-lifecycle claim. Automatic lowering should expose the qualified fast path without dropping validation.
3. **[#268](https://github.com/JueonPark/sym/issues/268): test graph replay after stable bindings are available.** Traces retain launch gaps and event ordering, but replay savings are unmeasured. Fresh inputs, revisions, caller isolation, and held outputs remain requirements; [#266](https://github.com/JueonPark/sym/issues/266) supplies the lifetime/memory constraints.
4. **[#267](https://github.com/JueonPark/sym/issues/267), [#269](https://github.com/JueonPark/sym/issues/269): separate kernel/fusion improvements from framework overhead.** Both controls already call the same deployed kernels. Improvements available to both must be applied to the matched control before attributing a gain to Sym.

Native group preparation/submission, output handling, extra CUDA event/API work, and interactions with consumer scheduling remain residual candidates. The frozen runtime lacks internal native CPU NVTX, so the native boundary is not further decomposed here. Their full host durations are not proven removable. No independent diagnostic costs are subtracted or summed to predict a production speedup.

This supports the planner work in [#271](https://github.com/JueonPark/sym/issues/271) and evaluation work in [#274](https://github.com/JueonPark/sym/issues/274): include host overhead, caller pressure, and lifecycle cost when selecting a mechanism. The evidence is one hardware/software configuration with small numbers of independent processes and a fixed-burst workload. It establishes a removable component and remaining work, not general superiority.

## Acceptance evidence and artifact audit

| #263 criterion | Evidence |
|---|---|
| Completed-request timelines and dependency chains without double counting | 23 raw Nsight/SQLite traces, request/thread/group correlation, event-record/wait witnesses, exclusive host phases, GPU unions, and the linked timelines above |
| Controlled removable-cost estimates with uncertainty | Separate unprofiled `baseline`, `no_report`, `no_check`, and `both` processes; five E2 and three E5 rounds; all per-process pairs and bootstrap definitions |
| Parity target distinguished from measured costs; follow-ups prioritized | E2 parity/effect table, direct burst-control table, and the prioritized residual work |
| Raw traces/settings/revisions, p50/p95, and unexplained residuals retained | `campaign/`, `summary.json`, `matched-trace-checks.json`, `final-input-verification.json`, protocol, source snapshot, runtime files, and checksums in the release |

The three-second isolation audit observed no foreign GPU/build processes during qualified runs; it is a sampled audit, not proof of an exclusively occupied machine. All numerical/lifetime checks passed. Validation passed 47 harness tests and 87 original control/frontend tests. The harness gate includes the enqueue-order regression.

The final review added strict execution-contract and transitive-source checks. Because the completed campaign predates those guards, `campaign/historical-qualification.json` binds all 150 original records and audits without rewriting them. Every recorded launch, software/GPU-name/affinity/thread/environment setting agrees after caller/cache normalization. Eight workload/timing helpers match the baseline in the current checkout and all four recorded harness commits, and both frozen E5 runner hashes match. These source checks were performed after measurement and cannot retrospectively exclude temporary dirty edits. The two additional pyreloc entry points were likewise verified after the campaign. New campaigns capture and enforce this provenance before launch.

The frozen artifact and runtime are unchanged. One controller interruption before accepted profiling is preserved under `excluded-runner-restart/` and its run was repeated. Accepted Nsight captures use a minimal environment. Pilot native reports containing unrelated workstation/session variables are excluded from the public bundle; copied pilot SQLite files remove only that environment metadata, retaining all event data. The release includes original accepted traces, excluded-run records, source/binary provenance, and a file-level manifest.
