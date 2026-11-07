# Issue 263 measurement protocol

The target is attribution of the matched E2/E5 regressions, not a claim that a production optimization is already qualified. Runtime commit: `2617ba166f98a7476a5f472d829a573317fc0cfb`. Frozen input: the published `experiments-2026-11-04-v1` artifact. New results live in a separate campaign directory.

## Questions and hypotheses

1. Does repeated report materialization/accounting extend the request's critical path? A diagnostic `no_report` intervention may bypass `FastCompletion._finish` report extraction/accounting while retaining `_done`, native completion, references, and output stream recording. It disables runtime accounting; it is not a production API proposal.
2. Does repeated Python signature validation contribute measurable completed latency for the already-qualified immutable source metadata? A `no_check` diagnostic may bypass only `_Program.check`, retaining native validation and all lifetimes. It is restricted to inputs whose original checks pass; it does not qualify unsafe production behavior.
3. Are these effects independent or interacting? A combined intervention is a separately measured contrast; effects are never summed.
4. What remains in native group preparation/submission, allocation, synchronization, copy/kernel execution, and consumer gaps after these candidates? Traces guide further controlled probes if the initial hypotheses fail; any new accepted contrast is registered before measurement.

## Separation of modes

- `off`: original callables, no phase/counter wrappers and no profiler; used for accepted paired latency.
- `counters`: thread-local CPU phase timing and counts, no NVTX/profiler; compared with `off` to measure instrumentation overhead.
- `nvtx`: request/group/phase annotations plus Nsight CUDA activity; diagnostic only. Profiler overhead is measured separately and never subtracted to manufacture production timings.

Host-exclusive phases partition the measured host request. GPU copy/kernel intervals use unions to avoid double counting streams. Their overlaps with host phases are reported separately. A GPU-busy interval is not proof that all CPU work is hidden; a host wait is not an additional independent GPU cost. Concurrent requests require launch correlation or burst-level aggregation; temporal containment alone cannot assign overlapping requests.

## Coverage and controls

- E2: all four FP16 workloads (`llm_decode`, `llm_prefill`, `moe_sparse`, `moe_dense`), eager and Inductor, native conversion/permutation, grouped KV, whole-request window, identical two-slot prefetch. Baseline Torch and baseline Sym share compute functions and input/oracle. Diagnostic variants change only named Sym overheads.
- E5 lifecycle: decode and sparse MoE, eager/Inductor; initial request, warm requests, input-shape update, scale-snapshot revision, retirement, retained outputs. Preserve the original 100 initial + 10 shape + 10 scale request sequence for accepted lifecycle comparisons.
- E5 concurrency: decode and sparse MoE, eager compute, 1/2/4 callers, eight requests per burst, 30 bursts, fixed eight logical CPUs, 128 MiB aggregate native scratch, two contexts per caller, and 1024 MiB physical free-VRAM budget. Preserve previous-burst outputs and queue/service/sojourn distinctions.
- Record actual native allocation/context deltas, Torch allocator counters, and retained memory separately. Sampled physical peaks remain lower bounds and counters with different scopes are not added.

## Pilots and acceptance

Pilot runs establish trace capability, numerical/lifetime correctness, and plausible interventions. They are excluded from accepted comparisons. Freeze the final matrix before acceptance; record any justified change here with its evidence and date.

E2 acceptance uses five independent processes, three warmups, and 60 completed samples per path/condition, rotating order by round and sample. E5 lifecycle/concurrency uses three independent rounds under the original sequences/bursts. Counter-overhead and profiled contrasts are separate runs with the same named workload and timing envelope. Source/build hashes, dependency versions, hardware, affinity, threads, budgets, stream settings, and cache paths are recorded.

Primary statistics are paired process-level latency reductions and throughput changes, with exploratory bootstrap intervals over processes. Request p50/p95 remain separate descriptive metrics. Require no warm recompilation for comparisons that assume a warmed fixed compute graph. Preserve all correctness failures, contamination, neutral results, and regressions outside accepted summaries.

The historical 4.6–10.9% Sym slowdown implies a 4.4–9.8% reduction of Sym's own completed latency to reach parity. Recompute the current paired parity requirement; compare it with measured intervention effects and uncertainty. State any residual that the evidence does not explain. This is fixed-burst analysis, not sustained serving.

## Execution and reporting

No production runtime changes are needed for the initial investigation. Keep the published evidence and runtime build intact. CPU/CUDA timelines, raw samples, SQLite exports, logs, source snapshots, and failure records are release assets; compact reports and reusable harness code belong in the PR. Fresh runs must accept explicit artifact/build paths rather than require the original machine's home directory.
