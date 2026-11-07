# Issue 264 measurement protocol

The baseline is Sym `2617ba166f98a7476a5f472d829a573317fc0cfb` and the published November E1–E5 inputs, with the #263 harness supplied separately. The candidate extends existing dispatch groups/resources. New-build legacy (`prepared_groups=False, report_mode='eager'`) separates changes to the build, owner locking and report lifetime from the two optimization switches. No validation, ownership, error handling, or accounting is bypassed.

## Grouped qualification

Five independent processes per policy, rotated by round: frozen baseline, new-build legacy/eager, legacy/native, prepared/eager, prepared/native. Each process measures two small INT8-to-FP16 weights, a six-matrix decode-sized weight group (four 1024×1024, one 1024×4096, one 4096×1024), and a two-member KV restore of float32 pairs shaped 64×16×32. Weights use the existing prepared wire format and fused FP16 row; KV uses the existing device layout kernel.

Each case reports common program/snapshot setup, its first completed group, five warmups, 300 completed samples, and 100 separate CPU-submission samples. Completed samples include successful native completion and CUDA synchronization. CPU submission is timed separately, with waiting outside its interval. Ten additional requests count actual legacy group/view construction calls or native prepared construction/rebind counters, without contaminating primary timing. Explicit prepared-handle creation and its first submission are additional diagnostics after all timing; implicit setup remains charged to the initial request.

Inputs are deterministic; numeric oracles run outside timing. Outputs are fresh; the last two groups remain held across subsequent requests and owner close, then are checked again. Report dictionaries are not requested during timing. Native aggregate accounting remains enabled. Warm native context and allocation counts must remain constant; native retained/live bytes must be zero after close.

Memory includes process RSS/PSS (whole-process scopes, including allocator caches), Torch allocated/reserved device bytes, snapshot sizes, and native retained/live resource bytes. These are separate scopes and must not be summed as though disjoint. Prepared handle/slot/construction counts describe metadata retention; RSS differences are noisy and not exact metadata allocation sizes.

Break-even counts charge setup and first-request premiums: `1 + max(0, ceil((new_setup + new_first - old_setup - old_first) / (old_warm - new_warm)))`. They are reported only for positive warm savings, with per-process values. Cold compiler/snapshot variability remains visible. A small warm benefit does not erase an expensive or uncertain cold path.

## Matched E2/E5 qualification

E2 uses the unchanged published `run_case`: decode, prefill, sparse MoE, dense MoE; Torch and Sym with eager and Inductor consumers; five rounds, three warmups, 60 warm samples. Each round rotates frozen baseline, new-build legacy, and the policy selected from grouped qualification. Torch controls appear in every process, so build/process drift is visible. FP16, exact fused kernels, grouped transfers, whole-layer prefetch, depth two, and 128 MiB aggregate native scratch remain matched. No warm consumer recompilation or native resource growth is admitted.

E5 uses the immutable lifecycle and concurrency programs through the published #263 wrapper. Decode and sparse MoE lifecycle each use eager/Inductor, three rounds, 100 initial requests plus ten shape-change and ten scale-change requests. Bursts use 1/2/4 callers, three rounds, 30 batches of eight requests with the previous burst retained. Arms are Torch on the baseline build, Sym on the baseline build, new-build legacy Sym, and selected-policy Sym, rotated by round. The original CPU/VRAM/scratch budgets, physical memory sampling, snapshot revision replacement, numerical checks and held-output retirement checks remain active.

CPU affinity is `4-7,20-23`, CUDA device 0, OMP/MKL threads eight for E2/groups; E5 partitions threads as its frozen program specifies. Interop threads are one, TF32 is off, and Inductor precision casts are emulated. Inductor caches are per process. GPU/build/packaging interference is audited before and every three seconds during a job; contaminated runs are rejected. No profiler is used for primary timing.

Request percentiles are descriptive. Effect uncertainty bootstraps independent process-paired percentage reductions (seed 264, 10,000 resamples); requests within one process are not independent replicates. A positive reduction means a faster candidate. Report p50 and p95, preparation/revision/retirement, memory, and unfavorable effects together. These are fixed synthetic offload conditions, not a sustained-serving or cross-hardware claim.
