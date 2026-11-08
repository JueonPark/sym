# Issue 268 replay qualification protocol

Base runtime: merged #267, b319a0ab94dbc8c8264e4fa093ffbbb796e2c046.
Only the explicit GPU-region wrapper changes. Native kernels, numerical workload
functions, grouped KV and whole-request two-slot prefetch are identical.

## Main matrix

Three rounds. E2: decode, prefill, sparse MoE, dense MoE × eager/Inductor.
E5: original decode/sparse MoE × 1/2/4 callers × eager, plus 1-caller Inductor.
Each cell compares Torch/Sym × unwrapped/slots-off/replay-on: six arms, each in a
fresh process. Mode blocks rotate by round, and runtime order reverses in
round 2, so every mode includes both Torch-first and Sym-first observations.
This is 48 cells and 288 arm processes.
The entire six-arm cell follows a measured 60-second quiet interval; continuous
process-tree GPU/build/CPU audits cover every arm. Rejected contaminated attempts
remain checksummed and cannot count as evidence. No timing retries for losses.
Fresh independent Inductor disk caches per arm keep compilation costs visible.

E2 uses original fixed seed 225 inputs, 5 warm and 60 timed completed requests. The earlier E2 protocol used 3;
this common increase satisfies the parent #247 minimum. The early 18-arm run
is retained as excluded protocol-development evidence and never analyzed as
acceptance. E5 retains its original 3 passes, at least 6 warm requests per caller.
E5 uses original eight changing inputs, 3 warm passes and 30 eight-request bursts,
persistent callers, round-robin queues and previous-burst retention. Ordinary
model outputs use the original exact token/routing check and FP16 rtol/atol .02.
Held outputs are additionally bit-checked across calls and after owner close.
E5 records service, queue, sojourn and completed burst latency; it is a finite-burst
experiment, not steady-state server throughput. E2 host submission is enqueue
return time with a final synchronize excluded (host routing synchronizations
inside a request remain included); completed latency includes the final fence.

All model GPU functions except mutating MoE combine get the same wrapper. CPU
routing, native transfer production, immutable snapshot preparation, grouping and
prefetch are outside capture. Captured output clones and all parameter/input
refresh copies are inside timed requests. No static-parameter exemption is used.
Unwrapped is the deployment continuity reference; slots-off vs replay-on isolates
replay with the same callable and input/output copies. Compare Sym/Torch within
all three modes; report absolute replay gains separately from Sym-relative gains.

During preparation, a device completion fence precedes every region preparation
to drain producer/prefetch work before capture; these fences remain in first/
warmup costs and never appear in frozen serving. Prepare every encountered bucket
during sequential warmup, including all eight
E5 inputs for each caller, then freeze before concurrent serving. All slots are
prepared, including slots not reached by sequential requests. Shape growth after
freeze falls back. Preserve bucket, busy-slot, mutation and budget exclusions.

## Resources and scope

GPU 0 RTX 2080 Ti; CPUs 4–7,20–23 (four physical cores, eight SMT threads), eight total
Torch threads; one interop thread; passive OpenMP; one Inductor compile worker;
TF32 off. E5 partitions affinity and Torch threads across callers; native gather
threads=1 and aggregate native scratch=128MiB, two contexts per caller. E2 native
gather threads=8, same original scratch/group policy. Precision and kernels are
unchanged in every replay comparison.

E2 has a 256 MiB per-region retained budget. E5 retains the original physical 1,024 MiB
free-capacity reservation. Its total declared replay-owner allocation is at most
256 MiB across callers: decoder/expert 192 MiB and other two regions 32 MiB each,
divided by caller count. This is an admission limit, not a total physical cap.
Private pool reservation and input payload count toward it; capture transient,
output copies, original transfer storage, allocator/context and library memory
are measured separately. Real OOM outcomes remain losses, not missing cells.

Per-arm 2 ms NVML traces cover fixture/oracle, snapshot preparation, worker
preparation, first requests, warmup, warm execution and retirement. Saved phase
boundaries reconstruct observed peaks separately from the whole-process peak.
An interval shorter than the sampling period may have no observation; report
its peak as missing, never zero. All sampled peaks are lower bounds. First
request peaks include capture/compilation rather than isolating CUDA capture
from every other first-request allocation. Torch allocated/reserved peaks and replay
owner inputs/private pools are separate, overlapping scopes and are never added
to NVML. The physical reserve precedes fixture/oracle creation, matching E5. Common oracle work precedes deployment measurement.
Preparation, first request (including compilation/capture), warmup and retirement
remain visible. Graph capture/warmup counters are separate owner measurements,
not subtractable model phases. Total(N)=preparation+sum(first calls)+sum(warmup costs)+N*warm-median
for N requests after preparation;
break-even uses measured intercept differences and is undefined without a
positive warm advantage. Shared driver startup/thermal state can persist between
fresh arm processes; arm rotation and three rounds limit, not eliminate, bias.

## Supplemental evidence

Use exact-shape and small-budget sweeps of a launch-heavy pure GPU region with
both eager/Inductor, explicit changing inputs/parameters and held outputs. Record
capture/first/submission/completed p50/p95, private pools, peak memory and fallback
counts, including shapes beyond capacity and budgets that refuse graphs.
Separate diagnostic Torch/CUPTI traces compare slots-off/replay-on and count
`cudaGraphLaunch` vs ordinary launch API calls and GPU kernels. Profiled timings
are never substituted for acceptance latencies.

The frozen manifest pins every job, source file, original artifact dependency,
native library and compiler hash. The stdlib analyzer rejects missing/duplicate
members, bad hashes, altered protocol or environment, short/missing quiet gates,
contaminated audit records, malformed timings, incomplete correctness or memory
observations. Raw failed/admission/OOM outcomes and negative comparisons remain
published. Canonical analysis uses explicit `math.fsum` aggregation where needed.

Supplemental membership is eager/Inductor × three rounds, each with stable,
varying-shape and 1 MiB owner-budget scenarios × plain/off/on, 100 checked warm calls
per row. Separate ten-call diagnostic traces are enabled in round 1. Supplemental
processes each have their own 60-second quiet gate and continuous audit.
Within each supplemental process, mode/scenario rows reset Dynamo but share an
Inductor disk cache. Their preparation/first costs are therefore not independent
cold-process costs. Varying shapes first reach the compiled callable during the
timed sweep; its distribution includes shape-change compilation penalties.

Separate model diagnostics run ten completed requests with profiling for E2
decode eager Sym and sparse MoE Inductor Torch, off/on, plus E5 decode Inductor
Sym. They are marked pilot/diagnostic and never enter acceptance timings. Their
traces retain kernel-name multisets and launch API counts.

The pre-fix campaign is preserved as excluded evidence after a temporary-input
lifetime defect was reproduced in graph fallback. The final runtime records
input use on the consumer stream for both replay and fallback, and all arms
are remeasured. The same restart adds runtime-order counterbalancing and explicit
phase markers; no timing outcome was used to select retained arms.
