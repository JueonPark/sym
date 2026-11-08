# Issue 266: qualified memory and latency tradeoffs

The opt-in memory policies recover Sym's previously failing E4 partial-residency
cell and substantially reduce four-caller decode memory. They do **not** provide
a general latency improvement. In the difficult E4 cell, fixed packing/recycling
is faster than the adaptive policy; in E5, adaptive Sym increases p95 in all six
workload/caller configurations. Defaults therefore keep ordinary output allocation.

The complete frozen campaign passed strict analysis: **360 jobs, 336 successes,
24 actual OOMs, zero terminal admission refusals, and zero contaminated acceptance
attempts**. All 120 predeclared arms and all three rounds are retained. The OOMs
are valid measured negative outcomes, not discarded test failures.

[Complete tables](TABLES.md) · [Frozen protocol](PROTOCOL.md) ·
[Runtime contract](../../docs/budgeted-transfer-memory.md) ·
[Review disposition](REVIEW.md) ·
[Versioned Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-266) ·
[Machine-readable index](INDEX.json)

## What was qualified

Measurement revision: `119bd7f587213967c1c3fd770dd6a71dcd14cc33`. The campaign ran
on 2026-11-08, from 02:57 to 03:51 KST after its initial idle check. It pins 72
measurement/source files and the complete input/runtime inventories. Numerical
workloads and Torch kernel controls come from the original E1–E5 artifact.
The fresh arms use the current runtime, including earlier prepared-group work
and existing native scratch reuse; they are not a replay of the old implementation.

One RTX 2080 Ti, Python 3.14.7, Torch 2.14.0+cu126, GPU0, CPU affinity 4–7,20–23
and eight aggregate CPU threads were fixed. E4 uses physical free budgets of
512/1024 MiB; E5 uses 1024 MiB. A 2 ms NVML sampler supplies observed lower bounds
on physical peaks. The trace baselines include the context and budget reservation.
E4 deployment peaks exclude the oracle interval; E5/group peaks cover the warm
loop, including validation gaps. These windows must not be compared as equivalent.

Tables use the median of three round medians or round p95s. E5 median time is
completed eight-request burst time divided by eight; its p95 is request sojourn,
including queueing. These finite bursts are not sustained-server throughput.
Three rounds do not establish statistical significance or cross-platform generality.

## E4: the failing 512 MiB partial-residency case

L32/context512, eight resident layers, 512 MiB free budget:

| Runtime | Policy | Success | Median ms | P95 ms | Deployment MiB | Torch reserved MiB | Native device retained MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch | fresh | 3/3 | 33.56 | 36.84 | 504 | 448.00 | 0.00 |
| torch | packed | 3/3 | 35.85 | 36.14 | 468 | 412.00 | 0.00 |
| torch | recycle | 3/3 | 38.05 | 41.08 | 468 | 412.00 | 0.00 |
| torch | adaptive | 3/3 | 46.76 | 49.02 | 444 | 388.00 | 0.00 |
| sym | fresh | 0/3 | — | — | 510 (OOM) | — | — |
| sym | packed | 3/3 | 39.54 | 40.20 | 484 | 390.00 | 30.82 |
| sym | recycle | 3/3 | 38.52 | 41.43 | 484 | 390.00 | 30.82 |
| sym | adaptive | 3/3 | 48.27 | 51.36 | 444 | 366.00 | 17.04 |

Sym fresh fails on its first request in all three rounds. Packing alone recovers
3/3 at 484 MiB and 39.54 ms; recycling keeps the same sampled peak at 38.52 ms.
Adaptive selects one slot/context instead of two, reducing the output window
from 72 to 48 MiB and native retained device scratch from 30.82 to 17.04 MiB.
Its sampled deployment peak falls to 444 MiB, with 48.27 ms completed latency.
This is a capacity/latency tradeoff: the adaptive policy is slower than fixed
packing or recycling in this cell. Matched adaptive Torch also reaches 444 MiB
and is faster at 46.76 ms. Ordinary Torch remains faster at 33.56 ms and fits
at a larger 504 MiB sampled peak.

The full E4 matrix has 192 successful jobs and 24 OOMs. Twenty-one OOMs are
full-resident Torch cases at 512 MiB: all ordinary resident configurations,
plus packed L16/context512 and both packed L32 configurations. Packing recovers
the L16/context16 resident control (3/3, 496 MiB, 10.80 ms), but cannot make every
resident configuration feasible. Packed L16/context512 fails on the first
request after preload; packed L32 fails when requesting its 768 MiB resident
allocation. The latter can OOM at only 90/94 MiB observed usage because the
requested allocation itself cannot fit. A low sampled peak does not imply
that the failed allocation was feasible. The remaining three OOMs are the
Sym fresh partial case above. No 1024 MiB E4 arm failed.

## E5: held outputs, callers and tail latency

Every E5 arm succeeds 3/3. The following table shows Sym fresh → adaptive and
the matched adaptive Torch control; full fixed-policy attribution is in TABLES.md.

| Workload | Callers | Sym median ms | Sym sojourn p95 ms | Sym warm MiB | Torch adaptive median ms | Torch adaptive p95 ms | Torch adaptive MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_decode | 1 | 19.13 → 23.10 | 152.94 → 184.85 | 332 → 304 | 21.64 | 173.32 | 290 |
| llm_decode | 2 | 17.38 → 24.83 | 140.13 → 201.21 | 540 → 444 | 21.75 | 175.50 | 414 |
| llm_decode | 4 | 19.72 → 30.43 | 157.82 → 243.97 | 956 → 526 | 23.89 | 192.13 | 566 |
| moe_sparse | 1 | 3.06 → 3.86 | 24.45 → 30.76 | 184 → 184 | 3.23 | 25.72 | 170 |
| moe_sparse | 2 | 2.48 → 3.25 | 19.72 → 26.03 | 270 → 270 | 2.89 | 23.01 | 242 |
| moe_sparse | 4 | 2.31 → 3.38 | 18.48 → 27.34 | 442 → 442 | 3.07 | 24.81 | 386 |

Four-caller decode is the clearest memory improvement: Sym falls from 956 to
526 MiB (45.0%), while p95 rises from 157.82 to 243.97 ms (54.6%). Compared with
matched adaptive Torch, Sym uses 40 MiB less sampled memory but takes 30.43 vs
23.89 ms per request-equivalent burst and has 243.97 vs 192.13 ms p95. One/two
caller decode saves 28/96 MiB against Sym fresh. MoE shows no sampled physical
improvement at any caller count, while its p95 increases. The optimization is
therefore useful for selected capacity constraints, not an unconditional fast path.

At four-caller decode warm end, Sym native retained device bytes fall from
110.28 to 32.05 MiB, and Torch reserved bytes fall from 726 to 398 MiB. The
adaptive arena accounts for 192 MiB of logical output storage inside those
Torch allocations. These numbers are separate scopes and must not be added
to reconstruct the 526 MiB physical peak. Context/library/workspace state is
not completely exposed by either allocator counter.

All 96 E5 processes check 240 completed requests each. The campaign contains
23,040 retained-output comparisons, including 768 checks after worker/snapshot
closure. All Sym native owners retire with zero live/retained bytes and active
contexts; packed arenas close without quarantine. Returned tensors remain
valid after closure. Torch cached blocks and caller-held outputs can still
occupy memory after native retirement.

## Attribution: reuse, packing, lifetime and selected depth

Fresh already reuses native scratch. At one-caller decode, Sym's sampled peak
is 332 MiB fresh and 304 MiB for each of packed/recycle/adaptive. Packing reduces
memory; lifetime recycling adds no further sampled reduction there. Packing
makes 3,840 warm arena allocations; recycling makes zero and reuses storage
3,840 times, but increases median time from 20.55 to 23.06 ms. Both have zero
warm native device allocations/context creations. At one-caller MoE, every
Sym policy remains at 184 MiB, including recycling with zero warm arena
allocations. Allocation-count improvements alone do not establish a physical
memory or latency advantage.

Group attribution uses 1 KiB small and 24 MiB large output groups. Each cell
has three rounds of 100 checked completed iterator deliveries:

| Size | Policy | Sym delivery ms | Torch delivery ms | Sym MiB | Torch MiB | Sym warm native device allocs | Sym warm arena allocs | Sym warm arena reuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | fresh | 0.2602 | 0.3905 | 24 | 22 | 0 | 0 | 0 |
| small | packed | 0.4629 | 0.4902 | 24 | 22 | 0 | 100 | 0 |
| small | recycle | 0.4563 | 0.4824 | 24 | 22 | 0 | 2 | 98 |
| small | adaptive | 0.4738 | 0.4766 | 24 | 22 | 0 | 1 | 99 |
| large | fresh | 1.4772 | 1.3179 | 176 | 142 | 8 | 0 | 0 |
| large | packed | 1.5937 | 1.4531 | 128 | 114 | 8 | 100 | 0 |
| large | recycle | 1.6013 | 1.4489 | 128 | 114 | 8 | 2 | 98 |
| large | adaptive | 1.6053 | 1.4699 | 86 | 90 | 0 | 1 | 99 |

Large Sym groups fall from 176 to 128 MiB with packing, remain at 128 MiB with
recycling, and fall to 86 MiB with adaptive depth. Delivery time increases
from 1.477 to 1.605 ms. Matched adaptive Torch uses 90 MiB and takes 1.470 ms.
The small Sym group increases from 0.260 to 0.474 ms with no sampled memory
reduction (24 MiB throughout). The conservative device barrier on recycling
and the extra allocation/planning machinery are not free.

The large fixed-depth group cases create eight native device buffers during
the timed loop: their serial warmup did not activate the second context.
Adaptive uses one context and creates no additional native device buffer.
This cold-context cost remains in the measured samples; it is not removed to
manufacture zero warm allocations. All 48 group jobs also verify a held output
after retirement.

**Deferred measurement limit:** group iterator construction submits its first
group before delivery timers start. The group metric is completed delivery
latency, not complete amortized submission cost. The physical warm-loop window
includes initialization and correctness checks. This limits conclusions about
very small operations; the original review's Minor finding remains disclosed.

## Preparation and retirement remain visible

In the difficult E4 cell, Sym packed/recycle/adaptive preparation takes about
939–946 ms plus 34 ms resident preload. Torch takes about 908–910 ms preparation
plus 40 ms packed preload; ordinary Torch preload is about 9.62 ms. Packed
preload uses one resident allocation and serial per-matrix loading with cache
clears during preparation only. Serving does not clear the cache. These costs
are excluded from steady request latency and are reported separately, not waived.

Four-caller decode lifecycle costs (ms):

| Runtime | Policy | Snapshot prep | Owner prep | Workers prep sum | First request max | Workers retire sum |
| --- | --- | --- | --- | --- | --- | --- |
| sym | fresh | 118.63 | 10.80 | 5.87 | 60.51 | 8.69 |
| sym | adaptive | 118.11 | 6.69 | 5.73 | 74.24 | 4.54 |
| torch | fresh | 117.12 | 0.00 | 4.77 | 24.69 | 0.24 |
| torch | adaptive | 116.90 | 0.00 | 4.60 | 39.25 | 0.53 |

Worker construction/retirement sums cover callers; first-request max is the
largest per-caller first service time, then the median across rounds. First
requests are part of warmup, not the timed bursts. Each caller's complete
warmup samples and phase boundaries are preserved. Small/large Sym group
preparation is about 14–16/41–42 ms, with first calls near 19–21 ms; matched
Torch preparation is about 6–7/33 ms, with first calls near 1.5–2.8 ms. Planning,
preparation, first-call and retirement columns for every arm are in TABLES.md.

## What the capacity model establishes

All 96 adaptive jobs succeed, with no actual OOM or terminal admission refusal.
At the difficult E4 cell, the declared serving estimate is 496 MiB within
512 MiB, selecting eight resident layers, one slot/context, a 48 MiB output
window, 16 MiB estimated device scratch and 32 MiB safety. Other E4 cells keep
two slots/contexts. E5 partitions a 256 MiB transfer allowance across callers;
the other 768 MiB of physical capacity is a separate fixture/compute/library
allowance. Four-caller decode selects one slot/context and single-member
staging: 48 MiB output + 8 MiB device scratch + 8 MiB safety per caller.
The predicted 64 MiB is transfer capacity, not a prediction of the complete
526 MiB application peak. MoE retains two slots/contexts at every caller count.

These finite choices demonstrate execution under the declared budgets, with
held consumers protected by strong ownership and a completion barrier before
reuse. They do not prove an exact whole-program liveness bound or latency
optimality. Workspace estimates are supplied by the caller; resident preload
staging, opaque library/allocator state and unannounced retention can still
cause a prediction to miss actual feasibility. The JSON preserves applied
plans separately from observed peaks and failure phases. Plans computed for
non-adaptive controls are marked as not applied. Speculative arena refusal
counters describe backpressure attempts that later progress, not terminal
admission outcomes.

## Verification, review and reproducibility

The acceptance matrix includes 31,680 checked timed results across E4, E5 and
groups. Strict standard-library reanalysis reproduces `analysis.json` and
`tables.md` byte for byte. The runner verifies actual source/runtime/input file hashes before and after
measurement. Saved-data analysis verifies matching recorded provenance, every
job, executed policy/configuration, physical reservations, trace checksums and
sampled phase peaks, finite timings, retained-output checks and retirement.
The final source snapshot retains the measurement files byte for byte.

Validation evidence: the full Python suite passed 1,445 tests with 13
backend/environment skips and two expected fork deprecation warnings; an
independent CMake CPU bridge build and 49 CPU cases passed; the current
analyzer/runner suite passed 37 tests. All seven CI checks passed on the frozen
measurement revision. Tests cover retained backing/storage aliases, detached
views, concurrent/post-close consumers, allocator pressure, fork guards,
subgroup errors and CUDA fault quarantine.

One independent implementation/harness review found five Important issues,
all corrected, and the group-timer Minor limitation above. There was no second
review. Later reporting and CPU-isolation/retry changes were verified by root
regressions and complete qualification. See REVIEW.md for each finding and
scope decision, including the absence of a dynamic reproduction claim for the
old CUDA race and the exclusion of raw-pointer/weak-storage resurrection after
the final strong consumer releases ownership.

Pilots are wholly excluded, including early fixture/control corrections and
runs with foreign build/CPU activity. The final pilot has 20 jobs (19 successes,
one actual OOM) and is still excluded by protocol. The accepted campaign has
no contaminated attempts. The Linux audit checks build/GPU activity and
sustained foreign CPU work; sampled auditing cannot exclude every transient
or low-utilization disturbance. No result was retried because it was slow or
because a clean policy OOM occurred.

The versioned Wiki package contains the complete raw records, traces, audits,
excluded pilots and source inventories, a compact final source snapshot,
verification logs, and a five-file runtime overlay. Unchanged compiler/native
files are pinned to the existing #265 Wiki runtime; GPU input/control files
are pinned to the original E1–E5 Wiki artifact. CPU saved-data analysis needs
only the new package and Python's standard library. The Wiki page supplies
archive/part/file hashes, dependency hashes, reconstruction commands and
anonymous-download verification.

This supplies concrete memory/lifetime evidence for #261, #270, #271 and #274.
It supports opt-in capacity control and records its costs; it does not complete
the broader semantics, joint optimization or generality contributions required
for a compiler/systems full paper. Issue #266 closes only when PR #279 merges.
