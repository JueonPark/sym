# Issue 266: frozen memory qualification

This protocol qualifies opt-in packed output lifetimes and budget-aware
prefetch, contexts and subgroup staging. It preserves the original E4/E5
mathematics, FP16 compute, source readiness and correctness oracles. It does
not equate zero warm native allocations with a physical memory improvement.

## Campaign membership

`support.matrix()` defines 360 fresh-process jobs over three rounds. Each job
has an immutable ID. All results, including actual OOMs and policy admission
refusals, remain in analysis. Failed validation, oracle/setup failures,
foreign GPU/build work or stale inputs invalidate a job rather than becoming
a successful fallback.

| Experiment | Matrix | Jobs |
| --- | --- | ---: |
| E4 main | L16/L32 × context16/512 × free budget512/1024 MiB × eight arms × three rounds | 192 |
| E4 fixed-policy attribution | L16/context16 and L32/context512 at512 MiB × Torch/Sym × packing/recycling × three rounds | 24 |
| E5 main | decode/sparse MoE ×1/2/4callers × Torch/Sym × fresh/adaptive × three rounds | 72 |
| E5 fixed-policy attribution | both workloads at1caller × Torch/Sym × packing/recycling × three rounds | 24 |
| Group attribution | small/large × Torch/Sym × fresh/packing/recycling/adaptive × three rounds | 48 |

E4's eight main arms are ordinary resident Torch, packed resident Torch,
ordinary partial Torch/Sym, ordinary full-offload Torch/Sym, and adaptive
partial Torch/Sym. Partial placement allows an eight-layer resident prefix.
Fixed packing/recycling attribution targets the original failed cell and a
smaller-operation control; it is not a full extra Cartesian grid. E5's
fixed packing/recycling attribution is limited to one caller. These limits
are part of the matrix, not postmeasurement exclusions.

## Common controls and ownership

One RTX2080Ti (GPU0), affinity4-7,20-23 and eight aggregate CPU threads are
fixed. E5 partitions those CPUs across persistent callers, uses one gather
thread per native owner and8/callers Torch threads. The original CUDA
reservation leaves at most512/1024MiB free after context materialization;
it is outside Torch's allocator. Achieved free capacity and NVML deltas must
agree within16MiB. The2ms NVML trace records sampled lower bounds, not exact
transient peaks. A process-tree audit checks foreign GPU/build/packaging work
every2seconds. Run no concurrent GPU tests, compilers or archive creation.

Fresh arms use the candidate runtime with ordinary independent output
allocations and existing bounded native scratch reuse. Packing adds one
allocation per group without storage recycling or an extra retirement fence.
Recycling adds strong storage/Tensor ownership tracking and a conservative device completion
barrier before address reuse. Adaptive uses the same recycling path and the
pure planner's selected capacities. CUDA errors are never retried. Native
conversion kernels, compiler binaries and the Torch control's fused
dequantization kernel remain unchanged; the Torch storage bridge is rebuilt.

Torch and Sym receive the same packed outputs, prefetch depth, resident prefix
and single-member staging opportunity. Torch uses ordinary copies and the
original kernel-only FP16 dequantizer into supplied views. Its completion
retains pinned sources until the event completes. GPU staging is released after
each kernel enqueue for safe reuse on the same producer stream, preserving the
fresh control's staging lifetime. Single-member staging
waits before releasing a member's temporary input. Sym uses its normal
validated submission/completion path. Full-group validation precedes any
Sym subgroup launch.

Packed resident preload gives both runtimes one contiguous resident allocation
and serial per-matrix loading. Preparation may clear Torch's cache after each
matrix, matching the original supplemental packed resident control. Serving
never clears the cache. Resident storage is separate from the bounded
nonresident output window. Preload time and peak remain visible; this is not
a free warm-path optimization.

Weight-specific arenas share the native owner with KV transfers. KV output
storage remains ordinary Torch storage and belongs in the workspace allowance.
There is no additional native owner hidden outside the caller's scratch share.
Returned request outputs and weight aliases remain valid after owner closure.
Native, Torch, arena, pinned-host, retained-output and global sampled counters
have overlapping or disjoint scopes and are not summed into a synthetic peak.

## Planner inputs frozen before acceptance

E4 uses24MiB output per layer group,16MiB qualified whole-group device scratch
and8MiB single-member scratch. Its workspace allowance is
`128MiB + layers*context*4096 + context*1024*16 + 32*context**2`, with32MiB
safety. This is a conservative declared allowance for fixture/compute/KV and
allocator/library state, not an exact liveness theorem. The physical free
budget is the plan's total capacity. At L32/context512/512MiB, the selected
plan keeps eight resident layers and chooses one slot/context. Fixed arms
keep two slots/contexts and64MiB host+device scratch quota per context.

E5 allocates256MiB of its1024MiB physical budget to transfer storage and
device scratch, divided equally across callers. The other768MiB is reserved
for shared fixture, KV/compute, held request outputs and library/allocator
state. Each transfer share includes8MiB safety. Decode uses24/16/8MiB
output/whole/single-member scratch estimates; MoE uses8/8/6MiB. Four-caller
decode therefore selects one slot/context and single-member staging. Fixed
arms retain the original128MiB aggregate native host+device scratch quota,
divided across callers and two contexts each. These are capacity policies,
not claims that the selected policy minimizes latency.

The group adaptive arm supplies capacity for two output groups plus one
whole-group scratch estimate and8MiB safety, selecting one slot. Small groups
contain four8×16INT8 matrices (1KiB FP16 total); large groups contain the
same12MiB INT8/24MiB FP16 payload as an E4 layer. Fixed group arms use two
slots. All preparation and first-call costs are recorded separately.

Plan predictions and actual outcomes are separate fields. The plan's device
scratch estimate is distinct from the native host+device quota. The plan
does not cover transient resident preload staging or prove freedom from OOM.
It accounts for an extra consumer group (`slots+1`) and explicit external
retention. The runtime refuses capacity exhausted by retained aliases before
launch; speculative prefetch may defer only while pending work can progress.

## Timing, validation and retirement

E4 retains the original first request, three warm requests and20completed
timed requests per process, each checked against the original CPU-copied
oracle with original tolerances. It records preparation/preload/first/warm
phases and actual policy-stage OOMs. Oracle and reservation failures are
invalid. The deployment physical window excludes the oracle phase.

E5 retains eight predeclared inputs, three warm passes and30eight-request
bursts. Per-caller construction, first request, all warmup requests and retirement
have separate timings and sampler-relative phase boundaries. Retired arena
statistics are recorded for both runtimes. Reports include completed burst time per request, service/sojourn
p95, queue delay and sampled warm physical peak. The previous burst remains
held and is checked bit-for-bit after the next burst; eight final outputs
are checked again after all workers and snapshots close. These are finite
bursts with validation gaps, not a sustained-server throughput measurement.

Group attribution times100completed iterator deliveries after three warm
groups. CPU bit checks occur outside each delivery timer, and the previous
group remains held across the next delivery. A final held group is checked
after retirement. The whole warm-loop physical trace includes validation
gaps. Small-operation losses and device-barrier costs remain in the report.

## Provenance and analysis

The runner pins the complete job list, measurement sources, transitive
repository workload sources, original artifact inputs/control binary, and
candidate Python/native/compiler hashes before launch and checks them again
at completion. The historical absolute repository path is replaced in memory
by a single AST assignment; numerical functions are unchanged. Runtime imports
must resolve inside the frozen candidate snapshot. Pilots are separately named
and cannot pass acceptance analysis. Existing pilot records remain available
with their exclusion reasons.

The standard-library analyzer requires complete, unique matrix membership,
matching provenance, the qualified hardware/runtime/thread configuration,
executed workload/policy settings, valid physical reservations, uncontaminated
jobs, finite timings, complete request identities, lifecycle timings and
retained-output checks. It verifies every trace checksum, ordered finite samples
and actual observations inside the required phase windows, and reconstructs
the recorded phase peaks. Missing observations never become zero memory use.
It retains negative outcomes in grouped tables; a timing
comparison with fewer than three successful rounds must say so. All raw
records/traces and scoped counters are published through the Wiki artifact.
