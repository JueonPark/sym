# Issue 265 qualification protocol

Baseline: merged main `ce0bbe3b2451737ec35765bb1051841f5d19fc3d`. Candidate:
compiler-derived automatic prepared execution. Both use identical native/compiler
binaries from the reviewed #264 build; only candidate frontend Python differs.
All automatic arms use real capture/import/compiler lowering. The explicit fast
controls alone use the frozen E3 handwritten recipes.

## Freeze and selection

Before any acceptance run, `workloads.py` and `support.py` fix disjoint selection
and held-out cases and shapes. E3 is also excluded from policy selection.
Selection evaluates four generic policies: always device-preferred, CPU typed rows (layout-only groups still use device layout),
4 KiB plus one previous occurrence, and 64 KiB plus one previous occurrence.
No workload-name or particular-shape branch enters production. Two independent
process rounds are exploratory selection evidence; acceptance uses three rounds.
Choose a single policy using the geometric mean repeated-sequence cost across
all four selection cases, retaining every result. Keep the feature opt-in unless
both the independent qualification and deployment costs justify a default change.
Record the selection decision before E3/held-out acceptance.

E3 preserves the original source functions, input generator, exact integer-view
oracle, live parameters/indices and 12-request sequence repeated five times:
`128,256,128,33,512,33,1,128,0,65,256,65`. The last request in each cycle changes
source stride in layout/cast cases. Eight arms cover eager Torch, Inductor,
legacy automatic eager/Inductor, candidate automatic eager/Inductor, explicit
fast reprepare and explicit fast cache. Held-out cases change rank, permutation,
cast order, parameter axis and indexed rank; they compare Torch, legacy automatic,
the frozen selected policy and always-prepared execution. Their singleton/empty/
stride cases and small tensors remain in the aggregate.

## Process isolation and measurement

Each arm/case/round runs in a separate process and fresh Inductor cache. This
provides an independent cold measurement, unlike the original E3 shared-process
campaign. Do not directly compare its first cycle with the old shared-process
first cycle. Arm order rotates by round. All builds, GPU work and packaging are
serialized. A process-tree-aware audit samples foreign builds/packaging and GPU
processes every two seconds; any observation invalidates the run. CPU affinity
is `4-7,20-23`, Torch/OpenMP/MKL threads 8, interop threads 1, GPU 0. CUDA graphs
and TF32 are disabled; Inductor emulates precision casts. Runtime, workload,
harness and environment hashes/settings are frozen per campaign.

Record launch/import/configuration, explicit CUDA initialization, owner setup,
compiler callbacks, automatic preparation/snapshots, completed calls and owner
retirement separately. The first oracle runs after the first measured request.
Each measured call completes with CUDA synchronization. Input generation and
oracle/held-output checks are timed separately and excluded from call latency.
Report first request, first cycle, all 60 requests and the final 48 requests;
compilation and preparation nested within calls must not be added twice.
`cold_first_ms` adds import/configuration, CUDA initialization and owner setup
to the first call. `accounted_full_ms` adds those phases and retirement to the
sum of completed calls. Process launch/import includes Python and Torch setup;
filesystem/OS and CUDA driver JIT caches are not flushed. Parent polling wall time is not latency.
All arms import the same frozen E3 module, which imports Sym even for Torch
controls. Thus cold launch/RSS includes common harness and unused Sym import
overhead in those controls; it is not a minimal Torch-only application footprint.

Phase memory snapshots report current physical device use, Torch allocated/
reserved peaks and process RSS. Physical boundary observations are lower bounds
on transient peaks. Process RSS includes compiler and verification overhead;
Torch's allocator counters exclude native allocations, which are also reported
by the resource owner. There is no memory polling thread or profiler in primary
timings. One output plus its oracle copy is retained throughout the stream and
checked after retirement. Caller outputs and allocator caches are then released
for a separate final memory snapshot. Native retired/live bytes must reach zero.

## Analysis and claims

The analysis unit is an independent process round. Pair matching rounds and
report mean percentage reductions with a deterministic process bootstrap (seed
265, 10,000 resamples). Three processes give weak uncertainty estimates; show
individual paired effects and every negative result. Requests within a process
are not independent samples. Report compilation avoidance separately from
latency and from actual prepared/native execution.

Backend runtime calls, original-region fallback events and capture-exclusion
region counts have different denominators. Preserve each map and the request
count rather than adding these into a fabricated coverage percentage. Report
prepared admission, revisions, hits/misses, evictions, snapshot costs, current
index binds, group constructions/rebinds and actual native rows. A speedup due
to fallback is not a prepared kernel speedup. All missing, duplicate, incomplete,
contaminated or changed-runtime records must fail CPU-only validation.
