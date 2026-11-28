# Paper-facing relocation claim ledger

Use the narrower code-generation thesis. For supported blocking CPU-to-CUDA
relocations, captured Torch operations lower to a typed plan that composes
layout changes while retaining ordered numerical stages. Checked plans select
bounded native kernels and can reuse preparation across compatible calls.
The evidence establishes specific materialization, kernel and reuse effects
under different controls. It also measures the costs of the automatic path.
The systems thesis remains a research hypothesis. [Issue #273](https://github.com/JueonPark/sym/issues/273) · [Historical claim ledger](claim-ledger.md)

An **implemented contract** describes a supported behavior and its executable
checks. A **measured effect** states the comparison and the observation
boundary. A **proposed hypothesis** states what an additional evaluation would
need to establish. Use the stable IDs below in contribution and prior-art
notes. Each campaign keeps its own hardware, baseline, timing and budget.
Original E1–E5 use source `2617ba166f98a7476a5f472d829a573317fc0cfb` and one
RTX 2080 Ti. Later RTX 4070 Ti SUPER measurements form separate campaigns.
[Original protocols, programs and raw measurements](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

| Paper location | Claim IDs and evidence to carry |
| --- | --- |
| Abstract and introduction | [C-ORDER](#c-order), [E-FOLD](#e-fold), [E-KERNEL](#e-kernel), [C-AUTO](#c-auto) and [E-REUSE](#e-reuse). The captured-plan timings include a completed transfer. The frontend uses its own staging and submission. |
| Relocation and implementation | [C-ORDER](#c-order) for expressible semantics, [C-AUTO](#c-auto) for the supported handoff, [E-REUSE](#e-reuse) for cache and group effects, [C-REGION](#c-region) for the experimental extension. |
| Related work | [Comparison boundaries](#comparison-boundaries) and the matching claim IDs. Use primary descriptions of the closest mechanisms. |
| Evaluation | [E-FOLD](#e-fold) through [E-PLANNER](#e-planner). Retain losing rows, fallback, OOM phases and cost boundaries. |
| Contributions and future work | Center the narrower contribution on [C-ORDER](#c-order), [E-FOLD](#e-fold), [E-KERNEL](#e-kernel) and [C-AUTO](#c-auto). [H-SYSTEMS](#h-systems) records the evidence needed for the broader thesis. |

### C-ORDER

**Implemented contract.** Within admitted plans, the compiler composes layout
maps while retaining ordered typed stages, policy-specific rounding, pad entry
stage and per-channel coordinates. A bound artifact fixes the symbols,
parameter revisions and representation at each legal cut. Runtime dispatch
checks actual views and native capability before executing a row. The
conformance harness checks a separate oracle across plan geometry, stage
boundaries, exceptional values, parameter coordinates and eligible CPU/CUDA
rows. Its cast comparisons use exact non-NaN bits and NaN classification.
Claims involving observable NaN payload bits require a stronger contract.
[Typed semantics](reloc-typed-semantics.md) · [Folding](reloc-typed-folding.md) · [Dispatch](runtime-dispatch.md) · [Conformance corpus](../libreloc/python/tests/test_typed_conformance.py) · [GPU conformance](../libreloc/python/tests/torch_frontend/test_typed_conformance_gpu.py)

### E-FOLD

**Measured effect.** Original E1's folded plan has a lower completed H2D
median in 35/36 configurations versus Sym materializing every source operator
into separate writable host storage. Both native arms use one native invocation
per request. All controls share one pinned final wire, one H2D copy, a
preallocated destination, synchronization and a one- or eight-worker budget.
Six chain families cover canceling layouts, rank-three permutations,
dequantization and staged padding. Five independent process rounds use repeated
buffers and check every timed result. Producer-only time is a separate
diagnostic and preparation is outside these warm timings.

The original manual loop has the lower median in 31/36 cells. The original
folded path leads the fastest manual/eager/Inductor alternative in 4/36, and
one rank-three cell is 8.30× slower than the fastest alternative. The folding
contrast evaluates materialization removal together with execution of the
folded plan. The separate `--no-canonicalize` control retains folding. None of
the 36 paired intervals for canonicalization alone lies wholly in favor of
canonicalization. These are explicit-recipe experiments on the 2080 Ti and
report producer-plus-copy with analytical byte counts.
[E1 protocol and results, archive directory `01_folding`](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

### E-KERNEL

**Measured effect.** #267 holds the folded plan, compiler and H2D path fixed
while changing the native implementation. The checked descriptor selects
prebuilt ordered register casts and dequantization, with adjacent short-row
transpose batching and bounded fallbacks. The current runtime improves
36/36 E1 completed medians over the prior folded runtime, leads manual fusion
in 30/36 and the fastest available manual/eager/Inductor alternative in
28/36. It improves 14/16 held-out completed medians and 3/4 completed consumer
comparisons. The eight-worker dequantizing consumer regresses 7.5%. The
original manual roundtrip arm uses an allowed idempotence simplification, so
its lead can include fewer requested conversions.

#267 reports plan-compile, bind and prepare distributions, first-call costs,
production ELF `.text` growth of 3,328 bytes and zero per-shape generated native
executable bytes. Its cold observations are shared-process first calls, and
native/manual builds sit outside request totals. The bounded AOT code uses
runtime-gated ISA variants. [#267 protocol](../bench/issue267/PROTOCOL.md) · [Complete results](../bench/issue267/RESULTS.md)

#272 links that quality comparison to actual Dynamo capture. Six fresh
processes extract the same 620-byte typed plan for an untuned rank-four
permutation and F32→F16→F32 chain. The isolated arms share one 700,640-byte
pinned wire within 1 MiB, one fresh destination, one H2D copy, and one worker.
Warm medians in milliseconds are 1.044 prior folded, 0.156 current folded,
0.541 materialized, 0.217 manual, 0.204 eager and 0.273 Inductor. A separate
probe exercises the register stages over all 175,160 elements. The real
automatic endpoint uses its own staging and submission and takes 0.467 ms
warm. The current captured-artifact arm's accounted 36-call total, including
artifact acquisition, is 1,911.781 ms versus 1,557.809 ms for eager. Keep the
single-chain warm gain and lifecycle result together.
[#272 captured-quality report and raw records](../bench/issue272/CAPTURED_QUALITY.md)

### C-AUTO

**Implemented contract.** Ordinary Torch functions passed to `torch.compile`
with the opt-in backend yield a captured graph, generated typed plan, guarded
specialization, selected legal native row and live binding. #272 saves this
provenance through the completed dispatch on two untuned chains. It checks
rounding and per-channel dequantization, fresh and held output metadata and
bits, shape revisits, live source values, in-place scale/zero-point revisions
and cached rebinding. A nonzero zero-point revision excludes the fused device
dequant row and takes a legal split row. Below-threshold and first-use calls
use the saved graph without native submission. CPU probes observe the
register-stage functions and the covered element counts, and the default device
policy exercises prebuilt device rows.

The caller specifies the backend and ordinary Torch operations. Compiler
generation covers the layout/value plan, with native row selection and
guarded preparation using prebuilt kernels. Handwritten `FastTransfers`
recipes, the pooled DLRM producer, opaque custom operators and explicit replay
wrappers have separate provenance. This contract covers supported blocking
inference with existing shape, device, alias and capability guards. Usability
evidence is the demonstrated API requirement for these callers. Quantified
engineering effort, maintenance cost and workload-wide automatic coverage
remain open. [Automatic execution contract](automatic-prepared-execution.md) · [#265 results](../bench/issue265/RESULTS.md) · [#272 execution witness and raw checks](../bench/issue272/WITNESS.md) · [Explicit transfer API](fast-transfers.md)

### E-REUSE

**Measured effect.** Original E3 replays the fixed 12-request sequence five
times, with revisits, empty/singleton/strided inputs and live values. No
additional Dynamo callback occurs after the first cycle. The original
automatic eager-compute path takes 2.06–8.29× eager Torch's repeated time, and
the automatic Inductor-compute path takes 1.90–7.42× against that same eager
baseline. The four streams record 45, 45, 50 and 55 automatic executions
out of 60 requests for layout cast, KV restore, live dequantization and
live indices, respectively. Capture exclusions count regions.
The original explicit layout-cast cache reduces preparations from 50 to six
and repeated time by a paired median of 37.7%. The 22.4% KV median includes
one losing round. Changing dequantization parameters yields zero explicit
cache hits. The two explicit index arms share their native descriptor cache.
The archived arms share a process and compiler cache, so their first cycle
cannot rank independent startup costs.
[E3 protocol and results, archive directory `03_dynamic_shapes`](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)

The #265 prepared frontend is opt-in. With its selected policy, repeated time
is 1.52–7.04× matched Torch, and its live-dequantization improvement uses the
saved-region fallback. #272 replays the original sequence with immediate
admission and a fixed CPU typed row on a 4070 Ti SUPER. Matched per-request
payload, modes, fallbacks and callback counts isolate cache reuse with groups
off, group reuse with the cache on, and the native build with both on. It
records cold first request, all 60 calls plus lifecycle costs, and the last
48 completed calls. Every one of 2,880 outputs passes the original oracle.
Caching saves 40 of 45 layout-cast preparations and reduces its repeated
time by 8.74% mean paired. KV improves 4.17%. Live dequantization has zero
hits, caps at 32 entries and 33,024 copied key bytes, and takes 2.45% longer
with prepared groups. All 55 live-index invocations rebuild their concrete
program. The indexed and layout-only controls have no claimed register
kernel effect. These results apply to three process pairs per factor and the
stated admission/resource settings. [#265 original-policy results](../bench/issue265/RESULTS.md) · [#272 protocol](../bench/issue272/PROTOCOL.md) · [#272 E3 results and raw records](../bench/issue272/RESULTS.md)

### C-REGION

**Implemented contract, research prototype.** #270 names the producer,
transfer, consumer, ownership, aliases, observable outputs and completion
edges around a typed plan. Its executable FX rule operates on a closed
region with a private dense host clone, typed relocation and leading-axis
concatenation. Forwarding the host source and writing into the fresh result's
suffix preserves coordinates, checked cuts and externally observable
lifetimes. An escaping intermediate retains its allocation. Mutations, mixed
readers, incompatible layouts or dtype promotion can retain the original
graph. External writes to in-flight host storage are outside the region
contract. Storage reuse requires ordering every prior consumer and releasing
external leases, with separate delayed-reader and held-output arena tests.

The same-engine pass-off/pass-on comparator keeps inputs, binding, native
runtime, row and payload equal. Exact output and pointer checks pass on
44 CUDA row/binding pairs and 19 host binding pairs from 16 artifacts.
For the first cast-transpose binding, two temporary tensors shrink from
144 bytes to zero while the wire stays 48 bytes. The measured quantities are
values, pointers and logical temporary bytes. Completed latency and physical
peaks were outside this qualification. [#270 region model, limits and ablation](relocation-regions.md) · [Region runner](../bench/issue270/ablate.py)

### E-REQUEST

**Measured effect.** Original E2's synthetic FP16 decode, prefill and MoE
compare explicit prepared Sym against Torch given the same native
dequantization/permutation kernels, grouping and two-slot prefetch. Inductor
arms share compiled consumer functions. Both meet a nominal 128 MiB
producer scratch/staging envelope, with separate capacity testing for actual
whole-device limits. The completed-request timer includes model compute,
transfer, routing or token selection, and synchronization after warmup.
Independent checks use exact tokens/expert choices and model-specific tolerance,
with separate bitwise kernel witnesses. Sym is 4.6–10.9% slower in all eight
principal matched comparisons. Native conversions, batching and a longer
prefetch window also benefit the matched controls, and their effects can
interact. The historical FP16 campaign lacked a kernel-only Torch arm and
used different grouping and prefetch opportunities.

#264's prepared group/native reporting changes improve completed median
latency 2.24–10.28% versus frozen Sym. Four of eight cells beat matched Torch;
the other four are 1.16–3.79% slower. #268 replay uses explicit pure-GPU
regions. Sym replay beats identical stable slots with replay disabled in
45/45 successful per-round comparisons, and beats plain Sym in 4/48.
It qualifies zero model variants at the predeclared 5% margin against both
Torch controls. #269's private KV-layout rewrite removes eager materialization
work, but matches Inductor's observed kernel/allocation counts and qualifies
0/24 against both Torch controls. These are explicit runtime and scoped
consumer results. [E2, archive directory `02_fp16_attribution`](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04) · [#263 overhead attribution](../bench/issue263/RESULTS.md) · [#264 prepared groups](../bench/issue264/RESULTS.md) · [#268 replay](../bench/issue268/RESULTS.md) · [#269 consumer lowering](../bench/issue269/RESULTS.md)

In #274, 18 fresh processes tested pretrained FP32 SqueezeNet 1.1 from
normalized CPU pixels to completed GPU logits. The default opt-in preparation
reused one group across changing image crops. Warm p95 was 4.154 ms with
Sym + Inductor and 5.022 ms with Inductor. The six paired ratios had a geometric
mean of 0.795 (95% bootstrap interval 0.634–0.904). Eager had the lowest
accounted 112-call lifecycle, 1.907 s versus 7.893 s with Sym + Inductor.
This endpoint includes the effect of the rewrite on the compiled consumer.
The result covers one fixed-shape application on the RTX 4070 Ti SUPER.
[#274 application and historical controls](../bench/issue274/README.md)

### E-CAPACITY

**Measured effect.** Original E4 sets physical free VRAM to 512 or 1024 MiB
after CUDA initialization using a direct reservation. The 120-process fixed
grid records 105 successes and 15 policy OOMs. Both complete-offload arms
succeed throughout. Matched offload arms share the native conversion kernel,
eager consumer, two-slot iterator and eligible resident-prefix overlap. In
the 32-layer, context-512, eight-resident-layer cell
at 512 MiB, Torch succeeds 3/3 and Sym fails during the first request 3/3.
Among successful matched process pairs, Sym's observed deployment peak is
12 or 32 MiB higher. At 1024 MiB, the resident control has the lowest warm
latency. A separate packed resident preload fits a 16-layer short-context
cell after the original two-slot preload fails. It still fails on the first
long-context request. Charge preload to setup and compare phases separately.
The sampled physical peak is a lower bound, and pinned-host usage and
allocator counters have distinct scopes.

#266's opt-in combined packing and resident preload recover the failing
partial cell 3/3 at an observed 484 MiB and a 39.54 ms warm median. Its
adaptive policy reaches 444 MiB at 48.27 ms, slower than fixed packing
and recycling. Matched adaptive Torch reaches 444 MiB at 46.76 ms.
These are capacity/latency tradeoffs for tested implementations under a
reservation, which applies capacity pressure without emulating a co-tenant's
compute or bandwidth use. [E4, archive directory `04_memory_budget`](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04) · [#266 paired results](../bench/issue266/RESULTS.md)

### E-LIFECYCLE

**Measured effect.** Original E5 matches native kernels, grouping and a
two-slot window in synthetic decode and sparse MoE. Its 120-call lifecycle
includes startup, snapshots and first execution, a shape change, a scale
revision, then owner retirement and held-output release. Sym is slower in
12/12 paired full-lifecycle rounds, with paired medians 0.71–3.51% higher.
Shape changes also change model work, and Inductor compilation remains
inside its first request. Memory ownership is checked on previously returned
outputs across revisions and close, and owned native counters drain at
retirement. This is process-accounted lifecycle cost, with- and without-setup
numbers retained separately from warm timing.

The concurrent comparison uses 1/2/4 callers, fixed eight-thread and
128 MiB aggregate scratch/staging budgets, 1024 MiB nominal free VRAM and
30 finite eight-request bursts per process. The matched burst arms retain
the same grouped KV and whole-request prefetch opportunities. This physical
reservation applies to the concurrent campaign. Sym has lower throughput and
higher enqueue-to-completion p95 in all 18 pairs. Paired median throughput
losses are 3.33–10.84%. Observed warm peaks rise by 14–222 MiB versus matched
Torch. #266 adaptive Sym increases p95 in all six tested workload/caller
configurations even where memory falls. Keep finite-burst results separate
from sustained serving SLOs. [E5, archive directory `05_lifecycle_concurrency`](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04) · [#266 resource and p95 tradeoffs](../bench/issue266/RESULTS.md)

### E-PLANNER

**Measured effect, research prototype.** #271 enumerates representation,
placement, residency, chunk/prefetch and scratch reuse choices from supplied
region facts and calibration. Its bounded simulator tracks CPU, H2D and GPU
dependencies and aligned storage lifetimes. An executable physical CUDA slab
checks the same storage contract, with CUDA context, libraries and pinned
host memory outside that slab. The fused arm uses explicitly implemented
Triton code, and the executable has one submitter and one transfer and compute
stream. It compares fixed, per-transfer, placement-only, schedule-only,
buffer-only and joint policies through the same executor and an exhaustive
measured oracle. All 358 admitted combinations out of 585 produce the
expected bits and reach the predicted live peak. The other 227 hit the
arena boundary.

Across nine workload/budget cells, median warm regret is 13.23% joint,
24.81% placement-only and 103.28% per-transfer. Five-call lifecycle regret
is 20.66% joint versus 19.63% placement-only. Worst joint warm regret is
313.21%, and joint selection meets the finite-trace 3 ms p95 deadline in only
one of four cells. Calibration takes 899 ms on the test host and deserves its
own charge, as do planning and lifecycle preparation. Application replay uses
recorded E2/E4/E5 endpoints rather than an executable new schedule. It picks
16/24 observed winners and incurs a 28.11% worst loss. Removing the 32 MiB
margin selects a recorded policy that OOMed in all three rounds.
[#271 executable catalog and protocol](../bench/issue271/README.md) · [All objective results](../bench/issue271/RESULTS.md) · [Application replay and sensitivity](../bench/issue271/HISTORICAL.md)

## Comparison boundaries

The direct baselines below reproduce the operation or policy contrast. The
closest systems place this work in context. A fair direct comparison with
those systems requires a common executable workload, precision and numerical
contract, input and output residency, and timing/budget boundary.

| Claim IDs | Closest precedent and mechanism | Reproducible comparison and present constraint |
| --- | --- | --- |
| C-ORDER, E-FOLD, E-KERNEL | TEMPI canonicalizes nested MPI datatypes and chooses a movement path. IREE propagates layout encodings and CuTe composes coordinate maps. Ladder transforms low-precision representations with hardware-aware scheduling. [TEMPI](https://carlpearson.net/pdf/20210624_pearson_hpdc.pdf) · [IREE tiling and encodings](https://iree.dev/blog/2025-08-25-data-tiling-walkthrough/) · [CuTe layout algebra](https://docs.nvidia.com/cutlass/4.5.2/media/docs/cpp/cute/02_layout_algebra.html) · [Ladder](https://www.usenix.org/conference/osdi24/presentation/wang-lei) | E1 and #272 hold final transfers and numerical stages fixed with native materialized, prior/current folded, manual and Torch arms. A cross-system test needs equivalent ordered stages and transfer completion. MPI pack/send, GPU operator layout and accelerator-oriented low-precision compute use different interfaces and endpoints in the cited results. |
| C-AUTO, E-REUSE, E-REQUEST | GraCE automates GPU graph coverage, parameter handling and selective replay in PyTorch. It is the OSDI 2026 version of the PyGraph line of work. [GraCE](https://www.usenix.org/conference/osdi26/presentation/ghosh) · [PyGraph preprint v3](https://arxiv.org/abs/2503.19779v3) | #272 uses matched capture, prepared reuse and native rows on the same H2D trace. #268 compares plain and stable slots with replay on/off. A head-to-head replay test needs the same capturable GPU compute region and stable-slot lifetimes. The blocking H2D admission trace and explicitly annotated replay wrappers cover different operations. |
| C-REGION, E-PLANNER | IREE Stream represents asynchronous resource transfers and timepoints. MAGIS and SwapAdvisor jointly reason about graph execution and memory or swap choices. Welder schedules tile-level traffic and reuse. [IREE Stream](https://iree.dev/reference/mlir-dialects/Stream/) · [MAGIS artifact](https://github.com/pku-liang/MAGIS) · [SwapAdvisor paper](https://par.nsf.gov/servlets/purl/10191573) · [SwapAdvisor proceedings](https://asplos-conference.org/wp-content/uploads/2020/abstracts/paper_13_3.html) · [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) | #270 holds the engine fixed and toggles only region forwarding. #271 runs all candidates in one executor and scores its measured oracle. The published experiments use other workloads and execution contracts. A port would need equivalent cuts, aliases, ownership and output readiness. |
| E-REQUEST, E-CAPACITY, E-LIFECYCLE | FFCV compiles and pipelines data loading. FlexGen chooses storage and transfer schedules for throughput-oriented LLM inference under constrained memory. [FFCV](https://openaccess.thecvf.com/content/CVPR2023/html/Leclerc_FFCV_Accelerating_Training_by_Removing_Data_Bottlenecks_CVPR_2023_paper.html) · [FlexGen](https://proceedings.mlr.press/v202/sheng23a.html) | The matched runtime arms in E2/E4/E5 share synthetic model work, native kernels and eligible scheduling opportunities. E4 and the concurrent E5 trial impose physical reservations. The cited loading and large-batch offload benchmarks use different workloads, memory tiers and objectives. A deployment comparison would need the same model, tokens, compute precision, arrival trace, residency/preload and physical device budget. |

For C-REGION, VTC is a close comparator for eliminating intermediate GPU
storage. Its compiler represents movement between compute operators with
index mappings and selects virtual tensors. A shared case could test whether
the same device consumer avoids the same materialization. Extending this to
the full H2D region requires a host operand, a real transfer, and equivalent
observable outputs and lifetimes. The existing #270 comparison isolates
the rule within Sym. [VTC, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/hu-muyan)

Layout composition, cost-guided placement, SIMD/pooling, compact INT8 wires,
prefetch and replay have established precedents. The narrower research
question is where checked composition of layout, ordered numerical stages
and boundary representation enables automatic specialization with measured
quality and cost. Novelty requires positive evidence about that contract
and the transformations it enables. Gaps in a feature matrix are prompts
for comparison.

### H-SYSTEMS

**Proposed hypothesis.** A deployment compiler could infer safe
producer–transfer–consumer regions, automatically lower an executable
candidate set and choose among them under physical memory and request
objectives. #270 establishes a bounded legality rule and #271 evaluates
choices in a supplied catalog. The planner's regret and deadline failures,
the automatic endpoint gap, and the matched model regressions bound the
claim today.

A full paper on the narrower thesis needs a reproducible direct comparison
with the closest supported compiler mechanisms, prospective coverage on
more untuned chains and parameter revisions, and portability or an explicit
architecture scope. The evaluation must retain register/fallback evidence,
compiler/setup/binding/code-size costs, and completed negative results.
The #274 application adds one pretrained FP32 model and one guarded input
transformation on Ada. Other architectures for that workload, BF16 models,
whole-program discovery and productivity gains remain open.

A broader systems paper additionally needs compiler-discovered realistic
regions and workload-wide coverage, executable matched alternatives under
actual physical and ownership budgets, positive and negative legality
evidence across streams, and calibrated makespan and tail predictions
tested under held-out traffic. Report calibration, planning and lifecycle
amortization, refusals, OOMs and worst-case losses. To claim deployment
benefit, compare sustained arrivals and SLOs against strong offload and
resident controls at the same budgets. The original E5 bursts and #271's
finite-trace trial establish their own observed outcomes.
