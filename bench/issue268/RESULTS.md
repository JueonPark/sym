# Issue 268: bounded GPU replay qualification

**Replay reduces launch overhead, but this integration does not establish a
qualified Sym model win.** Sym replay improves over the identical slots-off
control in all 45 successful per-round comparisons. It improves over unwrapped
Sym in only 4 of 48 per-round comparisons, and only one of 16 cross-round model
medians. None of the eight replay variants, or any of the 24 tested E2 variants,
passes the parent #247 requirement of at least 5% below both Torch controls in
every round. Keep this API opt-in.

The implementation provides explicit pure GPU regions with refreshed parameters,
private graph pools, bounded slots, guarded shapes/settings, copied slot outputs,
stream-safe fallback and retirement. CPU transforms, transfers, host routing and
mutating MoE combine remain outside capture. It does not implement automatic
arbitrary FX partitioning. See the [API contract](../../docs/gpu-replay.md),
[protocol](PROTOCOL.md), [complete tables](TABLES.md), and [review disposition](REVIEW.md).

## Evidence and scope

The frozen measurement/runtime revision is
`1f0578c8eb5089eff807f96907a32dc1cc5fdaf1`, on merged #267 native runtime
`b319a0ab94dbc8c8264e4fa093ffbbb796e2c046`. Native library SHA-256:
`28de8f98dd0f3f6d2f842de11e039c848641e5f2d375963fe37ebdd175c2cc58`.

- 288 fresh-process main arms over three rounds: 282 successes and six actual
  OOM outcomes; no contaminated attempts. The complete frozen matrix is present.
- 41,760 timed main output checks and the same number of retained-output checks.
  Six supplemental processes add 5,400 numerical and 5,400 retention checks.
  Main retention checks include 1,248 outputs checked after owner close.
- E2: four original model cases × eager/Inductor × Torch/Sym × plain/off/on.
  E5: decode/sparse MoE with 1/2/4 eager callers and one Inductor caller.
- Each cell follows 60 observed quiet seconds; process-tree audits cover every
  arm. Supplemental processes have independent quiet gates and audits. Precision,
  kernels, grouping, two-slot prefetch and resource assignments remain matched.
- RTX 2080 Ti, AMD EPYC 7351; CPUs 4–7,20–23 (four physical cores/eight SMT
  threads), eight aggregate compute threads; Python 3.14.7, Torch 2.14.0+cu126.

`plain` is unwrapped execution; `off` uses the same static inputs, copied outputs
and callable with ordinary kernels; `on` enables replay with guarded fallback.
Tables below use medians of three round-level medians. The target test uses
unrounded paired per-round values, never these displayed aggregate values.

## E2 completed requests

A ratio above 1 favors Sym replay. The fastest Torch column selects the lowest
observed cross-round median across both compilers and all three modes; it is a
deployment comparison distinct from same-mode attribution.

| Case | Sym compute | Sym plain ms | Sym off ms | Sym on ms | On p95 ms | On/off speedup | On/plain speedup | Fastest Torch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_decode | eager | 19.099 | 26.904 | 19.124 | 19.373 | 1.407× | 0.999× | 17.686 |
| llm_decode | inductor | 18.289 | 26.036 | 19.171 | 19.441 | 1.358× | 0.954× | 17.686 |
| llm_prefill | eager | 5.196 | 6.901 | 5.085 | 5.221 | 1.357× | 1.022× | 4.948 |
| llm_prefill | inductor | 5.013 | 6.448 | 5.161 | 5.286 | 1.249× | 0.971× | 4.948 |
| moe_dense | eager | 9.785 | 14.415 | 12.860 | 14.217 | 1.121× | 0.761× | 10.659 |
| moe_dense | inductor | 13.132 | 18.086 | 15.274 | 15.911 | 1.184× | 0.860× | 10.659 |
| moe_sparse | eager | 2.904 | 4.560 | 3.695 | 3.803 | 1.234× | 0.786× | 2.775 |
| moe_sparse | inductor | 3.856 | 5.670 | 4.143 | 4.238 | 1.368× | 0.931× | 2.775 |

Eager prefill is the sole positive Sym cross-round median against its own
unwrapped path: 5.196 → 5.085 ms (1.022×). Its direction changes across rounds,
and the fastest measured Torch median is 4.948 ms. Sym eager decode is essentially
unchanged against plain (19.099 → 19.124 ms); graph-enabled Torch reaches
17.686 ms. Sym dense MoE replay regresses against plain in both compiler modes.

Even an apparent unwrapped dense-MoE advantage must retain the paired-round
qualification: eager Sym's aggregate median is 9.785 ms versus Torch's 10.659 ms,
but only one of three rounds meets the 5% target. No robust model win is claimed.

## E5 finite bursts

The latency below is completed eight-request burst time divided by eight, not
individual request service latency or sustained server throughput. The full
tables separately retain worker submission, queue, service and sojourn p95s.

| Case | Compute | Callers | Sym plain ms | Sym off ms | Sym on ms | On p95 ms | On/plain speedup | Torch on ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_decode | eager | 1 | 19.295 | 26.361 | 20.682 | 21.216 | 0.933× | 19.477 |
| llm_decode | eager | 2 | 17.691 | 23.715 | 22.217 | 23.348 | 0.796× | 22.150 |
| llm_decode | eager | 4 | 19.637 | OOM (3/3) | 27.053 | 28.121 | 0.726× | 27.993 |
| llm_decode | inductor | 1 | 18.475 | 25.432 | 20.554 | 20.985 | 0.899× | 19.126 |
| moe_sparse | eager | 1 | 2.956 | 4.525 | 3.791 | 4.131 | 0.780× | 3.754 |
| moe_sparse | eager | 2 | 2.460 | 4.091 | 3.589 | 4.115 | 0.685× | 3.701 |
| moe_sparse | eager | 4 | 2.310 | 4.402 | 3.927 | 4.100 | 0.588× | 4.060 |
| moe_sparse | inductor | 1 | 3.876 | 5.805 | 4.162 | 4.308 | 0.931× | 4.181 |

Sym on is slower than plain in every E5 cross-round median, by approximately
7–70%. Four-caller sparse MoE reaches 3.927 ms with replay versus 2.310 ms plain;
its 1.034× advantage over graph-enabled Torch does not beat the unwrapped controls.

All six OOMs are four-caller decode slots-off: three Sym failures during first
request preparation and three Torch failures during concurrent warm execution.
Graph-enabled four-caller decode admits **zero retained graphs and zero warm
graph launches** because of the owner budget. Its completed results are fallback
outcomes. Plain already fits; fallback supplies no capacity gain over plain, and
its additional work increases latency. These measurements use two replay slots
per owner; they do not establish an optimal admission/slot policy.

## Preparation, memory and break-even

Capture, first-request, warmup, retirement, submission, completed p50/p95,
owner/pool/Torch/NVML memory and Total(N) are retained as cross-round aggregates
in the [complete tables](TABLES.md), with every per-round field in the raw package.

For Sym eager decode, replay retains 280.477 MiB across regions, including 40 MiB
of private pools, versus 240.477 MiB for slots-off. The median observed warm
physical increment is 878 MiB on, 932 MiB off and 332 MiB plain. These scopes
overlap and must not be added. Replay capture costs 19.563 ms; first request is
135.180 ms versus 55.777 ms plain. Median completed p95 is 19.373 ms on versus
19.330 ms plain. Reduced launch submission does not remove these ownership costs.

Sym eager prefill retains 123.008 MiB, including 56 MiB of private pools, and has
a 492 MiB warm physical increment versus 334 MiB plain. Capture costs 13.635 ms,
and first request is 60.443 ms versus 30.754 ms plain. Its aggregate linear
break-even estimate against plain is 435 post-preparation requests. Most Sym
on/plain comparisons have no positive warm advantage and therefore no break-even.

`Total(N) = preparation + sum(first calls) + sum(warmup costs) + N × warm p50`.
Recorded warmup elapsed time includes correctness validation; common fixture and
oracle creation precede deployment timing. This is the measured harness intercept,
not a universal cold-start cost. A zero break-even against off means the measured
intercept and warm slope already favor on; it does not mean capture is free.
E5 totals use amortized burst cost. Estimates are not separately observed long runs.

NVML samples every 2 ms across fixture, snapshot, worker preparation, first,
warmup, warm and retirement phases. All peaks are sampled lower bounds; intervals
with no observation are null, never zero. Warm scope includes retained outputs,
checking and allocator/context state. Whole-process E5 NVML includes the physical
reservation. The owner admission limit bounds input payload/private pool retention,
not transient capture, outputs or total device allocation; actual OOMs remain.

## Launch mechanism

Profiles cover ten calls/requests and never supply acceptance latency samples.

| Diagnostic | Off ordinary launch APIs | On ordinary launch APIs | On graph launches | GPU kernel events off → on |
| --- | ---: | ---: | ---: | ---: |
| Eager pointwise region | 240 | 0 | 10 | 240 → 240 |
| Inductor pointwise region | 10 | 0 | 10 | 10 → 10 |
| E2 eager decode, Sym | 4,020 | 740 | 240 | 4,020 → 4,180 |
| E2 Inductor sparse MoE, Torch | 700 | 420 | 60 | 700 → 700 |

Ordinary launch APIs sum the reported CUDA runtime/driver kernel-launch calls.
The sparse-MoE kernel-name multiset matches exactly. Decode retains every original
kernel-name count but adds 160 `memcpy32_post` events; its total kernel work is
not asserted identical. The traces establish reduced launch submission, not fewer
mathematical operations or a Sym-specific advantage.

## Shape and budget supplements

For the eager launch-heavy region, stable replay reduces p50 from 0.541 ms off
to 0.246 ms on; plain is 0.308 ms. CPU submission drops from 0.520 ms off to
0.226 ms on. Retention is 4.016 MiB, including 4 MiB of private pools.

Inductor fuses this region to one kernel per call. Replay improves its slots-off
control (0.359 → 0.246 ms) but is slower than unwrapped Inductor (0.120 ms).
Changing shapes makes 75 of 100 calls fall back. The 1 MiB budget refuses graph
pools, so all 100 timed on calls use ordinary execution and retain no graph storage.
These fallback rows remain slower than plain despite sometimes beating slots-off.

Supplemental rows share an Inductor disk cache inside each process, so their
preparation/first times are not independent cold-process costs. The varying-shape
distribution includes first-shape JIT penalties: its worst on sample is 1,041 ms;
the unwrapped control also reaches 1,052 ms. Maxima and total warm time preserve
these spikes rather than hiding them behind medians. Cumulative pressure fallback
counts include the first call and five warmups (106 total, 100 timed).

## Reproduction and review

The [public Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-268)
contains raw records/audits, traces, exact executed sources/runtime, original
controls, excluded development attempts, derived data and validation logs.
See [README.md](README.md) for saved-data analysis and qualified GPU reruns.
No Release or SVG file is used. Canonical saved-data analysis uses CPython 3.10.12.

Development campaigns were excluded for the E2 warmup minimum, fallback input
lifetime, precision guards and autograd/hidden-storage ownership fixes. All were
restarted before final reporting; no favorable timing selection was used. The
final source-only review found no remaining ownership blocker after the fixes.
Full Python tests: 1,502 passed, 15 skipped, five documented warnings; replay
suite: 54 passed, two skips; separate two-GPU guard: one passed; analyzer: 30 passed.
The [review disposition](REVIEW.md) records eligibility decisions and final evidence
assessment. This three-round single-machine study supports the opt-in mechanism
and its measured limits; it supplies no automatic-partitioning or full-paper
novelty claim and no qualified end-to-end Sym performance reversal.
