# Automatic prepared execution: issue 265

The automatic path now consumes compiler-generated artifacts and reuses guarded
native preparation for compatible revisions. It remains **opt-in**. E3 repeated-window changes versus legacy range from an
8.44% regression to an 18.70% reduction, while the candidate remains 1.52–7.04×
the completed-call time of matched Torch. Live dequantization gains come from original-region
fallback. This evidence does not justify enabling the feature by default. The selected
policy is a 64 KiB source-byte threshold plus one previous occurrence of the
exact shape/parameter revision. This is a conservative heuristic; a large indexed
source with a tiny output can still be admitted.

Measured implementation: `09db6b4771fa509e8c4ead9057503c655f550fbe`.
Qualification harness: `975e854829a3ac0b564c707d0c8b38fdcd1485c0`.
Baseline: merged main `ce0bbe3b2451737ec35765bb1051841f5d19fc3d`.
Native/compiler binaries are identical. [Protocol](PROTOCOL.md),
[frozen policy and selection hashes](POLICY.json), [machine-readable index](INDEX.json),
and [runtime contract](../../docs/automatic-prepared-execution.md) define scope.
Raw records, exact runtime/source snapshots and CPU-only reanalysis are in the
[versioned release](https://github.com/JueonPark/sym/releases/tag/automatic-prepared-issue265-2026-11-07-v1).
Repository read access is required for this private repository's release.

Final review added a constructor guard rejecting externally supplied prepared
adapters, whose global counters could contaminate backend-local diagnostics.
Resource sharing through independently owned adapters remains supported and
tested. This guard does not change the measured execution path; the release
preserves the exact measured candidate and identifies the final reviewed runtime
separately.

## Evidence and interpretation

E3 contains 96 independent processes (4 cases × 8 arms × 3 rounds), 5,760 exact
outputs; held-out qualification contains 48 processes and 2,880 outputs. Every
process preserves the fixed 60-request sequence, checks a held output after
retirement and verifies zero native retained/live bytes after close. No registered
process is omitted. Selection used 48 separate processes/2,400 outputs and four
disjoint chains; its aggregate selected-policy ratio was 2.99× Torch versus 3.47×
for legacy automatic execution. Selection is exploratory and is not acceptance.

All percentages below are mean process-paired reductions; positive means faster.
Brackets are 95% process-bootstrap intervals (10,000 resamples, seed 265). Three
process pairs provide weak uncertainty evidence. Requests within a process are
not treated as independent samples. All individual effects and unfavorable
comparisons remain in the release analysis. The maximum number of additional
Dynamo callbacks after the first cycle was **0** across E3 and held-out
arms. That compile count is separate from execution latency or preparation.

## E3 repeated 48 requests

| Case | Compute | Reduction vs legacy automatic [95% interval] | Candidate / matched Torch |
| --- | --- | --- | --- |
| layout_cast | eager | +0.24% [-2.25, +5.20] | 2.19× |
| layout_cast | inductor | +3.15% [+2.03, +4.98] | 1.52× |
| kv_restore | eager | -8.44% [-10.57, -5.94] | 2.30× |
| kv_restore | inductor | +1.76% [-4.50, +13.80] | 1.59× |
| dequant_live | eager | +18.10% [+15.91, +19.64] | 2.42× |
| dequant_live | inductor | +18.70% [+18.19, +19.55] | 2.10× |
| index_live | eager | +4.92% [+2.34, +8.02] | 7.04× |
| index_live | inductor | +2.17% [+0.84, +2.89] | 2.09× |

Matched Torch is eager Torch for eager compute and Inductor for Inductor compute.
E3 live dequantization takes the saved original region under the selected policy;
its reduction is a fallback/admission effect, not accelerated native dequantization.
The always-prepared held-out control separately exercises live parameter binding.

## E3 preparation and actual execution

Each row covers 60 requests. Counts are identical across the three rounds.

| Case | Compute | Native calls | Preparations / hits / misses | Entries / evictions | Plans / Dynamo callbacks | Actual native rows |
| --- | --- | --- | --- | --- | --- | --- |
| layout_cast | eager | 40 | 5/35/5 | 5/0 | 1/4 | {"cuda_relocate_f32": 40} |
| layout_cast | inductor | 40 | 5/35/5 | 5/0 | 1/4 | {"cuda_relocate_f32": 40} |
| kv_restore | eager | 40 | 5/35/5 | 5/0 | 1/4 | {"layout_cuda": 40} |
| kv_restore | inductor | 40 | 5/35/5 | 5/0 | 1/4 | {"layout_cuda": 40} |
| dequant_live | eager | 0 | 0/0/5 | 5/0 | 1/3 | {} |
| dequant_live | inductor | 0 | 0/0/5 | 5/0 | 1/3 | {} |
| index_live | eager | 48 | 6/42/7 | 7/0 | 3/4 | {"cpu_reference": 48} |
| index_live | inductor | 48 | 6/42/7 | 7/0 | 3/4 | {"cpu_reference": 48} |

Capture exclusions count graph regions; fallback counters count invocation events.
They are not added to native calls to invent an eligibility percentage. Complete
maps, adapter attempts, per-call snapshot/index binding and group construction/
rebind counts are retained in every request record. The first-use gate can defer
some preparation into the repeated window; repeated does not mean fully prewarmed.

## Cold and complete-stream costs

All numbers are mean milliseconds across processes. Cold-first includes process
launch/import/configuration, CUDA initialization, owner setup and the first call.
The first-cycle/full/repeated columns sum completed calls; compilation/preparation
inside those calls is already charged. Accounted-full additionally includes setup
and owner retirement. Input generation, oracle and diagnostic overhead are separate.
Fresh processes/Inductor caches differ from original E3's shared-process first cycle; OS and CUDA driver JIT caches are retained. All arms import the frozen E3 module, which imports Sym even for Torch controls, so cold/RSS includes shared harness and unused Sym import overhead; these are not minimal Torch-only deployment measurements.

| Case | Arm | Cold first | First 12 | Full 60 | Repeated 48 | Accounted full |
| --- | --- | --- | --- | --- | --- | --- |
| layout_cast | torch | 3012.987 | 8.261 | 25.015 | 16.754 | 3035.951 |
| layout_cast | inductor | 11623.474 | 9905.131 | 9928.919 | 23.789 | 13323.584 |
| layout_cast | sym_auto_eager | 3651.804 | 531.065 | 567.481 | 36.416 | 3963.934 |
| layout_cast | sym_auto_inductor | 5572.768 | 8956.671 | 8994.088 | 37.417 | 12389.527 |
| layout_cast | sym_prepared_eager | 3663.633 | 534.613 | 570.926 | 36.313 | 3980.609 |
| layout_cast | sym_prepared_inductor | 5574.274 | 8960.120 | 8996.352 | 36.233 | 12398.432 |
| layout_cast | sym_fast_reprepare | 3042.638 | 36.899 | 55.322 | 18.424 | 3071.095 |
| layout_cast | sym_fast_cache | 3047.313 | 37.274 | 51.148 | 13.874 | 3070.844 |
| kv_restore | torch | 3016.466 | 5.911 | 19.549 | 13.638 | 3034.347 |
| kv_restore | inductor | 11287.224 | 9270.180 | 9289.849 | 19.670 | 12694.711 |
| kv_restore | sym_auto_eager | 3634.168 | 508.636 | 537.560 | 28.923 | 3938.203 |
| kv_restore | sym_auto_inductor | 5569.190 | 8442.863 | 8474.921 | 32.058 | 11887.768 |
| kv_restore | sym_prepared_eager | 3637.920 | 512.482 | 543.816 | 31.335 | 3947.370 |
| kv_restore | sym_prepared_inductor | 5561.314 | 8444.446 | 8475.676 | 31.230 | 11881.720 |
| kv_restore | sym_fast_reprepare | 3026.873 | 14.226 | 29.044 | 14.818 | 3048.108 |
| kv_restore | sym_fast_cache | 3030.685 | 13.123 | 24.297 | 11.175 | 3047.333 |
| dequant_live | torch | 3018.186 | 5.476 | 16.949 | 11.473 | 3033.449 |
| dequant_live | inductor | 11714.756 | 9423.953 | 9437.309 | 13.356 | 12840.767 |
| dequant_live | sym_auto_eager | 3655.979 | 405.139 | 438.866 | 33.728 | 3836.762 |
| dequant_live | sym_auto_inductor | 5573.243 | 8230.313 | 8264.674 | 34.361 | 11660.795 |
| dequant_live | sym_prepared_eager | 3652.517 | 399.648 | 427.274 | 27.627 | 3826.609 |
| dequant_live | sym_prepared_inductor | 5588.105 | 8241.206 | 8269.143 | 27.937 | 11671.848 |
| dequant_live | sym_fast_reprepare | 3039.444 | 34.831 | 50.243 | 15.412 | 3060.615 |
| dequant_live | sym_fast_cache | 3045.521 | 33.526 | 49.065 | 15.538 | 3066.836 |
| index_live | torch | 3016.507 | 2.528 | 7.998 | 5.470 | 3023.575 |
| index_live | inductor | 11536.878 | 10227.382 | 10245.407 | 18.026 | 13650.970 |
| index_live | sym_auto_eager | 3620.827 | 428.467 | 467.610 | 39.143 | 3866.323 |
| index_live | sym_auto_inductor | 5552.194 | 2578.167 | 2616.569 | 38.403 | 6019.822 |
| index_live | sym_prepared_eager | 3626.005 | 427.932 | 465.112 | 37.180 | 3871.825 |
| index_live | sym_prepared_inductor | 5543.768 | 2574.079 | 2611.646 | 37.566 | 6013.612 |
| index_live | sym_fast_reprepare | 3052.443 | 33.662 | 43.285 | 9.622 | 3066.710 |
| index_live | sym_fast_cache | 3057.875 | 32.272 | 41.653 | 9.381 | 3071.894 |

## Held-out chains and small requests

The held-out chains change rank/permutation, cast order, parameter coordinates and
indexed rank. Neither their shapes nor chains influenced policy selection. The
always-device control demonstrates automatic native lowering even where selected
admission retains Torch. This is generated relocation/value-stage execution from
compiler artifacts, not fusion of a handwritten producer.

| Held-out case | Arm | Reduction vs legacy [95% interval] | Arm / Torch | Native calls / 60 |
| --- | --- | --- | --- | --- |
| held_cast_chain | sym_prepared_eager | +41.84% [+40.53, +43.07] | 3.15× | 17 |
| held_cast_chain | always_device | +37.51% [+36.75, +38.78] | 3.39× | 45 |
| held_dequant_axis0 | sym_prepared_eager | +42.84% [+42.11, +43.65] | 2.83× | 0 |
| held_dequant_axis0 | always_device | -12.33% [-15.62, -7.39] | 5.57× | 50 |
| held_index_rank3 | sym_prepared_eager | +6.21% [+5.41, +6.94] | 9.18× | 49 |
| held_index_rank3 | always_device | +6.05% [+4.51, +7.23] | 9.20× | 55 |
| held_layout | sym_prepared_eager | +14.60% [+14.17, +15.29] | 4.49× | 13 |
| held_layout | always_device | +2.55% [+1.73, +3.38] | 5.13× | 45 |

Small-request rows use repeated-window `n=1` or `n=7` requests. Values are means
of three process means in microseconds. The whole-call difference includes capture,
custom-op dispatch, guards and execution. Adapter bookkeeping is the measured
preflight/admission subset, including zero for requests rejected before the adapter;
it excludes earlier frontend guards and live-value rechecks at execution.

| Case | Torch | Legacy auto | Selected | Always prepared | Selected − Torch | Adapter bookkeeping |
| --- | --- | --- | --- | --- | --- | --- |
| held_cast_chain | 82.9 | 606.6 | 399.4 | 513.3 | +316.5 | 29.9 |
| held_dequant_axis0 | 143.6 | 694.4 | 424.2 | 824.8 | +280.7 | 28.3 |
| held_index_rank3 | 64.1 | 662.9 | 622.2 | 620.4 | +558.1 | 110.2 |
| held_layout | 63.1 | 421.7 | 367.4 | 479.3 | +304.3 | 31.6 |

## Memory and limitations

Mean MiB across process peaks/boundary observations are shown below. RSS includes
compiler and oracle overhead; physical CUDA free-memory observations include the
context and are lower bounds on transient peaks. Torch allocation counters exclude
native scratch. Native resource counters and all phase snapshots are in raw data.
Remaining memory after owner close can include held outputs, Torch allocator caches
and CUDA modules; native retired/live bytes reach zero in every accepted process.

| Case | Arm | Process peak RSS | Observed device use | Torch peak allocated |
| --- | --- | --- | --- | --- |
| layout_cast | torch | 840.0 | 224.6 | 4.9 |
| layout_cast | sym_auto_eager | 887.8 | 224.6 | 4.9 |
| layout_cast | sym_prepared_eager | 889.8 | 278.6 | 4.9 |
| layout_cast | inductor | 997.3 | 224.6 | 8.6 |
| layout_cast | sym_auto_inductor | 993.3 | 222.6 | 8.6 |
| layout_cast | sym_prepared_inductor | 997.1 | 278.6 | 8.6 |
| layout_cast | sym_fast_cache | 836.7 | 244.6 | 4.9 |
| kv_restore | torch | 829.0 | 224.6 | 5.5 |
| kv_restore | sym_auto_eager | 882.7 | 224.6 | 5.5 |
| kv_restore | sym_prepared_eager | 886.5 | 270.6 | 5.5 |
| kv_restore | inductor | 940.5 | 224.6 | 5.5 |
| kv_restore | sym_auto_inductor | 940.9 | 224.6 | 5.5 |
| kv_restore | sym_prepared_inductor | 942.8 | 270.6 | 5.5 |
| kv_restore | sym_fast_cache | 828.6 | 270.6 | 5.5 |
| dequant_live | torch | 816.8 | 200.6 | 0.4 |
| dequant_live | sym_auto_eager | 880.4 | 200.6 | 0.4 |
| dequant_live | sym_prepared_eager | 880.3 | 200.6 | 0.4 |
| dequant_live | inductor | 985.6 | 220.6 | 5.6 |
| dequant_live | sym_auto_inductor | 984.1 | 220.6 | 5.6 |
| dequant_live | sym_prepared_inductor | 983.3 | 220.6 | 5.6 |
| dequant_live | sym_fast_cache | 820.1 | 220.6 | 0.4 |
| index_live | torch | 834.8 | 200.6 | 0.4 |
| index_live | sym_auto_eager | 886.6 | 200.6 | 0.4 |
| index_live | sym_prepared_eager | 890.4 | 220.6 | 0.4 |
| index_live | inductor | 993.7 | 220.6 | 5.8 |
| index_live | sym_auto_inductor | 937.0 | 200.6 | 0.4 |
| index_live | sym_prepared_inductor | 938.4 | 220.6 | 0.4 |
| index_live | sym_fast_cache | 835.5 | 220.6 | 0.4 |

This is one-machine blocking H2D qualification (EPYC 7351, RTX 2080 Ti, eight CPU
threads), not a multi-device, overlap or sustained-serving claim. Private ordering
streams and metadata remain additional overhead. Small operations can regress, and
unchanged compilation counts do not establish a speedup. Indexed native programs
and their reports are rebuilt per call. Mutable parameter revisions are snapshotted
and rechecked; they never reuse stale values. Unsupported/configured legacy paths
remain explicit. Cache limits bound retained entries/key bytes, not outstanding
caller-held preflighted calls or all process memory.

## Acceptance mapping

| Issue criterion | Evidence |
| --- | --- |
| Fixed E3 sequence and eligibility/preparation/compile counts | `e3-v1/contract.json`, all 96 records, `e3-analysis.json`; tables above |
| Live values/parameters, metadata, held outputs and fallback | `test_auto_prepared.py`, real native failure/fork subprocesses, all exact/retirement checks |
| Cold/full/repeated latency and memory against all controls | Eight E3 arms, phase timings, process-paired analyses and memory snapshots |
| Unseen chains/shapes and small-operation costs | Frozen `workloads.py`, 48 held-out records, always-prepared ablation and small-request table |

Part of #261; this supplies evidence for #270, #272 and #274 and builds on #263/#264.
It does not by itself establish a full-paper contribution or a universal Sym benefit.
