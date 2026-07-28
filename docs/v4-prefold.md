# V4 / P4 — Pre-folded transform path (issue #98)

**Verdict: gate V4-G1 FAILS on this box (aprefold/B_xK = 2.17x at scatter
K=4 N=8192, bar 3.0x); V4-G2 PASSES (aprefold's DMA leg reaches 12.86 GB/s,
98% of the box's pinned H2D, bar 90%); V4-G3 PASSES (both counter-cases —
cold single-use and streaming — lose to Method A, and the `prefoldWins`
rule predicts every one of the 15 reuse/streaming rows correctly). The G1
miss is explained, not tuned away: B_xK at K=4 is a known-bimodal
measurement (R3 already saw min 8.66 / p95 16.66 ms at this cell), and this
run's median landed on the fast side of that split (9.13 ms) rather than
the slow side R3's published number used (15.21 ms) — same hardware, same
distribution, different median-of-20 draw.**

Measured 2026-07-28 on this box (bare-metal EPYC 7351, 4x RTX 2080 Ti,
PCIe gen3), performance governor, persistence mode Enabled (all 4 GPUs),
host threads pinned to `4-7,20-23` — R3's exact protocol
(`docs/r3-exp3-multigpu.md`), replicated here (`grep -n
'taskset\|governor\|persistence'` confirms R3 pinned to 4-7,20-23; no
new pinning invented). The harness itself did change from R3's, though:
each K iteration here holds a live pinned artifact resident (67 MB at
N=8192, 268 MB at N=16384) through the A/B_xK/B_staged timings and
inserts a fourth timed method (aprefold) alongside them; the G1
bimodality explanation below rests on the endpoint match (min/p95
against R3's own), so this harness change is a possible-but-unevidenced
contributor to that cell's draw, not something silently ruled out. Tool:
`bench-multigpu-reloc`
(`bench/rtrack/multigpu_reloc.cu`) with the Task 3 `--reuse`/`--streaming`
flags (issue #98); gate script `bench/rtrack/exp4v_gate.py` (Task 4,
committed before this data was collected). Data in `bench/results/v4_*`.
Binary + code at HEAD `8312884` (bench: pre-register V4-G1..G3 gate
bars, #98).

## Methods (same three as R3, plus the pre-folded path)

- **A**: one CPU transform pass → ship int8 (r·S) to the GPUs, re-run
  every load (R3's baseline).
- **Aprefold** (V4/P4, new): the CPU transform runs **once**, the folded
  int8 artifact is cached in pinned host memory; each load after the first
  is a bare `cudaMemcpyAsync` of r·S bytes with no CPU transform term.
- **B_xK**: ship fp32 (S) to the GPUs, transform on-device, K concurrent.
- **B_staged**: same as B_xK but DMAs serialized (secondary baseline).

K ∈ {1,2,4}, round-robin over the 4 GPUs. r = 0.25 (int8 artifact) for all
cells below.

## Gates (pre-registered in `exp4v_gate.py` before this run, Task 4)

| Gate | Bar | Measured | Verdict |
|---|---|---|---|
| V4-G1 aprefold/B_xK @ scatter K=4 N=8192 | ≥ 3.00x | **2.17x** | **FAIL** |
| V4-G2 aprefold DMA leg @ scatter K=1 N=8192 vs pinned H2D | ≥ 11.76 GB/s (0.90 × 13.07) | 12.86 GB/s | PASS |
| V4-G3 counter-cases lose AND rule matches every reuse/streaming row | all 15 rows | 15/15 match, both counter-cases lose | PASS |

Full gate script output (`bench/results/v4_gate_report.txt`):

```
V4-G1 aprefold/B_xK @ scatter K=4 N=8192: 2.17x (bar >= 3.0)  FAIL
V4-G2 aprefold DMA leg @ scatter K=1 N=8192: 12.86 GB/s (bar >= 11.76 = 0.9 x 13.07)  PASS
...
V4-G3 (counter-cases lose AND rule predicts every row): PASS
```

## Results (aggregate wall clock, median of 20)

### scatter N=8192 (the G1/G2 cell)

| K | A ms | Apre ms | B_xK ms | B_staged ms | Apre/B_xK | Apre/A |
|---|---|---|---|---|---|---|
| 1 | 17.64 | 5.22 | 21.29 | 21.20 | 4.08x | 3.38x |
| 2 | 16.10 | 3.99 | 15.69 | 21.22 | 3.94x | 4.04x |
| 4 | 15.89 | 4.21 | 9.13 | 30.61 | **2.17x** [G1] | 3.77x |

### scatter N=16384

| K | A ms | Apre ms | B_xK ms | B_staged ms | Apre/B_xK |
|---|---|---|---|---|---|
| 1 | 66.78 | 20.61 | 84.74 | 84.65 | 4.11x |
| 2 | 61.56 | 15.36 | 62.12 | 84.66 | 4.04x |
| 4 | 55.80 | 8.92 | 38.47 | 119.26 | 4.31x |

### broadcast N=8192

| K | A ms | Apre ms | B_xK ms | B_staged ms | Apre/B_xK |
|---|---|---|---|---|---|
| 1 | 86.28 | 5.22 | 22.38 | 22.29 | 4.29x |
| 2 | 88.95 | 7.74 | 32.27 | 44.60 | 4.17x |
| 4 | 94.61 | 12.21 | 50.44 | 125.66 | 4.13x |

Aprefold clears 3.9-4.3x everywhere **except** the one gated cell
(scatter K=4 N=8192), where it lands at 2.17x — see the explanation below.
N=16384's K=4 point (4.31x) and broadcast's K=4 point (4.13x) are the
same "scatter/broadcast K=4, fold pays only r·S" shape the gate expected;
G1 alone missed its bar. These aggregate numbers are the n_reuse→infinity
limit (the transform pass *and* the 67 MB pinned allocation are both
hoisted out of the timed loop), and the reuse sweep below puts this box's
break-even at n_reuse ≈ 6: below ~6 reuses aprefold loses outright to
plain Method A, and the 3.9-4.3x figures above only describe the
fully-amortized regime above that crossover.

## Why V4-G1 misses — B_xK K=4 N=8192 is bimodal, not slower here

Aprefold's own numbers at this cell are broadly stable and match the
model: DMA leg 4.21 ms (R3's DMA-only column measured 4.29 ms at the
identical cell — consistent), IQR/median 0.23% (tight — the bulk of the
20 samples cluster near the median). The same skepticism applied to
B_xK below is worth applying to the numerator too: aprefold's own
`wall_min_ms` at this cell is 2.23 ms against its 4.21 ms median (p95
4.22 ms), so a fast mode exists here as well — it's just rarer, with
median and p95 sitting close together and one or a few fast outliers
pulling the min down. This does not change the conclusion (B_xK's swing
is an order of magnitude larger and lands on the ratio's denominator),
but "stable" above should be read as "stable at the median/p95," not "no
fast mode observed anywhere in this row." The miss is still
overwhelmingly on the B_xK side of the ratio:

| source | wall_min | wall_median | wall_p95 | IQR/median |
|---|---|---|---|---|
| R3 (2026-07-27), scatter K=4 N=8192, B_xK | 8.66 ms | 15.21 ms | 16.66 ms | 19.7% |
| V4 (this run), scatter K=4 N=8192, B_xK | 9.01 ms | 9.13 ms | 16.60 ms | 36.0% |

Both runs sample from the **same bimodal distribution** (min ≈ 9 ms,
p95 ≈ 16.6 ms — essentially identical endpoints) — this is the K=4
multi-GPU PCIe-contention regime M0/R3 already characterized. R3's
median-of-20 happened to land near the slow mode (15.21 ms); this run's
median-of-20 landed near the fast mode (9.13 ms). `a["speedup_vs_bxk"]`
(Method A's own speedup vs B_xK — a different ratio than aprefold/B_xK
above, but built from the same B_xK samples) swinging between the two
runs (0.904448 in R3, 0.574705 here, i.e. `bxk.median/a.median`) is that
same draw. R3's own delivery-only column already predicted this cell:
B_xK's median (15.21 ms) over A's DMA-only leg (4.29 ms) = 15.21/4.29 =
3.55x — comfortably over the 3.0x bar. The model was never wrong about
aprefold's DMA leg (4.21 ms here vs R3's 4.29 ms, consistent); B_xK's
median was just drawn from the fast side of its own noise this time,
which is what turns 3.55x into 2.17x. **Reported as a FAIL, not re-run
or filtered**: rerunning with more iterations or a different seed could
plausibly move the ratio either side of 3.0x, which is itself the
finding — G1's bar is measuring a quantity with ~36% IQR/median noise at
this exact cell, so a single median-of-20 verdict here is not reproducible
without more reps or a variance-aware bar.

## Gate G1 and decision

**G1: scatter K=4 N=8192, aprefold/B_xK = 2.17x < 3.00x → FAIL** on this
box, for the noisy-B_xK-denominator reason explained above.

This FAIL does not block #94's validity-hardening track: that track
depends on prefoldArtifact's correctness properties, not on G1's ratio.
`prefoldWins` likewise remains consumable by #V3 as-is — its inputs are
`tTransformMs`/`tPrefoldMs`/`penaltyMs` (transform and fold times), none
of which is the G1 ratio, so a FAIL here does not invalidate the rule
V4-G3 already validated 15/15. The path to re-testing G1 is a
variance-aware bar (one that accounts for B_xK's ~36% IQR/median at this
cell) or simply more reps at the K=4 cell to pull the median away from
the bimodal split's boundary; neither is done in this run.

## Reuse sweep — amortization, per K, n_reuse ∈ {1,2,4,16}

**Rep count differs from the aggregate tables above**: these rows are
`sweepIters = 7` (median-of-7), not median-of-20 like the wall-clock
tables (`multigpu_reloc.cu`'s reuse-sweep and streaming loops both use
`sweepIters = 7`). No `wall_min`/`wall_p95`/IQR is recorded for reuse or
streaming rows — the JSON carries only the per-row median
(`a_per_load_ms`, `prefold_per_load_ms`); V4-G3 rests on 15 point
comparisons with no dispersion stats attached to any of them.

| mode | scenario | K | n_reuse | A ms/load | prefold ms/load | predicted | measured | rule match |
|---|---|---|---|---|---|---|---|---|
| reuse | scatter | 1 | 1 | 17.42 | 77.93 | A | A | ok |
| reuse | scatter | 1 | 2 | 17.48 | 41.63 | A | A | ok |
| reuse | scatter | 1 | 4 | 17.35 | 23.32 | A | A | ok |
| reuse | scatter | 1 | 16 | 17.35 | 9.79 | PREFOLD | PREFOLD | ok |
| reuse | scatter | 2 | 1 | 16.20 | 78.80 | A | A | ok |
| reuse | scatter | 2 | 2 | 16.11 | 41.04 | A | A | ok |
| reuse | scatter | 2 | 4 | 16.12 | 22.70 | A | A | ok |
| reuse | scatter | 2 | 16 | 16.07 | 8.67 | PREFOLD | PREFOLD | ok |
| reuse | scatter | 4 | 1 | 15.24 | 82.26 | A | A | ok |
| reuse | scatter | 4 | 2 | 16.39 | 44.24 | A | A | ok |
| reuse | scatter | 4 | 4 | 15.15 | 23.54 | A | A | ok |
| reuse | scatter | 4 | 16 | 15.71 | 8.48 | PREFOLD | PREFOLD | ok |

The crossover is between n_reuse=4 and n_reuse=16 at every K: below it, the
one-time fold cost (`t_prefold_cold_ms` ≈ 72-82 ms, see caveats) dominates
the per-load average and A wins; above it, the amortized fold cost drops
under A's flat per-load transform and prefold wins. All 12 rows match the
`prefoldWins` rule's prediction.

## Streaming counter-case

Same rep count as the reuse sweep: `sweepIters = 7` (median-of-7), no
min/p95/IQR recorded.

| mode | scenario | K | n_reuse | A ms/load | prefold ms/load | predicted | measured | rule match |
|---|---|---|---|---|---|---|---|---|
| streaming | scatter | 1 | 1 | 17.76 | 101.61 | A | A | ok |
| streaming | scatter | 2 | 1 | 16.47 | 103.49 | A | A | ok |
| streaming | scatter | 4 | 1 | 15.78 | 109.22 | A | A | ok |

Streaming (source mutated before every fold, one fresh fold per load) is
the worst case for the pre-fold path, and it loses to A by 5.7-6.9x
(`prefold_per_load_ms / a_per_load_ms`: 101.61/17.76 = 5.72,
103.49/16.47 = 6.28, 109.22/15.78 = 6.92 at K=1/2/4) — exactly as the
rule predicts, and the counter-case V4-G3 requires to lose.

**What the ~24-27 ms/load gap vs. plain reuse actually is.** Comparing
streaming's `prefold_per_load_ms` to the reuse sweep's n_reuse=1 row at
the same cell (same method, no mutation): 101.61 vs 77.93 ms/load at
K=1, 103.49 vs 78.80 at K=2, 109.22 vs 82.26 at K=4 — a gap of
23.7-27.0 ms/load. This is **not** the cold-fold cost itself:
`t_prefold_cold_ms` is nearly identical between the two modes at every K
(72.94 vs 72.11 ms at K=1, 75.67 vs 74.04 ms at K=2, 80.74 vs 77.84 ms at
K=4 — streaming/reuse respectively), a difference of only 0.8-2.9 ms;
the "≥100 ms/load" figure above is `prefold_per_load_ms`, a different
quantity than the cold-fold cost. What the code does support attributing:
(1) streaming calls `mutateSource` before every one of a trial's 4 fresh
folds (`loads = 4` in `multigpu_reloc.cu`), where the reuse n=1 loop
folds an unmutated, already-warm source once per trial with no analogous
per-load perturbation; and (2) streaming's reported per-load number is a
*mean* over those 4 per-trial mutate+fold+load repetitions before the
median-of-7 is taken across trials, whereas reuse n=1's per-trial number
is a single fold+load timed directly and then medianed across the same 7
trials — so one slow fold inside a streaming trial's 4-wide inner loop
drags that trial's average up in a way the reuse loop's
single-shot-per-trial median is not exposed to. Together these plausibly
explain part of the gap, but neither is quantified separately here, and
**most of the 23.7-27.0 ms/load residual is unattributed** by this run's
instrumentation — it shows up in neither `t_prefold_cold_ms` nor
`t_transform_ms` (both flat/near-identical between modes). It is tied to
the same open `allocStaging`-churn caveat below, not a newly-identified
mechanism.

## Memory-budget: what pre-folding costs beyond the reuse table

Pre-folding trades CPU-transform time for pinned-memory footprint. Holding
both the fp32 source and the folded int8 artifact pinned simultaneously
(the pipeline needs the source live until the fold's DMA is enqueued)
costs `(1 + r) · S` pinned bytes per tensor — with r = 0.25 here, that's
1.25 × S, a 25% pinned-memory tax on top of the artifact's own r·S once
the source can be released. Pre-folding **every** weight of a model
multiplies this footprint by the number of resident folded tensors; on a
host with a fixed pinned-memory budget, that ceiling is what eventually
forces eviction of a folded artifact to make room for another. When the
budget forces an eviction, the next load of the evicted tensor pays
`t_prefold_cold_ms` again (this run measured 72.1-77.8 ms/K for the
`reuse` mode's cold fold, 72.9-80.7 ms/K for `streaming`'s) — the
`prefoldWins` rule already carries this as a `penaltyMs` term for
exactly this re-fold cost, but **V4 does not measure an eviction harness**:
every number above is a single-tensor, single-residency measurement. An
eviction-pressure benchmark (multiple tensors competing for a fixed
pinned budget) is future work, not something this run's data speaks to.

## R4 tie-in

The pre-fold path has no host-transform term on the timed path (the fold
already happened); under the R4 hiding-ratio model
(`docs/r4-exp4-hiding-ratio.md`) a term with zero CPU-side work is trivially
transfer-bound regardless of the hiding ratio — its only cost is the r·S
DMA, so R4's model doesn't need to say anything new about it: aprefold's
3.99-20.61 ms wall times above are its DMA legs by construction, not by
measurement (`multigpu_reloc.cu` sets `d = w` for `Method::APrefold` —
"the whole iteration IS the DMA leg" — so `dma_ms` == `wall_median_ms`
in every aprefold row trivially, not as an empirical finding). The
transfer-bound argument instead comes from the code path itself: there is
no host-transform phase inside the timed loop for aprefold to overlap
with the reuse GPU work.

## Caveats

- **AVX2-only Zen1 host.** This box's CPU-transform time (feeding both
  Method A's flat-in-K cost and the one-time fold cost) is R3/R1's
  AVX2-Zen1 measurement; a faster host (7800X3D/AVX-512) shrinks the
  transform term and would change both the reuse-sweep crossover point
  (currently n_reuse ≈ 4-16) and how much of `t_prefold_cold_ms` is fold
  vs. `allocStaging` (see below). Every "X ms" number in this doc is this
  box's, not a portable constant.
- **Shared-root K=2 contention** (M0/#99's territory, not re-attributed
  here): `nvidia-smi topo -m` confirms GPU0+GPU1 share a PHB (PCIe host
  bridge); K=2 in every table above lands on that contended pair, so its
  numbers conflate GPU-count scaling with shared-root contention exactly
  as R3 flagged. K=1 and K=4 are the clean endpoints.
- **V4-G1's B_xK K=4 N=8192 cell is high-variance** (IQR/median 36.0% this
  run, 19.7% in R3) — see the explanation above. Treat the 2.17x FAIL as a
  measurement of a noisy quantity, not a stable regression from R3's
  0.90x-adjacent delivery-only prediction of ~3.5x.
- **`t_prefold_cold_ms` drifts with K** (72.1 → 74.0 → 77.8 ms for
  `reuse`'s K=1→2→4, 72.9 → 75.7 → 80.7 ms for `streaming`'s), while
  `t_transform_ms` stays flat (12.0-12.4 ms) across the same rows. A prior
  reviewer flagged `CudaBackend`'s per-fold `allocStaging` call as a
  possible source of this drift (pinned-allocation churn/contention
  scaling with concurrent K, distinct from the CPU transform itself,
  which does not scale with K). This run's numbers are consistent with
  that flag — the transform term is flat, the non-transform remainder of
  `t_prefold_cold_ms` is what grows with K — but this run did not
  instrument `allocStaging` separately, so it is reported here as an open
  variance source, not confirmed root cause.
- **The reuse grid {1,2,4,16} never tests the rule near its crossover.**
  `t_prefold_cold_ms / t_transform_ms` ≈ 5.84-6.38 across K=1/2/4
  (72.11/12.35, 74.04/12.21, 77.84/12.19) — `prefoldWins` predicts the
  crossover lands around n_reuse ≈ 6. The grid's endpoints (4 and 16) do
  coarsely bracket that range, but the tested grid jumps straight from
  n=4 (below it) to n=16 (well above it) with nothing in between. The
  15/15 measured/predicted match reported under V4-G3 is real, but every
  point sits comfortably on one side of the boundary or the other; this
  run does not test the rule near its actual crossover.
