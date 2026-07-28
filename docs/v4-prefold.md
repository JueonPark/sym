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
new pinning invented). Tool: `bench-multigpu-reloc`
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
N=16384's K=4 point (4.31x) and both broadcast K=4 points (4.13x) are the
same "scatter/broadcast K=4, fold pays only r·S" shape the gate expected;
G1 alone missed its bar.

## Why V4-G1 misses — B_xK K=4 N=8192 is bimodal, not slower here

Aprefold's own numbers at this cell are stable and match the model: DMA
leg 4.21 ms (R3's DMA-only column measured 4.29 ms at the identical cell —
consistent), IQR/median 0.23% (tight). The miss is entirely on the B_xK
side of the ratio:

| source | wall_min | wall_median | wall_p95 | IQR/median |
|---|---|---|---|---|
| R3 (2026-07-27), scatter K=4 N=8192, B_xK | 8.66 ms | 15.21 ms | 16.66 ms | 19.7% |
| V4 (this run), scatter K=4 N=8192, B_xK | 9.01 ms | 9.13 ms | 16.60 ms | 36.0% |

Both runs sample from the **same bimodal distribution** (min ≈ 9 ms,
p95 ≈ 16.6 ms — essentially identical endpoints) — this is the K=4
multi-GPU PCIe-contention regime M0/R3 already characterized. R3's
median-of-20 happened to land near the slow mode (15.21 ms); this run's
median-of-20 landed near the fast mode (9.13 ms). `bxk["speedup_vs_a"]`
swinging between the two runs (0.90x in R3, 0.57x here) is that same
draw. Predicted-from-R3's-DMA-only-column ratio was ~3.5x
(4.29/(15.21×0.25) scaled — see R3's "delivery-only column"); the model
was never wrong about aprefold's DMA leg, B_xK's median was just drawn
from the fast side of its own noise this time. **Reported as a FAIL, not
re-run or filtered**: rerunning with more iterations or a different seed
could plausibly move the ratio either side of 3.0x, which is itself the
finding — G1's bar is measuring a quantity with ~36% IQR/median noise at
this exact cell, so a single median-of-20 verdict here is not reproducible
without more reps or a variance-aware bar.

## Reuse sweep — amortization, per K, n_reuse ∈ {1,2,4,16}

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

| mode | scenario | K | n_reuse | A ms/load | prefold ms/load | predicted | measured | rule match |
|---|---|---|---|---|---|---|---|---|
| streaming | scatter | 1 | 1 | 17.76 | 101.61 | A | A | ok |
| streaming | scatter | 2 | 1 | 16.47 | 103.49 | A | A | ok |
| streaming | scatter | 4 | 1 | 15.78 | 109.22 | A | A | ok |

Streaming (never-reused tensors, one fold each) is the worst case for the
pre-fold path by construction: it pays the full cold-fold cost
(≥ 100 ms/load, worse than the plain reuse n=1 case because streaming's
staging buffer cannot be warmed by a prior fold in the same run) and never
gets to amortize it. A wins by 6-7x here, exactly as the rule predicts —
this is the counter-case V4-G3 requires to lose, and it does, at all
three K.

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
(`docs/r4-hiding-ratio.md`) a term with zero CPU-side work is trivially
transfer-bound regardless of the hiding ratio — its only cost is the r·S
DMA, so R4's model doesn't need to say anything new about it: aprefold's
5.2-20.6 ms wall times above are just its DMA legs (`dma_ms` ≈
`wall_median_ms` in every row), because there's no host phase left to
overlap with the reuse GPU work.

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
