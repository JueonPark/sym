# R3 / EXP-3 — Multi-GPU amortization (issue #84)

**Verdict: gate G5 FAILS on this box (scatter int8 K=4: A/B_xK = 0.92×,
bar 1.30×). Method A wins at K=1 (1.18–1.20×), ties at K=2, loses at K=4 —
a clean CPU-transform-bound crossover, not a model failure.**

Measured 2026-07-27 on `rebel-gpu1` (bare-metal EPYC 7351, 4× RTX 2080 Ti,
PCIe gen3), performance governor, persistence mode, host threads pinned to
4-7,20-23. Tool: `bench-multigpu-reloc` (`bench/rtrack/multigpu_reloc.cu`),
analysis `bench/rtrack/exp3_gate.py`, figure `exp3_figure.py`. Data in
`bench/results/r3_*`. Full experiment (not descoped): M0 fixed the
4-GPU aggregate at 3.35× < the 3.5× no-contention bar.

## Methods (both end at the identical int8 artifact)

- **A**: one CPU transform pass → ship int8 (r·S) to the GPUs.
- **B_xK**: ship fp32 (S) to the GPUs, transform on-device, K concurrent.
- **B_staged**: same as B_xK but DMAs serialized (secondary baseline).

Scenarios: **scatter** (tensor-parallel, one tensor sharded K ways, the G5
case — identity/quant plan so shards partition cleanly) and **broadcast**
(the same relocated+quantized tensor to all K GPUs — blocked-transpose
plan). K ∈ {1,2,4}, round-robin over the 4 GPUs.

## Concurrency methodology (issue's pilot decision)

**Single process, one host thread per GPU (own stream + device buffers),
spin-barrier start.** This is the M0-validated choice: M0 measured
cross-die pairs scaling 1.98× and 4-GPU aggregate 3.35×, both impossible
under CUDA context serialization, so a single process does not serialize
here. Process-per-GPU was rejected because Method A's defining feature —
one CPU transform shared across K GPUs (broadcast) / one folded pass
(scatter) — is naturally a single-process, shared-buffer fan-out.

## Results (aggregate wall clock, median of 20)

### scatter (G5)

| N | K | A ms | B_xK ms | B_staged ms | A/B_xK | A DMA-only ms |
|---|---|---|---|---|---|---|
| 8192 | 1 | 17.73 | 21.36 | 21.20 | **1.20×** | 5.38 |
| 8192 | 2 | 16.38 | 15.62 | 21.22 | 0.95× | 3.97 |
| 8192 | 4 | 16.81 | 15.21 | 32.79 | 0.90× | 4.29 |
| 16384 | 1 | 71.57 | 84.71 | 84.63 | **1.18×** | 20.71 |
| 16384 | 2 | 63.81 | 61.88 | 84.66 | 0.97× | 15.38 |
| 16384 | 4 | 63.04 | 58.03 | 118.81 | 0.92× | 14.91 |

### broadcast

| N | K | A ms | B_xK ms | B_staged ms | A/B_xK |
|---|---|---|---|---|---|
| 8192 | 1 | 78.91 | 22.43 | 22.29 | 0.28× |
| 8192 | 2 | 81.33 | 32.27 | 44.59 | 0.40× |
| 8192 | 4 | 82.62 | 37.06 | 90.99 | 0.45× |

## Why G5 fails — the crossover is CPU-transform-bound

**Method A's cost is flat in K** (one host transform pass; the K int8 DMAs
are cheap and parallel), while **B_xK accelerates with K** as more GPUs
share the aggregate PCIe. They cross between K=1 and K=2:

- scatter A wall ≈ 16.8 ms at N=8192 ≈ **15 GB/s** effective — the AVX2
  `quantize_pack` reading the full fp32 tensor (matches the R1 roofline).
  It does not improve with K.
- B_xK wall falls 21.4 → 15.6 → 15.2 ms (K=1→2→4): the 4-GPU aggregate
  PCIe (M0: ~42 GB/s) delivers S faster than the single-socket CPU can
  transform it.

So on this box Method A (scatter) wins iff `CPU_transform_BW >` the
effective aggregate delivery rate of B — which holds only at K=1. This is
the cost-model boundary, and it is the same AVX2-Zen1 weakness that failed
G3 in R1: the host transform, not the model, is the limiter.

**The delivery-only column proves the point.** A's int8 DMA alone is
4.3 ms at K=4 (N=8192) vs B_xK's full 15.2 ms — A moves ¼ the bytes and
would win ~3.5× if the transform were free or precomputed. The entire G5
miss is the ~12 ms CPU transform. A faster transform host (the 7800X3D
AVX-512 box) or an offline/amortized-across-many-loads transform flips G5.

## The contention regime is real (M0 confirmed)

`B_staged` (serialized DMAs) gets *worse* with K — 21 → 33 ms (N=8192),
85 → 119 ms (N=16384) — while `B_xK` (concurrent) *improves*. The gap is
exactly the multi-GPU transfer overlap EXP-3 was set up to probe: the win
of concurrent delivery over serialized is what B exploits, and it is what
outruns Method A's single CPU transform at K≥2.

## Broadcast

Method A loses at all K (0.28–0.45×) because its one CPU pass is the
**blocked-transpose fused gather+quant at ~3.5 GB/s** — the same
strided-gather wall that failed G3 in R1. B_xK, whose GPU relocate is
hidden under the transfer (R4: multiplier ≈ 1.4 ≪ ratio 42), pays only the
fp32 transfer. Broadcast on this host is CPU-gather-bound; it is a
win-condition-(a) case only where the host has a fast strided path.

## Gate G5 and decision

**G5: scatter int8 K=4, A/B_xK = 0.92× < 1.30× → FAIL** on this box.

Per the R-track plan this is not a blocker (G5 is a bonus multi-GPU claim,
not a paper gate). The result is a clean, quantified cost-model input: the
multi-GPU amortization win exists (K=1, and 3.5× on delivery alone) but is
gated by single-socket CPU transform throughput, which this AVX2 Zen1 host
lacks at K≥2. R2's Gen4 box (faster AVX-512 host) is where the end-to-end
gate could pass; that is a natural follow-on measurement, not a this-box
result.

## Caveats

- K=2 lands on GPU0+GPU1, which M0 showed share die 1's PCIe root (the
  contended pair). K=2 therefore conflates GPU-count scaling with
  shared-root contention; the K=1 and K=4 endpoints are the clean readings.
- Method A's CPU transform re-runs every timed iteration (no cross-load
  amortization); the DMA-only column is the "transform precomputed" bound.
- 4-GPU concurrency noise controlled by the barrier start + 20 reps; IQR/
  median stayed low (JSON `iqr_over_median_pct`).

## V5 addendum (issue #99) — pre-registered expectations

*This section is committed BEFORE the V5 data is taken (the `gates.py`
discipline, `a051a5a`). Measured results land in a follow-up commit below
it; the original R3 numbers above are unchanged.*

V5 closes two caveats flagged above: the contended K=2 point and the
indirect broadcast attribution. Harness: `--devices 2,3` (cross-die pair,
M0: 1.98x pair scaling vs 1.61x for the shared-root {0,1}) and
`--scenario broadcast_contig` (broadcast placement, identity plan, so
Method A's CPU stage is quantize-only at ~23 GB/s T=8 instead of the
blocked-transpose fused gather+quant at ~3.5 GB/s). Environment matches
R3 (governor, persistence, host threads pinned 4-7,20-23) for BOTH pairs,
so the only variable in the pair comparison is the PCIe root.

Expectations, stated before running:

1. **K=2 pair isolation (scatter + broadcast, N=8192).** If the K=2
   anomaly is shared-root contention, the delivery-bound legs on {2,3}
   should improve by roughly M0's pair-bandwidth ratio 23.69/21.02 ~ 1.13x
   (dma_ms down ~10-13%); scatter Method A's *wall* should move much less
   (< ~5%) because it is CPU-transform-bound on this host. If {2,3} shows
   no improvement, the K=2 dip is not root contention and the caveat gets
   revised, not confirmed.
2. **Broadcast attribution (broadcast_contig, N=8192, K=1,2,4).** If
   broadcast's 0.28-0.45x loss is CPU-gather-bound, replacing the strided
   gather with quantize-only should flip Method A to >= 1.0x vs B_xK at
   every K (A moves r*S = 67 MB/GPU vs B's S = 268 MB/GPU, and the 23 GB/s
   quant no longer starves the links). If it stays in the ~0.3-0.5x band,
   broadcast is fan-out-bound and R3's attribution is revised accordingly.

Both outcomes are reportable; the point is attribution, not confirmation.

### V5 measured results (data: `bench/results/v5_*_epyc_2080ti.json`)

**Expectation 2 CONFIRMED — broadcast is CPU-gather-bound, not
fan-out-bound.** `broadcast_contig` (same fan-out, quantize-only CPU
stage) flips Method A to a win at every K:

| scenario | K=1 | K=2 | K=4 |
|---|---|---|---|
| broadcast (gather+quant, R3) | 0.45x | 0.28-0.36x | 0.31x |
| broadcast (gather+quant, V5 re-run {0,1}) | — | 0.36x | — |
| **broadcast_contig (quant-only, V5)** | **1.26x** | **1.59x** | **2.02x** |

(A/B_xK, N=8192; V5 gather re-run reproduces R3's band, anchoring the
comparison on today's binary.) The attribution is upgraded from
consistent-with-rooflines to isolated: remove the strided gather and the
0.3-0.45x loss becomes a 1.3-2x win.

**Expectation 1 FALSIFIED AS DESIGNED — the clean K=2 point is not
obtainable by device selection on this harness.** Pair {2,3} measured
*worse*, not ~1.13x better: scatter A's DMA leg 3.97 ms on {0,1} vs
8.39 ms on {2,3} (2.1x), with instability (B_staged IQR 58.5% on {2,3}
vs 0.0% on {0,1}). Two controls locate the mechanism:

1. *Pinning-invariance*: re-running both pairs with host threads pinned
   to die 3 (12-15,28-31) leaves the ordering unchanged ({0,1} 4.01 ms,
   {2,3} 8.25 ms) — so it is not compute-thread NUMA locality.
2. *Ordinal remap*: `CUDA_VISIBLE_DEVICES=2,3,0,1` (physical {2,3} become
   ordinals {0,1}) improves physical {2,3} to 5.36 ms but physical {0,1}
   stays at 4.00 ms either way — pinned-page placement follows the
   allocation-time device context only partially, and pair {0,1} keeps an
   intrinsic advantage.

The intrinsic part is M0's topology: GPU0+GPU1's shared root lives on
die 1 *which has DRAM*; GPU2 sits on the memory-less die 2, so with a
single shared source tensor at least one {2,3} path always crosses the
IF. M0's 1.98x pair-{2,3} scaling came from `bench-multigpu-h2d`, which
gives every GPU its *own* source buffer — a semantics the delivery
workload (one tensor, K receivers) cannot adopt. **Consequence: the
original caveat is revised, not confirmed. R3's K=2 on {0,1} was not
penalized by root sharing in this harness — {0,1} is the favorable pair
here, because the source pages live on its die.** A genuinely
uncontended single-source K=2 needs NUMA-interleaved or per-die
replicated source allocation (`numactl` is not installed on this box);
recorded as a follow-up, out of V5's scope.

*Bonus row*: `aprefold` (P4, PR #101) held 2.9-4.4x over B_xK across
every V5 cell, including the remote-NUMA pair — pre-folding is robust to
the placement effect because it ships r*S bytes over the same paths.
