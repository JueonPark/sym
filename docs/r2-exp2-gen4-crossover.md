# R2 / EXP-2 — Gen4 repeat + crossover figure (issue #83)

**Verdict: R2-G2 PASS — the dtype-reduction crossover reproduces on Gen4
(quant A/B 2.1–4.2× at every N). But R2-G1 and R2-G4 FAIL and R2-G5
half-fails, all for one reason: the PCIe link is NOT Method B's floor on
this host. Method B is staging-copy-bound at ~16 GB/s while the Gen4 link
runs 24.6–26.8 GB/s. That single fact — BW_B ≈ 16, not H2D = 24.6 as the
pre-registered model assumed — flips G4 (pure relocation ties/wins instead
of losing ~0.6×) and pushes every measured r\* right of prediction (G5).
R2 is a reporting experiment; there is no go/no-go.**

Measured 2026-07-22/23 KST on `JueonAtHome` (**WSL2**, Linux
6.18.33.2-microsoft-standard-WSL2), AMD Ryzen 7 7800X3D (AVX-512, 8-core),
1× RTX 4070 Ti SUPER at PCIe **Gen4 x16** (confirmed gen4/x16 under load in
`bench/results/r2_gen4_calibration_7800x3d_4070tis.json`), CUDA 13.2 /
driver 595.79, STREAM triad 35.86 GB/s. Protocol: 5 warmup + 30 timed, best
chunk per method, `-arch=sm_89` standalone nvcc build.

**WSL2 environment-control caveats** (all recorded during the session, see
caveats): cpufreq governors unreadable on WSL2 → CPU governor
uncontrollable; `nvidia-smi -lgc` refused (no root) → GPU clocks unlocked;
persistence mode already on; nvbandwidth not installed → the calibration
JSON carries no `h2d_gbps` and the pinned-H2D number below is derived from
Method-B `h2d_ms` rows. Bare-metal control is the W7 item per issue #83;
the Gen3 report remains the clean bare-metal headline point.

Raw data (all committed):
`bench/results/r2_gen4_matrix_nsweep_7800x3d_4070tis.csv` (N sweep, T=8),
`r2_gen4_matrix_tsweep_7800x3d_4070tis.csv` (T sweep, N=8192),
`r2_gen4_rsweep_7800x3d_4070tis.csv` (r sweep, T=8),
`r2_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv` and
`r2_gen4_rsweep_rerun_7800x3d_4070tis.csv` (reruns of the six unstable
best-chunk N=16384 points — see the stabler-preference rule below),
`r2_rooflines/*.json` (stage rooflines), `r2_rstar_gen4.json` (r\* fit),
`r2_figure1_gen3_gen4.png` (Figure 1), `r2_figure_rstar_gen4.png` (r\*
figure). Gen3 comparison: `r1_gen3_nsweep_epyc_2080ti.csv`,
`docs/r1-exp1-gen3-gates.md`. Re-run the gate table with **exactly** the
N-sweep matrix CSV and its rerun — the tsweep CSV is a different config
family (T-sweep at fixed N=8192) and is **excluded** from the gate table:
`python3 bench/rtrack/gates.py --exp r2 --csv
bench/results/r2_gen4_matrix_nsweep_7800x3d_4070tis.csv
bench/results/r2_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv --rstar
bench/results/r2_rstar_gen4.json`. This reproduces the table below
digit-for-digit. Adding the tsweep CSV instead moves G1 to 26.81 GB/s (its
max Method-B row is transpose_quant N=8192 T=1) and quant N=8192 to 2.19× —
**no verdict changes**.

**Stabler-preference rerun rule (pre-declared).** Any best-chunk row with
IQR/median > 5% was flagged and re-measured; the analysis prefers, per
analysis point, whichever file's best-chunk row is stabler (lower IQR).
Exactly six N=16384 best-chunk points were flagged: T1 (transpose) r=1
methods a and b; T1b (blocked_transpose) r=0.5/0.25/0.125 method a; and
quant r=0.125 method a. Five became stable after rerun; T1b r=0.25 method a
remains borderline (5.08% IQR) and is flagged in the r\* figure's
`[UNSTABLE rows]` panel tag (it is an r-sweep point, not part of Figure 1).
All numbers
in this report use the merge (original CSVs with those six points replaced
by the stabler rerun rows).

## Pre-registered gates

| Gate | Claim | Bar | Result | Verdict |
|---|---|---|---|---|
| R2-G1 | Gen4 link is the floor | pinned H2D ∈ [20, 26] GB/s | 26.79 GB/s | **FAIL** |
| R2-G2 | dtype reduction wins (Case 1b) | T3 quant: A ≥ 1.5× B | 4.18 / 3.66 / 2.12 / 2.12× | **PASS** |
| R2-G3a | fused strided+quant loses | T2 transpose_quant: A/B < 0.95 | 0.40–0.20× | **PASS** |
| R2-G3b | fused strided+quant loses | T4 nchw_nhwc_quant: A/B < 0.95 | 0.69–0.46× | **PASS** |
| R2-G4 | pure relocation loses ~0.6× | T1b blocked_transpose: A/B ∈ [0.40, 0.80] | 1.40 / 1.20 / 1.19 / 0.91× | **FAIL** |
| R2-G5 | r\* model agrees per family | measured r\* within 2× of predicted | see below | **half-FAIL** |

Bars were fixed in `gates.py` before the Gen4 data was read. R2 is a
falsification experiment: the issue *predicts* the T2/T4 losses (G3) and
the T1b tie/loss (G4), and pre-registers where r\* should fall (G5). A/B is
the ratio of best effective-input GB/s per method at each N; the strict
reading requires a gate to hold at every measured N. R2-G5 verdicts per
family: blocked_transpose **FAIL** (measured r\* 0.499 vs predicted 0.143);
quant **FAIL** (0.992 vs 0.132); nchw_nhwc_quant **PASS** (both: no in-range
crossover); transpose_quant **PASS** (both: no in-range crossover).

## Decision / summary

The **headline claim of the paper reproduces**: dtype-reduction relocation
(T3 quant, r=0.25) is a decisive Method-A win on Gen4 — 4.18× at N=2048
falling to a stable 2.12× at N ≥ 8192, above the 1.5× bar at every N. T5
(fp16, r=0.5) is a second, ungated Case-1b win (1.77–2.87×). The Gen3→Gen4
crossover framing therefore holds.

The three failing gates are **all the same physical result** and are the
most informative part of R2: on this host Method B never reaches the link.
Its best effective input rate saturates at ~16 GB/s regardless of N, while
the DMA leg alone runs 24.6–26.8 GB/s. The pre-registered magnitude model
had set BW_B = H2D = 24.6 GB/s; the correct value is ~16. That one wrong
assumption is exactly why G1 misses high, why G4's predicted ~0.6× loss
turns into a tie/win, and why every measured r\* lands right of prediction.
The Gen3 box never exposed this because there its staging copy (~11.5 GB/s)
and its link (13 GB/s) were the same speed; on Gen4 the link outran the
single-socket staging memcpy. This is developed in the G4/G5 section.

## Why the gates land where they do — the 7800X3D rooflines

The AVX-512 7800X3D clears the strided-gather wall that sank the EPYC's
G3/G4 in R1 for the *contiguous* families, but the fused strided kernels
still bind. Stage rooflines (`in_gb_per_s`, `bench/results/r2_rooflines`,
N=8192; N=16384 T=8 quoted in text):

Strided CPU read bandwidth (GB/s):

| pattern | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| gather_f32 transpose (runs of 1 elem) | 0.88 | 1.60 | 2.94 | 4.59 |
| gather_f32 blocked (runs of N elems) | 13.61 | 13.94 | 16.48 | 15.47 |
| gather_f32 nchw (tiled) | 1.40 | 2.72 | 4.87 | 8.34 |
| gather_quantize transpose | 0.66 | 1.25 | 2.37 | 3.90 |
| gather_quantize blocked | 20.75 | 30.45 | 35.30 | 31.64 |
| gather_quantize nchw | 1.10 | 2.21 | 4.28 | 8.43 |

Contiguous CPU read bandwidth (GB/s, identity plan):

| kernel | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| quantize_pack | 28.57 | 32.62 | 37.79 | 36.61 |
| convert_f32_f16 | 20.89 | 25.71 | 27.51 | 26.20 |
| contig_read | 24.49 | 40.54 | 60.65 | 53.03 |
| pack_s8_s4 | 35.04 | 51.93 | 93.48 | 60.79 |

Reading the rooflines:

- **The contiguous stages clear the link.** `quantize_pack` hits 36.6 GB/s
  at T=8 (37.5 across the fastest plan) versus the EPYC's 23 — the R1
  "7800X3D may clear more gates" caveat is **confirmed for T3/T5**. These
  feed the two Method-A wins.
- **The single-element-stride transpose gather is still latency-bound**
  (0.88→4.59 GB/s over T=1→8, N=16384 T=8 only 3.63) — it never clears the
  link, so plain T1 and the fused T2 (transpose_quant) lose. The caveat is
  **not** cleared for the strided-fused families: gather stays 3.4–8.8 GB/s
  even with AVX-512 (nchw gather 8.34, N=16384 T=8 8.09).
- **The blocked gather does scale** (13.6→15.5 GB/s), but note it now tops
  out at ~15.5, *below* the Gen4 link — the same ~16 GB/s ceiling that binds
  Method B. `pack_s8_s4` is short-kernel noise at T=8 (60.79 at N=8192 but
  26.48 at N=16384, IQR up to 40%); cite with care.

So T2/T4 lose (G3 PASS-as-predicted) because their strided gather is
memory-latency-bound, exactly as R1's EPYC did — but T3/T5 win because
their contiguous quantize/convert stages run 26–37 GB/s. The surprise is
not the strided losses; it is that Method B, which should have ridden the
24.6 GB/s link, is itself pinned at ~16.

## The G4/G5 failure: Method B is staging-copy-bound, not link-bound

This is the central finding of R2 and it earns the same dedicated treatment
R1 gave its G3/G4 gather wall.

The pre-registered R2 model set the Method-B bandwidth equal to the
measured H2D link: BW_B = H2D = 24.6 GB/s. Under that assumption T1b should
lose ~0.6× (G4 bar [0.40, 0.80]) and the crossover r\* should sit near 0.13–
0.14. Neither happened. **Measured Method-B effective input rate is flat at
~16 GB/s** and never approaches 24.6:

| family / N | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| Method B blocked_transpose (GB/s) | 13.63 | 16.36 | 16.11 | 15.75 |

Meanwhile the DMA leg *inside* Method B runs 23–27 GB/s (from the same
rows' `h2d_ms`; pinned peak 26.79 GB/s). The gap is the host-side staging
memcpy into the pinned buffer: Method B stages the full fp32 tensor through
a single-socket copy that caps the whole pipeline at ~16 GB/s, so the fast
DMA never dominates. On Gen3 this was invisible — the EPYC's staging copy
(~11.5 GB/s) and its link (13 GB/s) matched, so BW_B = H2D held. On Gen4 the
link outran the memcpy and the assumption broke.

Consequences, all one mechanism:

- **G4 fails in the unexpected direction.** With B pinned at ~16 and blocked
  gather (Method A) reaching ~19 GB/s at N ≤ 8192, A/B is 1.40 / 1.20 /
  1.19 — A ties/wins instead of losing ~0.6×. Only at N=16384, where the
  1 GiB working set spills 7800X3D cache and Method A's gather drops to
  14.36 GB/s, does A/B fall to 0.91 (still above the [0.40, 0.80] bar).
- **G1 misses high, not low.** The 26.79 GB/s pinned H2D exceeds the [20,
  26] ceiling — the link is faster than pre-registered, which is fine for
  the hardware but confirms the link is not what binds Method B.

### Post-hoc model check (BW_B = 16 GB/s) — LABELLED POST-HOC

The following is a **post-hoc** correction, done *after* the data was read,
and is not a pre-registered result. The pre-registered `speedup_predicted`
scales as 1/BW_B; rescaling by the measured ratio (24.6/16 = 1.54) moves the
predicted curves toward the measured ones (N=16384, T=8):

| family | r | measured | predicted (BW_B=24.6) | post-hoc (×24.6/16) | serial |
|---|---|---|---|---|---|
| blocked_transpose | 0.25 | 1.98 | 1.22 | 1.87 | 0.93 |
| blocked_transpose | 0.125 | 1.94 | 0.95 | 1.46 | 0.85 |
| quant | 0.25 | 1.98 | 1.27 | 1.95 | 0.96 |
| quant | 0.125 | 1.93 | 0.98 | 1.51 | 0.87 |

At r=0.25 the post-hoc prediction matches measured almost exactly for both
families (1.87 vs 1.98; 1.95 vs 1.98). The residual gap at r=0.125 is the
plateau below. The takeaway: the r\* model's *shape* is right, but its
BW_B input was wrong by ~1.54×, which is the whole G5 miss.

## Full A-vs-B matrix (best chunk per method, effective input GB/s)

Columns: `A GB/s (chunk MiB, cpu_stage ms, h2d ms)`, `B GB/s (chunk MiB,
gpu_kernel ms)`, `A/B`. N=16384 T1 rows use the stabler rerun points.

| transform | N | A GB/s (C, cpu ms, h2d ms) | B GB/s (C, kern ms) | A/B |
|---|---|---|---|---|
| transpose (T1) | 2048 | 4.78 (16, 2.8, 0.6) | 12.46 (4, 0.06) | 0.38 |
| transpose (T1) | 4096 | 4.53 (16, 14.0, 2.5) | 17.18 (16, 0.27) | 0.26 |
| transpose (T1) | 8192 | 4.22 (16, 62.3, 10.1) | 16.18 (16, 1.13) | 0.26 |
| transpose (T1) | 16384 | 3.23 (256, 321.3, 40.7) | 16.05 (64, 4.02) | 0.20 |
| blocked_transpose (T1b) | 2048 | 19.04 (16, 0.2, 0.6) | 13.63 (4, 0.08) | **1.40** |
| blocked_transpose (T1b) | 4096 | 19.55 (16, 2.6, 2.5) | 16.36 (16, 0.26) | **1.20** |
| blocked_transpose (T1b) | 8192 | 19.16 (16, 12.8, 10.1) | 16.11 (16, 0.98) | **1.19** |
| blocked_transpose (T1b) | 16384 | 14.36 (4, 72.0, 40.4) | 15.75 (64, 3.98) | 0.91 |
| quant (T3) | 2048 | 47.67 (4, 0.1, 0.2) | 11.40 (4, 0.05) | **4.18** |
| quant (T3) | 4096 | 58.49 (16, 0.4, 0.6) | 15.98 (16, 0.24) | **3.66** |
| quant (T3) | 8192 | 33.64 (16, 7.1, 2.5) | 15.85 (16, 0.68) | **2.12** |
| quant (T3) | 16384 | 35.91 (16, 28.4, 10.2) | 16.91 (64, 2.30) | **2.12** |
| transpose_quant (T2) | 2048–16384 | 3.15–5.47 | 13.69–15.79 | 0.40–0.20 |
| nchw_nhwc_quant (T4) | 2048–16384 | 7.09–8.23 | 11.03–15.69 | 0.69–0.46 |
| convert_f16 (T5) | 2048–16384 | 29.45–34.74 | 11.61–17.27 | **2.87–1.77** |

Observations beyond the gates (for the paper):

- **T3 quant is the reproduced headline win.** The high small-N ratios
  (4.18/3.66) come from sub-millisecond CPU stages at N ≤ 4096 (unstable,
  hatched in Figure 1); the stable, cache-spilled N ≥ 8192 ratio is a
  clean 2.12×.
- **T5 (fp16, r=0.5) is an ungated Method-A win at every N** (1.77–2.87×),
  a second Case-1b point at a milder dtype ratio, useful for the r\* fit.
- **T1b confirms the G4 concept, wrong direction.** The coalesced blocked
  transpose *ties or beats* Method B at N ≤ 8192 and only degrades at
  N=16384 — because Method B is stuck at ~16, not because relocation is
  cheap. R1 saw T1b tie at small N too (0.96–1.02×); Gen4 pushes that tie
  into a win purely by handicapping B.
- **Bottleneck flip persists.** T3 Method A at N=16384 spends cpu 28.4 ms vs
  h2d 10.2 ms — the CPU quantize dominates the quarter-size DMA as r
  shrinks, the mechanism the P3b cost model must capture.

## r-sweep (critical r\*) — measured vs predicted per family

r\* fit at N=16384, T=8, from `bench/results/r2_rstar_gen4.json`. Speedup =
Method-A / Method-B effective input rate; r\* is the dtype ratio where the
measured curve crosses 1.0.

| family | r=1.0 | r=0.5 | r=0.25 | r=0.125 | r\* measured | r\* predicted |
|---|---|---|---|---|---|---|
| blocked_transpose | 0.92 | 1.00 | 1.98 | 1.94 | **0.499** | 0.143 |
| quant | 0.99 | 1.56 | 1.98 | 1.93 | **0.992** | 0.132 |
| nchw_nhwc_quant | 0.43 | 0.42 | 0.51 | 0.50 | none | none |
| transpose_quant | 0.20 | 0.19 | 0.19 | 0.19 | none | none |

- **Both winning families cross far right of prediction.** Predicted r\* ≈
  0.13–0.14 (pipelined, BW_B = H2D = 24.6); measured r\* = 0.499
  (blocked_transpose) and 0.992 (quant) — 3.5× and 7.5× larger, both well
  outside the 2× G5 tolerance. This is the BW_B = 16 error propagated into
  the crossover point.
- **The measured curve sits above the pipelined-predicted curve**, which in
  turn sits above the serial bound (blocked_transpose r=0.25: measured
  1.98, pipelined 1.22, serial 0.93; quant r=0.25: 1.98, 1.27, 0.96). The
  pipeline is real — measured beats serial — but the measured advantage is
  larger than the pipelined model expected because the model over-credited
  Method B.
- **Both winning families plateau at ~1.95–2.0× for r ≤ 0.25.** That
  plateau is not the model's H2D/r ceiling; it is
  BW_A(cpu-bound) / BW_B(staging-bound) ≈ 31–34 / 16 ≈ 1.9–2.1, where
  BW_A is the contiguous quantize/blocked-gather rate at low r (quant
  33.5–34.5, blocked 30.7–31.3 GB/s). Once r is small enough that the DMA
  is no longer the constraint, both methods hit their own memcpy walls and
  the ratio saturates.
- **The two loser families never cross** (T4 0.42–0.51×, T2 0.19–0.20× at
  all r), matching the model's "both none" prediction — those two G5
  sub-verdicts PASS. Their strided gather binds Method A regardless of r,
  so no dtype reduction can rescue them.

Figure 1 (`r2_figure1_gen3_gen4.png`) overlays the Gen3 and Gen4 A/B bars;
the r\* figure (`r2_figure_rstar_gen4.png`) plots measured vs predicted vs
serial per family. Unstable matrix bars in Figure 1 are hatched; the r\*
figure separately tags its unstable r-sweep points, including the
borderline T1b r=0.25 method a that stays flagged after rerun.

## sym#63 anchor re-baseline (R0 exit criterion)

- **Blocked gather (Method A CPU stage).** T1b method-a `cpu_stage_ms` at
  N=16384 best chunk gives 14.92 GB/s (1 GiB / 71.98 ms). sym#63 quoted
  ~14 GB/s; within 10%. Re-anchored.
- **Pinned H2D peak.** From Method-B `h2d_ms` rows, 26.79 GB/s (Gen4 x16).
  sym#63 quoted ~24 GB/s; this is the DMA-leg-only rate and re-baselines
  slightly higher on Gen4. Note this is *not* Method B's throughput — see
  below.
- **Method-B effective throughput** is flat across N (13.6 / 16.4 / 16.1 /
  15.7 GB/s at N=2048/4096/8192/16384) and sits far below the 26.79 GB/s
  DMA leg because it is staging-copy-bound (central finding). The R0
  re-baseline therefore records two distinct Method-B numbers on Gen4: the
  DMA leg (~26.8) and the delivered pipeline rate (~16).

## G1 note

Effective DMA bandwidth derived from Method B's `h2d_ms` peaks at 26.79
GB/s (max Method-B `h2d_ms` row in the gate-table CSV set: convert_f16
N=8192, T=8, chunk 256). The pre-registered
[20, 26] bar assumed the Gen4 link would be the floor and would peak inside
that window; instead the link exceeds it (miss high) *and*, more
importantly, is not the binding stage at all — Method B delivers only ~16
GB/s. G1 FAILs on both counts: the number is outside the bar, and the claim
"the link is the floor" is false for this host.

## Caveats

- **WSL2 host — stated caveat on every number.** cpufreq governors are
  unreadable so the CPU governor is uncontrollable; `nvidia-smi -lgc` was
  refused (no root) so GPU clocks are unlocked; nvbandwidth is not installed
  so the pinned-H2D figure is derived from Method-B `h2d_ms` rows rather
  than a dedicated micro-benchmark. Persistence mode was already on.
  **Bare-metal control (a T3/T1 subset) is deferred to W7 per issue #83**;
  the Gen3 bare-metal report stays the clean headline point and this Gen4
  report carries the WSL2 caveat.
- **Staging-copy bound is the load-bearing finding and is WSL2-tinged.** The
  ~16 GB/s Method-B ceiling could be partly WSL2 memcpy/pinning overhead;
  W7 bare-metal will separate host-memcpy cost from WSL2 cost. Either way it
  is not the PCIe link, which measures 24.6–26.8 GB/s.
- **Instability.** Unstable-row (IQR > 5%) share: nsweep 59/146, tsweep
  98/180, rsweep 129/235 — concentrated at small N (N=2048) and 4 MiB
  chunks, the known WSL2 dispatch-noise pattern (D3 ledger; sym#63
  addendum). At the analysis points (best-chunk, N=16384) only 2/12 matrix
  and 4/20 rsweep rows remain unstable after rerun, all 5.0–7.1% IQR. Every
  hatched Figure-1 bar traces to one of the 2/12 matrix points; among the
  r-sweep points, T1b r=0.25 method a is the lone one still flagged after
  rerun (5.08%), tagged in the r\* figure rather than hatched in Figure 1.
- **Scale-transfer excluded.** Only N ∈ {2048…16384} measured; no
  extrapolation to production tensor sizes is claimed here.
- **Two-pass CPU stages measured as-is.** The quantize/convert kernels are
  timed as implemented (read + write passes), not idealised single-pass;
  the roofline `in_gb_per_s` reflects that.
- The Gen3 comparison bars in Figure 1 are the R1 bare-metal EPYC/2080 Ti
  numbers (`r1_gen3_nsweep_epyc_2080ti.csv`); they are not re-measured here.
