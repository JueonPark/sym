# R1 / EXP-1 — Gen3 gate experiment (issue #82) — GO/NO-GO result

**Verdict: G2 PASS → win-condition (a). Proceed to R2 (Gen4 crossover).**

Measured 2026-07-20 on `rebel-gpu1` (bare-metal EPYC 7351, AVX2-only, 1×
RTX 2080 Ti at PCIe gen3 x16), performance governor, persistence mode on,
process pinned to GPU0's NUMA-affinity cores (4-7,20-23). Protocol: 5
warmup + 30 timed, best chunk per method. Raw data:
`bench/results/r1_gen3_nsweep_epyc_2080ti.csv` (N-sweep),
`r1_gen3_tsweep_epyc_2080ti.csv` (thread-sweep),
`r1_rooflines/*.json` (stage rooflines), `r1_figure1_gen3_epyc_2080ti.png`.
Re-run the gate check with `python3 bench/rtrack/gates.py --csv <csv>`.

## Pre-registered gates

| Gate | Claim | Bar | Result | Verdict |
|---|---|---|---|---|
| G1 | Gen3 link is the floor | pinned H2D ∈ [11, 14] GB/s | 13.06 GB/s | **PASS** |
| G2 | dtype reduction wins (Case 1b) | T3 quant: A ≥ 1.5× B | 1.74–2.43× (all N) | **PASS** |
| G3 | fused strided+quant ties (Case 1a) | T2/T4: A ≥ 0.95× B | 0.04–0.33× | **FAIL** |
| G4 | pure relocation ties | T1: A/B ∈ [0.85, 1.1] | 0.06–0.08× | **FAIL** |

Bars were fixed in `gates.py` before the data was read. A/B is the ratio
of best effective-input GB/s per method at each N; a gate passes only if
it holds at every measured N (strict reading).

## Decision

**G2 passes decisively** (≥1.74× at every N, growing to 2.43× at N=16384),
so the paper keeps **win-condition (a)** and R2 proceeds with the Gen3-vs-
Gen4 crossover framing. G3 is a bonus claim per the issue and is **not**
required for GO.

## Why G3 and G4 fail here (expected, machine-specific)

G3/G4 failing is **not** a model falsification — it is the AVX2-only Zen1
host hitting a strided-gather wall, exactly the risk the issue-#73 register
flagged for this box. The stage rooflines make the cause explicit:

Strided CPU read bandwidth (GB/s, `bench-cpu-rooflines`, N=8192):

| pattern | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| gather_f32 transpose (runs of 1 elem) | 0.18 | 0.32 | 0.65 | 0.53 |
| gather_f32 blocked (runs of N elems) | 7.90 | 11.25 | 11.51 | 11.44 |
| gather_f32 nchw (tiled) | 0.56 | 1.01 | 2.01 | 2.72 |
| gather_quantize transpose | 0.10 | 0.21 | 0.41 | 0.49 |
| gather_quantize nchw | 0.48 | 0.95 | 1.82 | 2.14 |

Contiguous CPU read bandwidth (GB/s):

| kernel | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| quantize_pack | 13.26 | 19.31 | 23.21 | 23.01 |
| convert_f32_f16 | 12.00 | 17.19 | 17.59 | 17.42 |
| contig_read | 12.61 | 24.11 | 36.12 | 37.15 |

The single-element-stride transpose gather is **~0.5 GB/s even at 8
threads** — latency-bound, not bandwidth-bound, and it does not scale with
cores. Method A for T1/T2/T4 is therefore dominated by that gather
(cpu_stage_ms ≫ h2d_ms), so it loses to Method B's full-fp32-DMA-then-GPU
path by 4–25×. The contiguous kernels feeding T3/T5 hit 13–23 GB/s and beat
the 13 GB/s link — which is exactly why G2 (T3) and the ungated T5 win.

Key roofline reading: **blocked-plan gather reaches 11.5 GB/s = the gen3
H2D ceiling** (M0: 13.06). So a coalesced strided read is link-bound here,
while the pathological single-element strides are memory-latency-bound. The
gap between them is the whole story of G3/G4.

## Full A-vs-B matrix (best chunk per method, effective input GB/s)

| transform | N | A GB/s (C, cpu ms, h2d ms) | B GB/s (C, kern ms) | A/B |
|---|---|---|---|---|
| transpose | 2048–16384 | 0.64–0.75 | 9.7–11.4 | 0.06–0.08 |
| blocked_transpose | 2048 | 8.36 (4M, 1.6, 1.4) | 8.73 (4M, 0.10) | **0.96** |
| blocked_transpose | 4096 | 9.95 (4M, 6.2, 5.5) | 9.77 (4M, 0.29) | **1.02** |
| blocked_transpose | 8192 | 7.99 | 10.32 | 0.77 |
| blocked_transpose | 16384 | 7.94 | 11.85 | 0.67 |
| transpose_quant (T2) | all | 0.49–0.90 | 8.5–11.3 | 0.04–0.11 |
| quant (T3) | 2048 | 18.06 | 9.14 | **1.98** |
| quant (T3) | 8192 | 18.73 (4M, 13.6, 5.8) | 10.39 (4M, 0.68) | **1.80** |
| quant (T3) | 16384 | 29.66 (4M, 34.2, 23.3) | 12.19 (4M, 2.53) | **2.43** |
| nchw_nhwc_quant (T4) | all | 1.24–2.74 | 8.4–11.3 | 0.11–0.33 |
| convert_f16 (T5) | 2048–16384 | 11.98–19.99 | 8.9–12.1 | **1.23–1.66** |

Observations beyond the gates (for R2/the paper):

- **T5 (fp16, r=0.5) is an ungated Method-A win at every N** (1.23–1.66×) —
  a second Case-1b data point at a milder dtype ratio, useful for the R2
  critical-r\* fit.
- **T1b (blocked transpose, r=1.0) ties at small N** (0.96–1.02× at N ≤
  4096) and degrades as N grows — the G4 "relocation ties, doesn't win"
  concept is confirmed for the *coalesced* access pattern; the plain-
  transpose G4 failure is purely the single-element-stride pathology.
- **Bottleneck flip (D-track evidence):** T3 Method A at N=16384 spends
  cpu 34.2 ms vs h2d 23.3 ms — the CPU quantize and the (quarter-size) DMA
  are within ~1.5× of each other, i.e. the stage decomposition shows the
  bottleneck moving toward the link as r shrinks, which is the mechanism
  the P3b cost model must capture.

## G1 note

Effective DMA bandwidth derived from Method B's `h2d_ms` peaks at 13.06
GB/s (N=16384), inside the [11, 14] bar and matching the M0 pinned-copy
number (13.08). The link is the floor, as claimed.

## Caveats

- **AVX2-only host.** These G3/G4 failures are this machine's strided-
  gather ceiling, not the model's. A host with AVX-512 gather + higher DRAM
  bandwidth (the 7800X3D) may clear G3 for the tiled T4 case; R2 measures
  it directly.
- Single 2080 Ti, GPU0, pinned. Not a WSL2 box (bare-metal), so this is a
  clean Gen3 headline point.
- The r-sweep (critical r\*) and the Gen4 repeat are R2 (issue for the next
  item).
