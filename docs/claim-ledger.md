# Claim ledger

Formalizes Build Document v3 Appendix C (issue #117): after three baseline
generations (staged `b` → `b_fair` #95 → `b_pipelined` #108), this table is
the single authoritative place a claim's current number and standing live.
Quote claims from here, not from the frozen experiment reports — those keep
their as-measured history, including numbers later withdrawn.

**Status vocabulary** (this document is the definition): `survives` — the
claim holds as originally stated under the strongest baseline; `narrowed` —
the direction holds but the originally quoted margin/bar does not;
`withdrawn` — the number must not be quoted (baseline artifact);
`refuted-as-stated` — the claim's direction failed under a fair baseline.

**A/B convention**: ratio of best effective-input GB/s per method
(equivalently t_B/t_A of best-chunk medians), stabler-preference merged
(docs/r2-exp2-gen4-crossover.md:48-58) for all `b_pipelined` figures —
derived from bench/results/cm5_eval_report.json's regen-locked machinery.

## The ledger

| claim | box | staged `b` | `b_fair` | `b_pipelined` | status | authoritative source |
|---|---|---|---|---|---|---|
| R1-G2: dtype reduction wins ≥1.5× (quant) | Gen3 | 1.74–2.43× PASS | 1.44/1.43/1.49/1.53× (issue #95, comment 2026-07-28) | 1.43–1.48× (below bar at every N) | narrowed | docs/r1-exp1-gen3-gates.md §BP restatement; boundary-law row below |
| R2-G2: dtype reduction wins ≥1.5× (quant) | Gen4 | 4.18/3.66/2.12/2.12× (R2); 3.23/3.58/2.21/2.24× (V1 session) | 2.20/2.60/1.46/1.54× | 1.94/2.39/1.52/1.42× (below bar at N=16384) | narrowed | docs/r2-exp2-gen4-crossover.md §BP restatement |
| R2-G4: pure relocation A/B ∈ [0.40, 0.80] (blocked_transpose) | Gen4 | **1.40/1.20/1.19/0.91× — WITHDRAWN, do not quote** (staged-baseline artifact; V1 found the baseline inadmissible) | 0.65–0.87× (direction reversed, in-band at N ≥ 8192) | 0.88/0.77/0.76/0.63× (loss direction confirmed, deeper) | withdrawn (staged number) / survives (loss direction, vs fair baselines) | docs/r2-exp2-gen4-crossover.md §§V1 + BP restatements |
| r\* (blocked_transpose) | Gen4 | 0.499 (R2 rsweep, Jul 22) / 0.604 (V1-session staged) | 0.374 (V1 rsweep) | 0.3956 serial / 0.3605 overlapped (BP rsweep; pred 0.3247/0.2914) | narrowed (each column is a DIFFERENT dataset — see note) | r2 doc r\*-tables; cm5_eval_report.json rstar_rows |
| r\* (quant) | Gen4 | 0.992 (R2) / none (V1 session) | 0.541 (V1 rsweep) | 0.6107 serial / 0.6150 overlapped (BP rsweep; pred none — one-sided) | narrowed (dataset caveat as above) | same |
| r\* (quant) | Gen3 | 0.6356 (V2 rsweep) | — | 0.7025 serial / 0.7024 overlapped (BP rsweep; overlapped pred 0.9966, Δ 0.2942) | narrowed (dataset caveat) | docs/v3-costmodel.md; cm5_eval_report.json rstar_rows |
| Boundary law: A/B ≈ BW_cpu/BW_link at the largest N | both | — | — | Gen3 1.4817, Gen4 1.4153 (quant, N=16384) | survives | this document, section below |
| cost-model v1 quality (MISCLASS / RSTAR rule-v1 / REGRET-p90, all-cells + held-out) | both | b_fair (Serial), b_pipelined (Overlapped); MISCLASS PASS (b_fair 2/48=0.0417, b_pipelined 2/40=0.0500; held-out 1/24=0.0417, 1/20=0.0500); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (serial max\|Δ\|=0.0709 + 2 one-sided mismatches; overlapped max\|Δ\|=0.2942 + 1 one-sided mismatch; overall FAIL) | | | narrowed | `bench/results/cm5_eval_report.json` (#113) |

**r\* dataset note**: the three r\* columns come from three different
measurement campaigns (R2 rsweep → V1 rsweep → BP rsweep), not re-fits of
one dataset; cross-column movement mixes baseline change with session
variance. Each cell is labeled accordingly.

## Boundary law — the headline

"host-side transform wins by the margin host memory bandwidth exceeds link
bandwidth — a margin that shrinks each PCIe generation."

| box | BW_cpu | BW_link (pinned, under load) | law BW_cpu/BW_link | measured A/B_pipelined (quant, N=16384) | residual |
|---|---|---|---|---|---|
| Gen3 | 23.2 (#108 pre-registered) / 23.01 (r1 rooflines, quantize_pack t8) | 13.06 | ≈1.78 / ≈1.76 | 1.4817 | −17% / −16% |
| Gen4 | 38.4 (#108 pre-registered) / 36.61 (r2 rooflines, quantize_pack t8) | 26.87 (BP session) | ≈1.43 / ≈1.36 | 1.4153 | −1% / +4% |

Both BW_cpu readings are shown — #108's pre-registered arithmetic verbatim
and the committed-roofline recomputation — precisely so no post-hoc choice
between them can tune the residual.

1. The margin shrink is measured (Gen3 1.48 → Gen4 1.42), matching the
   pre-registered direction.
2. Gen3's −16/−17% residual is a finding, not hidden: the naive law's
   BW_cpu term (isolated roofline) overstates the pipeline's effective CPU
   bandwidth. BP-G3's fair-baseline-corrected predictions (÷1.06:
   1.36–1.44) bracket the measured 1.43–1.48, so the law's shape holds
   with BW_cpu read as *pipelined effective* bandwidth — recorded as a
   definition refinement (no number adjusted; not a refit). Gen4's V-cache
   makes the two readings nearly coincide (−1/+4%).

Status: **survives** — both-box consistency in the pre-registered form;
the track's closing claim, in contrast to the narrowed G2 rows above.
