# Claim ledger

> Formalized by #BP4 (Build Document v3 Appendix C): one table, claim × box ×
> baseline × value × status, one authoritative quote per headline. This stub
> carries only the CM5 model-quality row so #113's deliverable lands with its
> verdicts; #BP4 expands the table.

| claim | box | baseline | value | status | source |
|---|---|---|---|---|---|
| cost-model v1 quality (MISCLASS / RSTAR rule-v1 / REGRET-p90, all-cells + held-out) | both | b_fair (Serial), b_pipelined (Overlapped) | MISCLASS PASS (b_fair 2/48=0.0417, b_pipelined 2/40=0.0500; held-out 1/24=0.0417, 1/20=0.0500); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (serial max\|Δ\|=0.0709 + 2 one-sided mismatches; overlapped max\|Δ\|=0.2942 + 1 one-sided mismatch; overall FAIL) | narrowed | `bench/results/cm5_eval_report.json` (#113) |
