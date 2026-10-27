# Registered bars for the cost-advice evidence

Committed before any evidence round of this directory was measured. The code
under test is this commit's parent chain on `feat/cost-advice`; the bars are
not re-tuned after data, and a failure is reported as a failure.

Arms (`compare.py`, default `float32` compute): `torch`, `torch_kernels`,
`sym_fast` (the fast lane pricing the compiler's advice with
`calibration/epyc7351-2080ti.cal`) and `sym_fast_noadvice`
(`CompilerClient(advice=False)`, `FastTransfers(advice=False)`: today's fast
lane). Three rounds through `run.py`, rotated path order, 30 samples after 5
warmups, idle host, CPUs 4-7/20-23.

- **NL1, never lose.** For every regime, the median of round medians
  p50(sym_fast) ≤ min(p50(torch_kernels), p50(sym_fast_noadvice)) × (1 + band),
  band = max(0.02, spread), spread = the faster control's
  (max − min) / min over its three round medians. Verdict: `never_lose.py`.
- **NL2, grouping kept.** GPT decode: |p50(sym_fast) − p50(sym_fast_noadvice)|
  ≤ band × p50(sym_fast_noadvice).
- **MISCLASS ≤ 0.15** of the 20 cells of `advice_sweep.py` (pooled DLRM bags,
  GraphSAGE gather rows, GPT weight columns, K/V group members; five sizes
  each), a cell counting as misclassified when the selected path is slower
  than the measured fastest by more than either path's own spread.
  **REGRET p90 ≤ 0.20**, regret = (t_selected − t_best) / t_best.
- **Attribution**, reported per regime and not a bar: `ablation.py`'s
  `advice = sym_fast_noadvice − sym_fast`, next to the #256 kernel share.
- **Intercept cells** (paths within the larger of the two path intercepts of
  each other) are listed separately; a MISCLASS failure concentrated there is
  a model-limits result and is written up as one.
- Expected, not a bar: headline speedups over Torch do not rise. The advice
  turns #256's six plan/runtime losses into ties at best and keeps decode.

Known before registration, from calibration (`bench/results/paths_epyc7351-2080ti.json`)
and profiling, not from evidence rounds: the direct path's Python
bookkeeping (signature guard, decision lookup, wire ring, accounting) costs
30–50 µs per call over the bare kernel-plus-copy that `torch_kernels` runs.
