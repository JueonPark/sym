# BP2 — Pre-registered gates: `gates.py --exp bp`

**Issue**: #115 (BP2), parent tracking #108. **Date**: 2026-08-07.
**Deliverable shape**: regular PR on branch `bp2-gates` — pure Python + docs, no
measurement; **must merge before any `bench/results/bp_*` artifact exists** (commit-order
acceptance); ~0.5 day.

## Context

BP3 measures both boxes with `Method::BPipelined` (BP1, merged). BP2 fixes the bars and
rules BEFORE that data exists: BP-G1 admissibility, BP-G2 hiding confirmation, BP-G3
expectation checks, the bimodal matched-percentile rule (jointly owned with CM4), and the
CM4 registration cross-link. `gates.py` already has the `--exp {r1,r2,v1}` mechanism,
module-constant bars declared before I/O, and the exit-0 "reports, never gates CI"
convention — BP2 adds a fourth experiment in the same shape.

**Baseline-anchoring correction (recorded on #108 before this spec, comment
`#issuecomment-5215567565`)**: #108's Gen3 expectation line divided R1's A/B_staged
ratios by 1.06; the arithmetically consistent anchor for a `b_pipelined` comparison is
the committed V1 A/**B_fair** baseline. BP-G3's registered per-cell predictions use the
fair baseline uniformly (Gen4 ÷1.09, Gen3 ÷1.06); the headline "Gen3 survives 1.5×"
expectation is evaluated as #108 wrote it and is now expected to MISS — an investigated
boundary-law finding per #108's own risk 1, not a tuned-away one. Decision taken with the
user.

## §1 `gates.py --exp bp`

**Registered constants** (module level, before any I/O — the `R1_GATES` pattern):

- `BP_G1_FRACTION = 0.95` — b_pipelined effective input BW on r=1.0 workloads ≥ 0.95× the
  session's pinned H2D (derived in-CSV per machine by the existing `pinned_h2d_gbps`,
  which gains `b_pipelined` via the `B_METHODS` tuple — same DMA path, so it legitimately
  joins the denominator's candidate set).
- `BP_G1_STRICT_MIN_N = {"epyc7351-2080ti": 0, "7800x3d-4070tis": 8192}` — strict at
  every N on Gen3 bare metal; N ≥ 8192 on Gen4/WSL2 (small-N dispatch overhead
  pre-excluded, not post-excused). Unknown machines default to strict (0).
- `BP_G2_EXPOSURE_BAR = 0.05`, `BP_G2_MIN_N = 8192` — per-chunk kernel exposure ≤ 5%.
  **Registered exposure metric: `1 − h2d_occupancy`** (a single committed CSV column;
  median of per-iteration event ratios). Its denominator is the GPU pipeline span, not
  the host wall — for B methods `cpu_stage_ms = 0` so span ≈ wall; the gate prints this
  definition so the number is never misread. Rows are additionally filtered to
  `n_chunks > 1` (BP1's structural caveat: a single-chunk config cannot overlap for any
  design — the filter reason is printed, never silent). `h2d_occupancy` and `n_chunks`
  are parsed with `.get()` so pre-BP1 CSVs degrade to "no data" rather than KeyError.
- `BP_G3_TOL = 0.10` (absolute, per cell) and **`BP_G3_PREDICTIONS`** — a literal dict
  `{(machine, family, N): predicted A/B_pipelined}` over 24 cells: both boxes ×
  {quant, convert_f16, blocked_transpose} × N ∈ {2048, 4096, 8192, 16384}. Values =
  the committed V1 A/B_fair best-chunk baselines (using the docs' pre-declared
  stabler-preference values where they applied — e.g. Gen4 blocked N=2048 uses the
  rerun's 0.8739, N=4096 the matrix file's 0.8053) ÷ 1.09 (Gen4) / ÷ 1.06 (Gen3),
  each line carrying a provenance comment (source CSV, stabler-preference note where
  applicable, and the #108 correction-comment link). The three families are exactly the
  ones #108's expectations name; `transpose`/`transpose_quant`/`nchw_nhwc_quant` are not
  gated (T2 is b_pipelined-N/A anyway). Divisor provenance: Gen4 "B improves ~9%",
  Gen3 "~6%", both quoted from #108.
- `B_METHODS` becomes `("b", "b_fair", "b_pipelined")` (comment: BP2, #115).

**`exp_bp(rows, bimodal_path, cm4_path)`** — grouped by `machine`, prints per-gate
verdicts (`PASS`/`FAIL`/`no data`), always returns 0:

1. **BP-G1**: per machine, `pinned = pinned_h2d_gbps(mrows)`, bar = 0.95×pinned; cells =
   r=1.0 `b_pipelined` rows with `N >= BP_G1_STRICT_MIN_N[machine]`, best
   `effective_input_GBps` per (transform, N); every cell ≥ bar → PASS.
2. **BP-G2**: cells = `b_pipelined` rows with kernel-bearing families (exclude
   `gpu_kernel_ms == 0` rows), `N ≥ 8192`, `n_chunks > 1`; exposure = `1 −
   h2d_occupancy` of the best-chunk row per (transform, N); every cell ≤ 0.05 → PASS.
3. **BP-G3**: per registered cell, the measured ratio follows `exp_gates`' existing
   convention — `best(effective_input_GBps, method=a) / best(effective_input_GBps,
   method=b_pipelined)` per (transform, N), best = max over the chunk sweep (the same
   quantity the V1 baseline ratios were computed with, so prediction and measurement
   share one definition); `|measured − predicted| ≤ 0.10` → cell OK, misses marked `!`
   and the summary prints "misses investigated, not tuned (#115)".
4. **Bimodal rule**: load `bench/results/cm4_bimodal_cells.json`; a measured cell is
   flagged iff it matches a listed identity (or the lineage rule `{method: bxk, K: 4}` —
   fields absent from rtrack CSVs, so BP's single-GPU matrix normally yields "0 flagged
   cells present (rule armed)"). If a flagged cell IS present: median-based verdicts are
   disallowed for it; the gate requires p50↔p50 AND p10↔p10 matched-percentile data —
   and since the rtrack CSV schema carries no p10 column, the gate hard-FAILs that cell
   with "flagged cell measured without p10 statistics — re-measure with increased reps
   and percentile emission" rather than silently falling back to medians. The
   pre-registration of this precondition is the deliverable.
5. **CM4 cross-link**: load `bench/results/cm4_registered_predictions.json`; for every
   BP-G3 cell, print alongside it the registration's `winner_vs_b_pipelined` and the
   model's predicted ratio `t_b_overlapped_ms / t_a_ms` for the joined
   `(box=machine, family, N)` cell — one dataset evaluates both tracks (#CM5 does the
   formal model scoring; this is the cross-reference #115 requires).
6. **`--selftest`**: inline synthetic rows exercise each gate's PASS, FAIL, no-data, and
   filter paths (G1 machine-conditional N, G2 n_chunks/occupancy filters, G3 tolerance
   edges at exactly ±0.10, bimodal armed-but-empty and flagged-without-p10 paths) —
   prints `SELFTEST PASS/FAIL` per case and exits nonzero on any selftest failure (the
   one deliberate exception to exit-0, since selftest is a developer check, not a
   verdict report). This is gates.py's first test of any kind; it runs pre-data by
   construction.

CLI: `--exp` gains `"bp"`; `--bimodal` and `--cm4` path flags default to the committed
artifact paths; `--selftest` runs without `--csv`.

## §2 Companions, hazards, verification

- **`figure_rstar.py`**: `--b-method` gains `b_pipelined`, `B_METHOD_PLACEMENT` gains
  `"b_pipelined": "overlapped"`. The legacy `"b": "overlapped"` mapping is untouched
  (historical record); a comment notes `cm4_registered_predictions.json`'s
  `placement_map` is the authority CM5 evaluates against.
- **README**: `--exp bp` usage line added next to the r1/r2/v1 examples; one paragraph on
  the registered constants and the exposure-metric definition.
- **Known hazards, checked in verification**: the lint CI guard greps `bench/**.py` for
  `min(bw`, `def cpu_bw(`, `1.0 / (1.0 / bw` — the new code must avoid those token
  shapes; ctest is untouched (pure Python); `ls bench/results/bp_*` must be empty at the
  merge commit.
- **Verification, end-to-end**: `--selftest` green; a synthetic bp CSV (hand-built rows
  with known ratios/occupancies) run through `--exp bp` produces the expected
  PASS/FAIL/filter lines for all three gates + the armed bimodal line + the CM4
  cross-link column; the 24 registered predictions re-derived by an independent script
  from the committed V1 CSVs match the hardcoded constants exactly; full C++ + pytest
  suites and `v3_gate.py` REGEN checks stay green; lint-guard grep locally returns
  status 1; `bench-rtrack-test` untouched.

**Acceptance mapping** (#115): BP-G1/G2/G3 bars = the registered constants + gate
functions; bimodal statistic rule = §1.4 (with the p10 precondition made explicit);
CM4 cross-link = §1.5; "merged standalone before any bp_* exists" = this PR while
`bench/results/bp_*` is empty.
