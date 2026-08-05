# CM4 — Tightened RSTAR rule + registered predictions + held-out split

**Issue**: #112 (CM4), parent tracking #107; discharges `docs/v3-costmodel.md` §5's
obligation and registers everything #CM5 is judged by. **Date**: 2026-08-05.
**Deliverable shape**: regular PR on branch `cm4-registration` — ~0.5 day, no measurement;
**must merge before any `bench/results/bp_*` artifact exists** (commit-order verifiable).

## Context

V3 §5 documented that `gate_rstar` excludes `mismatch_one_sided` rows from the RSTAR bar,
so a *more wrong* prediction can escape by reclassification, and obligated any future
re-run to pre-register a tightened rule first. CM3 made this concrete: under the current
(CM1+CM2) model, three committed `_pyreloc` files carry live
`both_exist`→`mismatch_one_sided` instances (the gen4 quant overlapped grid sits entirely
above 1.0). CM4 registers, before the BP dataset exists: (a) the tightened rule, (b) the
held-out split, (c) the model's per-cell predictions for the BP matrix, (d) the
pre-declared bimodal cell list #BP2's statistic rule applies to.

Decision taken with the user: **tightened-rule mechanics** — any `mismatch_one_sided` row
forces the RSTAR verdict to FAIL; each such row also contributes `RSTAR_ABS_BAR` (0.15)
to the reported max-|Δ| statistic so the printed number stays bounded and comparable.
Bars themselves are unchanged (0.15 / 0.15 / 0.20 — #107: changing them now would be
tuning).

## §1 Tightened rule + integrity

**`bench/rtrack/v3_gate.py` `gate_rstar`** (currently lines 92-116; the exclusion is
`diffs = [r["abs_diff"] for r in both]`):

- Each `mismatch_one_sided` row contributes `RSTAR_ABS_BAR` to `diffs`.
- Verdict: PASS iff `worst <= RSTAR_ABS_BAR` **and** `len(mismatched) == 0`; any
  one-sided mismatch ⇒ FAIL regardless of the max.
- `both_no_crossing` remains a qualitative PASS outcome (unchanged).
- Output labels the gate "rule v1 (CM4-tightened, v3-costmodel.md §5)" and prints the
  mismatch count it FAILed on.
- Header docstring gains the rationale, citing §5 verbatim ("treating
  `mismatch_one_sided` as an automatic FAIL contribution rather than an exclusion").

**Retroactivity note (documented, not avoided)**: running the tightened gate on the
frozen v0 report may print harsher verdicts than the v0 record. The v0 verdicts stand as
recorded in `docs/v3-costmodel.md` (untouched); the script is forward-looking, and the
"rule v1" label prevents conflation. The PR discloses the v0-report output under the new
rule.

**CM4-REGEN check** (new, in `v3_gate.py`, CALIBRATION-REGEN pattern): re-run
`cm4_register.py` to a temp path and byte-compare against the committed
`bench/results/cm4_registered_predictions.json` — the registration must be
deterministically reproducible from the committed model + calibrations. Commit-order for
the acceptance criterion is verifiable from git history alone (the registration commit
precedes the first `bp_*` commit); no extra machinery.

## §2 Registration file, split, bimodal list

**Generator** — new `bench/rtrack/cm4_register.py` (REPO_ROOT/`sys.exit("error: ...")`
conventions; imports pyreloc via `PYTHONPATH=build/sym/python`, hard-exits with the
figure_rstar-style message when unavailable). Deterministic output (sorted keys, fixed
float formatting via `json.dump(..., indent=2)` on values pyreloc returns — no
timestamps).

**`bench/results/cm4_registered_predictions.json`** schema:

- Header block: `generated_by`, calibration paths, `model` description string
  ("reloc::costmodel as of CM1(#109)+CM2(#110)+CM3(#111)"), and the **split declaration**:
  `{"train_n": [2048, 8192], "test_n": [4096, 16384], "heldout_bars":
  {"misclass": 0.15, "regret_p90": 0.20}}` plus a caveat string recording honestly that
  the committed calibrations' `overhead.{a,b}_ms` intercepts were two-point-fit on
  N∈{2048, 16384} endpoints, so the split stratifies *evaluation* (per #87's
  reconciliation in #107), not calibration inputs — noted, not fixed; bars unchanged.
- `cells`: one entry per (box × family × N) over the BP matrix — families
  {transpose, blocked_transpose, transpose_quant, nchw_nhwc_quant, quant, convert_f16},
  N ∈ {2048, 4096, 8192, 16384}, boxes {epyc7351-2080ti, 7800x3d-4070tis} (48 cells).
  Per cell: `pattern`, `r_native` (test_prediction.py's FAMILY_MAP), `src_bytes`,
  `split` ("train"/"test"), `t_a_ms`, `t_b_serial_ms`, `t_b_overlapped_ms`,
  `threshold_bytes_serial`, `threshold_bytes_overlapped`,
  `winner_vs_b_fair` (A vs Serial-B), `winner_vs_b_pipelined` (A vs Overlapped-B).
  Unmodelable cells (missing calibration keys, e.g. epyc r=0.125 tier): explicit nulls
  with a `reason` string — V3's loud-omission convention.
- `rstar`: per (box × family × placement): the predicted speedup grid over the measured
  r points {1.0, 0.5, 0.25, 0.125} (speedup(r) = t_b(r=1)/t_a(r), same quantity as
  figure_rstar/test_prediction) and `rstar_predicted = crossing(grid)`; unmodelable r
  points recorded as nulls with the grid actually used.

**`bench/results/cm4_bimodal_cells.json`** (hand-authored, evidence-cited — deliberately
outside the REGEN check since it is a declaration, not a generated artifact):

- The B_xK-K=4 lineage: `{scenario: "scatter", method: "bxk", K: 4, N: 8192}` with the
  R3 evidence (2026-07-27: min 8.66 / p95 16.66 ms, IQR 19.7%) and V4's re-observation
  (same bimodal distribution; `docs/v4-prefold.md:100-121`), plus a lineage rule
  extending the flag to B_xK K=4 at every N.
- A `rule` block restating #BP2's matched-percentile requirement (median-of-20
  disallowed on flagged cells; p50↔p50 and p10↔p10 reporting, increased reps) and the
  joint-ownership note (Build Doc v3 §5.3; #BP2 cross-links this file).
- Committed now — before any measurement — even though BP's single-GPU matrix may never
  touch these cells; fixing the list pre-data is the point.

**Verification, end-to-end**: full C++ + pytest suites green; `v3_gate.py` runs green on
all checks — CALIBRATION-REGEN ×2, REPORT-REGEN, and the new CM4-REGEN; frozen report,
calibrations, and all prior artifacts untouched; the tightened gate's output on the v0
report captured in the PR body; registration file spot-checked against direct
`pyreloc.predict` calls for at least one cell per box and one r* grid.

**Acceptance mapping** (#112): tightened rule = §1 (bars unchanged); held-out split
registration = the split declaration block; registered per-cell predictions = the cells
+ rstar sections (winner, T_A, T_B per placement, r* per family); pre-declared bimodal
list = `cm4_bimodal_cells.json`; "merged standalone, commit-order verifiably before the
first `bp_*` artifact" = this PR merging while no `bp_*` exists.

## Amendment (2026-08-05, during implementation)

Two drifts surfaced between this spec and the committed artifacts during the final
review of PR #121. In both cases the artifact is correct and this spec text is updated
to match it, not the other way around:

1. **§2 model string.** This spec originally wrote the `model` header field as
   `"reloc::costmodel as of CM1(#109)+CM2(#110)"`, omitting CM3. The committed
   `bench/rtrack/cm4_register.py` and `bench/results/cm4_registered_predictions.json`
   correctly emit `"reloc::costmodel as of CM1(#109)+CM2(#110)+CM3(#111)"` -- CM3
   (#111) landed before CM4 and is part of the model CM4 registers against. §2 above
   has been corrected in place to read `+CM3(#111)`.
2. **Bimodal-list "rule block" naming.** §2 above describes
   `bench/results/cm4_bimodal_cells.json` as carrying "a `rule` block restating #BP2's
   matched-percentile requirement." The committed artifact names this field `purpose`,
   not `rule` (see the file's top-level `"purpose"` key). The artifact's naming is
   correct as committed; this spec's prose used the wrong field name and is recorded
   here rather than silently reworded away.
