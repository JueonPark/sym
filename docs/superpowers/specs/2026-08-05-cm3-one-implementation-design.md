# CM3 — One implementation: retire figure_rstar.py's internal model

**Issue**: #111 (CM3), parent tracking #107. **Date**: 2026-08-05.
**Deliverable shape**: regular PR on branch `cm3-one-implementation` — no new measurement;
~0.5 day.

## Context

V3 documented (`docs/v3-costmodel.md` §"Two-sided implementation note") that
`bench/rtrack/figure_rstar.py` carries its own same-formula-in-spirit cost model whose
stored `rstar_predicted` values diverge 1.6–2.9× from `pyreloc.predict` on the same
family/box, and declared the C++ `reloc::costmodel` the maintained implementation. #111
retires the duplicate: predictions come exclusively from `pyreloc.predict`; stored
predicted artifacts are regenerated as new files alongside the originals; a CI guard
prevents a second roofline formula from reappearing under `bench/`.

Evidence gathered during brainstorming: the internal model (`figure_rstar.py:81-97`
`cpu_bw` + `:153-159` pred/serial lines) lacks the intercepts, the B-side HBM term, CM1's
`BPlacement`, and CM2's `cpu_pipe` override. The only live reader of stored
`rstar_predicted` is `bench/rtrack/gates.py:228` (R2-G5), which reads the *originals* by
path — unaffected by alongside-files. `test_prediction.py` and `test_f16_overcredit_pin.py`
read only measured fields. No bench script imports pyreloc today.

Decision taken with the user: **b_method→b_placement mapping** — artifacts measured with
`--b-method b_fair` (all four in scope) get predictions computed with
`b_placement="serial"`; `b` maps to `"overlapped"`. Measured-serial-B vs predicted-serial-B
is the honest comparison CM1's term was built for.

## §1 Script surgery + artifact regeneration

**`figure_rstar.py` changes** (one arithmetic, the script keeps its role — measured
curves, figure, JSON):

1. `import pyreloc` guarded at startup: on ImportError,
   `sys.exit("error: pyreloc not importable -- build it (ninja -C build/sym) and run "
   "with PYTHONPATH=build/sym/python")` (bench `sys.exit("error: ...")` convention).
2. Delete the internal model: `cpu_bw()`, `FAMILY_PLAN`, `load_rooflines()`, the
   `--rooflines` argument, and the pred/serial computation lines.
3. New argument `--calibration <path>` (a `.cal` file, loaded via
   `pyreloc.load_calibration`). Optional: absent → measured-only output (the same graceful
   degradation `--rooflines`-absent produced). Present → predicted curves via
   `pyreloc.predict`.
4. Prediction: pattern per family from the same map `test_prediction.py` uses
   (`quant`→contiguous, `blocked_transpose`→blocked, `transpose_quant`→single_element,
   `nchw_nhwc_quant`→tiled);
   `speedup_predicted[r] = predict(r=1.0, b_placement=mapped).t_b_ms / predict(r,
   b_placement=mapped).t_a_ms`; `rstar_predicted = crossing(...)` (existing function,
   unchanged). Missing calibration keys (ValueError) → that family's predicted fields are
   omitted, mirroring `test_prediction.py`'s unmodelable handling.
5. `speedup_serial` is dropped from the JSON and the "serial bound" series from the
   figure — the A-side serial bound (`1/BW_cpu + r/H2D`) is an internal-model-only
   quantity with no pyreloc counterpart (CM1's `Serial` placement serializes the *B* side).
6. matplotlib becomes optional: on ImportError, warn to stderr and skip the figure (the
   JSON is already written; today the unconditional import crashes after the JSON write).

**Regenerated artifacts** — four new files alongside the untouched originals, `_pyreloc`
suffix, each with a companion PNG:

- `bench/results/v1_gen4_rstar_bfair_pyreloc.json`
- `bench/results/v2_isa_rstar_avx2_pyreloc.json`
- `bench/results/v2_isa_rstar_avx512_pyreloc.json`
- `bench/results/v2_isa_gen3_rstar_avx2_epyc7351-2080ti_pyreloc.json`

All four originals are `b_fair` → predictions computed with `b_placement="serial"`, using
the box's committed calibration (`calibration/{7800x3d-4070tis,epyc7351-2080ti}.cal`).
Invariant: the new files' `speedup_measured`, `rstar_measured`, `h2d_gbps`, `n`, `threads`,
`b_method`, `unstable` are numerically identical to the originals (they derive from the
same CSVs); only the predicted fields differ (and `speedup_serial` is absent).
`r2_rstar_gen4.json` (pre-`b_fair`, R2-era) stays untouched — a historical record,
mentioned in the reconcile note. `gates.py` R2-G5 keeps reading the originals — no
behavior change.

## §2 Reconcile note, CI guard, verification

**Reconcile note** — new doc `docs/cm3-one-implementation.md` covering the three axes with
old→new tables:

1. **V3-era implementation divergence** (the issue's 1.6–2.9×): what the internal model
   lacked (intercepts, B-side HBM term, K/broadcast) and how that decomposes the gen4
   blocked 0.1807-vs-0.2914 and quant 0.1640-vs-0.4806 gaps.
2. **CM1/CM2 model drift since V3**: gen3 quant `rstar_predicted` 0.9989→0.9966 (CM2's
   `cpu_pipe` key); **gen4 quant's crossing disappears** (r=0.5 predicted speedup
   0.978→1.195 lifts the whole grid above 1) — a live instance of the
   `both_exist`→`mismatch_one_sided` reclassification fragility V3 §5 flagged; recorded
   as a CM4 hand-off (the tightened rule must handle it).
3. **The placement axis**: `b_fair` artifacts are serial-B measurements, so the new files
   predict with `b_placement="serial"`; the frozen V3 report computed overlapped — the
   numbers legitimately differ, and neither is "wrong."

`docs/v3-costmodel.md` is not modified (as-measured record); the note cites it.

**CI guard** — a new step in `.github/workflows/lint.yml` (pure grep, no build):

- Negative check: grep `bench/` `*.py` for the A-path roofline signatures (a `def cpu_bw`
  definition; the `min(<bw>, <h2d>/r)`-shaped pipelined composition). Exit-code
  discipline per the MLIR-free precedent (`build.yml` "Assert libreloc is MLIR-free"):
  grep's status must be exactly 1 — 0 (match found) fails the build with the offending
  lines printed, ≥2 (scan error) also fails; never `! grep`, which converts scan failure
  into a false PASS. Tuned against known-legitimate arithmetic: `make_calibration.py`'s
  measurement reductions, `hiding_model.py`'s R4 `m = hbm/bw`, `gates.py`'s threshold
  bars.
- Positive check: `figure_rstar.py` imports `pyreloc` (grep for the import line; its
  absence fails the step) — "prediction paths import the model from exactly one module."

**Verification, end-to-end**:

1. A comparison script (throwaway, in the PR verification notes — not committed) confirms
   the four new files' measured-side fields equal the originals'.
2. Full C++ suite + full pytest green; `v3_gate.py` CALIBRATION-REGEN ×2 + REPORT-REGEN
   PASS (this change touches neither calibrations, the C++ model, nor the frozen report).
3. Guard self-test: locally, the new lint step exits clean on the final tree; injecting a
   dummy `def cpu_bw` into a bench `.py` makes it fail (then remove the dummy).
4. `gates.py` R2-G5 on the original artifacts produces the same verdict as before.

**Acceptance mapping** (#111): one arithmetic, two consumers = internal model deleted +
figure_rstar consumes pyreloc (gate side already does); reconcile note merged =
`docs/cm3-one-implementation.md`; guard in CI = the lint.yml step.

## Amendment (2026-08-05, during implementation)

§2 point 2's Axis-2 example mixed the retired internal model's stored value (0.864,
`v1_gen4_rstar_bfair.json`'s stored `quant` `speedup_predicted["0.5"]` — Axis-1 material)
into what was meant to be a pyreloc-vs-pyreloc drift comparison; corrected to the verified
V3-era `pyreloc.predict` value 0.978 (`docs/v3-costmodel.md:386-387`, which reproduces the
frozen 0.4806 crossing), so the point now reads 0.978→1.195.
