# CM3 — Retire figure_rstar's Internal Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #111 (CM3): `figure_rstar.py` obtains predictions exclusively from `pyreloc.predict`; four stored r* artifacts regenerated as new `_pyreloc` files alongside the originals; a three-axis reconcile note; a CI grep guard against a second roofline formula under `bench/`.

**Architecture:** The spec is `docs/superpowers/specs/2026-08-05-cm3-one-implementation-design.md` — read it first. Four pieces: (1) script surgery — delete `cpu_bw`/`FAMILY_PLAN`/`load_rooflines`/`--rooflines` and the pred/serial lines, add `--calibration` + guarded pyreloc use + b_method→b_placement mapping, drop `speedup_serial`, make matplotlib optional; (2) regenerate 4 JSON+PNG pairs with `b_placement="serial"` (all four originals are `b_fair`); (3) `docs/cm3-one-implementation.md` reconcile note; (4) lint.yml guard job with exit-code-1 discipline.

**Tech Stack:** Python 3 (stdlib + pyreloc + matplotlib for figures), GitHub Actions YAML.

## Global Constraints

- The A/B cost model arithmetic must exist in exactly one place: `reloc::costmodel` (consumed via `pyreloc.predict`). No calibration, C++ model, or frozen-report changes anywhere in this plan — `v3_gate.py` CALIBRATION-REGEN ×2 and REPORT-REGEN must stay PASS untouched.
- The four original artifacts (`v1_gen4_rstar_bfair.json`, `v2_isa_rstar_avx2.json`, `v2_isa_rstar_avx512.json`, `v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json`) and `r2_rstar_gen4.json` are NEVER modified — new files land alongside with the `_pyreloc` suffix.
- Invariant on every regenerated file: `h2d_gbps`, `n`, `threads`, `b_method`, and per-family `speedup_measured`/`rstar_measured`/`unstable` are identical to the original's; `speedup_serial` is absent; only `speedup_predicted`/`rstar_predicted` differ.
- pyreloc lives at `PYTHONPATH=build/sym/python` (built via `ninja -C build/sym`). matplotlib is NOT installed system-wide on this box — Task 2 creates a venv for it.
- Grep-guard exit-code discipline (MLIR-free precedent, `build.yml`): grep status must be exactly 1; 0 = violation found (fail with lines printed); ≥2 = scan failed (fail); never `! grep`.
- Full suites must stay green: `build/sym/libreloc/test/libreloc-test`; `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q`.
- Branch: `cm3-one-implementation` (exists, spec committed). Deliverable: regular PR.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: figure_rstar.py surgery

**Files:**
- Modify: `bench/rtrack/figure_rstar.py`

**Interfaces:**
- Consumes: `pyreloc.load_calibration(path)` (raises ValueError with parser diagnostic), `pyreloc.predict(cal, pattern=, src_bytes=, r=, threads=, b_placement=)` → dict with `t_a_ms`/`t_b_ms`, raising ValueError on missing calibration keys.
- Produces: CLI `--calibration <path>` (replaces `--rooflines`); JSON schema per family = `{speedup_measured, speedup_predicted, rstar_measured, rstar_predicted, unstable}` (NO `speedup_serial`); module-level maps `FAMILY_PATTERN` and `B_METHOD_PLACEMENT` (Task 4's guard and Task 2's invocations rely on the CLI shape).

- [ ] **Step 1: Rewrite the module docstring** (lines 2-26) to:

```python
"""R2 / EXP-2 (issue #83): A/B speedup vs r per family + measured and
model-predicted critical r*.

  PYTHONPATH=build/sym/python python3 bench/rtrack/figure_rstar.py \
      --csv rsweep.csv [--calibration calibration/<machine>.cal] \
      [--h2d GBPS] [--n N] [--threads T] [--b-method b|b_fair] \
      [--out figure_rstar.png] [--json rstar.json]

Measured: variant=rsweep rows at --n/--threads (defaults: largest present
per family), best chunk per (method, r); speedup(r) = median_ms(B at
r=1.0) / median_ms(A at r). r*_measured = the 1.0 crossing, interpolated
linearly in log2(r); None when the curve never crosses in [0.125, 1.0].
--b-method picks the Method-B baseline rows (issue #95): "b" is the
staged baseline, "b_fair" the admissible pinned-source one; the choice is
recorded in the JSON as "b_method".

Predictions (issue #111): computed exclusively by pyreloc.predict -- the
maintained reloc::costmodel, the same C++ arithmetic decide() uses --
from the --calibration .cal file. speedup_predicted(r) =
t_b_pred(r=1.0) / t_a_pred(r), with b_placement mapped from --b-method
(b -> "overlapped", b_fair -> "serial": a serial-B measurement is
compared against a serial-B prediction, CM1's placement term). Without
--calibration the output is measured-only. The pre-#111 standalone
roofline model (and its A-side "serial bound" series, which has no
pyreloc counterpart) is retired; see docs/cm3-one-implementation.md.
"""
```

- [ ] **Step 2: Replace the imports/constants block** (lines 28-37):

```python
import argparse
import csv
import json
import math
import sys
from collections import defaultdict

try:
    import pyreloc  # the single cost-model implementation (issue #111)
except ImportError:
    pyreloc = None  # only fatal when --calibration asks for predictions

R_POINTS = [1.0, 0.5, 0.25, 0.125]
# transform family -> reloc::costmodel Pattern name (the same map
# libreloc/python/tests/test_prediction.py uses).
FAMILY_PATTERN = {"quant": "contiguous", "blocked_transpose": "blocked",
                  "transpose_quant": "single_element",
                  "nchw_nhwc_quant": "tiled"}
# --b-method -> pyreloc b_placement: measured-serial-B (b_fair) is
# compared against a serial-B prediction, staged b against overlapped.
B_METHOD_PLACEMENT = {"b": "overlapped", "b_fair": "serial"}
```

(`FAMILY_PLAN` is deleted.)

- [ ] **Step 3: Delete `load_rooflines` (lines 66-78) and `cpu_bw` (lines 81-97)** — `load_rows` and `crossing` stay unchanged.

- [ ] **Step 4: Update `main()`**:

1. argparse: delete the `--rooflines` line; add in its place:

```python
    ap.add_argument("--calibration", default=None,
                    help=".cal file for pyreloc.predict predictions "
                         "(issue #111); absent -> measured-only output")
```

2. After `args = ap.parse_args()`, load the calibration:

```python
    cal = None
    if args.calibration:
        if pyreloc is None:
            sys.exit("error: pyreloc not importable -- build it "
                     "(ninja -C build/sym) and run with "
                     "PYTHONPATH=build/sym/python")
        try:
            cal = pyreloc.load_calibration(args.calibration)
        except ValueError as e:
            sys.exit(f"error: {e}")
```

3. Delete `rooflines = load_rooflines(args.rooflines)` (line 138).

4. Replace the prediction block (lines 153-159, `pred, serial = {}, {}` through the `serial[rr] = ...` line) with:

```python
        pred = {}
        if cal is not None and fam in FAMILY_PATTERN:
            placement = B_METHOD_PLACEMENT[args.b_method]
            pattern = FAMILY_PATTERN[fam]
            try:
                t_b1 = pyreloc.predict(
                    cal, pattern=pattern, src_bytes=n * n * 4, r=1.0,
                    threads=threads, b_placement=placement)["t_b_ms"]
                for rr in sorted(meas):
                    t_a = pyreloc.predict(
                        cal, pattern=pattern, src_bytes=n * n * 4, r=rr,
                        threads=threads, b_placement=placement)["t_a_ms"]
                    pred[rr] = t_b1 / t_a
            except ValueError:
                # Missing calibration keys: the family is unmodelable --
                # omit ALL predicted points (test_prediction.py's
                # all-or-nothing modelable convention), never a partial
                # grid.
                pred = {}
```

5. In the `result["families"][fam]` dict (lines 161-168): delete the `"speedup_serial": ...` line. `"speedup_predicted"` and `"rstar_predicted"` stay as-is (empty dict / None when `pred` is empty — `crossing(sorted(pred.items()))` returns None on an empty list).

- [ ] **Step 5: Make matplotlib optional** — replace lines 177-179 with:

```python
    try:
        import matplotlib
    except ImportError:
        print("figure_rstar: matplotlib not available -- skipping figure",
              file=sys.stderr)
        return 0
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
```

and in the plot series tuple (lines 188-190), delete the `("speedup_serial", "^:", "serial bound")` entry, leaving measured + model.

- [ ] **Step 6: Behavior checks** (this script has no pytest; these runs are its test cycle):

```bash
# (a) measured-only mode: no --calibration -> JSON without predicted points
python3 bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv \
        bench/results/v2_isa_gen3_bfair_rsweep_epyc7351-2080ti.csv \
  --b-method b_fair --json /tmp/cm3_check_a.json --out /tmp/cm3_check_a.png
python3 - <<'EOF'
import json
d = json.load(open("/tmp/cm3_check_a.json"))
q = d["families"]["quant"]
assert q["speedup_predicted"] == {} and q["rstar_predicted"] is None
assert "speedup_serial" not in q
assert q["speedup_measured"]["0.5"] == 1.0601163292018307  # unchanged measured side
print("measured-only OK")
EOF
# (b) predictions via pyreloc (serial placement for b_fair):
PYTHONPATH=build/sym/python python3 bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv \
        bench/results/v2_isa_gen3_bfair_rsweep_epyc7351-2080ti.csv \
  --b-method b_fair --calibration calibration/epyc7351-2080ti.cal \
  --json /tmp/cm3_check_b.json --out /tmp/cm3_check_b.png
python3 - <<'EOF'
import json
d = json.load(open("/tmp/cm3_check_b.json"))
q = d["families"]["quant"]
assert q["speedup_predicted"], "expected predicted points"
assert "speedup_serial" not in q
print("pyreloc predictions OK:", q["speedup_predicted"], q["rstar_predicted"])
EOF
# (c) --calibration without pyreloc on PYTHONPATH -> clean error:
python3 bench/rtrack/figure_rstar.py --csv bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv \
  --b-method b_fair --calibration calibration/epyc7351-2080ti.cal --json /tmp/x.json 2>&1 | grep "pyreloc not importable"
```

Expected: (a) prints "measured-only OK" and the script warns matplotlib is missing but still exits 0 with the JSON written; (b) prints predicted points computed under the serial placement (record them — Task 2/3 will see the same numbers in the regenerated gen3 file; note serial raises t_b(r=1), so these speedups sit ABOVE the overlapped ones, e.g. above 1.1072 at r=0.5); (c) prints the error line.

- [ ] **Step 7: Commit**

```bash
git add bench/rtrack/figure_rstar.py
git commit -m "feat(bench): figure_rstar predictions come exclusively from pyreloc.predict (#111)"
```

---

### Task 2: Regenerate the four artifacts

**Files:**
- Create: `bench/results/{v1_gen4_rstar_bfair,v2_isa_rstar_avx2,v2_isa_rstar_avx512,v2_isa_gen3_rstar_avx2_epyc7351-2080ti}_pyreloc.json` + matching `*_pyreloc.png` figures (names below)

**Interfaces:**
- Consumes: Task 1's CLI (`--calibration`, b_fair→serial mapping).
- Produces: the four `_pyreloc` artifacts Task 3's reconcile note quotes.

- [ ] **Step 1: matplotlib venv** (not installed system-wide; the venv's python sees pyreloc via PYTHONPATH):

```bash
python3 -m venv build/cm1-tools/plot 2>/dev/null; build/cm1-tools/plot/bin/pip install -q matplotlib
PY=build/cm1-tools/plot/bin/python3
```

- [ ] **Step 2: Regenerate** (input CSVs verified: `v1_gen4_rsweep_7800x3d_4070tis.csv` carries a/b/b_fair rows itself; the v2 gen4 arms pair each avx CSV with the shared bfair CSV; gen3 pairs per the committed invocation in `docs/v2-isolation.md:152-156`):

```bash
export PYTHONPATH=build/sym/python
$PY bench/rtrack/figure_rstar.py \
  --csv bench/results/v1_gen4_rsweep_7800x3d_4070tis.csv \
  --b-method b_fair --calibration calibration/7800x3d-4070tis.cal \
  --json bench/results/v1_gen4_rstar_bfair_pyreloc.json \
  --out bench/results/v1_gen4_figure_rstar_bfair_pyreloc.png
$PY bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_rsweep_avx2_7800x3d_4070tis.csv \
        bench/results/v2_isa_bfair_rsweep_7800x3d_4070tis.csv \
  --b-method b_fair --calibration calibration/7800x3d-4070tis.cal \
  --json bench/results/v2_isa_rstar_avx2_pyreloc.json \
  --out bench/results/v2_isa_figure_rstar_avx2_pyreloc.png
$PY bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_rsweep_avx512_7800x3d_4070tis.csv \
        bench/results/v2_isa_bfair_rsweep_7800x3d_4070tis.csv \
  --b-method b_fair --calibration calibration/7800x3d-4070tis.cal \
  --json bench/results/v2_isa_rstar_avx512_pyreloc.json \
  --out bench/results/v2_isa_figure_rstar_avx512_pyreloc.png
$PY bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv \
        bench/results/v2_isa_gen3_bfair_rsweep_epyc7351-2080ti.csv \
  --b-method b_fair --calibration calibration/epyc7351-2080ti.cal \
  --json bench/results/v2_isa_gen3_rstar_avx2_epyc7351-2080ti_pyreloc.json \
  --out bench/results/v2_isa_gen3_figure_rstar_avx2_epyc7351-2080ti_pyreloc.png
```

- [ ] **Step 3: Verify the measured-side invariant against every original** (the decisive check for correct input CSVs — a mismatch means the wrong CSV set, NOT a tolerance to accept; on mismatch for the v1_gen4 pair, retry it with `--csv .../v1_gen4_rsweep_7800x3d_4070tis.csv .../v1_gen4_rsweep_rerun_7800x3d_4070tis.csv` and re-verify):

```bash
python3 - <<'EOF'
import json
pairs = [("v1_gen4_rstar_bfair", "v1_gen4_rstar_bfair_pyreloc"),
         ("v2_isa_rstar_avx2", "v2_isa_rstar_avx2_pyreloc"),
         ("v2_isa_rstar_avx512", "v2_isa_rstar_avx512_pyreloc"),
         ("v2_isa_gen3_rstar_avx2_epyc7351-2080ti",
          "v2_isa_gen3_rstar_avx2_epyc7351-2080ti_pyreloc")]
for orig, new in pairs:
    a = json.load(open(f"bench/results/{orig}.json"))
    b = json.load(open(f"bench/results/{new}.json"))
    for k in ("h2d_gbps", "n", "threads", "b_method"):
        assert a[k] == b[k], (orig, k, a[k], b[k])
    assert sorted(a["families"]) == sorted(b["families"]), orig
    for fam, d in a["families"].items():
        e = b["families"][fam]
        assert d["speedup_measured"] == e["speedup_measured"], (orig, fam)
        assert d["rstar_measured"] == e["rstar_measured"], (orig, fam)
        assert d["unstable"] == e["unstable"], (orig, fam)
        assert "speedup_serial" not in e, (orig, fam)
    print(f"{orig}: measured side identical; new rstar_predicted =",
          {f: b["families"][f]["rstar_predicted"] for f in sorted(b["families"])})
EOF
```

Expected: four "measured side identical" lines. Record the printed `rstar_predicted` values verbatim in your report — Task 3's tables need them.

- [ ] **Step 4: Confirm originals untouched + suites green**

```bash
git status --short bench/results/ | grep -v '_pyreloc'   # MUST print nothing
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
```

Expected: no non-`_pyreloc` changes; all suites green; REGEN all PASS.

- [ ] **Step 5: Commit (8 new files)**

```bash
git add bench/results/*_pyreloc.json bench/results/*_pyreloc.png
git commit -m "bench(results): regenerate r* artifacts with the maintained model, serial placement (#111)"
```

---

### Task 3: Reconcile note

**Files:**
- Create: `docs/cm3-one-implementation.md`

**Interfaces:**
- Consumes: Task 2's regenerated values (from its report / the committed JSONs).
- Produces: the note the PR and #111 acceptance reference.

- [ ] **Step 1: Write the note** with this exact structure (fill the `<new>` cells from the regenerated JSONs — never leave them symbolic):

```markdown
# CM3 — one cost-model implementation (issue #111)

`bench/rtrack/figure_rstar.py`'s standalone roofline model is retired; predictions now
come exclusively from `pyreloc.predict` (`reloc::costmodel`, the implementation
`decide()` uses). The four `b_fair` r* artifacts are regenerated alongside their
originals with the `_pyreloc` suffix; originals are unchanged (`gates.py`'s R2-G5 keeps
reading them; `r2_rstar_gen4.json` is the pre-`b_fair` R2-era record and is not
regenerated). Three distinct effects separate the old stored predictions from the new
files — conflating them would misattribute the change:

## Axis 1 — V3-era implementation divergence (the issue's "1.6–2.9×")

`figure_rstar.py`'s internal model had no intercepts (`overhead.{a,b}_ms`), no B-side
HBM term (`hbm.m.*`, CM1 #109), and no K/broadcast handling. Documented at
`docs/v3-costmodel.md` "Two-sided implementation note": stored gen4
`rstar_predicted` 0.1807 (blocked_transpose) / 0.1640 (quant) vs the V3-era
`pyreloc.predict` 0.2914 / 0.4806 on the same family/box.

## Axis 2 — model drift since V3 (CM1 + CM2)

The maintained model itself moved after the V3 snapshot: CM2's
`cpu_pipe.t8.contiguous.convert_f32_f16_gbps` keys shift every contiguous r=0.5
prediction (gen3 quant overlapped r* 0.9989 → 0.9966), and on gen4 the quant
crossing **disappears entirely** (r=0.5 predicted speedup 0.864 → 1.195 lifts the
whole grid above 1.0 → `rstar_predicted = None`). That is a live instance of the
`both_exist` → `mismatch_one_sided` reclassification fragility `docs/v3-costmodel.md`
§5 flagged; **CM4's tightened RSTAR rule must handle it** (hand-off).

## Axis 3 — the placement axis (new in these files)

All four artifacts were measured with `--b-method b_fair` — serial-B. The new files
therefore predict with `b_placement="serial"` (CM1's term; `--b-method` maps
b→overlapped, b_fair→serial). The frozen `v3_prediction_report.json` computed the
same families with the overlapped default. The numbers legitimately differ; neither
is wrong — they answer different questions (which B implementation is being priced).

## Old → new, per artifact (rstar_predicted)

| artifact | family | stored (internal model) | new `_pyreloc` (serial) | rstar_measured |
|---|---|---|---|---|
| v1_gen4_rstar_bfair | blocked_transpose | 0.180653 | <new> | 0.373802 |
| v1_gen4_rstar_bfair | quant | 0.164007 | <new> | 0.540927 |
| v2_isa_rstar_avx2 | quant | 0.497078 | <new> | 0.579018 |
| v2_isa_rstar_avx512 | blocked_transpose | 0.333408 | <new> | 0.395783 |
| v2_isa_rstar_avx512 | quant | 1.0 | <new> | 0.597010 |
| v2_isa_gen3_rstar_avx2_epyc… | quant | 1.0 | <new> | 0.635650 |

(Families whose stored and new values are both None are omitted. `speedup_serial` —
the internal model's A-side serial bound — has no pyreloc counterpart and is absent
from the new files; CM1's `Serial` placement serializes the *B* side.)

Also note: the gen4 arms (v1/avx2/avx512) now share one prediction per family — the
maintained model prices the box (its calibration), not the per-artifact roofline
JSONs the internal model consumed. Per-arm measured curves still differ.
```

- [ ] **Step 2: Cross-check every number** you filled against the committed `_pyreloc` JSONs and the originals (open them; do not trust the Task 2 report alone). The stored/measured columns above were pre-extracted from the originals — verify at least two spot values.

- [ ] **Step 3: Commit**

```bash
git add docs/cm3-one-implementation.md
git commit -m "docs: CM3 reconcile note — three axes separating old and new r* predictions (#111)"
```

---

### Task 4: CI guard + PR

**Files:**
- Modify: `.github/workflows/lint.yml`

**Interfaces:**
- Consumes: Task 1's final `figure_rstar.py` (must contain `import pyreloc`, must not contain the retired signatures).
- Produces: the CI guard; the PR.

- [ ] **Step 1: Add a new job to `.github/workflows/lint.yml`** (after the `clang-format` job, same indentation level):

```yaml
  cost-model-single-implementation:
    name: cost-model single implementation
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v5

      - name: Assert no second roofline formula under bench/
        run: |
          # Issue #111: the A/B cost model lives ONLY in reloc::costmodel
          # (consumed via pyreloc.predict). Signatures of the retired
          # figure_rstar internal model must not reappear in bench/
          # python. grep's status must be exactly 1: 0 means a duplicate
          # model was found, >=2 means the scan itself failed. Never
          # `! grep` -- that converts scan failure into a false PASS
          # (see the MLIR-free guard in build.yml).
          set +e
          grep -rEn --include='*.py' \
            'def cpu_bw\(|min\(\s*bw\s*,|1\.0\s*/\s*\(\s*1\.0\s*/\s*bw' \
            bench/
          scan=$?
          set -e
          if [ "$scan" -eq 0 ]; then
            echo "ERROR: a second cost-model implementation appeared under bench/ (issue #111)" >&2
            exit 1
          elif [ "$scan" -ne 1 ]; then
            echo "ERROR: scan failed (exit $scan; missing dir?)" >&2
            exit 1
          fi

      - name: Assert figure_rstar consumes pyreloc
        run: |
          # The prediction path imports the model from exactly one module.
          grep -q 'import pyreloc' bench/rtrack/figure_rstar.py || {
            echo "ERROR: bench/rtrack/figure_rstar.py no longer imports pyreloc (issue #111)" >&2
            exit 1
          }
```

- [ ] **Step 2: Guard self-test, locally**

```bash
# clean tree passes:
bash -c "set +e; grep -rEn --include='*.py' 'def cpu_bw\(|min\(\s*bw\s*,|1\.0\s*/\s*\(\s*1\.0\s*/\s*bw' bench/; echo scan=\$?"
# expected: no output, scan=1
# injected violation is caught:
echo 'def cpu_bw(k, f, r): pass' >> bench/rtrack/gates.py
bash -c "grep -rEn --include='*.py' 'def cpu_bw\(' bench/; echo scan=\$?"
# expected: the injected line printed, scan=0
git checkout bench/rtrack/gates.py   # remove the injection
grep -q 'import pyreloc' bench/rtrack/figure_rstar.py && echo positive-check-ok
```

- [ ] **Step 3: Final verification + commit**

```bash
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
git status --short   # clean (untracked .claude/ is pre-existing)
git add .github/workflows/lint.yml
git commit -m "ci(lint): guard -- no second cost-model implementation under bench/ (#111)"
```

- [ ] **Step 4: Push and open the PR (regular)**

```bash
git push -u origin cm3-one-implementation
gh pr create --title "feat(bench, ci): CM3 — one cost-model implementation, figure_rstar via pyreloc (#111)" --body "<body>"
```

PR body, in order: (1) verdict-first: internal model deleted, predictions exclusively via `pyreloc.predict`, 4 artifacts regenerated alongside originals (serial placement per b_fair mapping), CI guard live; all suites green, REGEN checks PASS, originals byte-untouched; (2) the three-axis reconcile summary with a link to `docs/cm3-one-implementation.md` — call out the gen4 quant crossing disappearance as a CM4 hand-off; (3) what changed per file; (4) notes: `gates.py` R2-G5 unaffected (reads originals), `r2_rstar_gen4.json` left as the R2-era record, `speedup_serial` retired with rationale; (5) `Refs #111, #107`; (6) the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. Task 2 Step 3's invariant script: four "measured side identical" lines.
2. `git diff main -- bench/results/ | grep '^---\|^+++' | grep -v _pyreloc` shows nothing (originals untouched; only new files).
3. Both suites green; `v3_gate.py` CALIBRATION-REGEN ×2 + REPORT-REGEN PASS (nothing this plan touches feeds them).
4. Guard self-test: clean scan=1; injected `def cpu_bw` caught (scan=0); positive pyreloc-import check passes.
5. Issue #111 acceptance: one arithmetic two consumers = internal model deleted + figure path imports pyreloc (gate path already did); reconcile note merged = `docs/cm3-one-implementation.md`; guard in CI = the lint.yml job.
