# CM4 — Tightened RSTAR Rule + Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #112 (CM4): tightened RSTAR rule in `v3_gate.py`, deterministic registered per-cell predictions for the 48-cell BP matrix (`cm4_register.py` → `cm4_registered_predictions.json` with embedded held-out split), the pre-declared bimodal cell list, and a CM4-REGEN check — all merged before any `bench/results/bp_*` artifact exists.

**Architecture:** The spec is `docs/superpowers/specs/2026-08-05-cm4-registration-design.md` — read it first. Four pieces: (1) `gate_rstar` tightening (mismatch_one_sided ⇒ automatic FAIL, contributing bar-value 0.15 to the reported max; "rule v1" label; §5 rationale in the header); (2) `bench/rtrack/cm4_register.py` generating the registration JSON from pyreloc + committed calibrations; (3) hand-authored `cm4_bimodal_cells.json` with R3/V4 evidence; (4) CM4-REGEN check in `v3_gate.py` + PR.

**Tech Stack:** Python 3 (v3_gate.py, new bench script), pyreloc via `PYTHONPATH=build/sym/python`.

## Global Constraints

- Bars unchanged: `MISCLASS_BAR = 0.15`, `RSTAR_ABS_BAR = 0.15`, `REGRET_P90_BAR = 0.20` (v3_gate.py:45-47 stay byte-identical).
- Frozen artifacts untouched: `bench/results/v3_prediction_report.json`, both `calibration/*.cal`, all prior bench/results files. `docs/v3-costmodel.md` is not modified.
- Known fact (verified pre-plan): the frozen v0 report has ZERO `mismatch_one_sided` rows (agreements: 3× both_exist, 3× both_no_crossing), so the tightened gate prints identical verdicts on it — the PR discloses exactly this.
- The registration JSON must be byte-deterministic: sorted iteration order, `json.dump(..., indent=2)`, no timestamps, values straight from pyreloc.
- pyreloc: `PYTHONPATH=build/sym/python`; suites: `build/sym/libreloc/test/libreloc-test`, `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q`; gate: `python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json`.
- Branch: `cm4-registration` (exists, spec committed). Deliverable: regular PR, merged while no `bench/results/bp_*` exists (commit-order acceptance).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: gate_rstar tightened rule (rule v1)

**Files:**
- Modify: `bench/rtrack/v3_gate.py` (docstring lines 2-36; `gate_rstar` lines 92-116)

**Interfaces:**
- Consumes: the report JSON schema (`rstar` rows with `agreement` ∈ {both_exist, both_no_crossing, mismatch_one_sided} and `abs_diff` on both_exist rows).
- Produces: `gate_rstar` under rule v1 — Tasks 2-4 don't call it, but Task 4's PR discloses its v0-report output.

- [ ] **Step 1: Update the module docstring** — replace the `RSTAR_ABS_BAR` description block (lines 12-15) with:

```
  RSTAR_ABS_BAR = 0.15   |r*_pred - r*_meas| over families where BOTH a
                         measured and a predicted r* exist. RULE v1
                         (CM4, issue #112, discharging v3-costmodel.md
                         S5's obligation): `mismatch_one_sided` rows
                         (one side has a crossing, the other doesn't)
                         are an AUTOMATIC FAIL contribution counted at
                         the bar value -- S5: "treating
                         `mismatch_one_sided` as an automatic FAIL
                         contribution rather than an exclusion" -- so a
                         more-wrong prediction can no longer escape the
                         bar by reclassification. `both_no_crossing`
                         remains a qualitative PASS outcome (a correctly
                         predicted "no crossing"). The v0 verdicts in
                         docs/v3-costmodel.md stand as recorded; this
                         rule applies to every run from CM4 onward.
```

- [ ] **Step 2: Rewrite `gate_rstar`** (keep the table printing; change the statistic and verdict):

```python
def gate_rstar(report):
    rows = report.get("rstar", [])
    both = [r for r in rows if r.get("agreement") == "both_exist"]
    no_crossing = [r for r in rows if r.get("agreement") == "both_no_crossing"]
    mismatched = [r for r in rows if r.get("agreement") == "mismatch_one_sided"]
    # Rule v1 (CM4, #112): every one-sided mismatch contributes the bar
    # value to the reported statistic AND forces the verdict to FAIL --
    # v3-costmodel.md S5's tightening, registered before any bp_* data.
    diffs = [r["abs_diff"] for r in both] + [RSTAR_ABS_BAR for _ in mismatched]
    worst = max(diffs) if diffs else None
    if worst is None:
        verdict = "no data"
    elif mismatched or worst > RSTAR_ABS_BAR:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    print(f"\nV3-RSTAR [rule v1, CM4-tightened, v3-costmodel.md S5]: "
          f"max |r*_pred - r*_meas| over {len(both)} both-exist + "
          f"{len(mismatched)} one-sided-mismatch families = "
          f"{'--' if worst is None else f'{worst:.4f}'} "
          f"(bar <= {RSTAR_ABS_BAR}; any one-sided mismatch => FAIL)  "
          f"{verdict}")
    print(f"  (+ {len(no_crossing)} family/box pairs agree on 'no crossing' "
          "-- correct winner-shaped outcome, not counted in the bar above)")
    print("\n| family | machine | source | r*_meas | r*_pred | agreement |")
    print("|---" * 6 + "|")
    for r in rows:
        meas = "--" if r["rstar_measured"] is None else f"{r['rstar_measured']:.4f}"
        pred = "--" if r["rstar_predicted"] is None else f"{r['rstar_predicted']:.4f}"
        agreement = r.get("agreement", "unmodelable")
        print(f"| {r['family']} | {r['machine']} | {r['source']} | {meas} "
              f"| {pred} | {agreement} |")
    return worst, verdict
```

- [ ] **Step 3: Behavior checks** (v3_gate has no pytest; a synthetic report is the test):

```bash
python3 - <<'EOF'
import json, pathlib
pathlib.Path("build/cm1-tools").mkdir(parents=True, exist_ok=True)
synth = {
  "cells": [], "rstar": [
    {"family": "f1", "machine": "m", "source": "s", "modelable": True,
     "rstar_measured": 0.5, "rstar_predicted": 0.55,
     "agreement": "both_exist", "abs_diff": 0.05},
    {"family": "f2", "machine": "m", "source": "s", "modelable": True,
     "rstar_measured": 0.6, "rstar_predicted": None,
     "agreement": "mismatch_one_sided"},
  ], "ablation": {}, "summary": {}}
pathlib.Path("build/cm1-tools/cm4_synth_report.json").write_text(json.dumps(synth))
EOF
python3 bench/rtrack/v3_gate.py --report build/cm1-tools/cm4_synth_report.json 2>&1 | grep 'V3-RSTAR'
# Expected: "... 1 both-exist + 1 one-sided-mismatch families = 0.1500 (bar <= 0.15; any one-sided mismatch => FAIL)  FAIL"
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'V3-(MISCLASS|RSTAR|REGRET).*(PASS|FAIL)|REGEN.*(PASS|FAIL)'
# Expected: MISCLASS PASS, RSTAR FAIL 0.3632 (v0 report has zero mismatch rows -- same verdict as v0, now labeled rule v1), REGRET PASS, CALIBRATION-REGEN PASS x2, REPORT-REGEN PASS.
```

Also verify the synthetic FAIL case with the mismatch row REMOVED prints PASS (0.0500) — confirming the rule fires on the mismatch, not on the both_exist row:

```bash
python3 - <<'EOF'
import json, pathlib
d = json.loads(pathlib.Path("build/cm1-tools/cm4_synth_report.json").read_text())
d["rstar"] = [r for r in d["rstar"] if r["agreement"] == "both_exist"]
pathlib.Path("build/cm1-tools/cm4_synth_pass.json").write_text(json.dumps(d))
EOF
python3 bench/rtrack/v3_gate.py --report build/cm1-tools/cm4_synth_pass.json 2>&1 | grep 'V3-RSTAR'
# Expected: "... 1 both-exist + 0 one-sided-mismatch families = 0.0500 ... PASS"
```

- [ ] **Step 4: Commit**

```bash
git add bench/rtrack/v3_gate.py
git commit -m "feat(bench): RSTAR rule v1 -- mismatch_one_sided is an automatic FAIL at bar value (#112)"
```

---

### Task 2: cm4_register.py + registered predictions + CM4-REGEN

**Files:**
- Create: `bench/rtrack/cm4_register.py`
- Create (generated): `bench/results/cm4_registered_predictions.json`
- Modify: `bench/rtrack/v3_gate.py` (add `cm4_regen_check()` after `calibration_regen_check`, call it in `main()`)

**Interfaces:**
- Consumes: `pyreloc.load_calibration(path)`, `pyreloc.predict(cal, pattern=, src_bytes=, r=, threads=, b_placement=)` → dict with `method`/`t_a_ms`/`t_b_ms`/`threshold_bytes`, raising ValueError on missing keys.
- Produces: `cm4_register.py --out PATH` (deterministic); the committed registration JSON #BP2 cross-links and #CM5 evaluates against; `cm4_regen_check()` in the gate.

- [ ] **Step 1: Write `bench/rtrack/cm4_register.py`**:

```python
#!/usr/bin/env python3
"""CM4 (issue #112): register the fixed model's per-cell predictions for
the BP matrix -- both boxes, both B placements -- BEFORE any
bench/results/bp_* artifact exists. #CM5 evaluates the BP measurement
against THIS file only; #BP2 cross-links it. Deterministic: values come
straight from pyreloc.predict over the committed calibrations, iteration
order is sorted, no timestamps -- v3_gate.py's CM4-REGEN check re-runs
this script and byte-compares.

  PYTHONPATH=build/sym/python python3 bench/rtrack/cm4_register.py \
      --out bench/results/cm4_registered_predictions.json

Held-out split (registered here per #87's reconciliation in #107):
train N in {2048, 8192} -> test N in {4096, 16384}; held-out bars
misclass <= 0.15, regret p90 <= 0.20. The split stratifies EVALUATION:
the committed calibrations' overhead.{a,b}_ms intercepts were
two-point-fit on N in {2048, 16384} endpoints, so test-N data touched
calibration inputs -- recorded honestly in the output's caveat field,
not silently ignored; bars unchanged.
"""
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import pyreloc
except ImportError:
    sys.exit("error: pyreloc not importable -- build it (ninja -C build/sym) "
             "and run with PYTHONPATH=build/sym/python")

REPO_ROOT = Path(__file__).resolve().parents[2]

# transform -> (native r, pattern); the same map test_prediction.py and
# figure_rstar.py use.
FAMILY_MAP = {
    "transpose": (1.0, "single_element"),
    "blocked_transpose": (1.0, "blocked"),
    "transpose_quant": (0.25, "single_element"),
    "nchw_nhwc_quant": (0.25, "tiled"),
    "quant": (0.25, "contiguous"),
    "convert_f16": (0.5, "contiguous"),
}
NS = [2048, 4096, 8192, 16384]
TRAIN_N = [2048, 8192]
TEST_N = [4096, 16384]
R_POINTS = [1.0, 0.5, 0.25, 0.125]  # the measured r-sweep grid
RSTAR_N = 16384                     # r-sweeps are measured at N=16384
THREADS = 8
PLACEMENTS = ("serial", "overlapped")  # b_fair <-> serial, b_pipelined <-> overlapped
BOXES = {
    "epyc7351-2080ti": "calibration/epyc7351-2080ti.cal",
    "7800x3d-4070tis": "calibration/7800x3d-4070tis.cal",
}


def crossing(points):
    """points: [(r, speedup)] sorted ascending in r -> r* or None.
    Verbatim from bench/rtrack/figure_rstar.py (the shared interpolation,
    linear in log2(r))."""
    for (r0, s0), (r1, s1) in zip(points, points[1:]):
        if (s0 - 1.0) * (s1 - 1.0) <= 0 and s0 != s1:
            t = (1.0 - s0) / (s1 - s0)
            return 2 ** (math.log2(r0) + t * (math.log2(r1) - math.log2(r0)))
    return None


def predict(cal, pattern, src_bytes, r, placement):
    """pyreloc.predict, or None on the missing-key ValueError (the
    unmodelable signal -- recorded as explicit nulls, never invented)."""
    try:
        return pyreloc.predict(cal, pattern=pattern, src_bytes=src_bytes,
                               r=r, threads=THREADS, b_placement=placement)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells, rstar = [], []
    for box, cal_rel in sorted(BOXES.items()):
        cal = pyreloc.load_calibration(str(REPO_ROOT / cal_rel))
        for fam in sorted(FAMILY_MAP):
            r_native, pattern = FAMILY_MAP[fam]
            for n in NS:
                src = n * n * 4
                cell = {"box": box, "family": fam, "N": n,
                        "pattern": pattern, "r_native": r_native,
                        "src_bytes": src,
                        "split": "train" if n in TRAIN_N else "test"}
                ps = predict(cal, pattern, src, r_native, "serial")
                po = predict(cal, pattern, src, r_native, "overlapped")
                if ps is None or po is None:
                    cell.update(t_a_ms=None, t_b_serial_ms=None,
                                t_b_overlapped_ms=None,
                                threshold_bytes_serial=None,
                                threshold_bytes_overlapped=None,
                                winner_vs_b_fair=None,
                                winner_vs_b_pipelined=None,
                                reason="missing calibration keys")
                else:
                    if ps["t_a_ms"] != po["t_a_ms"]:
                        sys.exit(f"error: t_a differs across placements at "
                                 f"{box}/{fam}/N={n} -- placement must only "
                                 "touch the B side")
                    cell.update(
                        t_a_ms=ps["t_a_ms"],
                        t_b_serial_ms=ps["t_b_ms"],
                        t_b_overlapped_ms=po["t_b_ms"],
                        threshold_bytes_serial=ps["threshold_bytes"],
                        threshold_bytes_overlapped=po["threshold_bytes"],
                        winner_vs_b_fair=(
                            "a" if ps["method"] in ("a", "a_prefold") else "b"),
                        winner_vs_b_pipelined=(
                            "a" if po["method"] in ("a", "a_prefold") else "b"))
                cells.append(cell)

            src = RSTAR_N * RSTAR_N * 4
            for placement in PLACEMENTS:
                pb1 = predict(cal, pattern, src, 1.0, placement)
                grid = {}
                for r in R_POINTS:
                    if pb1 is None:
                        grid[str(r)] = None
                        continue
                    pr = predict(cal, pattern, src, r, placement)
                    grid[str(r)] = (None if pr is None
                                    else pb1["t_b_ms"] / pr["t_a_ms"])
                pts = sorted((float(k), v) for k, v in grid.items()
                             if v is not None)
                rstar.append({"box": box, "family": fam,
                              "placement": placement, "n": RSTAR_N,
                              "speedup_predicted": grid,
                              "grid_used": [p[0] for p in pts],
                              "rstar_predicted": crossing(pts)})

    doc = {
        "generated_by": "bench/rtrack/cm4_register.py",
        "issue": "#112 (CM4); evaluated by #CM5; cross-linked by #BP2 (#115)",
        "model": "reloc::costmodel as of CM1(#109)+CM2(#110)+CM3(#111)",
        "calibrations": sorted(BOXES.values()),
        "placement_map": {"b_fair": "serial", "b_pipelined": "overlapped"},
        "heldout_split": {
            "train_n": TRAIN_N, "test_n": TEST_N,
            "heldout_bars": {"misclass": 0.15, "regret_p90": 0.20},
            "caveat": ("the committed calibrations' overhead.{a,b}_ms "
                       "intercepts were two-point-fit on N in {2048, 16384} "
                       "endpoints, so test-N data touched calibration "
                       "inputs; the split stratifies evaluation (per #87's "
                       "reconciliation in #107), not calibration -- "
                       "recorded, not fixed; bars unchanged"),
        },
        "cells": cells,
        "rstar": rstar,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate and sanity-check**

```bash
PYTHONPATH=build/sym/python python3 bench/rtrack/cm4_register.py \
  --out bench/results/cm4_registered_predictions.json
PYTHONPATH=build/sym/python python3 - <<'EOF'
import json
d = json.load(open("bench/results/cm4_registered_predictions.json"))
cells, rstar = d["cells"], d["rstar"]
assert len(cells) == 48, len(cells)                      # 2 boxes x 6 families x 4 N
assert len(rstar) == 24, len(rstar)                      # 2 x 6 x 2 placements
assert sum(1 for c in cells if c["t_a_ms"] is None) == 0  # all r_native tiers modelable
assert {c["split"] for c in cells if c["N"] in (2048, 8192)} == {"train"}
assert {c["split"] for c in cells if c["N"] in (4096, 16384)} == {"test"}
# epyc has no pack_s8_s4 keys: its r=0.125 grid points must be null.
epyc_r0125 = [r["speedup_predicted"]["0.125"] for r in rstar
              if r["box"] == "epyc7351-2080ti"]
assert all(v is None for v in epyc_r0125), epyc_r0125
gen4_r0125 = [r["speedup_predicted"]["0.125"] for r in rstar
              if r["box"] == "7800x3d-4070tis"]
assert all(v is not None for v in gen4_r0125)
print("48 cells / 24 rstar rows OK; epyc 0.125 nulls OK")
# Spot-check one cell per box against direct predict:
import pyreloc
for box, cal_path in [("epyc7351-2080ti", "calibration/epyc7351-2080ti.cal"),
                      ("7800x3d-4070tis", "calibration/7800x3d-4070tis.cal")]:
    cal = pyreloc.load_calibration(cal_path)
    c = next(x for x in cells if x["box"] == box and x["family"] == "quant"
             and x["N"] == 16384)
    p = pyreloc.predict(cal, pattern="contiguous", src_bytes=16384**2*4,
                        r=0.25, threads=8, b_placement="serial")
    assert c["t_a_ms"] == p["t_a_ms"] and c["t_b_serial_ms"] == p["t_b_ms"]
    print(box, "spot-check OK:", c["winner_vs_b_fair"], c["winner_vs_b_pipelined"])
EOF
# Determinism: regenerate and byte-compare.
PYTHONPATH=build/sym/python python3 bench/rtrack/cm4_register.py --out build/cm1-tools/cm4_regen.json
cmp bench/results/cm4_registered_predictions.json build/cm1-tools/cm4_regen.json && echo deterministic-OK
```

- [ ] **Step 3: Add `cm4_regen_check()` to `v3_gate.py`** (after `calibration_regen_check`, called in `main()` between it and `report_regen_check`):

```python
def cm4_regen_check():
    """CM4-REGEN (issue #112): the registered predictions must stay
    byte-reproducible from the committed model + calibrations. Re-runs
    cm4_register.py to a temp path and compares against the committed
    file. Skips loudly (not silently) when the file or pyreloc is
    absent -- e.g. on a checkout predating CM4."""
    print("\n=== CM4-REGEN ===")
    committed = REPO_ROOT / "bench" / "results" / "cm4_registered_predictions.json"
    if not committed.exists():
        print("CM4-REGEN: SKIP (no committed registration file)")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cm4.json"
        proc = subprocess.run(
            [sys.executable, "bench/rtrack/cm4_register.py",
             "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"CM4-REGEN: FAIL (regen errored)\n{proc.stderr.strip()}")
            return
        if out.read_text() == committed.read_text():
            print("CM4-REGEN: PASS (matches committed "
                  "bench/results/cm4_registered_predictions.json)")
        else:
            print("CM4-REGEN: FAIL (registration no longer reproduces from "
                  "the committed model + calibrations)")
```

Note: the subprocess inherits the caller's environment, so run the gate with `PYTHONPATH=build/sym/python` set (the check FAILs loudly with the pyreloc error otherwise — acceptable; the error text says exactly what to do).

- [ ] **Step 4: Run the gate + suites**

```bash
PYTHONPATH=build/sym/python python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL|SKIP)'
# Expected: CALIBRATION-REGEN PASS x2, CM4-REGEN PASS, REPORT-REGEN PASS
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
git status --short bench/results/ | grep -v cm4_   # MUST print nothing
```

- [ ] **Step 5: Commit**

```bash
git add bench/rtrack/cm4_register.py bench/results/cm4_registered_predictions.json bench/rtrack/v3_gate.py
git commit -m "feat(bench): CM4 registered predictions -- 48-cell BP matrix, both placements + CM4-REGEN (#112)"
```

---

### Task 3: Pre-declared bimodal cell list

**Files:**
- Create: `bench/results/cm4_bimodal_cells.json` (hand-authored — deliberately OUTSIDE the CM4-REGEN check: it is a declaration citing history, not a generated artifact)

**Interfaces:**
- Consumes: committed R3 artifacts (`bench/results/r3_scatter_n{8192,16384}_epyc_2080ti.json`, `bench/results/r3_broadcast_n8192_epyc_2080ti.json`) and `docs/v4-prefold.md:100-121`.
- Produces: the list #BP2's matched-percentile rule applies to.

- [ ] **Step 1: Enumerate every committed `method=bxk, K=4` row** and record its actual stats (values from the artifacts, never invented):

```bash
python3 - <<'EOF'
import json, glob
for p in sorted(glob.glob("bench/results/r3_*_epyc_2080ti.json") +
                glob.glob("bench/results/v5_*_epyc_2080ti.json")):
    d = json.load(open(p))
    cfg = d.get("config", {})
    for r in d.get("rows", []):
        if r.get("method") == "bxk" and r.get("K") == 4:
            print(p, cfg.get("scenario"), cfg.get("N"),
                  {k: r[k] for k in ("wall_min_ms", "wall_median_ms", "wall_p95_ms")
                   if k in r})
EOF
```

Record the output verbatim in your report — it populates Step 2's `cells` array (one entry per printed row).

- [ ] **Step 2: Write `bench/results/cm4_bimodal_cells.json`** with this structure (fill `cells` from Step 1's actual numbers; the scatter-N=8192 entry shown here carries the known values and MUST match what Step 1 printed):

```json
{
  "issue": "#112 (CM4), jointly owned with #115 (BP2) -- Build Doc v3 s5.3",
  "declared": "before any bench/results/bp_* artifact exists (commit-order verifiable)",
  "purpose": "Cells with historically bimodal timing distributions. For flagged cells, median-of-20 comparison is disallowed; #BP2's matched-percentile rule applies: p50<->p50 and p10<->p10 reporting, verdict from matched percentiles, increased reps.",
  "lineage_rule": "B_xK at K=4 is flagged at every N and scenario: R3/V4 evidence shows the bimodality is a property of the 4-GPU delivery path, not of one tensor size. Any future cell matching {method: bxk, K: 4} inherits the flag.",
  "cells": [
    {
      "scenario": "scatter", "method": "bxk", "K": 4, "N": 8192,
      "wall_min_ms": 8.66338, "wall_median_ms": 15.2059, "wall_p95_ms": 16.6569,
      "evidence": [
        {"source": "bench/results/r3_scatter_n8192_epyc_2080ti.json",
         "note": "R3 (2026-07-27): min/median/p95 above; the direct bimodal observation"},
        {"source": "docs/v4-prefold.md:100-121",
         "note": "V4 re-observation: same bimodal distribution (min ~9 ms, upper mode ~15-17 ms); V4-G1 miss explained by endpoint match (min/p95), IQR 19.7%"}
      ]
    }
  ]
}
```

(Additional entries: one per remaining Step-1 row, with `"evidence": [{"source": "<that artifact>", "note": "flagged by lineage_rule; stats recorded as committed"}]`. Do not editorialize about whether those look bimodal — the lineage rule is the declaration.)

- [ ] **Step 3: Validate + commit**

```bash
python3 -c "import json; d=json.load(open('bench/results/cm4_bimodal_cells.json')); assert d['cells'][0]['wall_min_ms']==8.66338 and d['cells'][0]['wall_p95_ms']==16.6569; assert all(c['method']=='bxk' and c['K']==4 for c in d['cells']); print(len(d['cells']), 'cells OK')"
git add bench/results/cm4_bimodal_cells.json
git commit -m "feat(bench): CM4 pre-declared bimodal cell list -- B_xK-K=4 lineage (#112)"
```

---

### Task 4: Final verification + PR

**Files:** none beyond the above.

- [ ] **Step 1: Full verification**

```bash
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
PYTHONPATH=build/sym/python python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json > /tmp/cm4_gate_out.txt 2>&1
grep -E 'V3-RSTAR|REGEN' /tmp/cm4_gate_out.txt
ls bench/results/bp_* 2>/dev/null; echo "bp_* count: $(ls bench/results/bp_* 2>/dev/null | wc -l)"   # MUST be 0
git status --short   # clean except pre-existing .claude/
```

Expected: suites green; V3-RSTAR line shows rule-v1 label with 0 mismatches and the historical 0.3632 FAIL unchanged; CALIBRATION-REGEN ×2 + CM4-REGEN + REPORT-REGEN PASS; zero `bp_*` files.

- [ ] **Step 2: Push and open the PR (regular)**

```bash
git push -u origin cm4-registration
gh pr create --title "feat(bench): CM4 — tightened RSTAR rule + registered predictions + held-out split (#112)" --body "<body>"
```

PR body, in order: (1) verdict-first — rule v1 live (mismatch_one_sided ⇒ automatic FAIL at bar value, §5 discharged), 48-cell registration + 24 r* grids committed and CM4-REGEN-guarded, bimodal list declared, all while zero `bp_*` artifacts exist (the commit-order acceptance — state the HEAD commit and the empty `ls bench/results/bp_*`); (2) the v0-report disclosure: the frozen report has zero mismatch_one_sided rows, so rule v1 prints identical verdicts on it (RSTAR FAIL 0.3632 as recorded) — no retroactive change; (3) what CM5 consumes: the registration schema summary (cells fields, placement map b_fair↔serial / b_pipelined↔overlapped, r* grids with epyc-0.125 nulls, held-out split + overhead-intercept caveat quoted); (4) the CM3 hand-off note: three live mismatch_one_sided instances exist in `_pyreloc` artifacts — under rule v1 those would FAIL, which is the rule working as intended (link `docs/cm3-one-implementation.md`); (5) `Refs #112, #107, #115` and a note that #BP2 should cross-link `cm4_registered_predictions.json` + `cm4_bimodal_cells.json`; (6) the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. Both suites green; gate: CALIBRATION-REGEN ×2, CM4-REGEN, REPORT-REGEN all PASS; V3-RSTAR carries the rule-v1 label and the unchanged 0.3632 FAIL on the v0 report.
2. Synthetic-report checks: a mismatch_one_sided row forces FAIL with the max at 0.1500; removing it restores PASS at the both_exist value.
3. Registration: 48 cells / 24 r* rows; zero cell-level nulls; epyc r=0.125 grid points null; spot-checks equal direct pyreloc.predict; regeneration byte-identical.
4. Bimodal list: scatter-K4-N8192 entry matches the R3 artifact's 8.66338/15.2059/16.6569; every entry is method=bxk K=4.
5. `ls bench/results/bp_*` is empty at the merge commit — #112's commit-order acceptance.
6. Frozen artifacts untouched: `git diff main -- bench/results/v3_prediction_report.json calibration/` is empty.
