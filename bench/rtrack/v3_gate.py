#!/usr/bin/env python3
"""V3 pre-registered prediction gate (issue #97): evaluate the calibrated
cost model's predictions against every committed measured cell. Bars are
FIXED here, BEFORE any prediction run happens against data -- this file is
committed standalone first; `libreloc/python/tests/test_prediction.py`
only runs afterward, producing the report this reads. Structure mirrors
`bench/rtrack/exp4v_gate.py`.

  MISCLASS_BAR = 0.15    winner misclassification rate over all
                         modelable single-GPU (a vs b_fair) and multi-GPU
                         (a vs bxk) cells
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
  REGRET_P90_BAR = 0.20  p90 of regret = (T_chosen - T_oracle)/T_oracle
                         over all modelable cells, T_chosen = the measured
                         time of whichever method the model picked

Also runs a CALIBRATION-REGEN check (Task-6 review addendum): re-invokes
bench/rtrack/make_calibration.py for both machines to temp paths and diffs
the output against the committed calibration/*.cal files, so the
committed numbers stay honest.

Also runs a REPORT-REGEN check (final-review addendum, #97): `git diff
--exit-code` against the committed bench/results/v3_prediction_report.json
-- a dirty report means the committed verdicts above no longer match the
code that produced them. Does NOT re-run test_prediction.py (the report
must stay byte-identical); this only inspects the working tree.

Exit code is always 0 -- this prints verdicts, it does not gate CI.
Whatever the bars say IS the result; misses are reported and explained
(Task 9's doc), never refit.

  python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

MISCLASS_BAR = 0.15
RSTAR_ABS_BAR = 0.15
REGRET_P90_BAR = 0.20

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATIONS = {
    "epyc7351-2080ti": REPO_ROOT / "calibration" / "epyc7351-2080ti.cal",
    "7800x3d-4070tis": REPO_ROOT / "calibration" / "7800x3d-4070tis.cal",
}


def load_report(path):
    with open(path) as f:
        return json.load(f)


def percentile(vals, p):
    """Nearest-rank percentile over an ascending-sorted list; None if
    empty (mirrors the same helper in test_prediction.py)."""
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return s[k]


def gate_misclass(report):
    cells = [c for c in report["cells"] if c["modelable"]]
    n = len(cells)
    miss = [c for c in cells if c["winner_predicted"] != c["winner_measured"]]
    rate = len(miss) / n if n else 0.0
    verdict = "PASS" if (n and rate <= MISCLASS_BAR) else (
        "no data" if not n else "FAIL")
    print(f"V3-MISCLASS: {len(miss)}/{n} = {rate:.3f}  "
          f"(bar <= {MISCLASS_BAR})  {verdict}")
    if miss:
        print("\n| cell_id | machine | family | winner_meas | winner_pred "
              "| t_a_meas | t_b_meas | t_a_pred | t_b_pred |")
        print("|---" * 9 + "|")
        for c in miss:
            print(f"| {c['cell_id']} | {c['machine']} | {c['family']} | "
                  f"{c['winner_measured']} | {c['winner_predicted']} | "
                  f"{c['t_a_meas']:.4f} | {c['t_b_meas']:.4f} | "
                  f"{c['t_a_pred']:.4f} | {c['t_b_pred']:.4f} |")
    return rate, verdict


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


def gate_regret(report):
    cells = [c for c in report["cells"] if c["modelable"]]
    regrets = sorted(c["regret"] for c in cells)
    p90 = percentile(regrets, 0.90)
    verdict = "PASS" if (p90 is not None and p90 <= REGRET_P90_BAR) else (
        "no data" if p90 is None else "FAIL")
    print(f"\nV3-REGRET-P90: {'--' if p90 is None else f'{p90:.4f}'}  "
          f"(bar <= {REGRET_P90_BAR})  {verdict}")
    return p90, verdict


def print_ablation(report):
    a = report.get("ablation", {})
    print("\n| policy | mean regret |")
    print("|---|---|")
    for label, key in (("model", "model_mean_regret"),
                       ("always-A", "always_a_mean_regret"),
                       ("always-B", "always_b_mean_regret")):
        v = a.get(key)
        print(f"| {label} | {'--' if v is None else f'{v:.4f}'} |")
    print(f"(n={a.get('n_cells', 0)} modelable winner-vs-winner cells)")


def print_unmodelable(report):
    cells = [c for c in report["cells"] if not c["modelable"]]
    print(f"\nUnmodelable cells (missing calibration keys): {len(cells)}")
    for c in cells:
        print(f"  {c['cell_id']}")
    rstar_unmod = [r for r in report.get("rstar", []) if not r.get("modelable")]
    if rstar_unmod:
        print(f"Unmodelable r* families: {len(rstar_unmod)}")
        for r in rstar_unmod:
            print(f"  {r['family']} ({r['source']})")


def calibration_regen_check():
    print("\n=== CALIBRATION-REGEN ===")
    with tempfile.TemporaryDirectory() as td:
        for machine, committed in sorted(CALIBRATIONS.items()):
            out = Path(td) / f"{machine}.cal"
            proc = subprocess.run(
                [sys.executable, "bench/rtrack/make_calibration.py",
                 "--machine", machine, "--out", str(out)],
                cwd=REPO_ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"CALIBRATION-REGEN {machine}: FAIL (regen errored)\n"
                      f"{proc.stderr.strip()}")
                continue
            regen_text = out.read_text()
            committed_text = committed.read_text()
            if regen_text == committed_text:
                print(f"CALIBRATION-REGEN {machine}: PASS "
                      f"(matches committed {committed.relative_to(REPO_ROOT)})")
            else:
                regen_lines = regen_text.splitlines()
                committed_lines = committed_text.splitlines()
                first_diff = next(
                    (i for i, (a, b) in enumerate(
                        zip(regen_lines, committed_lines)) if a != b),
                    min(len(regen_lines), len(committed_lines)))
                print(f"CALIBRATION-REGEN {machine}: FAIL (diverges from "
                      f"{committed.relative_to(REPO_ROOT)} at line "
                      f"{first_diff + 1})")


# added post-run for integrity (commit history: bars untouched;
# 5b9572e..this-commit diff shows no bar change): a dirty
# bench/results/v3_prediction_report.json means the committed prediction
# verdicts no longer match the code that produced them. Deliberately does
# NOT re-run test_prediction.py -- that report is frozen and must stay
# byte-identical; this only diffs the working tree against HEAD.
def report_regen_check(report_path):
    print("\n=== REPORT-REGEN ===")
    rel = Path(report_path).resolve().relative_to(REPO_ROOT)
    proc = subprocess.run(
        ["git", "diff", "--exit-code", "--", str(rel)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"REPORT-REGEN: PASS (no uncommitted diff in {rel})")
    else:
        print(f"REPORT-REGEN: FAIL (uncommitted diff in {rel} -- the "
              "committed verdicts no longer match the code)\n"
              f"{proc.stdout.strip()}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = load_report(args.report)

    gate_misclass(report)
    gate_rstar(report)
    gate_regret(report)
    print_ablation(report)
    print_unmodelable(report)
    calibration_regen_check()
    report_regen_check(args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
