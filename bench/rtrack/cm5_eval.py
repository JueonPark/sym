#!/usr/bin/env python3
# bench/rtrack/cm5_eval.py
"""CM5 (issue #113): evaluate the BP measurement (#116/#124) against the
CM4-registered predictions (#112/#121) ONLY -- no calibration is read, no
prediction is recomputed. Bars are fixed before this tool existed:
all-cells MISCLASS <= 0.15 / RSTAR |dr*| <= 0.15 (rule v1: any one-sided
mismatch => FAIL) / REGRET-p90 <= 0.20; held-out bars come from the
registration's heldout_split block. The stabler-preference merge
(docs/r2-exp2-gen4-crossover.md:48-58) is implemented HERE --
gates.py --exp bp's best-of-max merge is NOT the rule (both session PRs
record this).

  python3 bench/rtrack/cm5_eval.py --out bench/results/cm5_eval_report.json

Deterministic: sorted iteration, sort_keys dump, no timestamps --
v3_gate.py's CM5-REPORT-REGEN and test_cm5_eval.py's regen twin re-run
this and byte-compare.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_rstar import crossing  # noqa: E402  shared interpolation (#111)
from v3_gate import percentile     # noqa: E402  nearest-rank, shared with v0

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "bench" / "results"
REGISTRATION = RESULTS / "cm4_registered_predictions.json"
BOXES = ("7800x3d-4070tis", "epyc7351-2080ti")
PAIRINGS = (("b_fair", "winner_vs_b_fair", "serial"),
            ("b_pipelined", "winner_vs_b_pipelined", "overlapped"))
METHOD_FOR_PLACEMENT = {"serial": "b_fair", "overlapped": "b_pipelined"}
RSWEEP_FAMILIES = ("blocked_transpose", "nchw_nhwc_quant", "quant",
                   "transpose_quant")
MISCLASS_BAR = 0.15
RSTAR_ABS_BAR = 0.15
REGRET_P90_BAR = 0.20


def fmt_r(x):
    return "%g" % x


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def point_key(row):
    return (row["transform"], int(row["N"]), row["method"], row["variant"],
            row["r"])


def best_chunk(rows):
    return min(rows, key=lambda r: float(r["median_ms"]))


def merge_points(orig_rows, rerun_rows):
    """Stabler-preference merge: per analysis point (transform, N, method,
    variant, r), take each file's best-chunk row (lowest median), adopt the
    lower-IQR one; tie -> original. Rerun-only points are an input error."""
    by_point = defaultdict(list)
    for r in orig_rows:
        by_point[point_key(r)].append(r)
    rerun_by_point = defaultdict(list)
    for r in rerun_rows:
        rerun_by_point[point_key(r)].append(r)
    unknown = sorted(set(rerun_by_point) - set(by_point))
    if unknown:
        sys.exit(f"error: rerun rows with no original point: {unknown[:3]}")
    merged, audit = {}, []
    for key in sorted(by_point):
        o = best_chunk(by_point[key])
        if key not in rerun_by_point:
            merged[key] = o
            continue
        n = best_chunk(rerun_by_point[key])
        chose = ("rerun" if float(n["iqr_over_median_pct"])
                 < float(o["iqr_over_median_pct"]) else "original")
        merged[key] = n if chose == "rerun" else o
        transform, num, method, variant, rr = key
        audit.append({"transform": transform, "N": num, "method": method,
                      "variant": variant, "r": rr, "chose": chose,
                      "orig_chunk": o["chunk_req_mib"],
                      "orig_median_ms": float(o["median_ms"]),
                      "orig_iqr": float(o["iqr_over_median_pct"]),
                      "rerun_chunk": n["chunk_req_mib"],
                      "rerun_median_ms": float(n["median_ms"]),
                      "rerun_iqr": float(n["iqr_over_median_pct"])})
    return merged, audit


def eval_winner_cells(merged_by_box, registration):
    """Per pairing, measured winner = argmin of best-chunk medians at the
    family's native-r matrix rows vs the registered winner field. A cell
    whose measured method is absent is excluded from that pairing's
    denominator and disclosed (the V3 unmodelable-cells pattern)."""
    cells_by_pairing = {p: [] for p, _, _ in PAIRINGS}
    excluded = []
    for cell in registration["cells"]:
        box, family, n = cell["box"], cell["family"], cell["N"]
        merged = merged_by_box[box]
        r_str = fmt_r(cell["r_native"])
        a = merged.get((family, n, "a", "matrix", r_str))
        for pairing, field, _ in PAIRINGS:
            b = merged.get((family, n, pairing, "matrix", r_str))
            if a is None or b is None:
                missing = "a" if a is None else pairing
                excluded.append({"box": box, "family": family, "N": n,
                                 "pairing": pairing,
                                 "reason": f"no measured {missing} matrix row"})
                continue
            t_a, t_b = float(a["median_ms"]), float(b["median_ms"])
            cells_by_pairing[pairing].append(
                {"box": box, "family": family, "N": n, "split": cell["split"],
                 "pairing": pairing, "winner_pred": cell[field],
                 "winner_meas": "a" if t_a <= t_b else "b",
                 "t_a_meas_ms": t_a, "t_b_meas_ms": t_b})
    return cells_by_pairing, excluded


def misclass(cells, bar):
    wrong = sum(1 for c in cells if c["winner_pred"] != c["winner_meas"])
    rate = wrong / len(cells) if cells else None
    return {"n_cells": len(cells), "n_wrong": wrong, "rate": rate, "bar": bar,
            "verdict": None if rate is None else
            ("PASS" if rate <= bar else "FAIL")}


def cell_regret(cell, policy):
    chosen = {"model": cell["winner_pred"], "always_a": "a",
              "always_b": "b"}[policy]
    t = cell["t_a_meas_ms"] if chosen == "a" else cell["t_b_meas_ms"]
    return t / min(cell["t_a_meas_ms"], cell["t_b_meas_ms"]) - 1.0


def regret_gate(cells, bar):
    p90 = percentile([cell_regret(c, "model") for c in cells], 0.9)
    return {"n_cells": len(cells), "p90": p90, "bar": bar,
            "verdict": None if p90 is None else
            ("PASS" if p90 <= bar else "FAIL")}


def eval_rstar(merged_rsweep_by_box, registration):
    """Measured r* per registered (box, family, placement) at N=16384 from
    speedup(r) = median_B(r=1) / median_A(r), restricted to the registered
    grid_used, via the shared crossing(). Families outside the BP r-sweep
    protocol and rows with missing measured rsweep rows are excluded with
    reason (never silently classified)."""
    rows_out, excluded = [], []
    for reg in registration["rstar"]:
        box, family = reg["box"], reg["family"]
        placement, n = reg["placement"], reg["n"]
        method = METHOD_FOR_PLACEMENT[placement]
        base = {"box": box, "family": family, "placement": placement}
        if family not in RSWEEP_FAMILIES:
            excluded.append(dict(base, reason=(
                "family not in the BP r-sweep protocol (T1b/T2/T3/T4 only)")))
            continue
        merged = merged_rsweep_by_box[box]
        b1 = merged.get((family, n, method, "rsweep", "1"))
        a_rows = {g: merged.get((family, n, "a", "rsweep", fmt_r(g)))
                  for g in reg["grid_used"]}
        if b1 is None or any(v is None for v in a_rows.values()):
            excluded.append(dict(base, reason="missing measured rsweep rows"))
            continue
        pts = sorted((g, float(b1["median_ms"]) / float(a_rows[g]["median_ms"]))
                     for g in reg["grid_used"])
        meas = crossing(pts)
        pred = reg["rstar_predicted"]
        if pred is not None and meas is not None:
            cls, delta = "both_exist", abs(pred - meas)
        elif pred is None and meas is None:
            cls, delta = "both_no_crossing", None
        else:
            cls, delta = "mismatch_one_sided", None
        rows_out.append(dict(base, rstar_pred=pred, rstar_meas=meas,
                             classification=cls, abs_delta=delta))
    return rows_out, excluded


def rstar_gate(rows, placement, bar):
    rows = [r for r in rows if r["placement"] == placement]
    one_sided = sum(1 for r in rows
                    if r["classification"] == "mismatch_one_sided")
    deltas = [r["abs_delta"] for r in rows
              if r["classification"] == "both_exist"]
    max_delta = max(deltas) if deltas else None
    if not rows:
        verdict = None
    elif one_sided or (max_delta is not None and max_delta > bar):
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"n_rows": len(rows), "n_both_exist": len(deltas),
            "n_one_sided_mismatch": one_sided, "max_abs_delta": max_delta,
            "bar": bar, "verdict": verdict}


def ablation(cells):
    out = {}
    for policy in ("model", "always_a", "always_b"):
        vals = [cell_regret(c, policy) for c in cells]
        out[policy] = {"mean": sum(vals) / len(vals) if vals else None,
                       "p90": percentile(vals, 0.9)}
    out["oracle"] = {"mean": 0.0, "p90": 0.0}
    return out


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(registration_path=None):
    registration_path = (Path(registration_path).resolve()
                         if registration_path else REGISTRATION)
    registration = json.loads(registration_path.read_text())
    heldout = registration["heldout_split"]
    ho_mis_bar = heldout["heldout_bars"]["misclass"]
    ho_reg_bar = heldout["heldout_bars"]["regret_p90"]

    inputs = [registration_path]
    merged_matrix, merged_rsweep, audit = {}, {}, []
    for box in sorted(BOXES):
        pairs = {"matrix": (RESULTS / f"bp_matrix_nsweep_{box}.csv",
                            RESULTS / f"bp_matrix_nsweep_rerun_{box}.csv"),
                 "rsweep": (RESULTS / f"bp_rsweep_{box}.csv",
                            RESULTS / f"bp_rsweep_rerun_{box}.csv")}
        for sweep, (orig, rerun) in sorted(pairs.items()):
            inputs += [orig, rerun]
            merged, entries = merge_points(load_rows(orig), load_rows(rerun))
            (merged_matrix if sweep == "matrix" else merged_rsweep)[box] = merged
            audit += [dict(e, box=box, sweep=sweep) for e in entries]

    cells_by_pairing, excluded_cells = eval_winner_cells(
        merged_matrix, registration)
    rstar_rows, rstar_excluded = eval_rstar(merged_rsweep, registration)

    gates = {"all_cells": {}, "held_out": {"caveat": heldout["caveat"]},
             "rstar": {"rule": ("v1 (CM4): any one-sided mismatch => FAIL; "
                                "bar on max |dr*| over both_exist")}}
    ab, pooled = {}, []
    for pairing, _, placement in PAIRINGS:
        cells = cells_by_pairing[pairing]
        test_cells = [c for c in cells if c["split"] == "test"]
        pooled += cells
        gates["all_cells"][pairing] = {
            "misclass": misclass(cells, MISCLASS_BAR),
            "regret": regret_gate(cells, REGRET_P90_BAR)}
        gates["held_out"][pairing] = {
            "misclass": misclass(test_cells, ho_mis_bar),
            "regret": regret_gate(test_cells, ho_reg_bar)}
        gates["rstar"][placement] = rstar_gate(rstar_rows, placement,
                                               RSTAR_ABS_BAR)
        ab[pairing] = {
            "all_cells": ablation(cells),
            "all_cells_by_box": {
                box: ablation([c for c in cells if c["box"] == box])
                for box in sorted(BOXES)},
            "held_out": ablation(test_cells),
            "held_out_by_box": {
                box: ablation([c for c in test_cells if c["box"] == box])
                for box in sorted(BOXES)},
        }
    gates["all_cells"]["pooled_informational_no_bar"] = \
        misclass(pooled, MISCLASS_BAR) | {"bar": None, "verdict": None}
    verdicts = [gates["rstar"][p]["verdict"] for _, _, p in PAIRINGS]
    gates["rstar"]["overall_verdict"] = ("FAIL" if "FAIL" in verdicts
                                         else "PASS")
    gates["held_out"]["note"] = ("no separate held-out RSTAR: the r-sweep "
                                 "exists only at N=16384, so RSTAR is "
                                 "already an all-test-N metric")

    misses = {
        "misclass": sorted(
            (c for cs in cells_by_pairing.values() for c in cs
             if c["winner_pred"] != c["winner_meas"]),
            key=lambda c: (c["pairing"], c["box"], c["family"], c["N"])),
        "rstar": [r for r in rstar_rows
                  if r["classification"] == "mismatch_one_sided"
                  or (r["abs_delta"] is not None
                      and r["abs_delta"] > RSTAR_ABS_BAR)],
        "regret_over_bar": sorted(
            (dict(c, regret=cell_regret(c, "model"))
             for cs in cells_by_pairing.values() for c in cs
             if cell_regret(c, "model") > REGRET_P90_BAR),
            key=lambda c: (c["pairing"], c["box"], c["family"], c["N"]))}

    return {
        "generated_by": "bench/rtrack/cm5_eval.py",
        "issue": ("#113 (CM5): BP measurement (#116/#124) vs CM4-registered "
                  "predictions (#112/#121); evaluation only"),
        "provenance": {str(p.relative_to(REPO_ROOT)): sha256(p)
                       for p in inputs},
        "merge_audit": audit,
        "gates": gates,
        "rstar_rows": rstar_rows,
        "excluded_cells": excluded_cells + rstar_excluded,
        "ablation": ab,
        "misses": misses,
    }


def render(report):
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registration", default=str(REGISTRATION))
    args = ap.parse_args()
    report = build_report(registration_path=args.registration)
    Path(args.out).write_text(render(report))
    g = report["gates"]
    for pairing, _, placement in PAIRINGS:
        a = g["all_cells"][pairing]
        h = g["held_out"][pairing]
        print(f"{pairing}: MISCLASS {a['misclass']['n_wrong']}/"
              f"{a['misclass']['n_cells']} {a['misclass']['verdict']} | "
              f"REGRET-P90 {a['regret']['p90']:.4f} {a['regret']['verdict']}"
              f" | held-out MISCLASS {h['misclass']['verdict']} "
              f"REGRET {h['regret']['verdict']} | RSTAR({placement}) "
              f"{g['rstar'][placement]['verdict']}")
    print(f"RSTAR overall: {g['rstar']['overall_verdict']}")


if __name__ == "__main__":
    main()
