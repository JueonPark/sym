"""V3 (issue #97): the pre-registered prediction test. Walks the frozen
cell inventory (task-7 brief) over committed R-track measurements, calls
the identical C++ cost model through `pyreloc.predict` for each cell, and
writes the report to a pytest scratch path; the committed
`bench/results/v3_prediction_report.json` is V3's frozen as-measured record
(#107) that `bench/rtrack/v3_gate.py` judges/guards.

Ordering discipline: `v3_gate.py` (with its bars fixed in code) is
committed BEFORE this file ever runs against data. This file only
measures -- it asserts structural sanity (>=30 modelable cells, report
written), never a bar. Judgment lives entirely in v3_gate.py.
"""
import csv
import json
import math
import pathlib

import pytest

pyreloc = pytest.importorskip("pyreloc")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "bench" / "results"
CAL_DIR = REPO_ROOT / "calibration"

THREADS = 8

# transform -> (native r, pattern). Shared by the single-GPU N-sweep
# matrix (items 1/2, which use both fields) and the r*-family model
# (item 3, which uses only the pattern -- r sweeps there instead).
FAMILY_MAP = {
    "transpose": (1.0, "single_element"),
    "blocked_transpose": (1.0, "blocked"),
    "transpose_quant": (0.25, "single_element"),
    "nchw_nhwc_quant": (0.25, "tiled"),
    "quant": (0.25, "contiguous"),
    "convert_f16": (0.5, "contiguous"),
}

# multi-GPU scenario -> (pattern, broadcast). r=1.0 throughout (these rows
# ship the full fp32 tensor; there is no dtype-reduction axis in R3/V5).
MULTIGPU_PATTERN = {
    "scatter": ("contiguous", False),
    "broadcast": ("blocked", True),
    "broadcast_contig": ("contiguous", True),
}

MULTIGPU_FILES = [
    "r3_scatter_n8192_epyc_2080ti.json",
    "r3_scatter_n16384_epyc_2080ti.json",
    "r3_broadcast_n8192_epyc_2080ti.json",
    "v5_broadcast_contig_epyc_2080ti.json",
]

RSTAR_SOURCES = [
    ("v1_gen4_rstar_bfair.json", "7800x3d-4070tis"),
    ("v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json", "epyc7351-2080ti"),
]


# ------------------------------------------------------------- helpers --

def load_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def best_chunk(rows, transform, n, method):
    """Best (min median_ms) row across chunk sizes for (transform, N,
    method); None when no row matches."""
    cand = [r for r in rows
            if r["transform"] == transform and int(r["N"]) == n
            and r["method"] == method]
    if not cand:
        return None
    return min(cand, key=lambda r: float(r["median_ms"]))


def merged_best_chunk(orig_rows, rerun_rows, transform, n, method):
    """Stabler (lower iqr_over_median_pct) of the two files' independently
    -recomputed best-chunk rows for this (transform, N, method) point --
    the pre-declared merge rule in docs/r2-exp2-gen4-crossover.md:48-58,
    generalized here from the calibration overhead fit's quant-only
    application to every (transform, N, method) analysis point."""
    o = best_chunk(orig_rows, transform, n, method)
    r = best_chunk(rerun_rows, transform, n, method)
    if o is None:
        return r
    if r is None:
        return o
    return r if float(r["iqr_over_median_pct"]) < float(o["iqr_over_median_pct"]) else o


def predict_cell(cal, pattern, src_bytes, r, k=1, broadcast=False):
    """The C++ cost model's decision, or None on a missing-key ValueError
    -- that is the unmodelable signal (per Task 5's pyreloc surface)."""
    try:
        return pyreloc.predict(cal, pattern=pattern, src_bytes=src_bytes, r=r,
                               threads=THREADS, k=k, broadcast=broadcast)
    except ValueError:
        return None


def crossing(points):
    """points: [(r, speedup)] sorted ascending in r -> r* or None. Verbatim
    formula from bench/rtrack/figure_rstar.py, inlined per the task-7
    brief (the model's speedup crosses 1.0, interpolated linearly in
    log2(r))."""
    for (r0, s0), (r1, s1) in zip(points, points[1:]):
        if (s0 - 1.0) * (s1 - 1.0) <= 0 and s0 != s1:
            t = (1.0 - s0) / (s1 - s0)
            return 2 ** (math.log2(r0) + t * (math.log2(r1) - math.log2(r0)))
    return None


def percentile(vals, p):
    """Nearest-rank percentile over an ascending-sorted list; None if
    empty."""
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return s[k]


def make_cell(cell_id, machine, family, winner_measured, t_a_meas, t_b_meas,
             pred, N=None, K=None, scenario=None):
    row = {
        "cell_id": cell_id, "machine": machine, "family": family,
        "N": N, "K": K, "scenario": scenario,
        "t_a_meas": t_a_meas, "t_b_meas": t_b_meas,
        "winner_measured": winner_measured,
    }
    if pred is None:
        row.update(modelable=False, winner_predicted=None,
                   t_a_pred=None, t_b_pred=None, regret=None)
        return row
    # decide() only activates the prefold arm when n_reuse>=1; every call
    # here leaves it at the default (-1), so pred["method"] is always "a"
    # or "b" -- "a_prefold" is handled defensively anyway.
    winner_predicted = "a" if pred["method"] in ("a", "a_prefold") else "b"
    t_chosen = t_a_meas if winner_predicted == "a" else t_b_meas
    t_oracle = min(t_a_meas, t_b_meas)
    regret = (t_chosen - t_oracle) / t_oracle if t_oracle > 0 else 0.0
    row.update(modelable=True, winner_predicted=winner_predicted,
               t_a_pred=pred["t_a_ms"], t_b_pred=pred["t_b_ms"], regret=regret)
    return row


# --------------------------------------------------------- items 1 & 2 --

def matrix_cells(label, machine, cal, orig_path, rerun_path=None):
    """Best-chunk a-vs-b_fair winner per (transform, N), item 1 (gen3,
    rerun_path=None) / item 2 (gen4, merged with its rerun companion)."""
    orig_rows = load_csv_rows(orig_path)
    rerun_rows = load_csv_rows(rerun_path) if rerun_path else []
    all_rows = orig_rows + rerun_rows

    ns = sorted({int(r["N"]) for r in all_rows if r["transform"] in FAMILY_MAP})
    cells = []
    for transform in sorted(FAMILY_MAP):
        r_fam, pattern = FAMILY_MAP[transform]
        for n in ns:
            if rerun_path:
                a_row = merged_best_chunk(orig_rows, rerun_rows, transform, n, "a")
                b_row = merged_best_chunk(orig_rows, rerun_rows, transform, n,
                                          "b_fair")
            else:
                a_row = best_chunk(orig_rows, transform, n, "a")
                b_row = best_chunk(orig_rows, transform, n, "b_fair")
            if a_row is None or b_row is None:
                continue  # no measured point here (not a modeling gap)
            t_a, t_b = float(a_row["median_ms"]), float(b_row["median_ms"])
            winner_measured = "a" if t_a <= t_b else "b"
            pred = predict_cell(cal, pattern, n * n * 4, r_fam)
            cell_id = f"{label}:{transform}:N={n}"
            cells.append(make_cell(cell_id, machine, transform, winner_measured,
                                   t_a, t_b, pred, N=n))
    return cells


# ------------------------------------------------------------------ item 3 --

def rstar_cells(path, machine, cal):
    doc = json.loads(path.read_text())
    n = doc["n"]
    src_bytes = n * n * 4
    rows = []
    for family, d in sorted(doc["families"].items()):
        if family not in FAMILY_MAP:
            continue
        _, pattern = FAMILY_MAP[family]
        grid = sorted(float(k) for k in d["speedup_measured"])

        pred_b = predict_cell(cal, pattern, src_bytes, 1.0)
        modelable = pred_b is not None
        pts = []
        if modelable:
            t_b1 = pred_b["t_b_ms"]
            for r in grid:
                pred_r = predict_cell(cal, pattern, src_bytes, r)
                if pred_r is None:
                    modelable = False
                    break
                pts.append((r, t_b1 / pred_r["t_a_ms"]))
        rstar_pred = crossing(pts) if modelable else None
        rstar_meas = d.get("rstar_measured")

        row = {
            "family": family, "machine": machine, "source": path.name,
            "n": n, "modelable": modelable,
            "rstar_measured": rstar_meas, "rstar_predicted": rstar_pred,
        }
        if modelable:
            if rstar_meas is None and rstar_pred is None:
                row["agreement"] = "both_no_crossing"
            elif rstar_meas is None or rstar_pred is None:
                row["agreement"] = "mismatch_one_sided"
            else:
                row["agreement"] = "both_exist"
                row["abs_diff"] = abs(rstar_pred - rstar_meas)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ item 4 --

def multigpu_cells(path, cal):
    doc = json.loads(path.read_text())
    cfg = doc["config"]
    n, scenario = cfg["N"], cfg["scenario"]
    machine = "epyc7351-2080ti"
    pattern, broadcast = MULTIGPU_PATTERN[scenario]
    src_bytes = n * n * 4
    rows = doc["rows"]
    ks = sorted({r["K"] for r in rows})

    cells = []
    for k in ks:
        a_row = next((r for r in rows
                     if r["method"] == "a" and r["K"] == k), None)
        b_row = next((r for r in rows
                     if r["method"] == "bxk" and r["K"] == k), None)
        if a_row is None or b_row is None:
            continue
        t_a, t_b = a_row["wall_median_ms"], b_row["wall_median_ms"]
        winner_measured = "a" if t_a <= t_b else "b"
        pred = predict_cell(cal, pattern, src_bytes, 1.0, k=k, broadcast=broadcast)
        cell_id = f"multigpu:{scenario}:N={n}:K={k}"
        cells.append(make_cell(cell_id, machine, scenario, winner_measured,
                               t_a, t_b, pred, N=n, K=k, scenario=scenario))
    return cells


# --------------------------------------------------------------- item 5 --

def ablation(cells):
    modelable = [c for c in cells if c["modelable"]]

    def mean_regret(policy):
        vals = []
        for c in modelable:
            t_oracle = min(c["t_a_meas"], c["t_b_meas"])
            if t_oracle <= 0:
                continue
            if policy == "model":
                t_chosen = (c["t_a_meas"] if c["winner_predicted"] == "a"
                           else c["t_b_meas"])
            elif policy == "always_a":
                t_chosen = c["t_a_meas"]
            else:
                t_chosen = c["t_b_meas"]
            vals.append((t_chosen - t_oracle) / t_oracle)
        return sum(vals) / len(vals) if vals else None

    return {
        "n_cells": len(modelable),
        "model_mean_regret": mean_regret("model"),
        "always_a_mean_regret": mean_regret("always_a"),
        "always_b_mean_regret": mean_regret("always_b"),
    }


# ------------------------------------------------------------------- main --

def build_report():
    cal_epyc = pyreloc.load_calibration(str(CAL_DIR / "epyc7351-2080ti.cal"))
    cal_gen4 = pyreloc.load_calibration(str(CAL_DIR / "7800x3d-4070tis.cal"))

    cells = []
    cells += matrix_cells("gen3", "epyc7351-2080ti", cal_epyc,
                          RESULTS / "v1_gen3_nsweep_epyc_2080ti.csv")
    cells += matrix_cells(
        "gen4", "7800x3d-4070tis", cal_gen4,
        RESULTS / "v1_gen4_matrix_nsweep_7800x3d_4070tis.csv",
        rerun_path=RESULTS / "v1_gen4_matrix_nsweep_rerun_7800x3d_4070tis.csv")

    rstar_rows = []
    for fname, machine in RSTAR_SOURCES:
        cal = cal_gen4 if machine == "7800x3d-4070tis" else cal_epyc
        rstar_rows += rstar_cells(RESULTS / fname, machine, cal)

    for fname in MULTIGPU_FILES:
        cells += multigpu_cells(RESULTS / fname, cal_epyc)

    modelable = [c for c in cells if c["modelable"]]
    unmodelable = [c for c in cells if not c["modelable"]]
    misclassified = [c for c in modelable
                     if c["winner_predicted"] != c["winner_measured"]]
    regrets = [c["regret"] for c in modelable]

    report = {
        "generated_by": "libreloc/python/tests/test_prediction.py",
        "cells": cells,
        "rstar": rstar_rows,
        "ablation": ablation(cells),
        "summary": {
            "n_modelable": len(modelable),
            "n_unmodelable": len(unmodelable),
            "unmodelable_cell_ids": [c["cell_id"] for c in unmodelable],
            "n_misclassified": len(misclassified),
            "misclassified_cell_ids": [c["cell_id"] for c in misclassified],
            "misclass_rate": (len(misclassified) / len(modelable)
                             if modelable else None),
            "regret_p90": percentile(regrets, 0.90),
        },
    }
    return report


def test_write_prediction_report(tmp_path):
    # CM2 (#110): the report is written to a scratch path, never to the
    # committed bench/results/v3_prediction_report.json -- that file is
    # V3's as-measured record against the model that produced it (#107;
    # v3_gate.py's REPORT-REGEN guards it). Re-registering predictions
    # against the corrected model is #CM4's job, not a pytest side
    # effect.
    report = build_report()
    report_path = tmp_path / "v3_prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Structural sanity only -- the BARS are judged by v3_gate.py, never
    # here (measurement and judgment stay separate).
    assert report_path.exists()
    assert report["summary"]["n_modelable"] >= 30, (
        f"only {report['summary']['n_modelable']} modelable cells "
        "(expected >= 30)")
