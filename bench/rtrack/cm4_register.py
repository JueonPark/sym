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
                t_b_r1 = None if pb1 is None else pb1["t_b_ms"]
                grid = {}
                t_a_grid = {}
                for r in R_POINTS:
                    if pb1 is None:
                        grid[str(r)] = None
                        t_a_grid[str(r)] = None
                        continue
                    pr = predict(cal, pattern, src, r, placement)
                    if pr is None:
                        grid[str(r)] = None
                        t_a_grid[str(r)] = None
                    else:
                        grid[str(r)] = t_b_r1 / pr["t_a_ms"]
                        t_a_grid[str(r)] = pr["t_a_ms"]
                pts = sorted((float(k), v) for k, v in grid.items()
                             if v is not None)
                rstar.append({"box": box, "family": fam,
                              "placement": placement, "n": RSTAR_N,
                              "speedup_predicted": grid,
                              "t_a_ms": t_a_grid,
                              "t_b_r1_ms": t_b_r1,
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
        "speedup_definition": (
            "speedup_predicted[r] = t_b_r1_ms / t_a_ms[r] (the "
            "docs/v3-costmodel.md S4 quantity); speedup > 1.0 means method "
            "a is predicted to win at that r; rstar_predicted = the 1.0 "
            "crossing interpolated linearly in log2(r) (crossing())"),
        "cells": cells,
        "rstar": rstar,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())
