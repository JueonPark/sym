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
