#!/usr/bin/env python3
# bench/rtrack/r6_bind_demo.py
"""R6 (issue #87): cross-box no-recompile bind demo.

Loads the MLIR-folded symbolic wire blob
(libreloc/test/corpus/blocked_transpose_sym.bin -- the V3 wire-row plan),
binds it at runtime symbol values N in {2048, 4096, 8192, 16384} with
this box's committed calibration as the bind model, and records the
bind-time placement decision (t8 + Overlapped -- libreloc/src/Bind.cpp
step 8) plus a serial-priced predict() check per cell, for wire ratios
r in {0.25, 0.5, 1.0} (the measured rsweep tiers s8/f16/f32; r=0.125 is
excluded -- no blocked s4 calibration key exists, a CM-track boundary,
see docs/r6-crossbox-bind.md).

The output is FULLY DETERMINISTIC: sorted keys, no timestamps, no
hostnames, no git revs; inputs sha256-pinned inside the artifact.
Byte-equality between the artifact a box commits and a CI in-process
regeneration is the demo's no-recompile portability bar (spec AC3).

  PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
      --machine epyc7351-2080ti
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import pyreloc
except ImportError:
    sys.exit("error: pyreloc not importable -- build it (ninja -C build/sym "
             "pyreloc_ext) and run with PYTHONPATH=build/sym/python")

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = (REPO_ROOT / "libreloc" / "test" / "corpus" /
          "blocked_transpose_sym.bin")
MACHINES = ("epyc7351-2080ti", "7800x3d-4070tis")
R_GRID = (0.25, 0.5, 1.0)      # measured rsweep wire tiers s8/f16/f32
N_GRID = (2048, 4096, 8192, 16384)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_demo(machine):
    cal_path = REPO_ROOT / "calibration" / f"{machine}.cal"
    cal = pyreloc.load_calibration(str(cal_path))
    plan = pyreloc.load_plan(CORPUS.read_bytes())
    cells = []
    for r in R_GRID:
        for n in N_GRID:
            bound = pyreloc.bind(plan, {"N": n}, model=cal, wire_ratio=r)
            decision = bound.decision
            assert decision is not None, f"no decision at r={r} N={n}"
            serial = pyreloc.predict(
                cal, pattern=decision["pattern"],
                src_bytes=bound.total_bytes, r=r, b_placement="serial")
            cells.append({
                "r": r, "N": n,
                "bound": {
                    "extents": list(bound.extents),
                    "src_strides": list(bound.src_strides),
                    "dst_strides": list(bound.dst_strides),
                    "total_bytes": bound.total_bytes,
                    "strategy": bound.strategy,
                },
                "bind_decision": decision,
                "serial_check": serial,
            })
    return {
        "generated_by": "bench/rtrack/r6_bind_demo.py",
        "issue": "#87 (R6): cross-box no-recompile bind demo",
        "machine": machine,
        "inputs": {
            str(CORPUS.relative_to(REPO_ROOT)): sha256(CORPUS),
            str(cal_path.relative_to(REPO_ROOT)): sha256(cal_path),
        },
        "cells": cells,
    }


def render(report):
    return json.dumps(report, indent=1, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="R6 cross-box bind demo (issue #87)")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: bench/results/r6_bind_demo_<machine>.json")
    args = ap.parse_args()
    out = args.out or (REPO_ROOT / "bench" / "results" /
                       f"r6_bind_demo_{args.machine}.json")
    report = build_demo(args.machine)
    out.write_text(render(report))
    for c in report["cells"]:
        d = c["bind_decision"]
        print(f"r={c['r']:<5g} N={c['N']:<6d} -> {d['method']}  "
              f"(t_a={d['t_a_ms']:.4g} ms, t_b={d['t_b_ms']:.4g} ms, "
              f"threshold_bytes={d['threshold_bytes']:.6g})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
