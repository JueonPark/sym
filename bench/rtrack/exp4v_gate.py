#!/usr/bin/env python3
"""V4 / P4 pre-fold gates (issue #98): evaluate the pre-registered bars
from bench-multigpu-reloc JSON. Bars are FIXED here, before the data is
read:

  V4-G1  aprefold/B_xK >= 3.0x on scatter int8 K=4 N=8192 (R3's DMA-only
         column predicts ~3.5x; the bar allows pipeline overhead).
  V4-G2  admissibility in reverse: aprefold's DMA leg at K=1 scatter
         reaches >= 0.90 x the box's pinned H2D for r*S bytes (r=0.25:
         N*N int8 bytes over dma_ms). K=1 isolates a single link (the
         K=2/K=4 aggregates are confounded by shared PCIe roots, M0).
  V4-G3  both measured counter-cases (reuse n=1 cold single-use, and
         streaming) LOSE, and prefoldWins' prediction (recorded by the
         bench as predicted_prefold_wins) matches the measured verdict
         on EVERY reuse/streaming row.

  python3 bench/rtrack/exp4v_gate.py --json scatter.json [more.json ...]
          [--pinned-h2d 13.07]
"""

import argparse
import json
import sys

G1_BAR = 3.0
G2_FRACTION = 0.90
PINNED_H2D_DEFAULT = 13.07  # GB/s, bench/results/v1_gate_report.txt


def load(paths):
    rows, reuse = [], []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        rows += doc.get("rows", [])
        reuse += doc.get("reuse_rows", [])
    return rows, reuse


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--pinned-h2d", type=float, default=PINNED_H2D_DEFAULT)
    args = ap.parse_args()
    rows, reuse = load(args.json)
    if not rows:
        sys.exit("error: no rows")

    def cell(scenario, k, n, method):
        for r in rows:
            if (r["scenario"] == scenario and r["K"] == k and r["N"] == n
                    and r["method"] == method):
                return r
        return None

    # V4-G1
    ap4 = cell("scatter", 4, 8192, "aprefold")
    bxk4 = cell("scatter", 4, 8192, "bxk")
    if ap4 and bxk4:
        ratio = bxk4["wall_median_ms"] / ap4["wall_median_ms"]
        print(f"V4-G1 aprefold/B_xK @ scatter K=4 N=8192: {ratio:.2f}x "
              f"(bar >= {G1_BAR})  "
              f"{'PASS' if ratio >= G1_BAR else 'FAIL'}")
    else:
        print("V4-G1: no data")

    # V4-G2
    ap1 = cell("scatter", 1, 8192, "aprefold")
    if ap1 and ap1["dma_ms"] > 0:
        n = ap1["N"]
        bw = n * n / (ap1["dma_ms"] * 1e-3) / 1e9  # int8: N*N bytes
        bar = G2_FRACTION * args.pinned_h2d
        print(f"V4-G2 aprefold DMA leg @ scatter K=1 N=8192: {bw:.2f} GB/s "
              f"(bar >= {bar:.2f} = {G2_FRACTION} x {args.pinned_h2d})  "
              f"{'PASS' if bw >= bar else 'FAIL'}")
    else:
        print("V4-G2: no data")

    # V4-G3
    if reuse:
        losses_ok, pred_ok = True, True
        print("\n| mode | scenario | K | n_reuse | A ms/load | prefold "
              "ms/load | predicted | measured | rule match |")
        print("|---" * 9 + "|")
        for r in reuse:
            match = r["predicted_prefold_wins"] == r["measured_prefold_wins"]
            pred_ok &= match
            if r["mode"] == "streaming" or r["n_reuse"] == 1:
                losses_ok &= not r["measured_prefold_wins"]
            print(f"| {r['mode']} | {r['scenario']} | {r['K']} | "
                  f"{r['n_reuse']} | {r['a_per_load_ms']:.2f} | "
                  f"{r['prefold_per_load_ms']:.2f} | "
                  f"{'PREFOLD' if r['predicted_prefold_wins'] else 'A'} | "
                  f"{'PREFOLD' if r['measured_prefold_wins'] else 'A'} | "
                  f"{'ok' if match else 'MISMATCH'} |")
        verdict = "PASS" if (losses_ok and pred_ok) else "FAIL"
        print(f"\nV4-G3 (counter-cases lose AND rule predicts every row): "
              f"{verdict}")
    else:
        print("V4-G3: no reuse/streaming data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
