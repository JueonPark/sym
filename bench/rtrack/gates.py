#!/usr/bin/env python3
"""R1 / EXP-1 (issue #82): evaluate the pre-registered Gen3 gates from
rtrack CSVs. Bars are FIXED here, before the data is read:

  G1  Gen3 link is the floor        pinned H2D in [11, 14] GB/s
  G2  dtype reduction wins (1b)     T3 (quant):    A >= 1.50x B
  G3  fused strided+quant ties (1a) T2, T4:        A >= 0.95x B
  G4  pure relocation ties          T1 (transpose): A/B in [0.85, 1.10]

A/B is the ratio of best effective_input_GBps per method (best chunk,
per the issue's best-C-per-method rule) at each N. The overall verdict
for a gate is the STRICT reading: it must hold at every measured N; the
per-N table is printed so a looser reading can be argued explicitly.
Decision rule: G2 pass -> win-condition (a), proceed R2; G2 fail ->
pivot to win-condition (b). G3 is a bonus claim.

  python3 bench/rtrack/gates.py --csv r1_gen3_nsweep.csv [more.csv ...]
"""

import argparse
import csv
import sys
from collections import defaultdict

GATE_TRANSFORMS = {
    "G2": ("quant", 1.50, None),
    "G3a": ("transpose_quant", 0.95, None),
    "G3b": ("nchw_nhwc_quant", 0.95, None),
    "G4": ("transpose", 0.85, 1.10),
}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(l for l in f if not l.startswith("#"))
            for row in reader:
                row["N"] = int(row["N"])
                for k in ("median_ms", "effective_input_GBps", "h2d_ms"):
                    row[k] = float(row[k])
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no data rows")

    # G1: DMA-only bandwidth from Method B rows (h2d_ms is the summed
    # per-chunk DMA event time; input is the full fp32 tensor).
    h2d = [r["N"] * r["N"] * 4 / (r["h2d_ms"] * 1e-3) / 1e9
           for r in rows if r["method"] == "b" and r["h2d_ms"] > 0]
    g1 = max(h2d) if h2d else 0.0
    g1_pass = 11.0 <= g1 <= 14.0
    print(f"G1 pinned H2D: {g1:.2f} GB/s (bar [11, 14])  "
          f"{'PASS' if g1_pass else 'FAIL'}")

    # Best effective input GB/s per (transform, N, method).
    best = defaultdict(float)
    for r in rows:
        key = (r["transform"], r["N"], r["method"])
        best[key] = max(best[key], r["effective_input_GBps"])

    ns = sorted({r["N"] for r in rows})
    verdicts = {}
    print(f"\n| gate | transform | bar | " +
          " | ".join(f"N={n}" for n in ns) + " | verdict |")
    print("|---" * (4 + len(ns)) + "|")
    for gate, (transform, lo, hi) in GATE_TRANSFORMS.items():
        cells, ok_all, seen = [], True, False
        for n in ns:
            a, b = best[(transform, n, "a")], best[(transform, n, "b")]
            if a <= 0 or b <= 0:
                cells.append("--")
                continue
            seen = True
            ratio = a / b
            ok = ratio >= lo and (hi is None or ratio <= hi)
            ok_all &= ok
            cells.append(f"{ratio:.2f}x{'' if ok else ' !'}")
        verdict = "PASS" if (seen and ok_all) else ("no data" if not seen
                                                    else "FAIL")
        verdicts[gate] = verdict
        bar = f">= {lo}" if hi is None else f"[{lo}, {hi}]"
        print(f"| {gate} | {transform} | {bar} | " + " | ".join(cells) +
              f" | {verdict} |")

    g3 = ("PASS" if verdicts.get("G3a") == verdicts.get("G3b") == "PASS"
          else "FAIL")
    print(f"\nG3 overall (T2 AND T4): {g3}")
    g2 = verdicts.get("G2", "no data")
    print(f"DECISION (G2): {g2} -> " +
          ("win-condition (a): proceed R2 with the crossover framing."
           if g2 == "PASS" else
           "pivot to win-condition (b): hiding-ratio model + machinery."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
