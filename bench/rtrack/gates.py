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

R2 / EXP-2 (issue #83) bars, FIXED before the Gen4 data is read
(--exp r2; falsification gates -- the issue PREDICTS the losses):

  R2-G1  Gen4 link is the floor      pinned H2D in [20, 26] GB/s
  R2-G2  T3 quant wins (Case 1b)     quant:            A >= 1.50x B
  R2-G3  fused strided+quant loses   T2, T4:           A/B < 0.95
  R2-G4  T1b loses ~0.6x (sym#63)    blocked_transpose: A/B in [0.40, 0.80]
  R2-G5  r* model agrees             per family: measured r* within 2x of
                                     predicted (both-none also agrees);
                                     needs --rstar from figure_rstar.py

Only variant=matrix rows feed R2-G1..G4 (rows without the column count as
matrix). R2-G4 is barred on T1b because the issue's ~0.6x prediction is
anchored on sym#63's *blocked* gather; plain T1 is reported ungated.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

R1_GATES = {
    "G2": ("quant", 1.50, None),
    "G3a": ("transpose_quant", 0.95, None),
    "G3b": ("nchw_nhwc_quant", 0.95, None),
    "G4": ("transpose", 0.85, 1.10),
}
R2_GATES = {
    "R2-G2": ("quant", 1.50, None),
    "R2-G3a": ("transpose_quant", None, 0.95),
    "R2-G3b": ("nchw_nhwc_quant", None, 0.95),
    "R2-G4": ("blocked_transpose", 0.40, 0.80),
}
G1_BARS = {"r1": (11, 14), "r2": (20, 26)}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(l for l in f if not l.startswith("#"))
            for row in reader:
                row["N"] = int(row["N"])
                for k in ("median_ms", "effective_input_GBps", "h2d_ms"):
                    row[k] = float(row[k])
                row["variant"] = row.get("variant") or "matrix"
                rows.append(row)
    rows = [r for r in rows if r["variant"] == "matrix"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--exp", choices=["r1", "r2"], default="r1")
    ap.add_argument("--rstar")
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no data rows")

    gate_transforms = R1_GATES if args.exp == "r1" else R2_GATES
    g1_lo, g1_hi = G1_BARS[args.exp]

    # G1: DMA-only bandwidth from Method B rows (h2d_ms is the summed
    # per-chunk DMA event time; input is the full fp32 tensor).
    h2d = [r["N"] * r["N"] * 4 / (r["h2d_ms"] * 1e-3) / 1e9
           for r in rows if r["method"] == "b" and r["h2d_ms"] > 0]
    g1 = max(h2d) if h2d else 0.0
    g1_pass = g1_lo <= g1 <= g1_hi
    print(f"G1 pinned H2D: {g1:.2f} GB/s (bar [{g1_lo}, {g1_hi}])  "
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
    for gate, (transform, lo, hi) in gate_transforms.items():
        cells, ok_all, seen = [], True, False
        for n in ns:
            a, b = best[(transform, n, "a")], best[(transform, n, "b")]
            if a <= 0 or b <= 0:
                cells.append("--")
                continue
            seen = True
            ratio = a / b
            ok = (lo is None or ratio >= lo) and (hi is None or ratio <= hi)
            ok_all &= ok
            cells.append(f"{ratio:.2f}x{'' if ok else ' !'}")
        verdict = "PASS" if (seen and ok_all) else ("no data" if not seen
                                                    else "FAIL")
        verdicts[gate] = verdict
        bar = (f"< {hi}" if lo is None else
               (f">= {lo}" if hi is None else f"[{lo}, {hi}]"))
        print(f"| {gate} | {transform} | {bar} | " + " | ".join(cells) +
              f" | {verdict} |")

    if args.exp == "r1":
        g3 = ("PASS" if verdicts.get("G3a") == verdicts.get("G3b") == "PASS"
              else "FAIL")
        print(f"\nG3 overall (T2 AND T4): {g3}")
        g2 = verdicts.get("G2", "no data")
        print(f"DECISION (G2): {g2} -> " +
              ("win-condition (a): proceed R2 with the crossover framing."
               if g2 == "PASS" else
               "pivot to win-condition (b): hiding-ratio model + machinery."))
    else:
        g3 = ("PASS" if verdicts.get("R2-G3a") == verdicts.get("R2-G3b") ==
              "PASS" else "FAIL")
        print(f"\nR2-G3 overall (T2 AND T4 lose): {g3}")
        print("R2: reporting experiment (no go/no-go); see "
              "docs/r2-exp2-gen4-crossover.md")

        if args.rstar:
            with open(args.rstar) as f:
                rstar = json.load(f)["families"]
            print("\nR2-G5 (measured r* within 2x of predicted):")
            for fam, d in sorted(rstar.items()):
                m, p = d.get("rstar_measured"), d.get("rstar_predicted")
                if m is None and p is None:
                    verdict = "PASS (both: no in-range crossover)"
                elif m is None or p is None:
                    verdict = "FAIL (one side has no crossover)"
                else:
                    verdict = ("PASS" if max(m, p) / min(m, p) <= 2.0
                               else "FAIL") + \
                        f" (measured {m:.3f} vs predicted {p:.3f})"
                print(f"  {fam}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
