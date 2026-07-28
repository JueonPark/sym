#!/usr/bin/env python3
"""Pre-registered rtrack gates. Bars are FIXED in code, before any data is
read. Select the experiment with --exp:

  --exp r1  (default)  R1 / EXP-1 Gen3 gates (issue #82)
  --exp r2             R2 / EXP-2 Gen4 gates (issue #83)
  --exp v1             V1 baseline-admissibility bar (issue #95)

R1 / EXP-1 (issue #82):

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

R2 / EXP-2 (issue #83) bars, FIXED before the Gen4 data is read
(--exp r2; falsification gates -- the issue PREDICTS the losses):

  R2-G1  Gen4 link is the floor      pinned H2D in [20, 26] GB/s
  R2-G2  T3 quant wins (Case 1b)     quant:            A >= 1.50x B
  R2-G3  fused strided+quant loses   T2, T4:           A/B < 0.95
  R2-G4  T1b loses ~0.6x (sym#63)    blocked_transpose: A/B in [0.40, 0.80]
  R2-G5  r* model agrees             per family: measured r* within 2x of
                                     predicted (both-none also agrees);
                                     needs --rstar from figure_rstar.py

Only variant=matrix rows feed the R1/R2 gates (rows without the column
count as matrix). R2-G4 is barred on T1b because the issue's ~0.6x
prediction is anchored on sym#63's *blocked* gather; plain T1 is reported
ungated.

V1 admissibility bar (issue #95), derived from R4's validated model, not
invented: a Method-B baseline is ADMISSIBLE iff, on an r=1.0 workload
whose transform multiplier m < ratio (all of R0.2 qualifies), it reaches
effective input bandwidth >= 0.90 x the measured pinned H2D of the SAME
box. Pinned H2D is read from the DMA-only leg (h2d_ms) of the B-type rows
on that box -- the same quantity G1 checks. The gate reports pass/fail per
(machine, method) so b (staged) and b_fair are graded side by side; a
ratio computed against an inadmissible B is provisional and excluded from
headline claims.

  python3 bench/rtrack/gates.py [--exp r1|r2|v1] --csv run.csv [more.csv ...]
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

# Method tags that transfer the full fp32 tensor (S bytes), so h2d_ms is the
# pinned-H2D DMA leg and effective_input_GBps is comparable to the link rate.
B_METHODS = ("b", "b_fair")
V1_BAR_FRACTION = 0.90  # effective input BW must reach this x pinned H2D


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(l for l in f if not l.startswith("#"))
            for row in reader:
                row["N"] = int(row["N"])
                for k in ("r", "median_ms", "effective_input_GBps", "h2d_ms"):
                    row[k] = float(row[k])
                row["variant"] = row.get("variant") or "matrix"
                rows.append(row)
    return rows


def pinned_h2d_gbps(rows):
    """Best DMA-only H2D from the B-type rows: input is the full fp32 tensor
    (N*N*4), h2d_ms is the summed per-chunk DMA event time."""
    vals = [r["N"] * r["N"] * 4 / (r["h2d_ms"] * 1e-3) / 1e9
            for r in rows if r["method"] in B_METHODS and r["h2d_ms"] > 0]
    return max(vals) if vals else 0.0


def exp_v1(rows):
    """V1 admissibility bar, reported per (machine, method)."""
    by_machine = defaultdict(list)
    for r in rows:
        by_machine[r["machine"]].append(r)

    any_fail = False
    for machine in sorted(by_machine):
        mrows = by_machine[machine]
        pinned = pinned_h2d_gbps(mrows)
        bar = V1_BAR_FRACTION * pinned
        print(f"\n=== {machine} ===")
        print(f"pinned H2D (DMA leg): {pinned:.2f} GB/s  "
              f"-> admissibility bar {V1_BAR_FRACTION:g} x = {bar:.2f} GB/s")

        # r=1.0 workloads only (m < ratio holds for all of R0.2).
        r1 = [r for r in mrows if r["r"] == 1.0]
        if not r1:
            print("  (no r=1.0 rows on this box; bar not applicable)")
            continue
        methods = [m for m in B_METHODS if any(r["method"] == m for r in r1)]
        ns = sorted({r["N"] for r in r1})
        transforms = sorted({r["transform"] for r in r1})

        print("| method | transform | " +
              " | ".join(f"N={n}" for n in ns) + " | verdict |")
        print("|---" * (3 + len(ns)) + "|")
        for method in methods:
            method_ok = True
            for transform in transforms:
                cells, ok_all, seen = [], True, False
                for n in ns:
                    best = max((r["effective_input_GBps"] for r in r1
                                if r["method"] == method
                                and r["transform"] == transform
                                and r["N"] == n), default=0.0)
                    if best <= 0:
                        cells.append("--")
                        continue
                    seen = True
                    ok = best >= bar
                    ok_all &= ok
                    cells.append(f"{best:.2f}{'' if ok else ' !'}")
                verdict = ("ADMISSIBLE" if (seen and ok_all)
                           else "no data" if not seen else "INADMISSIBLE")
                if seen and not ok_all:
                    method_ok = False
                print(f"| {method} | {transform} | " + " | ".join(cells) +
                      f" | {verdict} |")
            if not method_ok:
                any_fail = True
    print("\nAdmissibility bar is a per-baseline verdict; b_fair is the "
          "intended admissible baseline, b (staged) is expected to fail.")
    # Exit 0 regardless: this gate reports admissibility, it does not decide
    # a GO/NO-GO. The caller reads the per-method verdicts.
    return 0


def exp_gates(rows, exp, rstar_path=None):
    """R1/R2 pre-registered gate tables (A/B ratio bars per transform)."""
    # Only variant=matrix rows feed the R1/R2 gates.
    rows = [r for r in rows if r["variant"] == "matrix"]
    if not rows:
        sys.exit("error: no matrix rows")
    gate_transforms = R1_GATES if exp == "r1" else R2_GATES
    g1_lo, g1_hi = G1_BARS[exp]

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

    if exp == "r1":
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

        if rstar_path:
            with open(rstar_path) as f:
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=("r1", "r2", "v1"), default="r1")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--rstar")
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no data rows")
    if args.exp == "v1":
        return exp_v1(rows)
    return exp_gates(rows, args.exp, args.rstar)


if __name__ == "__main__":
    sys.exit(main())
