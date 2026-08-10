#!/usr/bin/env python3
"""Pre-registered rtrack gates. Bars are FIXED in code, before any data is
read. Select the experiment with --exp:

  --exp r1  (default)  R1 / EXP-1 Gen3 gates (issue #82)
  --exp r2             R2 / EXP-2 Gen4 gates (issue #83)
  --exp v1             V1 baseline-admissibility bar (issue #95)
  --exp bp             BP / two-stream overlap-fair gates (issue #115)

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

BP two-stream overlap-fair gates (issue #115), registered pre-data: G1 checks
b_pipelined effective input BW >= 0.95x pinned H2D (r=1.0 only, dispatch-
overhead caution per machine); G2 assesses per-chunk kernel exposure
(1 - h2d_occupancy, GPU pipeline span denominator) <= 5% on multi-chunk rows
N >= 8192; G3 measures A/B_pipelined ratio vs 24 registered predictions
(fair baseline / 1.06 or 1.09 per machine, see issue #108#issuecomment-5215567565)
within ±0.10 tolerance. Always exits 0 (reports, never gates CI).

  python3 bench/rtrack/gates.py [--exp r1|r2|v1|bp] --csv run.csv [more.csv ...]
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
B_METHODS = ("b", "b_fair", "b_pipelined")  # b_pipelined joins in BP2 (#115) -- identical DMA path
V1_BAR_FRACTION = 0.90  # effective input BW must reach this x pinned H2D

# --- BP (issue #115) registered bars, FIXED before any bp_* data exists ---
# BP-G1: b_pipelined effective input BW on r=1.0 workloads >= 0.95x the
# session's pinned H2D (derived in-CSV per machine, pinned_h2d_gbps).
BP_G1_FRACTION = 0.95
# Strict at every N on Gen3 bare metal; N >= 8192 on Gen4/WSL2 (small-N
# dispatch overhead pre-excluded, not post-excused -- issue #115). Unknown
# machines default to strict (0).
BP_G1_STRICT_MIN_N = {"epyc7351-2080ti": 0, "7800x3d-4070tis": 8192}
# BP-G2: per-chunk kernel exposure <= 5% at N >= 8192. Registered exposure
# metric: 1 - h2d_occupancy (single committed CSV column; median of
# per-iteration event ratios; denominator is the GPU pipeline span -- for
# B methods cpu_stage_ms = 0, so span ~= wall). n_chunks > 1 only: a
# single-chunk config cannot overlap for ANY design (BP1, issue #114).
BP_G2_EXPOSURE_BAR = 0.05
BP_G2_MIN_N = 8192
# BP-G3: measured best(effective_input_GBps, a)/best(..., b_pipelined) per
# (transform, N) within +-0.10 (absolute) of the registered prediction.
BP_G3_TOL = 0.10
# Predictions = committed V1 A/B_fair best-chunk baselines (per-analysis-
# point stabler-preference between the gen4 matrix and rerun CSVs -- the
# docs/r2-exp2-gen4-crossover.md:48-51 pre-declared rule) divided by the
# #108 improvement factors: Gen4 / 1.09 ("B improves ~9%"), Gen3 / 1.06
# ("~6%"). The Gen3 anchor is the FAIR baseline -- #108's staged-anchored
# range was a slip, corrected pre-data on the issue:
# https://github.com/JueonPark/sym/issues/108#issuecomment-5215567565
# Families are exactly the three #108's expectations name; T2 is
# b_pipelined-N/A (chunkability audit, #114).
BP_G3_DIVISOR = {"epyc7351-2080ti": 1.06, "7800x3d-4070tis": 1.09}
BP_G3_PREDICTIONS = {
    # (machine, transform, N): predicted A/B_pipelined
    # -- Gen3, from v1_gen3_nsweep_epyc_2080ti.csv A/B_fair / 1.06:
    ("epyc7351-2080ti", "quant", 2048): 1.3598,
    ("epyc7351-2080ti", "quant", 4096): 1.3479,
    ("epyc7351-2080ti", "quant", 8192): 1.4042,
    ("epyc7351-2080ti", "quant", 16384): 1.4389,
    ("epyc7351-2080ti", "convert_f16", 2048): 0.9322,
    ("epyc7351-2080ti", "convert_f16", 4096): 1.0259,
    ("epyc7351-2080ti", "convert_f16", 8192): 1.0644,
    ("epyc7351-2080ti", "convert_f16", 16384): 1.0801,
    ("epyc7351-2080ti", "blocked_transpose", 2048): 0.7075,
    ("epyc7351-2080ti", "blocked_transpose", 4096): 0.8253,
    ("epyc7351-2080ti", "blocked_transpose", 8192): 0.6196,
    ("epyc7351-2080ti", "blocked_transpose", 16384): 0.6405,
    # -- Gen4, from v1_gen4_matrix_nsweep(+_rerun) A/B_fair / 1.09:
    ("7800x3d-4070tis", "quant", 2048): 2.0161,
    ("7800x3d-4070tis", "quant", 4096): 2.385,
    ("7800x3d-4070tis", "quant", 8192): 1.3434,
    ("7800x3d-4070tis", "quant", 16384): 1.4133,
    ("7800x3d-4070tis", "convert_f16", 2048): 1.3547,
    ("7800x3d-4070tis", "convert_f16", 4096): 1.2708,
    ("7800x3d-4070tis", "convert_f16", 8192): 1.1211,
    ("7800x3d-4070tis", "convert_f16", 16384): 1.1546,
    ("7800x3d-4070tis", "blocked_transpose", 2048): 0.8018,
    ("7800x3d-4070tis", "blocked_transpose", 4096): 0.7388,
    ("7800x3d-4070tis", "blocked_transpose", 8192): 0.7324,
    ("7800x3d-4070tis", "blocked_transpose", 16384): 0.5937,
}
BP_BIMODAL_PATH = "bench/results/cm4_bimodal_cells.json"
BP_CM4_PATH = "bench/results/cm4_registered_predictions.json"


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


def exp_bp(rows, bimodal_path=BP_BIMODAL_PATH, cm4_path=BP_CM4_PATH):
    """BP pre-registered gates (issue #115). Prints verdicts and returns
    {(machine, gate): verdict} so --selftest can assert on them. Always
    exit 0 via main() -- reports, never gates CI."""
    # sections 4-5 land with the selftest (#115 Task 2)
    verdicts = {}
    by_machine = defaultdict(list)
    for r in rows:
        by_machine[r["machine"]].append(r)

    for machine in sorted(by_machine):
        mrows = by_machine[machine]
        print(f"\n=== {machine} ===")

        # BP-G1: admissibility of b_pipelined against the session's link.
        pinned = pinned_h2d_gbps(mrows)
        bar = BP_G1_FRACTION * pinned
        min_n = BP_G1_STRICT_MIN_N.get(machine, 0)
        g1_rows = [r for r in mrows if r["method"] == "b_pipelined"
                   and r["r"] == 1.0 and r["N"] >= min_n]
        best = defaultdict(float)
        for r in g1_rows:
            key = (r["transform"], r["N"])
            best[key] = max(best[key], r["effective_input_GBps"])
        g1_ok, g1_seen = True, False
        print(f"BP-G1: pinned H2D {pinned:.2f} GB/s -> bar "
              f"{BP_G1_FRACTION:g} x = {bar:.2f} GB/s"
              + (f" (N >= {min_n} only)" if min_n else " (every N)"))
        for (t, n) in sorted(best):
            g1_seen = True
            ok = best[(t, n)] >= bar
            g1_ok &= ok
            print(f"  {t} N={n}: {best[(t, n)]:.2f} GB/s"
                  f"{'' if ok else ' !'}")
        v = "PASS" if (g1_seen and g1_ok) else (
            "no data" if not g1_seen else "FAIL")
        verdicts[(machine, "BP-G1")] = v
        print(f"BP-G1 verdict: {v}")

        # BP-G2: per-chunk kernel exposure = 1 - h2d_occupancy of the
        # best-chunk row, kernel-bearing b_pipelined rows only,
        # N >= BP_G2_MIN_N and n_chunks > 1 (BP1: single-chunk configs
        # cannot overlap by construction -- filtered, reason printed).
        g2_ok, g2_seen, g2_skipped = True, False, 0
        best_rows = {}
        for r in mrows:
            if r["method"] != "b_pipelined" or r["N"] < BP_G2_MIN_N:
                continue
            if float(r.get("gpu_kernel_ms") or 0) <= 0:
                continue  # DMA-only rows have no kernel to expose
            if int(r.get("n_chunks") or 0) <= 1:
                g2_skipped += 1
                continue
            occ = r.get("h2d_occupancy")
            if occ is None or occ == "":
                continue  # pre-BP1 CSV: no occupancy column
            key = (r["transform"], r["N"])
            prev = best_rows.get(key)
            if prev is None or r["effective_input_GBps"] > \
                    prev["effective_input_GBps"]:
                best_rows[key] = r
        print(f"BP-G2: exposure = 1 - h2d_occupancy (event-derived; "
              f"denominator is the GPU pipeline span, ~= wall for B "
              f"methods) <= {BP_G2_EXPOSURE_BAR:g} at N >= {BP_G2_MIN_N}; "
              f"{g2_skipped} single-chunk rows filtered (no overlap "
              "possible by construction, issue #114)")
        for (t, n) in sorted(best_rows):
            g2_seen = True
            exposure = 1.0 - float(best_rows[(t, n)]["h2d_occupancy"])
            ok = exposure <= BP_G2_EXPOSURE_BAR
            g2_ok &= ok
            print(f"  {t} N={n}: exposure {exposure:.3f}"
                  f"{'' if ok else ' !'}")
        v = "PASS" if (g2_seen and g2_ok) else (
            "no data" if not g2_seen else "FAIL")
        verdicts[(machine, "BP-G2")] = v
        print(f"BP-G2 verdict: {v}")

        # BP-G3: measured a/b_pipelined vs the registered prediction.
        best = defaultdict(float)
        for r in mrows:
            key = (r["transform"], r["N"], r["method"])
            best[key] = max(best[key], r["effective_input_GBps"])
        g3_ok, g3_seen = True, False
        print(f"BP-G3: |measured A/B_pipelined - prediction| <= "
              f"{BP_G3_TOL:g} per registered cell (fair baseline / "
              f"{BP_G3_DIVISOR.get(machine, '?')}; misses investigated, "
              "not tuned -- #115)")
        for (pm, t, n), pred in sorted(BP_G3_PREDICTIONS.items()):
            if pm != machine:
                continue
            a = best[(t, n, "a")]
            b = best[(t, n, "b_pipelined")]
            if a <= 0 or b <= 0:
                print(f"  {t} N={n}: -- (pred {pred:.4f})")
                continue
            g3_seen = True
            ratio = a / b
            ok = abs(ratio - pred) <= BP_G3_TOL
            g3_ok &= ok
            print(f"  {t} N={n}: measured {ratio:.4f} vs pred {pred:.4f}"
                  f"{'' if ok else ' !'}")
        v = "PASS" if (g3_seen and g3_ok) else (
            "no data" if not g3_seen else "FAIL")
        verdicts[(machine, "BP-G3")] = v
        print(f"BP-G3 verdict: {v}")

    return verdicts


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
    ap.add_argument("--exp", choices=("r1", "r2", "v1", "bp"), default="r1")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--rstar")
    ap.add_argument("--bimodal", default=BP_BIMODAL_PATH)
    ap.add_argument("--cm4", default=BP_CM4_PATH)
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no data rows")
    if args.exp == "bp":
        exp_bp(rows, args.bimodal, args.cm4)
        return 0
    if args.exp == "v1":
        return exp_v1(rows)
    return exp_gates(rows, args.exp, args.rstar)


if __name__ == "__main__":
    sys.exit(main())
