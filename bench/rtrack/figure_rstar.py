#!/usr/bin/env python3
"""R2 / EXP-2 (issue #83): A/B speedup vs r per family + measured and
model-predicted critical r*.

  PYTHONPATH=build/sym/python python3 bench/rtrack/figure_rstar.py \
      --csv rsweep.csv [--calibration calibration/<machine>.cal] \
      [--h2d GBPS] [--n N] [--threads T] [--b-method b|b_fair] \
      [--out figure_rstar.png] [--json rstar.json]

Measured: variant=rsweep rows at --n/--threads (defaults: largest present
per family), best chunk per (method, r); speedup(r) = median_ms(B at
r=1.0) / median_ms(A at r). r*_measured = the 1.0 crossing, interpolated
linearly in log2(r); None when the curve never crosses in [0.125, 1.0].
--b-method picks the Method-B baseline rows (issue #95): "b" is the
staged baseline, "b_fair" the admissible pinned-source one; the choice is
recorded in the JSON as "b_method".

Predictions (issue #111): computed exclusively by pyreloc.predict -- the
maintained reloc::costmodel, the same C++ arithmetic decide() uses --
from the --calibration .cal file. speedup_predicted(r) =
t_b_pred(r=1.0) / t_a_pred(r), with b_placement mapped from --b-method
(b -> "overlapped", b_fair -> "serial": a serial-B measurement is
compared against a serial-B prediction, CM1's placement term). Without
--calibration the output is measured-only. The pre-#111 standalone
roofline model (and its A-side "serial bound" series, which has no
pyreloc counterpart) is retired; see docs/cm3-one-implementation.md.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

try:
    import pyreloc  # the single cost-model implementation (issue #111)
except ImportError:
    pyreloc = None  # only fatal when --calibration asks for predictions

R_POINTS = [1.0, 0.5, 0.25, 0.125]
# transform family -> reloc::costmodel Pattern name (the same map
# libreloc/python/tests/test_prediction.py uses).
FAMILY_PATTERN = {"quant": "contiguous", "blocked_transpose": "blocked",
                  "transpose_quant": "single_element",
                  "nchw_nhwc_quant": "tiled"}
# --b-method -> pyreloc b_placement: measured-serial-B (b_fair) is
# compared against a serial-B prediction, staged b against overlapped.
B_METHOD_PLACEMENT = {"b": "overlapped", "b_fair": "serial"}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(l for l in f if not l.startswith("#"))
            for row in reader:
                if (row.get("variant") or "matrix") != "rsweep":
                    continue
                row["N"] = int(row["N"])
                row["threads"] = int(row["threads"])
                for k in ("median_ms", "r", "h2d_ms"):
                    row[k] = float(row[k])
                row["unstable"] = row["unstable"] == "1"
                rows.append(row)
    return rows


def crossing(points):
    """points: [(r, speedup)] sorted ascending in r -> r* or None."""
    for (r0, s0), (r1, s1) in zip(points, points[1:]):
        if (s0 - 1.0) * (s1 - 1.0) <= 0 and s0 != s1:
            t = (1.0 - s0) / (s1 - s0)
            return 2 ** (math.log2(r0) + t * (math.log2(r1) - math.log2(r0)))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--calibration", default=None,
                    help=".cal file for pyreloc.predict predictions "
                         "(issue #111); absent -> measured-only output")
    ap.add_argument("--h2d", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--b-method", default="b", choices=("b", "b_fair"))
    ap.add_argument("--out", default="figure_rstar.png")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    cal = None
    if args.calibration:
        if pyreloc is None:
            sys.exit("error: pyreloc not importable -- build it "
                     "(ninja -C build/sym) and run with "
                     "PYTHONPATH=build/sym/python")
        try:
            cal = pyreloc.load_calibration(args.calibration)
        except ValueError as e:
            sys.exit(f"error: {e}")

    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no rsweep rows (variant column?)")

    n = args.n or max(r["N"] for r in rows)
    rows = [r for r in rows if r["N"] == n]
    threads = args.threads or max(r["threads"] for r in rows)
    rows = [r for r in rows if r["threads"] == threads]

    # H2D floor: explicit, or the best full-fp32 DMA rate from B rows.
    h2d = args.h2d
    if h2d is None:
        cands = [r["N"] * r["N"] * 4 / (r["h2d_ms"] * 1e-3) / 1e9
                 for r in rows
                 if r["method"] == args.b_method and r["h2d_ms"] > 0]
        if not cands:
            sys.exit(f"error: no --h2d and no method-{args.b_method} rows "
                     "to derive it")
        h2d = max(cands)
    print(f"figure_rstar: H2D = {h2d:.2f} GB/s, N={n}, T={threads}",
          file=sys.stderr)

    fams = defaultdict(lambda: defaultdict(list))  # fam -> (method, r) -> rows
    for r in rows:
        fams[r["transform"]][(r["method"], r["r"])].append(r)

    result = {"h2d_gbps": h2d, "n": n, "threads": threads,
              "b_method": args.b_method, "families": {}}
    for fam, grp in sorted(fams.items()):
        if (args.b_method, 1.0) not in grp:
            print(f"figure_rstar: skipping {fam} (no B row)", file=sys.stderr)
            continue
        best_b = min(grp[(args.b_method, 1.0)], key=lambda r: r["median_ms"])
        meas, unstable = {}, best_b["unstable"]
        for rr in R_POINTS:
            if ("a", rr) not in grp:
                continue
            best_a = min(grp[("a", rr)], key=lambda r: r["median_ms"])
            meas[rr] = best_b["median_ms"] / best_a["median_ms"]
            unstable |= best_a["unstable"]
        pred = {}
        if cal is not None and fam in FAMILY_PATTERN:
            placement = B_METHOD_PLACEMENT[args.b_method]
            pattern = FAMILY_PATTERN[fam]
            try:
                t_b1 = pyreloc.predict(
                    cal, pattern=pattern, src_bytes=n * n * 4, r=1.0,
                    threads=threads, b_placement=placement)["t_b_ms"]
                for rr in sorted(meas):
                    t_a = pyreloc.predict(
                        cal, pattern=pattern, src_bytes=n * n * 4, r=rr,
                        threads=threads, b_placement=placement)["t_a_ms"]
                    pred[rr] = t_b1 / t_a
            except ValueError:
                # Missing calibration keys: the family is unmodelable --
                # omit ALL predicted points (test_prediction.py's
                # all-or-nothing modelable convention), never a partial
                # grid.
                pred = {}
        pts = sorted(meas.items())
        result["families"][fam] = {
            "speedup_measured": {str(k): v for k, v in pts},
            "speedup_predicted": {str(k): v for k, v in sorted(pred.items())},
            "rstar_measured": crossing(pts),
            "rstar_predicted": crossing(sorted(pred.items())),
            "unstable": unstable,
        }

    if not result["families"]:
        sys.exit("error: no family had both methods")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"figure_rstar: wrote {args.json_out}", file=sys.stderr)

    try:
        import matplotlib
    except ImportError:
        print("figure_rstar: matplotlib not available -- skipping figure",
              file=sys.stderr)
        return 0
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams_sorted = sorted(result["families"])
    fig, axes = plt.subplots(1, len(fams_sorted),
                             figsize=(4 * len(fams_sorted), 4), sharey=True)
    if len(fams_sorted) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams_sorted):
        d = result["families"][fam]
        for key, style, label in (("speedup_measured", "o-", "measured"),
                                  ("speedup_predicted", "s--", "model")):
            if d[key]:
                xs = [float(k) for k in d[key]]
                ax.plot(xs, list(d[key].values()), style, label=label)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("r (wire bytes / input bytes)")
        title = fam + ("  [UNSTABLE rows]" if d["unstable"] else "")
        for which, mark in (("rstar_measured", "r*"),
                            ("rstar_predicted", "r*_pred")):
            if d[which] is not None:
                ax.axvline(d[which], color="gray", linewidth=0.8)
                ax.annotate(f"{mark}={d[which]:.3f}", (d[which], 1.02),
                            fontsize=7, rotation=90, va="bottom")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("speedup  median_ms(B) / median_ms(A)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Method A vs B across r (N={n}, T={threads}, "
                 f"H2D={h2d:.1f} GB/s, baseline={args.b_method})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure_rstar: wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
