#!/usr/bin/env python3
"""R2 / EXP-2 (issue #83): A/B speedup vs r per family + measured and
model-predicted critical r*.

  python3 bench/rtrack/figure_rstar.py --csv rsweep.csv \
      [--rooflines r2_rooflines/*.json] [--h2d GBPS] [--n N] [--threads T] \
      [--b-method b|b_fair] [--out figure_rstar.png] [--json rstar.json]

Measured: variant=rsweep rows at --n/--threads (defaults: largest present
per family), best chunk per (method, r); speedup(r) = median_ms(B at
r=1.0) / median_ms(A at r). r*_measured = the 1.0 crossing, interpolated
linearly in log2(r); None when the curve never crosses in [0.125, 1.0].
--b-method picks the Method-B baseline rows (issue #95): "b" is the
staged baseline (default, the R2 figure), "b_fair" the admissible
pinned-source one; the choice is recorded in the JSON as "b_method".

Model (stage rooflines, effective INPUT GB/s, H2D from --h2d or derived
from the CSV's method-b h2d_ms):
  pipelined: BW_A(r) = min(BW_cpu(family, r), H2D / r); BW_B = H2D
  serial:    1 / BW_A(r) = 1 / BW_cpu + r / H2D
BW_cpu comes from cpu_rooflines JSONs; two-pass stages compose
harmonically in source-normalized GB/s (pack reads S/4 bytes, so its
source-normalized BW is 4x its measured input BW). A crossover exists in
range iff BW_cpu(family, r) > BW_B somewhere -- the host transform must
beat the link.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict

R_POINTS = [1.0, 0.5, 0.25, 0.125]
FAMILY_PLAN = {"quant": "identity", "blocked_transpose": "blocked",
               "transpose_quant": "transpose", "nchw_nhwc_quant": "nchw"}


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


def load_rooflines(paths):
    """-> {(plan, threads): {kernel: in_gb_per_s}} at the largest N seen."""
    best = {}
    for path in paths or []:
        with open(path) as f:
            doc = json.load(f)
        cfg = doc["config"]
        key = (cfg["plan"], cfg["threads"])
        if key in best and best[key][0] >= cfg["N"]:
            continue
        best[key] = (cfg["N"], {k: v["in_gb_per_s"]
                                for k, v in doc["kernels"].items()})
    return {k: v[1] for k, v in best.items()}


def cpu_bw(kernels, family, r):
    """Source-normalized CPU GB/s for the family's Method-A stage at r."""
    def h(*bws):  # harmonic composition of sequential passes
        return 1.0 / sum(1.0 / b for b in bws)
    contig = family == "quant"
    if r == 1.0:
        return kernels["contig_read"] if contig else kernels["gather_f32"]
    if r == 0.5:
        return (kernels["convert_f32_f16"] if contig else
                h(kernels["gather_f32"], kernels["convert_f32_f16"]))
    if r == 0.25:
        return kernels["quantize_pack"] if contig else kernels["gather_quantize"]
    if r == 0.125:
        base = (kernels["quantize_pack"] if contig
                else kernels["gather_quantize"])
        return h(base, 4.0 * kernels["pack_s8_s4"])
    raise ValueError(f"no model for r={r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--rooflines", nargs="*", default=[])
    ap.add_argument("--h2d", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--b-method", default="b", choices=("b", "b_fair"))
    ap.add_argument("--out", default="figure_rstar.png")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

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

    rooflines = load_rooflines(args.rooflines)
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
        pred, serial = {}, {}
        kernels = rooflines.get((FAMILY_PLAN.get(fam), threads))
        for rr in sorted(meas):
            if kernels:
                bw = cpu_bw(kernels, fam, rr)
                pred[rr] = min(bw, h2d / rr) / h2d
                serial[rr] = (1.0 / (1.0 / bw + rr / h2d)) / h2d
        pts = sorted(meas.items())
        result["families"][fam] = {
            "speedup_measured": {str(k): v for k, v in pts},
            "speedup_predicted": {str(k): v for k, v in sorted(pred.items())},
            "speedup_serial": {str(k): v for k, v in sorted(serial.items())},
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

    import matplotlib
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
                                  ("speedup_predicted", "s--", "model"),
                                  ("speedup_serial", "^:", "serial bound")):
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
