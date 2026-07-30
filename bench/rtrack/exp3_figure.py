#!/usr/bin/env python3
"""R3 / EXP-3 (issue #84): the scatter-win K-sweep figure.

Plots Method-A / B_xK speedup vs K for the scatter (tensor-parallel) case,
with the 1.0 break-even and the 1.3 G5-gate lines. A second panel shows the
aggregate wall times (A, B_xK, B_staged) so the CPU-transform-bound plateau
of A and the concurrency win of B_xK over B_staged are both visible.

  python3 bench/rtrack/exp3_figure.py --json r3_scatter_n8192.json \
      [--n 8192] [--out exp3_scatter.png]
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="exp3_scatter.png")
    args = ap.parse_args()

    rows = []
    gpu = None
    for path in args.json:
        doc = json.load(open(path))
        gpu = doc["config"]["gpu"]
        rows += [r for r in doc["rows"] if r["scenario"] == "scatter"]
    if not rows:
        sys.exit("error: no scatter rows")
    n = args.n or max(r["N"] for r in rows)
    rows = [r for r in rows if r["N"] == n]
    ks = sorted({r["K"] for r in rows})

    def wall(method, k):
        for r in rows:
            if r["method"] == method and r["K"] == k:
                return r["wall_median_ms"]
        return float("nan")

    speed = [wall("bxk", k) / wall("a", k) for k in ks]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.plot(ks, speed, "o-", color="#1f77b4", label="A / B_xK")
    ax0.axhline(1.0, color="black", ls="--", lw=0.8, label="break-even")
    ax0.axhline(1.3, color="crimson", ls=":", lw=1.0, label="G5 gate (1.3x)")
    for k, s in zip(ks, speed):
        ax0.annotate(f"{s:.2f}x", (k, s), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)
    ax0.set_xticks(ks)
    ax0.set_xlabel("K (GPUs)")
    ax0.set_ylabel("speedup  wall(B_xK) / wall(A)")
    ax0.set_title(f"scatter Method-A amortization vs K (N={n})")
    ax0.set_ylim(0, max(1.4, max(speed) * 1.15))
    ax0.legend(fontsize=8)

    for m, style, lab in (("a", "o-", "A (CPU transform + int8 DMA)"),
                          ("bxk", "s-", "B_xK (fp32 DMA + GPU transform)"),
                          ("bstaged", "^--", "B_staged (serialized)")):
        ax1.plot(ks, [wall(m, k) for k in ks], style, label=lab)
    ax1.set_xticks(ks)
    ax1.set_xlabel("K (GPUs)")
    ax1.set_ylabel("aggregate wall (ms)")
    ax1.set_title("aggregate wall clock")
    ax1.legend(fontsize=8)

    fig.suptitle(f"EXP-3 scatter — {gpu}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"exp3_figure: wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
