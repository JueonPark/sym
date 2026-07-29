#!/usr/bin/env python3
"""V2 (issue #96): Figure 1 re-drawn with the ISOLATED VARIABLE labelled
per series, per the issue's acceptance criterion — instead of "machine A
vs machine B", each series names what actually changes:

  - the two gen4-box arms differ ONLY in ISA dispatch (same box, same
    link, same session): the within-box ISA contrast;
  - the gen3-box arm holds the ISA fixed (avx2) and changes the box: the
    cross-box link+host-generation contrast (the V2 plan change recorded
    in docs/v2-isolation.md means link is NOT isolated from host here).

Input: the per-arm figure_rstar JSONs (speedup_measured per family/r).
Color follows the box (entity); linestyle carries the ISA arm, so the
ISA-null shows as two same-color curves lying on top of each other.

  python3 bench/rtrack/v2_figure.py \
      --gen4-avx512 bench/results/v2_isa_rstar_avx512.json \
      --gen4-avx2   bench/results/v2_isa_rstar_avx2.json \
      --gen3-avx2   bench/results/v2_isa_gen3_rstar_avx2_<M>.json \
      --out bench/results/v2_figure1_isolated_variables.png
"""

import argparse
import json

# Okabe-Ito colorblind-safe pair; color = box, linestyle = ISA arm.
STYLE = {
    "gen4_avx512": ("#0072B2", "-", "gen4 box (7800X3D) · avx512"),
    "gen4_avx2": ("#0072B2", "--", "gen4 box (7800X3D) · avx2"),
    "gen3_avx2": ("#E69F00", "--", "gen3 box (EPYC/2080Ti) · avx2"),
}
FAMILIES = ("quant", "blocked_transpose")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen4-avx512", required=True)
    ap.add_argument("--gen4-avx2", required=True)
    ap.add_argument("--gen3-avx2", required=True)
    ap.add_argument("--out", default="v2_figure1_isolated_variables.png")
    args = ap.parse_args()

    arms = {
        "gen4_avx512": json.load(open(args.gen4_avx512)),
        "gen4_avx2": json.load(open(args.gen4_avx2)),
        "gen3_avx2": json.load(open(args.gen3_avx2)),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for key, (color, ls, label) in STYLE.items():
            fam = arms[key]["families"].get(family)
            if fam is None:
                continue
            pts = sorted(
                (float(r), s) for r, s in fam["speedup_measured"].items()
            )
            rs = [p[0] for p in pts]
            sp = [p[1] for p in pts]
            ax.plot(rs, sp, ls, color=color, marker="o", ms=5, label=label)
            rstar = fam.get("rstar_measured")
            if rstar is not None:
                ax.axvline(rstar, color=color, ls=":", lw=1, alpha=0.5)
                # Per-arm offsets so the two same-box annotations (whose
                # r* values nearly coincide -- the ISA null) don't collide.
                dy = {"gen4_avx512": 6, "gen4_avx2": 18, "gen3_avx2": -14}
                ax.annotate(
                    f"r*={rstar:.2f}",
                    (rstar, 1.0),
                    textcoords="offset points",
                    xytext=(4, dy[key]),
                    fontsize=8,
                    color=color,
                )
        ax.axhline(1.0, color="#666666", lw=1)
        ax.set_title(family)
        ax.set_xlabel("r (bytes shipped by A / bytes shipped by B)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("A / B_fair speedup (>1: A wins)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "V2: same-color pair = ISA contrast at fixed box/link (null); "
        "color change = box contrast at fixed ISA (link+host-generation)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"v2_figure: wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
