#!/usr/bin/env python3
"""R0.3 (issue #76): Figure 1 -- A/B speedup per transform per machine.

  python3 bench/rtrack/figure1.py --csv gen3.csv gen4.csv \
      [--n N] [--threads T] [--out figure1.png]

For each (machine, transform): pick the row set at --n (default: the
largest N present for that machine) and --threads (default: the largest
present), take the BEST chunk size per method independently (min
median_ms -- issue #76: Method A and B may prefer different C), and plot
speedup = median_ms(B at best C) / median_ms(A at best C). Bars above
the 1.0 line mean Method A (CPU transform + reduced DMA) wins. Bars
built from any unstable-flagged row are hatched.
"""

import argparse
import csv
import sys
from collections import defaultdict

TRANSFORM_ORDER = ["transpose", "blocked_transpose", "transpose_quant",
                   "quant", "nchw_nhwc_quant", "convert_f16"]


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(l for l in f if not l.startswith("#"))
            for row in reader:
                for k in ("N", "threads"):
                    row[k] = int(row[k])
                for k in ("median_ms", "chunk_req_mib"):
                    row[k] = float(row[k])
                row["unstable"] = row["unstable"] == "1"
                rows.append(row)
    return rows


def best_per_method(rows):
    """rows -> {method: best row by median_ms} (best chunk per method)."""
    best = {}
    for row in rows:
        m = row["method"]
        if m not in best or row["median_ms"] < best[m]["median_ms"]:
            best[m] = row
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="figure1.png")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        sys.exit("error: no data rows")

    groups = defaultdict(list)  # (machine, transform) -> rows
    for row in rows:
        groups[(row["machine"], row["transform"])].append(row)

    # (machine, transform) -> (speedup, bestA, bestB, unstable)
    results = {}
    chosen = defaultdict(set)  # machine -> {(N, threads)} actually plotted
    for (machine, transform), grp in sorted(groups.items()):
        n = args.n or max(r["N"] for r in grp)
        grp = [r for r in grp if r["N"] == n]
        threads = args.threads or max((r["threads"] for r in grp), default=0)
        grp = [r for r in grp if r["threads"] == threads]
        best = best_per_method(grp)
        if "a" not in best or "b" not in best:
            print(f"figure1: skipping {machine}/{transform} (need both "
                  f"methods at N={n} T={threads})", file=sys.stderr)
            continue
        a, b = best["a"], best["b"]
        results[(machine, transform)] = (
            b["median_ms"] / a["median_ms"], a, b,
            a["unstable"] or b["unstable"])
        chosen[machine].add((n, threads))

    if not results:
        sys.exit("error: no (machine, transform) group has both methods")

    # The per-group max-N default can silently mix problem sizes when a
    # sweep aborted partway (e.g. only some transforms reached N=16384).
    machine_label = {}
    for machine, nts in chosen.items():
        if len(nts) == 1:
            (n, threads), = nts
            machine_label[machine] = f"{machine} (N={n}, T={threads})"
        else:
            machine_label[machine] = f"{machine} (MIXED N/T)"
            print(f"figure1: WARNING {machine} mixes configs {sorted(nts)} "
                  "in one figure — the CSV is likely from an aborted sweep; "
                  "pass --n/--threads to pin one config", file=sys.stderr)

    machines = sorted({m for m, _ in results})
    transforms = [t for t in TRANSFORM_ORDER
                  if any((m, t) in results for m in machines)]
    # Never silently drop transforms the frozen order predates.
    transforms += sorted({t for _, t in results} - set(TRANSFORM_ORDER))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(transforms), 4.5))
    width = 0.8 / max(1, len(machines))
    for mi, machine in enumerate(machines):
        xs, ys, hatches, labels = [], [], [], []
        for ti, transform in enumerate(transforms):
            key = (machine, transform)
            if key not in results:
                continue
            speedup, a, b, unstable = results[key]
            xs.append(ti + (mi - (len(machines) - 1) / 2) * width)
            ys.append(speedup)
            hatches.append("//" if unstable else None)
            labels.append(f"A:{a['chunk_req_mib']:.0f}M\n"
                          f"B:{b['chunk_req_mib']:.0f}M")
        bars = ax.bar(xs, ys, width * 0.9, label=machine_label[machine])
        for bar, h, lab, y in zip(bars, hatches, labels, ys):
            if h:
                bar.set_hatch(h)
            ax.annotate(f"{y:.2f}x\n{lab}",
                        (bar.get_x() + bar.get_width() / 2, y),
                        ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    top = max(s for s, _, _, _ in results.values())
    ax.set_ylim(0, max(1.1, top) * 1.25)  # headroom for the annotations
    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels(transforms, rotation=15)
    ax.set_ylabel("speedup  median_ms(B) / median_ms(A)")
    ax.set_title("Method A vs Method B (best chunk per method); "
                 "hatched = unstable rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure1: wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
