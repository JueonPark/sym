#!/usr/bin/env python3
"""R3 / EXP-3 (issue #84): summarize the multi-GPU amortization matrix and
evaluate gate G5 (scatter, int8, K=4: Method A >= 1.3x B_xK aggregate).

Reads one or more multigpu_reloc.cu JSONs and prints, per (scenario, N), a
K-sweep table of A / B_xK / B_staged aggregate wall times and the A/B_xK
speedup, then the G5 verdict. Also reports the delivery-only (DMA) reading,
which separates the CPU-transform cost from the transfer.

  python3 bench/rtrack/exp3_gate.py --json r3_scatter_n8192.json ...
"""

import argparse
import json
import sys
from collections import defaultdict

G5_BAR = 1.3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", required=True)
    args = ap.parse_args()

    # (scenario, N, K) -> {method: row}
    rows = defaultdict(dict)
    gpu = None
    for path in args.json:
        doc = json.load(open(path))
        gpu = doc["config"]["gpu"]
        for r in doc["rows"]:
            rows[(r["scenario"], r["N"], r["K"])][r["method"]] = r

    print(f"# R3 / EXP-3 multi-GPU amortization — {gpu}\n")
    g5 = None
    scen_n = sorted({(s, n) for (s, n, _) in rows})
    for scenario, n in scen_n:
        print(f"## {scenario}  N={n}")
        print("| K | A wall ms | B_xK wall ms | B_staged wall ms | "
              "A/B_xK | A DMA-only ms |")
        print("|---|---|---|---|---|---|")
        ks = sorted({k for (s, nn, k) in rows if s == scenario and nn == n})
        for k in ks:
            cell = rows[(scenario, n, k)]
            a = cell.get("a", {})
            bk = cell.get("bxk", {})
            bs = cell.get("bstaged", {})
            sp = a.get("speedup_vs_bxk", 0.0)
            print(f"| {k} | {a.get('wall_median_ms', 0):.2f} | "
                  f"{bk.get('wall_median_ms', 0):.2f} | "
                  f"{bs.get('wall_median_ms', 0):.2f} | {sp:.2f}x | "
                  f"{a.get('dma_ms', 0):.2f} |")
            if scenario == "scatter" and k == 4:
                g5 = sp
        print()

    if g5 is not None:
        verdict = "PASS" if g5 >= G5_BAR else "FAIL"
        print(f"## Gate G5 (scatter int8 K=4: A >= {G5_BAR}x B_xK)")
        print(f"measured A/B_xK = {g5:.2f}x  ->  **{verdict}**")
    else:
        print("## Gate G5: no scatter K=4 row found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
