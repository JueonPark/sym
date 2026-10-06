#!/usr/bin/env python3
"""Gates and tables for stack_fusion.py results (torch.stack support).

Gate 1: every output exact and the run complete.
Gate 2: at >= 8 MiB, FD p50 <= 1.10 x S1 p50 in every configuration.
Gate 3: at >= 32 MiB, FB p50 < min(B1, B3) p50 in every configuration.
Gate 4: the default threshold is the smallest measured size from which gate
        3's condition holds in every configuration.
Exit status 1 when gate 1, 2 or 3 fails.
"""
import json
import statistics
import sys
from collections import defaultdict

METHODS = ("B1", "B2", "B3", "B4", "B5", "S1", "FD", "FB", "GB")


def medians(data):
    cells = defaultdict(list)
    for row in data["rows"]:
        if "error" not in row:
            cells[(row["total_mib"], row["count"], row["dim"], row["method"])].append(row["p50_ms"])
    return {key: statistics.median(values) for key, values in cells.items()}


def evaluate(data):
    med = medians(data)
    configs = sorted({key[:3] for key in med})

    def as_written(c):
        return min(med[(*c, m)] for m in ("B1", "B3") if (*c, m) in med)

    def wins(c):
        return med[(*c, "FB")] < as_written(c)

    gate1 = bool(data.get("correct")) and bool(data.get("complete"))
    gate2 = all(med[(*c, "FD")] <= 1.10 * med[(*c, "S1")] for c in configs if c[0] >= 8)
    gate3 = all(wins(c) for c in configs if c[0] >= 32)
    sizes = sorted({c[0] for c in configs})
    threshold = next((s for s in sizes if all(wins(c) for c in configs if c[0] >= s)), None)
    return med, configs, gate1, gate2, gate3, threshold, as_written


def main(path):
    data = json.load(open(path))
    med, configs, gate1, gate2, gate3, threshold, as_written = evaluate(data)
    meta = data["metadata"]
    print("# Fused host `torch.stack` → H2D: completed-call latency\n")
    print(f"Source `{meta['source_revision']}` (dirty: {meta['source_dirty']}); torch {meta['torch']}, "
          f"CUDA {meta['cuda']}; GPU {meta['gpu']}; affinity {meta['affinity']}; "
          f"{meta['threads']} threads; Inductor compile threads {meta['inductor_compile_threads']}. "
          f"{meta['clocks']}. Median of {meta['args']['rounds']} shuffled-round p50s "
          f"({meta['args']['samples']} samples after {meta['args']['warmup']} warmups); every output "
          "compared with `torch.equal`. Raw samples: [latency.json](latency.json). "
          "Harness: [bench/stack_fusion](../../stack_fusion/README.md).\n")
    print("## Gates\n")
    print("| Gate | Condition | Result |\n|---|---|---|")
    print(f"| 1 | every output exact, run complete | {'PASS' if gate1 else 'FAIL'} |")
    print(f"| 2 | ≥ 8 MiB: FD ≤ 1.10 × S1 | {'PASS' if gate2 else 'FAIL'} |")
    print(f"| 3 | ≥ 32 MiB: FB < min(B1, B3) | {'PASS' if gate3 else 'FAIL'} |")
    print(f"| 4 | smallest size where FB < min(B1, B3) from there up | "
          f"{'none' if threshold is None else f'{threshold} MiB'} |\n")
    print("## Latency (ms)\n")
    print("| MiB | N | dim | " + " | ".join(METHODS) + " | FD/S1 | as-written/FB |")
    print("|" + "---:|" * (3 + len(METHODS) + 2))
    for c in configs:
        cells = [f"{med[(*c, m)]:.3f}" if (*c, m) in med else "—" for m in METHODS]
        fd = f"{med[(*c, 'FD')] / med[(*c, 'S1')]:.2f}" if (*c, "FD") in med else "—"
        fb = f"{as_written(c) / med[(*c, 'FB')]:.2f}×" if (*c, "FB") in med else "—"
        print(f"| {c[0]} | {c[1]} | {c[2]} | " + " | ".join(cells) + f" | {fd} | {fb} |")
    errors = [row for row in data["rows"] if "error" in row]
    if errors:
        print("\nInductor baselines that failed to compile (excluded):")
        for row in errors:
            print(f"- {row['total_mib']} MiB N={row['count']} dim={row['dim']} {row['method']}: {row['error']}")
    return 0 if gate1 and gate2 and gate3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
