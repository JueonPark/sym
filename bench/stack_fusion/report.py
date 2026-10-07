#!/usr/bin/env python3
"""Gates and tables for stack_fusion.py results (torch.stack support).

Gate 1: every output exact and the run complete.
Gate 2: at >= 8 MiB, FD p50 <= 1.10 x S1 p50 in every configuration that
        stacks at dim 0 or 1 (the last dim is reported descriptively).
Gate 3: at >= 32 MiB, FB (tuned transfer options) p50 < min(B1, B3) p50 in
        every configuration.
Gate 4: two rows by one rule, the smallest measured size from which the
        method's p50 < min(B1, B3) p50 in every configuration at every size
        from there up (none when no size qualifies): FB (tuned options), and
        FBD (RelocBackend's default options), which sets
        DEFAULT_MIN_STACK_BYTES.
Exit status 1 when gate 1, 2 or 3 fails; gate 4 never changes it.
A deviations.md next to the JSON is appended to the output verbatim.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METHODS = ("B1", "B2", "B3", "B4", "B5", "S1", "FD", "FB", "FBD", "GB")
LAST_DIM = 2  # stack_fusion.py stacks rank-2 inputs into rank-3 outputs; dim 2 is the last axis


def medians(data):
    cells = defaultdict(list)
    for row in data["rows"]:
        if "error" not in row:
            cells[(row["total_mib"], row["count"], row["dim"], row["method"])].append(row["p50_ms"])
    return {key: statistics.median(values) for key, values in cells.items()}


def evaluate(data):
    med = medians(data)
    configs = sorted({key[:3] for key in med})
    sizes = sorted({c[0] for c in configs})

    def as_written(c):
        return min(med[(*c, m)] for m in ("B1", "B3") if (*c, m) in med)

    def wins(c, method="FB"):
        return med[(*c, method)] < as_written(c)

    def at_or_above(size, predicate, extra=lambda c: True):
        """(result, note): FAIL with a note instead of a vacuous PASS when no
        configuration reaches `size` (after any `extra` filter)."""
        subset = [c for c in configs if c[0] >= size and extra(c)]
        if not subset:
            return False, f"no configuration at or above {size} MiB"
        return all(predicate(c) for c in subset), None

    def crossover(method):
        """Gate 4's rule for `method`; a configuration without its row is a loss."""
        return next((s for s in sizes
                     if all((*c, method) in med and wins(c, method) for c in configs if c[0] >= s)), None)

    gate1 = bool(data.get("correct")) and bool(data.get("complete"))
    gate2, gate2_note = at_or_above(
        8, lambda c: med[(*c, "FD")] <= 1.10 * med[(*c, "S1")], lambda c: c[2] != LAST_DIM)
    gate3, gate3_note = at_or_above(32, wins)
    thresholds = {method: crossover(method) for method in ("FB", "FBD")}
    last_dim_fd_s1 = [med[(*c, "FD")] / med[(*c, "S1")] for c in configs if c[0] >= 8 and c[2] == LAST_DIM]
    return med, configs, gate1, gate2, gate2_note, gate3, gate3_note, thresholds, as_written, last_dim_fd_s1


def main(path):
    path = Path(path)
    data = json.loads(path.read_text())
    (med, configs, gate1, gate2, gate2_note, gate3, gate3_note, thresholds, as_written,
     last_dim_fd_s1) = evaluate(data)
    meta = data["metadata"]

    def result(passed, note):
        return "PASS" if passed else ("FAIL" if note is None else f"FAIL ({note})")

    def size(threshold):
        return "none" if threshold is None else f"{threshold} MiB"

    print("# Fused host `torch.stack` → H2D: completed-call latency\n")
    print(f"Source `{meta['source_revision']}` (dirty: {meta['source_dirty']}); torch {meta['torch']}, "
          f"CUDA {meta['cuda']}; GPU {meta['gpu']}; Inductor compile threads {meta['inductor_compile_threads']}. "
          f"{meta['clocks']}. Median of {meta['args']['rounds']} shuffled-round p50s "
          f"({meta['args']['samples']} samples after {meta['args']['warmup']} warmups); every output "
          "compared with `torch.equal`. Raw samples: [latency.json](latency.json). "
          "Harness: [bench/stack_fusion](../../stack_fusion/README.md).\n")
    staging = data.get("backend_stats", {}).get("FBD", {}).get("staging_decisions")
    print(f"Measured configuration: {meta['threads']} Torch threads ({meta['interop_threads']} interop) on "
          f"CPUs {meta['affinity']}; Sym's methods run on warm, retained transfer resources. "
          f"FB, FD and S1 pass the tuned transfer options `{meta['tuned_transfer_options']}`. "
          "FBD and GB pass no `transfer_options`, so they run `RelocBackend()`'s default transfer options "
          f"`{meta['default_transfer_options']}`"
          + (f" (FBD's staging decisions: `{staging}`)" if staging else "")
          + f". GB's `min_stack_bytes` is {meta['gb_min_stack_bytes']} bytes, above every measured size, "
          "so every GB call falls back to PyTorch.\n")
    print("## Gates\n")
    print("| Gate | Condition | Result |\n|---|---|---|")
    print(f"| 1 | every output exact, run complete | {'PASS' if gate1 else 'FAIL'} |")
    print(f"| 2 | ≥ 8 MiB, dim 0 or 1: FD ≤ 1.10 × S1 | {result(gate2, gate2_note)} |")
    print(f"| 3 | ≥ 32 MiB: FB (tuned options) < min(B1, B3) | {result(gate3, gate3_note)} |")
    print(f"| 4 (tuned options) | FB: smallest size where FB < min(B1, B3) from there up | "
          f"{size(thresholds['FB'])} |")
    print(f"| 4 (default options) | FBD: smallest size where FBD < min(B1, B3) from there up; sets "
          f"`DEFAULT_MIN_STACK_BYTES` | {size(thresholds['FBD'])} |\n")
    if last_dim_fd_s1:
        print(f"At ≥ 8 MiB, last-dim (dim {LAST_DIM}) stacks ranged FD/S1 "
              f"{min(last_dim_fd_s1):.3f}–{max(last_dim_fd_s1):.3f} (reported descriptively, not gated).\n")
    else:
        print(f"No last-dim (dim {LAST_DIM}) configuration at or above 8 MiB.\n")
    print("## Latency (ms)\n")
    print("| MiB | N | dim | " + " | ".join(METHODS) + " | FD/S1 | as-written/FB | as-written/FBD |")
    print("|" + "---:|" * (3 + len(METHODS) + 3))
    for c in configs:
        cells = [f"{med[(*c, m)]:.3f}" if (*c, m) in med else "—" for m in METHODS]
        fd = f"{med[(*c, 'FD')] / med[(*c, 'S1')]:.2f}" if (*c, "FD") in med else "—"
        speedups = [f"{as_written(c) / med[(*c, m)]:.2f}×" if (*c, m) in med else "—" for m in ("FB", "FBD")]
        print(f"| {c[0]} | {c[1]} | {c[2]} | " + " | ".join(cells) + f" | {fd} | " + " | ".join(speedups) + " |")
    errors = [row for row in data["rows"] if "error" in row]
    if errors:
        print("\nInductor baselines that failed to compile (excluded):")
        for row in errors:
            print(f"- {row['total_mib']} MiB N={row['count']} dim={row['dim']} {row['method']}: {row['error']}")
    deviations = path.with_name("deviations.md")
    if deviations.is_file():
        print("\n" + deviations.read_text().strip("\n"))
    return 0 if gate1 and gate2 and gate3 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: report.py <latency.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
