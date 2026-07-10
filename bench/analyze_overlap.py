#!/usr/bin/env python3
"""Overlap-trace analysis for bench-poc-transpose nsys captures (issues #47, #67).

Reads a `nsys stats --report cuda_gpu_trace --format csv` export and computes,
for the pipeline phase (the two streams carrying the chunked H2D copies) and
the baseline phase (the single-stream whole-tensor copies):

  - H2D copy-engine utilization over the timed span (busy / span)
  - chunk-issue period (start-to-start gap; the producer cadence)
  - per-chunk copy duration
  - per-iteration wall from first-chunk starts

Warmup iterations are excluded. Chunks per iteration is inferred from the
copy count: pipeline copies = (warmup + timed) * chunks_per_iter.
"""
import argparse
import csv
import statistics
from collections import defaultdict


def load_h2d(path):
    by_stream = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "host-to-device" not in row["Name"].lower():
                continue
            by_stream[row["Strm"]].append(
                (int(row["Start (ns)"]), int(row["Duration (ns)"])))
    return {s: sorted(v) for s, v in by_stream.items()}


def phase_stats(copies, label):
    busy = sum(d for _, d in copies)
    span = copies[-1][0] + copies[-1][1] - copies[0][0]
    print(f"{label}: {len(copies)} H2D copies, busy {busy/1e6:.2f} ms over "
          f"span {span/1e6:.2f} ms -> utilization {100*busy/span:.1f}%")
    return busy, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--timed-iters", type=int, default=5)
    args = ap.parse_args()

    by_stream = load_h2d(args.csv_path)
    if len(by_stream) < 3:
        raise SystemExit(f"expected >=3 streams with H2D copies "
                         f"(2 pipeline + 1 baseline), got {list(by_stream)}")
    # Pipeline = the two busiest streams; baseline = the next one.
    ranked = sorted(by_stream, key=lambda s: -len(by_stream[s]))
    pipe = sorted(by_stream[ranked[0]] + by_stream[ranked[1]])
    base = by_stream[ranked[2]]

    total_iters = args.warmup_iters + args.timed_iters
    if len(pipe) % total_iters:
        raise SystemExit(f"pipeline copy count {len(pipe)} not divisible by "
                         f"{total_iters} iterations")
    chunks = len(pipe) // total_iters
    print(f"inferred chunks/iteration: {chunks} "
          f"(streams: pipeline {ranked[0]},{ranked[1]}; baseline {ranked[2]})")

    timed = pipe[args.warmup_iters * chunks:]
    phase_stats(timed, "pipeline (timed)")

    periods = [b[0] - a[0] for a, b in zip(timed, timed[1:])]
    durs = [d for _, d in timed]
    print(f"chunk-issue period ms: mean {statistics.mean(periods)/1e6:.3f}, "
          f"median {statistics.median(periods)/1e6:.3f}, "
          f"min {min(periods)/1e6:.3f}, max {max(periods)/1e6:.3f}")
    print(f"copy duration ms: mean {statistics.mean(durs)/1e6:.3f}, "
          f"median {statistics.median(durs)/1e6:.3f}")

    starts = [timed[i * chunks][0] for i in range(args.timed_iters)]
    walls = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
    print("per-iteration wall ms (first-chunk-start deltas): "
          + ", ".join(f"{w:.2f}" for w in walls))

    base_timed = base[args.warmup_iters:]
    phase_stats(base_timed, "baseline (timed)")


if __name__ == "__main__":
    main()
