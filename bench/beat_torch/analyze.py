#!/usr/bin/env python3
"""Attribute an Nsight Systems capture of bench/beat_torch/compare.py --trace.

For every NVTX range ``beat_torch/<case>/<path>/<sample>`` the report gives
the wall time, the union of DMA (memcpy) activity, the union of kernel
activity, the time the GPU had neither, bytes per direction, copy/kernel
counts, the delay until the first DMA and the longest DMA gap. Medians over
samples are printed per path, so the tables say *where* each path spends
its time rather than merely which one is faster.

    python bench/beat_torch/analyze.py capture.sqlite [--json out.json]
"""
import argparse
import json
import sqlite3
import statistics
from collections import defaultdict

COPY_KIND = {1: 'H2D', 2: 'D2H', 8: 'D2D', 10: 'D2D'}


def union(intervals):
    """Total length of the union of [start, end) intervals (ns)."""
    total, current = 0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        total += current[1] - current[0]
    return total


def overlapping(rows, start, end):
    return [(max(s, start), min(e, end)) for s, e in rows if s < end and e > start]


def load(path):
    db = sqlite3.connect(path)
    ranges = db.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'beat_torch/%' AND end IS NOT NULL").fetchall()
    copies = db.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    kernels = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName").fetchall()
    return ranges, copies, kernels


def attribute(ranges, copies, kernels):
    rows = defaultdict(list)
    for start, end, text in ranges:
        _, case, path, _ = text.split('/')
        inside = [(s, e, b, k) for s, e, b, k in copies if s < end and e > start]
        kernel_rows = [(s, e, n) for s, e, n in kernels if s < end and e > start]
        dma = overlapping([(s, e) for s, e, _, _ in inside], start, end)
        gpu = overlapping([(s, e) for s, e, _ in kernel_rows], start, end)
        gaps = []
        ordered = sorted(dma)
        for (s0, e0), (s1, e1) in zip(ordered, ordered[1:]):
            gaps.append(max(0, s1 - e0))
        top = defaultdict(int)
        for s, e, name in kernel_rows:
            top[name] += min(e, end) - max(s, start)
        rows[(case, path)].append(dict(
            wall_ms=(end - start) / 1e6,
            dma_busy_ms=union(dma) / 1e6,
            kernel_busy_ms=union(gpu) / 1e6,
            gpu_idle_ms=((end - start) - union(dma + gpu)) / 1e6,
            h2d_mib=sum(b for _, _, b, k in inside if COPY_KIND.get(k) == 'H2D') / 2 ** 20,
            d2h_mib=sum(b for _, _, b, k in inside if COPY_KIND.get(k) == 'D2H') / 2 ** 20,
            copies=len(inside), kernels=len(kernel_rows),
            first_dma_ms=((min(s for s, _ in dma) - start) / 1e6) if dma else None,
            max_dma_gap_ms=(max(gaps) / 1e6) if gaps else 0.0,
            top_kernels={name: ns / 1e6 for name, ns in sorted(top.items(), key=lambda kv: -kv[1])[:5]},
        ))
    return rows


def summarize(rows):
    out = {}
    for (case, path), samples in sorted(rows.items()):
        summary = {}
        for key in ('wall_ms', 'dma_busy_ms', 'kernel_busy_ms', 'gpu_idle_ms', 'h2d_mib', 'd2h_mib',
                    'copies', 'kernels', 'max_dma_gap_ms'):
            summary[key] = statistics.median(s[key] for s in samples)
        first = [s['first_dma_ms'] for s in samples if s['first_dma_ms'] is not None]
        summary['first_dma_ms'] = statistics.median(first) if first else None
        kernels = defaultdict(list)
        for s in samples:
            for name, ms in s['top_kernels'].items():
                kernels[name].append(ms)
        summary['top_kernels'] = {n: statistics.median(v) for n, v in
                                  sorted(kernels.items(), key=lambda kv: -statistics.median(kv[1]))[:5]}
        summary['samples'] = len(samples)
        out[f'{case}/{path}'] = summary
    return out


def table(summary):
    header = ['case/path', 'wall', 'dma busy', 'kernel busy', 'gpu idle', 'H2D MiB', 'D2H MiB', 'copies',
              'kernels', 'first DMA', 'max gap']
    lines = [header]
    for name, s in summary.items():
        lines.append([name, f"{s['wall_ms']:.2f}", f"{s['dma_busy_ms']:.2f}", f"{s['kernel_busy_ms']:.2f}",
                      f"{s['gpu_idle_ms']:.2f}", f"{s['h2d_mib']:.1f}", f"{s['d2h_mib']:.2f}",
                      f"{s['copies']:.0f}", f"{s['kernels']:.0f}",
                      '-' if s['first_dma_ms'] is None else f"{s['first_dma_ms']:.2f}",
                      f"{s['max_dma_gap_ms']:.2f}"])
    widths = [max(len(row[i]) for row in lines) for i in range(len(header))]
    return '\n'.join('  '.join(cell.ljust(w) for cell, w in zip(row, widths)) for row in lines)


def timeline(ranges, copies, kernels, which, sample=0, merge_ns=30_000):
    """One sample's GPU activity as bursts: DMA copies and kernels merged when
    closer than merge_ns, printed with offsets from the range start (ms)."""
    matches = [(s, e, t) for s, e, t in ranges if t.startswith(f'beat_torch/{which}/')]
    if not matches:
        raise SystemExit(f'no range for {which}')
    start, end, text = sorted(matches)[min(sample, len(matches) - 1)]
    events = ([(s, e, 'dma', f"{COPY_KIND.get(k, k)} {b / 2**20:.2f}MiB") for s, e, b, k in copies if s < end and e > start] +
              [(s, e, 'krn', n) for s, e, n in kernels if s < end and e > start])
    events.sort()
    bursts = []
    for s, e, kind, name in events:
        if bursts and kind == bursts[-1][2] and s - bursts[-1][1] < merge_ns:
            bursts[-1][1] = max(bursts[-1][1], e)
            bursts[-1][3] += 1
            bursts[-1][4].add(name.split(' ')[0])
        else:
            bursts.append([s, e, kind, 1, {name.split(' ')[0]}])
    print(f'{text}: {(end - start) / 1e6:.2f} ms')
    previous = {'dma': None, 'krn': None}
    for s, e, kind, count, names in bursts:
        gap = '' if previous[kind] is None else f' (idle {(s - previous[kind]) / 1e6:.2f})'
        previous[kind] = e
        print(f"  {kind} {(s - start) / 1e6:7.2f} +{(e - s) / 1e6:5.2f} x{count:<3d}{gap} {', '.join(sorted(names))[:60]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sqlite')
    parser.add_argument('--json')
    parser.add_argument('--timeline', help='case/path whose first sample is printed as GPU bursts')
    args = parser.parse_args()
    loaded = load(args.sqlite)
    if args.timeline:
        timeline(*loaded, args.timeline)
        return
    summary = summarize(attribute(*loaded))
    print(table(summary))
    print('(ms; medians over samples; busy = union of activity inside the range; idle = neither DMA nor kernel)')
    for name, s in summary.items():
        print(f"{name}: top kernels " + ', '.join(f"{k} {v:.2f}" for k, v in s['top_kernels'].items()))
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
