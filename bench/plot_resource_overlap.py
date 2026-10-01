#!/usr/bin/env python3
"""Render two analyzed Nsight captures; requires matplotlib, outside benchmarking."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('multiple', type=Path)
    parser.add_argument('single', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text())['requests'][1] for path in [args.multiple, args.single]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    maximum = max((max(c['dma_end_ns'] for c in row['chunks']) - min(w['start_ns'] for c in row['chunks'] for w in c['gather'])) / 1e6 for row in records)
    for ax, row in zip(axes, records):
        threads = sorted({w['thread'] for c in row['chunks'] for w in c['gather']})
        streams = sorted({c['stream'] for c in row['chunks']})
        origin = min(w['start_ns'] for c in row['chunks'] for w in c['gather'])
        for chunk in row['chunks']:
            for work in chunk['gather']:
                ax.barh(threads.index(work['thread']), (work['end_ns']-work['start_ns'])/1e6,
                        left=(work['start_ns']-origin)/1e6, height=.65, color=colors[chunk['chunk']])
            ax.barh(len(threads)+streams.index(chunk['stream']), (chunk['dma_end_ns']-chunk['dma_start_ns'])/1e6,
                    left=(chunk['dma_start_ns']-origin)/1e6, height=.65, color=colors[chunk['chunk']])
        labels = [f'CPU gather {i}' for i in range(len(threads))] + [f'GPU DMA queue {i}' for i in range(len(streams))]
        ax.set_yticks(range(len(labels)), labels)
        ax.axhline(len(threads) - .5, color='0.6', linewidth=.7)
        ax.invert_yaxis()
        ax.set_xlim(0, maximum * 1.03)
        ax.set_xlabel('Time from first gather (ms)')
        ax.grid(axis='x', alpha=.2)
        observed = 'overlap observed' if row['observed_overlap'] else 'next gather waits for DMA'
        ax.set_title(f"{row['buffers']} buffer(s): {observed}", loc='left')
    fig.suptitle('Same four 4 MiB chunks; CPU gather and GPU H2D activity\nRepresentative profiled request; durations include profiler overhead', fontsize=12)
    fig.legend(handles=[Patch(color=colors[i], label=f'Chunk {i}') for i in range(4)], loc='outside lower center', ncol=4)
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
