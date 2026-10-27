#!/usr/bin/env python3
"""Attribute the sym_fast win to kernels versus plan/runtime from ablation rounds.

    python bench/beat_torch/ablation.py <rounds-or-evidence-dir> [--csv out.csv]

Reads workloads-r*.json(.gz) produced by run.py with the arms ``torch``,
``torch_kernels``, ``sym_fast_scalar``, ``sym_fast_nopipe``, ``sym_fast_kvsplit``,
``sym_fast_noadvice`` and ``sym_fast`` (any subset) and prints, per regime, the median of the
round medians for every arm and related contrasts (not independent shares):

- ``torch - torch_kernels`` includes fused producer/dequantize changes;
- ``sym_fast_scalar - sym_fast`` isolates vectorization only in GPT/MoE;
  elsewhere those arms are identical implementations and differences are noise;
- ``torch_kernels - sym_fast`` compares higher-level execution, retaining
  compiler/plan use in the GraphSAGE kernel control and differing prefetch
  windows in decode;
- ``torch - sym_fast_scalar`` includes fusion and is NOT a runtime-only estimate;
- ``sym_fast_nopipe - sym_fast`` tests producer overlap in DLRM/GraphSAGE;
  ``sym_fast_kvsplit - sym_fast`` tests KV grouping in decode;
- ``sym_fast_noadvice - sym_fast`` tests the path-selection policy, not
  whether its structural information must originate in a compiler.

Positive values mean the second arm is faster. These interacting contrasts
must not be added as independent causal contributions. Legacy CSV field names
are retained for existing evidence readers.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

ARMS = ('torch', 'torch_kernels', 'sym_fast_scalar', 'sym_fast_nopipe', 'sym_fast_kvsplit', 'sym_fast_noadvice',
        'sym_fast')


def load(directory):
    rounds = []
    for path in sorted(Path(directory).glob('workloads-r*.json*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as handle:
            rounds.append(json.load(handle))
    if not rounds:
        raise SystemExit(f'no rounds in {directory}')
    return rounds


def medians(rounds):
    table = {}
    for case in rounds[0]['cases']:
        row = {}
        for arm in rounds[0]['cases'][case]['paths']:
            row[arm] = statistics.median(r['cases'][case]['paths'][arm]['p50_ms'] for r in rounds)
        table[case] = row
    return table


def shares(row):
    def diff(a, b):
        return row[a] - row[b] if a in row and b in row else None
    return {
        'kernel_a': diff('torch', 'torch_kernels'),
        'kernel_b': diff('sym_fast_scalar', 'sym_fast'),
        'runtime_a': diff('torch_kernels', 'sym_fast'),
        'runtime_b': diff('torch', 'sym_fast_scalar'),
        'overlap': diff('sym_fast_nopipe', 'sym_fast'),
        'kv_grouping': diff('sym_fast_kvsplit', 'sym_fast'),
        'advice': diff('sym_fast_noadvice', 'sym_fast'),
        'total': diff('torch', 'sym_fast'),
    }


def fmt(value):
    return '' if value is None else f'{value:+.3f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('directory')
    p.add_argument('--csv')
    args = p.parse_args()
    rounds = load(args.directory)
    table = medians(rounds)
    arms = [a for a in ARMS if a in next(iter(table.values()))]
    print('| Regime | ' + ' | '.join(arms) + ' |')
    print('| --- | ' + ' | '.join('---:' for _ in arms) + ' |')
    for case, row in table.items():
        print(f'| {case} | ' + ' | '.join(f'{row[a]:.3f}' for a in arms) + ' |')
    print()
    columns = [('total', 'torch − sym_fast'), ('kernel_a', 'kernels A: torch − torch_kernels'),
               ('kernel_b', 'kernels B: scalar − sym_fast'), ('runtime_a', 'runtime A: torch_kernels − sym_fast'),
               ('runtime_b', 'total vs scalar (includes fusion): torch − scalar'), ('overlap', 'chunk overlap: nopipe − sym_fast'),
               ('kv_grouping', 'KV grouping: kvsplit − sym_fast'),
               ('advice', 'policy contrast: noadvice − sym_fast')]
    print('| Regime | ' + ' | '.join(label for _, label in columns) + ' |')
    print('| --- | ' + ' | '.join('---:' for _ in columns) + ' |')
    rows = []
    for case, row in table.items():
        share = shares(row)
        rows.append(dict(case=case, **{a: row.get(a) for a in arms}, **share))
        print(f'| {case} | ' + ' | '.join(fmt(share[key]) for key, _ in columns) + ' |')
    print('(ms, medians of round medians; positive = second arm faster; contrasts are not independent)')
    if args.csv:
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
