#!/usr/bin/env python3
"""Registered verdicts NL1 / NL2 (evidence-advice/REGISTERED.md) from rounds.

    python bench/beat_torch/never_lose.py <rounds-or-evidence-dir>

NL1, every regime: p50(sym_fast) <= min(p50(torch_kernels), p50(sym_fast_noadvice)) * (1 + band),
band = max(0.02, spread), spread = the faster control's (max - min) / min over its round medians.
NL2, GPT decode: |p50(sym_fast) - p50(sym_fast_noadvice)| <= band * p50(sym_fast_noadvice).
p50 is the median of the round medians. Exit status 1 when any verdict fails.
"""
import statistics
import sys

from ablation import load

CONTROLS = ('torch_kernels', 'sym_fast_noadvice')


def verdicts(rounds):
    rows = []
    for case in rounds[0]['cases']:
        per_round = {arm: [r['cases'][case]['paths'][arm]['p50_ms'] for r in rounds]
                     for arm in rounds[0]['cases'][case]['paths']}
        p50 = {arm: statistics.median(values) for arm, values in per_round.items()}
        controls = [arm for arm in CONTROLS if arm in p50]
        control = min(controls, key=p50.get)
        spread = (max(per_round[control]) - min(per_round[control])) / min(per_round[control])
        band = max(0.02, spread)
        nl2 = None
        if case == 'llm_decode':
            nl2 = abs(p50['sym_fast'] - p50['sym_fast_noadvice']) <= band * p50['sym_fast_noadvice']
        rows.append(dict(case=case, sym_fast=p50['sym_fast'], torch_kernels=p50.get('torch_kernels'),
                         sym_fast_noadvice=p50.get('sym_fast_noadvice'), control=control, band=band,
                         nl1=p50['sym_fast'] <= p50[control] * (1 + band), nl2=nl2))
    return rows


def main(directory):
    rows = verdicts(load(directory))
    print('| Regime | `sym_fast` | `torch_kernels` | `sym_fast_noadvice` | faster control | band | NL1 | NL2 |')
    print('| --- | ---: | ---: | ---: | --- | ---: | --- | --- |')
    for row in rows:
        nl2 = '' if row['nl2'] is None else ('PASS' if row['nl2'] else 'FAIL')
        print(f"| {row['case']} | {row['sym_fast']:.3f} | {row['torch_kernels']:.3f} | {row['sym_fast_noadvice']:.3f} | "
              f"`{row['control']}` | {row['band'] * 100:.1f}% | {'PASS' if row['nl1'] else 'FAIL'} | {nl2} |")
    failed = [row['case'] for row in rows if not row['nl1'] or row['nl2'] is False]
    print(f"NL1/NL2: {'PASS' if not failed else 'FAIL in ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
