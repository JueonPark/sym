#!/usr/bin/env python3
"""Summarize independent matched rounds; retain every original sample file."""
import argparse
import json
import statistics
from pathlib import Path

PREPARATION = ('frontend_and_other_validation', 'binding', 'descriptors',
               'fresh_validation', 'native_preparation')


def summarize(root):
    data = {variant: [json.loads(path.read_text()) for path in sorted(root.glob(variant + '-round*.json'))]
            for variant in ('main', 'candidate')}
    assert len(data['main']) == len(data['candidate']) >= 3
    result = {'aggregation': 'Median of per-process-round statistics; p95 is not pooled.',
              'preparation_scope': 'Sum of exclusive host categories excluding output allocation, native execution/completion and explicit end sync. Instrumented, not headline timing.',
              'rounds_per_variant': len(data['main']), 'cases': {}}
    for case in data['main'][0]['cases']:
        entry = {}
        for variant, runs in data.items():
            rows = [run['cases'][case] for run in runs]
            assert all(row['correctness'] for row in rows)
            entry[variant] = {
                'warm': {path: {stat: statistics.median(row['warm'][path][stat] for row in rows)
                                for stat in ('p50_ms', 'p95_ms')}
                         for path in ('sym', 'torch')},
                'first_completed_ms': {path: [row['first_completed_ms'][path] for row in rows]
                                       for path in ('sym', 'torch')},
                'exclusive_phase_mean_ms': {
                    category: statistics.median(statistics.mean(
                        sample['exclusive_ms'].get(category, 0) for sample in row['diagnostic_phases'])
                        for row in rows)
                    for category in (*PREPARATION, 'output_allocation', 'native_including_completion', 'explicit_completion')},
                'preparation_mean_ms': statistics.median(statistics.mean(
                    sum(sample['exclusive_ms'].get(category, 0) for category in PREPARATION)
                    for sample in row['diagnostic_phases']) for row in rows),
            }
        entry['sym_reduction_percent'] = {
            stat: 100 * (1 - entry['candidate']['warm']['sym'][stat] / entry['main']['warm']['sym'][stat])
            for stat in ('p50_ms', 'p95_ms')}
        entry['preparation_reduction_percent'] = 100 * (1 - entry['candidate']['preparation_mean_ms'] / entry['main']['preparation_mean_ms'])
        result['cases'][case] = entry
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    (args.directory / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, case in report['cases'].items():
        print(name, json.dumps({k: case[k] for k in ('sym_reduction_percent', 'preparation_reduction_percent')}))
