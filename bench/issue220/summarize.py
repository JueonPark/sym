#!/usr/bin/env python3
"""Summarize independent rounds without pooling their percentile samples."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(root):
    rounds = {kind: [json.loads((root / f'{kind}-round{i}.json').read_text())
                     for i in (1, 2, 3)] for kind in ('main', 'candidate')}
    result = {'aggregation': 'Median of three independent per-round p50/p95 statistics; raw rounds retained.',
              'cases': {}}
    paths = {'main_individual': ('main', 'individual'),
             'candidate_individual': ('candidate', 'individual'),
             'grouped': ('candidate', 'grouped'),
             'torch_with_main': ('main', 'torch_batched'),
             'torch_with_candidate': ('candidate', 'torch_batched')}
    for case in rounds['main'][0]['cases']:
        row = {'latency': {}, 'diagnostics': {}}
        for label, (revision, path) in paths.items():
            cases = [r['cases'][case] for r in rounds[revision]]
            assert all(c['correctness'] for c in cases)
            values = {p: [c['warm'][path][p + '_ms'] for c in cases] for p in ('p50', 'p95')}
            row['latency'][label] = {p + '_ms': median(v) for p, v in values.items()}
            row['latency'][label].update({p + '_ms_rounds': v for p, v in values.items()})
            row['latency'][label]['first_completed_ms_rounds'] = [c['first_completed_ms'][path] for c in cases]
            diagnostics = []
            for c in cases:
                d = c['diagnostics'][path]
                before, after = d['resources_before'], d['resources_after']
                counters = {'torch_peak_extra_allocated_bytes': d['torch_peak_extra_allocated_bytes'],
                            'torch_output_bytes': d['torch_output_bytes'], 'packing_bytes': d['packing_bytes']}
                if after:
                    counters.update(
                        native_requests=(after['requests'] + after['typed']['requests'] -
                                         before['requests'] - before['typed']['requests']),
                        native_host_capacity_bytes=after['allocated_staging_bytes'] + after['typed']['host_bytes'],
                        native_device_capacity_bytes=after['typed']['device_bytes'])
                if path == 'grouped':
                    counters.update({key: value for key, value in d['report'].items()
                                     if isinstance(value, int)})
                diagnostics.append(counters)
            row['diagnostics'][label] = diagnostics
        for baseline in ('main_individual', 'candidate_individual'):
            for p in ('p50', 'p95'):
                row[f'{p}_reduction_vs_{baseline}_pct'] = 100 * (
                    1 - row['latency']['grouped'][p + '_ms'] / row['latency'][baseline][p + '_ms'])
        result['cases'][case] = row
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = summarize(args.directory)
    (args.directory / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')
    for case, row in data['cases'].items():
        def cell(path):
            v = row['latency'][path]
            return f"{v['p50_ms']:.4f} / {v['p95_ms']:.4f}"
        print(f"| {case} | {cell('main_individual')} | {cell('candidate_individual')} | "
              f"{cell('grouped')} | {row['p50_reduction_vs_main_individual_pct']:.1f}% | "
              f"{cell('torch_with_candidate')} |")
