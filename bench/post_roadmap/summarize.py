#!/usr/bin/env python3
"""Validate accepted rounds, emit aggregates, and archive only measurement JSON."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

PATHS = ['torch', 'inductor', 'sym', 'sym_inductor', 'sym_placement_inductor']
ROOT = Path(__file__).resolve().parents[2]


def summarize(source, destination):
    rounds = [json.loads((source / f'workloads-r{r}.json').read_text()) for r in (1, 2, 3)]
    assert len({r['source_revision'] for r in rounds}) == 1
    destination.mkdir(parents=True, exist_ok=True)
    rows, wins = [], []
    for case in rounds[0]['cases']:
        cases = [r['cases'][case] for r in rounds]
        for row in cases:
            assert not row['dynamo_counters'].get('graph_break')
            assert set(row['paths']) == set(PATHS)
            for path in PATHS:
                assert row['paths'][path]['warm_compile_callbacks'] == 0
                assert len(row['paths'][path]['samples_ms']) == 30
        for path in PATHS:
            measurements = [row['paths'][path] for row in cases]
            record = dict(case=case, path=path,
                p50_ms=median(m['p50_ms'] for m in measurements),
                p95_ms=median(m['p95_ms'] for m in measurements),
                min_round_p50_ms=min(m['p50_ms'] for m in measurements),
                max_round_p50_ms=max(m['p50_ms'] for m in measurements),
                transfer_p50_ms=median(m['transfer']['p50_ms'] for m in measurements),
                kv_p50_ms=median(m['kv_roundtrips']['p50_ms'] for m in measurements)
                    if 'kv_roundtrips' in measurements[0] else '',
                first_min_ms=min(m['first_completed_ms'] for m in measurements),
                first_max_ms=max(m['first_completed_ms'] for m in measurements),
                max_abs_error=max(m['max_abs_error'] for m in measurements))
            for control in ('torch', 'inductor'):
                for i, row in enumerate(cases, 1):
                    record[f'latency_ratio_vs_{control}_r{i}'] = (
                        row['paths'][path]['p50_ms'] / row['paths'][control]['p50_ms'])
            rows.append(record)
            if path.startswith('sym'):
                ratios = {control: [row['paths'][path]['p50_ms'] / row['paths'][control]['p50_ms']
                                    for row in cases] for control in ('torch', 'inductor')}
                if all(value <= .95 for values in ratios.values() for value in values):
                    wins.append(dict(case=case, path=path))
    with (destination / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    manifest = {}
    for r in (1, 2, 3):
        for name in (f'workloads-r{r}.json', f'calibration-r{r}.json', f'calibration-r{r}.profile.json'):
            data = (source / name).read_bytes()
            parsed = json.loads(data)
            if 'isolation' in parsed:
                assert not parsed['isolation']['foreign_busy_processes']
            compressed = gzip.compress(data, mtime=0)
            (destination / (name + '.gz')).write_bytes(compressed)
            manifest[name] = dict(bytes=len(data), gzip_bytes=len(compressed),
                sha256=hashlib.sha256(data).hexdigest())
    # #225's main rewrites metadata.benchmark_sha256 with its own local files.
    # Preserve the complete, unchanged harness dependency set here as well.
    dependencies = [p for directory in ('issue219', 'issue221', 'issue223',
        'issue224', 'issue225', 'issue227', 'issue229', 'post_roadmap')
        for p in (ROOT / 'bench' / directory).glob('*.py') if p != Path(__file__).resolve()]
    result = dict(source_revision=rounds[0]['source_revision'],
                  harness_sha256={str(p.relative_to(ROOT)):
                      hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  criterion='At least 5% lower p50 than both Torch controls in all three rounds',
                  qualifying_wins=wins, files=manifest)
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('qualifying wins:', wins)
    for case in rounds[0]['cases']:
        print(case, {r['path']: round(r['p50_ms'], 3) for r in rows if r['case'] == case})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    summarize(args.source, args.destination)
