#!/usr/bin/env python3
"""Pack scalar samples/provenance, preserving all 126 independent path runs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def percentile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(args.input.glob('r*.json'))}
    assert len(runs) == 144, len(runs)
    paths = {k:v for k,v in runs.items() if 'samples_ms' in v}
    assert len(paths) == 126
    assert all(not v['isolation']['foreign_processes'] for v in runs.values())
    assert all(v['checks'] == 199 and len(v['samples_ms']) == 100 for v in paths.values())
    groups = {}
    for name, run in paths.items():
        key = (run['case'], 'initialized' if run['initialized'] else 'cold', run['path'])
        stats = dict(preparation_ms=run['preparation_ms'], first_ms=run['samples_ms'][0],
                     warm_p50_ms=statistics.median(run['samples_ms'][1:]),
                     warm_p95_ms=percentile(run['samples_ms'][1:], .95),
                     retirement_ms=run['retirement_ms'],
                     launch_to_first_ms=run['launch_to_script_ms'] + run['readiness_from_script_ms'] + run['samples_ms'][0],
                     **{'total_'+n+'_ms':v for n,v in run['total_horizons_ms'].items()})
        for n in ('1','10','100'):
            stats['process_total_'+n+'_ms'] = (run['launch_to_script_ms'] + run['readiness_from_script_ms']
                - run['preparation_ms'] + run['total_horizons_ms'][n])
        groups.setdefault(key, []).append(stats)
    rows = []
    for key, values in sorted(groups.items()):
        assert len(values) == 3
        row = dict(zip(('case','process','path'), key))
        for metric in values[0]:
            samples = [v[metric] for v in values]
            row[metric] = statistics.median(samples)
            row[metric+'_min'] = min(samples)
            row[metric+'_max'] = max(samples)
        rows.append(row)
    with (args.output/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    compiler_runs = {p.stem: json.loads(p.read_text()) for p in sorted((args.input/'compilers').glob('*.json'))}
    assert len(compiler_runs) == 27
    assert all(r['checks'] and not r['foreign_processes'] for r in compiler_runs.values())
    payload = (json.dumps(dict(transfers=runs, compilers=compiler_runs), separators=(',', ':'))+'\n').encode()
    (args.output/'samples.json.gz').write_bytes(gzip.compress(payload, mtime=0))
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.output.iterdir()) if p.name in ('samples.json.gz','summary.csv')}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for row in rows:
        print(row['case'], row['process'], row['path'],
              *(f"{row[k]:.3f}" for k in ('preparation_ms','first_ms','warm_p50_ms','total_1_ms','total_10_ms','total_100_ms')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
