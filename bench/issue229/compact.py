#!/usr/bin/env python3
"""Keep samples/provenance and a table of distributions, excluding run logs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics


def distributions(value, path=()):
    if isinstance(value, dict):
        if 'p50_ms' in value and 'p95_ms' in value:
            yield '/'.join(path), value['p50_ms'], value['p95_ms']
        for key, child in value.items():
            yield from distributions(child, (*path, str(key)))


def main(args):
    paths = sorted(args.input.glob('*-r[123].json'))
    assert len(paths) == 18, 'three suites x two revisions x three rounds'
    runs = {p.stem: json.loads(p.read_text()) for p in paths}
    assert all(not r['isolation']['foreign_busy_processes'] for r in runs.values())
    # #223 stores individual observations in CSV rather than its summary JSON.
    csv_paths = sorted(args.input.glob('rawmodels-*-r[123].csv'))
    assert len(csv_paths) == 6
    for path in csv_paths:
        run = runs[path.stem]
        with path.open() as file:
            rows = [r | dict(sample=int(r['sample']), completed_ms=float(r['completed_ms']))
                    for r in csv.DictReader(file)]
        for case, result in run['cases'].items():
            for name, summary in result['paths'].items():
                values = [r['completed_ms'] for r in rows if r['case'] == case and r['path'] == name]
                assert len(values) == run['configuration']['samples']
                assert statistics.median(values) == summary['p50_ms']
        run['raw_samples'] = rows
    args.output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(runs, separators=(',', ':')).encode()
    (args.output/'samples.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    table = io.StringIO()
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['run', 'series', 'p50_ms', 'p95_ms'])
    for name, result in runs.items():
        for row in distributions(result):
            writer.writerow([name, *row])
    (args.output/'distributions.csv.gz').write_bytes(gzip.compress(table.getvalue().encode(), mtime=0))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + csv_paths}
    for name in ('samples.json.gz', 'distributions.csv.gz'):
        manifest[name] = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    if args.traces:
        selected = {'dlrm/2048/pinned', 'dlrm/2048/pageable', 'gnn/35180/pinned',
            'gnn/35180/pageable', 'kv_restore/33/pinned', 'kv_evict/33/pinned',
            'weight/(1024, 4096)/pageable', 'weight/(4096, 1024)/pageable', 'weight/(31, 33)/pageable'}
        for path in sorted(args.traces.glob('*-trace.json.gz')):
            trace = json.loads(gzip.decompress(path.read_bytes()))
            trace['first_call_witnesses'] = [r for r in trace['first_call_witnesses']
                if r['label'].removeprefix('request219/') in selected
                or r['label'].endswith('/sym_pinned_inductor')]
            source = path.with_name(path.name.removesuffix('-trace.json.gz')+'.json')
            profile = json.loads(source.read_text())
            trace['metadata'] = profile['metadata']
            trace['profile_json_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            trace['host_phase_samples'] = ({kind: {size: row['phases'] for size, row in family['cases'].items()}
                for kind, family in profile['families'].items()} if 'families' in profile else
                {case: row['phases'] for case, row in profile['cases'].items()})
            target = args.output/path.name
            target.write_bytes(gzip.compress(json.dumps(trace, separators=(',', ':')).encode(), mtime=0))
            manifest[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--traces', type=Path, help='directory containing the four analyzed #229 profiles')
    main(p.parse_args())
