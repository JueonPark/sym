#!/usr/bin/env python3
"""Keep samples/provenance and a table of distributions, excluding run logs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


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
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in ('samples.json.gz', 'distributions.csv.gz'):
        manifest[name] = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    main(p.parse_args())
