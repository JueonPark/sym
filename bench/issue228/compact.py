#!/usr/bin/env python3
"""Retain raw timings and provenance without execution logs or compile caches."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    samples = io.StringIO()
    writer = csv.writer(samples)
    writer.writerow(['run', 'configuration', 'path', 'phase', 'sample', 'milliseconds'])
    runs = []
    for file in sorted(args.input.glob('*-r[123].json')):
        run = json.loads(file.read_text())
        for row in run['rows']:
            for path, result in row['paths'].items():
                for phase in ('batch', 'calls', 'cold_owner'):
                    for i, value in enumerate(result[phase].pop('samples_ms')):
                        writer.writerow([file.stem, row['name'], path, phase, i, value])
        runs.append(run)
    assert len(runs) == 6, 'expected three rounds for baseline and candidate'
    for name, payload in [('samples.csv.gz', samples.getvalue()),
                          ('runs.json.gz', json.dumps(runs, separators=(',', ':'))+'\n')]:
        (args.output/name).write_bytes(gzip.compress(payload.encode(), mtime=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
