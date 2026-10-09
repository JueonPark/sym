#!/usr/bin/env python3
"""Keep raw timing samples and scalar provenance; omit execution logs/caches."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path


def without_samples(value):
    if isinstance(value, dict):
        return {k: without_samples(v) for k, v in value.items() if k != 'samples_ms'}
    if isinstance(value, list):
        return [without_samples(v) for v in value]
    return value


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer, lineterminator='\n')
    writer.writerow(['build', 'round', 'threads', 'case', 'direction', 'path', 'state', 'sample', 'ms'])
    provenance = []
    for path in sorted(args.input.glob('*-r*-t*.json')):
        data = json.loads(path.read_text())
        provenance.append(without_samples(data))
        for row in data['rows']:
            key = [data['label'], data['round'], row['threads'], row['case'], row['direction']]
            def samples(name, state, distribution):
                for i, value in enumerate(distribution['samples_ms']):
                    writer.writerow([*key, name, state, i, f'{value:.6f}'])
            if 'kernel' in row:
                samples('kernel', 'warm', row['kernel'])
            for name, values in row['paths'].items():
                samples(name, 'warm', values['warm'])
                if values['cold_owner']:
                    samples(name, 'cold_owner', values['cold_owner'])
    assert len(provenance) == 18, 'expected both builds, three rounds, three worker budgets'
    (args.output / 'samples.csv.gz').write_bytes(gzip.compress(csv_buffer.getvalue().encode(), mtime=0))
    (args.output / 'runs.json.gz').write_bytes(gzip.compress(
        (json.dumps(provenance, separators=(',', ':')) + '\n').encode(), mtime=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
