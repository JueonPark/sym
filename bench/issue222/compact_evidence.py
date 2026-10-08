#!/usr/bin/env python3
"""One CSV row per distribution, retaining lossless samples and provenance.

With --trace, also summarize actual H2D activity inside every indexed222 range.
Profiler databases and generated Inductor sources remain outside the repository.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--trace', type=Path)
    p.add_argument('inputs', nargs='+', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, provenance, diagnostics = [], [], {}
    for path in args.inputs:
        run = json.loads(path.read_text())
        samples = {}
        with path.with_suffix('.csv').open() as f:
            for row in csv.DictReader(f):
                samples.setdefault((int(row['selected']), row['path']), []).append(float(row['ms']))
        provenance.append(dict(run=path.stem, input_sha256=digest(path),
                               metadata=run['metadata'], configuration=run['configuration']))
        for case in run['results']:
            count = case.get('selected', case.get('batch'))
            if 'allocation_diagnostic' in case:
                diagnostics[f'{path.stem}/{count}'] = case['allocation_diagnostic']
            for name, result in case['paths'].items():
                resources = result.get('resources', {})
                rows.append(dict(run=path.stem, selected=count, path=name,
                    wire_bytes=case['wire_bytes'],
                    first_completed_ms=result.get('first_completed_ms', case.get('first_completed_ms', '')),
                    p50_ms=result['p50_ms'], p95_ms=result['p95_ms'],
                    native_executions=result.get('native_executions', ''),
                    host_scratch_bytes=resources.get('host_bytes', result.get('pinned_logical_bytes', '')),
                    native_peak_scratch_bytes=resources.get('peak_live_bytes', ''),
                    samples_ms=json.dumps(samples[count, name], separators=(',', ':'))))
    with (args.output / 'timings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'provenance.jsonl').write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in provenance))
    if args.trace:
        db = sqlite3.connect(args.trace)
        db.row_factory = sqlite3.Row
        names = {r['id']: r['value'] for r in db.execute('select id,value from StringIds')}
        copies = list(db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1'))
        records = []
        for root in db.execute('select * from NVTX_EVENTS where end is not null order by start'):
            label = root['text'] or names.get(root['textId'], '')
            if not label.startswith('indexed222/'):
                continue
            dma = [copy for copy in copies if root['start'] <= copy['start'] <= copy['end'] <= root['end']]
            records.append(dict(path=label, copies=len(dma), bytes=sum(copy['bytes'] for copy in dma),
                                dma_ms=sum(copy['end']-copy['start'] for copy in dma) / 1e6))
        diagnostics['trace'] = dict(sqlite_sha256=digest(args.trace), actual_h2d=records)
    (args.output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')


if __name__ == '__main__':
    main()
