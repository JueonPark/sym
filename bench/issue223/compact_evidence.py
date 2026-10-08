#!/usr/bin/env python3
"""Lossless samples and bounded trace intervals; exclude logs and profiler databases."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compact(value):
    return json.dumps(value, separators=(',', ':'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--traces', type=Path, nargs='*', default=[])
    p.add_argument('--trace-runs', type=Path, nargs='*', default=[])
    p.add_argument('runs', type=Path, nargs='+')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timing, diagnostics, provenance = [], [], []
    for path in args.runs:
        run = json.loads(path.read_text())
        samples = defaultdict(list)
        with path.with_suffix('.csv').open() as file:
            for row in csv.DictReader(file):
                samples[row['case'], row['path']].append(float(row['completed_ms']))
        provenance.append(dict(run=path.stem, input_sha256=digest(path),
            metadata=run['metadata'], configuration=run['configuration'], limits=run['limits'],
            artifact_setup_ms=run['artifact_setup_ms']))
        for case, record in run['cases'].items():
            for name, r in record['paths'].items():
                timing.append(dict(run=path.stem, case=case, path=name,
                    first_completed_ms=r['first_completed_ms'], p50_ms=r['p50_ms'], p95_ms=r['p95_ms'],
                    tokens_per_second=r['tokens_per_second'], throughput_unit=record['throughput_unit'],
                    samples_ms=compact(samples[case,name])))
                diagnostics.append(dict(run=path.stem, case=case, path=name,
                    pinned_source_bytes=record['pinned_source_bytes'], exact_checks=record['exact_checks'],
                    startup_host_ready_ms=compact(r.get('startup_host_ready_ms',[])),
                    drain_ms=compact(r.get('drain_ms',[])), queue=compact(r['queue']),
                    allocator=compact(r['allocator_diagnostic'])))
    write_csv(args.output/'timings.csv', timing)
    write_csv(args.output/'diagnostics.csv', diagnostics)
    for path in args.trace_runs:
        run = json.loads(path.read_text())
        provenance.append(dict(run=path.stem, input_sha256=digest(path),
            metadata=run['metadata'], configuration=run['configuration']))
    (args.output/'provenance.jsonl').write_text(''.join(compact(row)+'\n' for row in provenance))
    summaries, intervals, sources = [], [], []
    for path in args.traces:
        trace = json.loads(path.read_text())
        summaries.extend(trace['calls'])
        sources.append(dict(trace=path.stem, database_sha256=trace['database_sha256'],
                            extractor_output_sha256=digest(path)))
        by_kind = defaultdict(list)
        with path.with_suffix('.csv').open() as file:
            for row in csv.DictReader(file):
                by_kind[row['case'],row['path'],int(row['sample']),row['kind']].append([
                    int(row['start_ns']), int(row['end_ns']),
                    *[int(row[k]) if row[k] else None for k in ('stream','correlation_id','bytes')],
                    row['kernel'] or None])
        for (case, name, sample, kind), values in by_kind.items():
            intervals.append(dict(case=case, path=name, sample=sample, kind=kind, intervals=compact(values)))
    if summaries:
        write_csv(args.output/'overlap.csv', summaries)
        write_csv(args.output/'intervals.csv', intervals)
        (args.output/'trace-sources.json').write_text(json.dumps(sources, indent=2)+'\n')


if __name__ == '__main__':
    main()
