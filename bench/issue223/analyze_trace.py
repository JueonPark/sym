#!/usr/bin/env python3
"""Correlate CUDA API launches to actual H2D DMA and model GEMM intervals.

Only actual device activity counts. GEMMs exclude transfer conversion kernels;
this is a conservative subset of model compute. Keep one model call per path's
raw intervals and all calls' overlap totals; no profiler database is committed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.database)
    db.row_factory = sqlite3.Row
    names = dict(db.execute('select id,value from StringIds'))
    tables = {r[0] for r in db.execute('select name from sqlite_master')}
    apis = []
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            apis += list(db.execute(f'select * from {table}'))
    copies = list(db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1'))
    kernels = list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'))
    roots = list(db.execute('select * from NVTX_EVENTS where end is not null order by start'))
    rows, calls, retained = [], [], set()
    for root in roots:
        label = root['text'] or names.get(root['textId'], '')
        if not label.startswith('issue223/'):
            continue
        _, case, path, sample = label.split('/')
        correlated = {a['correlationId'] for a in apis if a['globalTid'] == root['globalTid'] and
                      root['start'] <= a['start'] <= a['end'] <= root['end']}
        dma = [c for c in copies if c['correlationId'] in correlated]
        compute = [k for k in kernels if k['correlationId'] in correlated and
                   any(t in names[k['demangledName']].lower() for t in ('gemm', 'gemv'))]
        assert dma and compute, f'missing correlated DMA or model GEMMs: {label}'
        assert all(root['start'] <= r['start'] <= r['end'] <= root['end'] for r in dma + compute)
        intervals = union((r['start'], r['end']) for r in compute)
        overlap = sum(max(0, min(c['end'], b)-max(c['start'], a))
                      for c in dma for a, b in intervals)
        calls.append(dict(case=case, path=path, sample=int(sample), h2d_bytes=sum(c['bytes'] for c in dma),
            h2d_calls=len(dma), gemms=len(compute), dma_ms=sum(c['end']-c['start'] for c in dma)/1e6,
            model_gemm_ms=sum(b-a for a,b in intervals)/1e6, dma_gemm_overlap_ms=overlap/1e6))
        if path.endswith(('_serial', '_blocking')):
            assert overlap == 0, f'serial control overlapped: {label}'
        if (case, path) in retained:
            continue
        retained.add((case, path))
        for kind, events in (('h2d', dma), ('model_gemm', compute)):
            for r in events:
                rows.append(dict(case=case, path=path, sample=sample, kind=kind,
                    start_ns=r['start']-root['start'], end_ns=r['end']-root['start'],
                    stream=r['streamId'], correlation_id=r['correlationId'],
                    bytes=r['bytes'] if kind=='h2d' else '',
                    kernel=names[r['demangledName']] if kind=='model_gemm' else ''))
    assert any(c['dma_gemm_overlap_ms'] > 0 for c in calls if c['path']=='sym_prefetch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(database_sha256=hashlib.sha256(args.database.read_bytes()).hexdigest(),
                  intervals_csv=args.output.with_suffix('.csv').name, calls=calls)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    for call in calls:
        print(call)


if __name__ == '__main__':
    main()
