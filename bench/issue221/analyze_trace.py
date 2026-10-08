#!/usr/bin/env python3
"""Correlate real CPU worker intervals with next-chunk GPU DMA; emit compact CSV.

No CUDA API duration or worker-barrier time is counted as transform/DMA overlap.
Worker intervals are unioned, so parallel workers are never double-counted.
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
            result.append([start,end])
    return result


def analyze(path, output, first_per_path=False, prefix='typed221/'):
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    names = {r['id']:r['value'] for r in db.execute('select id,value from StringIds')}
    nvtx = [dict(r) for r in db.execute('select * from NVTX_EVENTS where end is not null order by start')]
    for event in nvtx: event['label'] = event['text'] or names.get(event['textId'], '')
    roots = [r for r in nvtx if r['label'].startswith(prefix)]
    apis = [dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
    copies = {r['correlationId']:dict(r) for r in db.execute(
        'select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1')}
    rows, requests = [], []
    for request, root in enumerate(roots):
        contained = [r for r in nvtx if root['start'] <= r['start'] <= r['end'] <= root['end']]
        work, dma = {}, {}
        for r in contained:
            if r['label'] in ('reloc.typed.transform.work','reloc.typed.fill'):
                work.setdefault(r['uint64Value'],[]).append([r['start'],r['end']])
            if r['label'] == 'reloc.typed.h2d.submit':
                matched = [a for a in apis if a['globalTid']==r['globalTid'] and
                           r['start'] <= a['start'] <= a['end'] <= r['end'] and
                           a['correlationId'] in copies]
                assert len(matched)==1, (root['label'],r,matched)
                dma[r['uint64Value']] = copies[matched[0]['correlationId']]
        if not dma: continue  # whole/Torch controls have no typed chunk ranges
        assert set(dma)==set(work), 'each chunk needs both real CPU work and DMA'
        total_overlap = 0
        for chunk, copy in sorted(dma.items()):
            intervals = union(work[chunk])
            next_work = union(work.get(chunk+1,[]))
            overlap = sum(max(0,min(end,copy['end'])-max(start,copy['start'])) for start,end in next_work)
            total_overlap += overlap
            base = root['start']
            rows.append(dict(request=request,path=root['label'],chunk=chunk,
                cpu_union_ns=json.dumps([[a-base,b-base] for a,b in intervals],separators=(',',':')),
                dma_start_ns=copy['start']-base,dma_end_ns=copy['end']-base,
                correlation_id=copy['correlationId'],bytes=copy['bytes'],next_transform_overlap_ns=overlap))
        record = dict(request=request,path=root['label'],chunks=len(dma),
            completed_ms=(root['end']-root['start'])/1e6,
            dma_ms=sum(r['end']-r['start'] for r in dma.values())/1e6,
            next_transform_overlap_ms=total_overlap/1e6)
        if root['label'].endswith('/ring1'):
            assert total_overlap==0, 'one-buffer control must serialize transforms and copies'
        requests.append(record)
    assert requests, 'no native typed pipeline activity captured'
    assert any(r['next_transform_overlap_ms']>0 for r in requests if r['path'].endswith('/ring2'))
    if first_per_path:
        first = {}
        for request in requests: first.setdefault(request['path'],request['request'])
        rows = [row for row in rows if row['request'] in first.values()]
    with output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    result = dict(sqlite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  intervals_csv=output.with_suffix('.csv').name,
                  interval_requests=sorted({row['request'] for row in rows}),requests=requests)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(requests,indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--first-per-path',action='store_true',help='retain all summaries but only the first call per path in the interval CSV')
    p.add_argument('--prefix',default='typed221/',help='outer request NVTX label prefix')
    args = p.parse_args(); analyze(args.sqlite,args.output,args.first_per_path,args.prefix)
