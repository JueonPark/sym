#!/usr/bin/env python3
"""Export request-correlated Nsight CUDA APIs/GPU activity without nesting sums."""
import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def analyze(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    names = {r['id']: r['value'] for r in db.execute('select id,value from StringIds')}
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    requests = []
    for request in db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'request219/%'"):
        start, end, tid = request['start'], request['end'], request['globalTid']
        phases = [dict(r) for r in db.execute('select start,end,text from NVTX_EVENTS where start>=? and end<=? and globalTid=? and text like ? order by start', (start,end,tid,'phase219/%'))]
        native = [r for r in phases if r['text'].startswith('phase219/native_including_completion/')]
        assert len(native) == 1
        native = native[0]
        apis = [dict(r) for r in db.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? and globalTid=? order by start', (start,end,tid))]
        for api in apis:
            api['name'] = names[api.pop('nameId')]
        waits = [api for api in apis if 'Synchronize' in api['name'] and api['start']>=native['start'] and api['end']<=native['end']]
        # Runtime API events on this CPU thread are sequential; do not include
        # nested driver APIs, nor add GPU durations to this host partition.
        assert all(a['end']<=b['start'] for a,b in zip(waits, waits[1:]))
        wait_ns = sum(r['end']-r['start'] for r in waits)
        correlations = {a['correlationId'] for a in apis if a['start']>=native['start'] and a['end']<=native['end']}
        gpu = []
        for table, kind in [('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'), ('CUPTI_ACTIVITY_KIND_KERNEL','kernel')]:
            if table not in tables:
                continue
            for r in db.execute(f'select * from {table} where start>=? and end<=?',(start,end)):
                if r['correlationId'] in correlations:
                    activity = {k:r[k] for k in ('start','end','deviceId','streamId','correlationId')}
                    activity['kind'] = kind
                    if kind == 'memcpy':
                        activity['bytes'] = r['bytes']
                        activity['copy_kind'] = r['copyKind']
                    else:
                        activity['name'] = names[r['demangledName']]
                    gpu.append(activity)
        requests.append({'label':request['text'], 'start':start,'end':end, 'phases':phases,
                         'cuda_runtime_apis':apis, 'native_completion_wait_ms':wait_ns/1e6,
                         'native_other_ms':(native['end']-native['start']-wait_ns)/1e6,
                         'gpu_activity_not_additive':gpu})
    assert requests and all(r['gpu_activity_not_additive'] for r in requests)
    return {'sqlite_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'scope':'Instrumented diagnostic only. Runtime synchronization API durations are a subset of the native host interval. GPU activity overlaps host intervals and is not additive. Raw timestamps are Nsight nanoseconds.',
            'mean_ms':{k:statistics.mean(r[k] for r in requests) for k in ('native_completion_wait_ms','native_other_ms')},
            'requests':requests}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    args.output.write_text(json.dumps(analyze(args.sqlite),indent=2)+'\n')
