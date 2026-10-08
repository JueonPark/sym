#!/usr/bin/env python3
"""Request-correlated CUDA API/event counts; GPU intervals are not host timings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def analyze(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    names={r['id']:r['value'] for r in db.execute('select id,value from StringIds')}
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    requests=[]
    for root in db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'group220/%' order by start"):
        apis=[dict(r) for r in db.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? and globalTid=? order by start',
                                         (root['start'],root['end'],root['globalTid']))]
        for api in apis: api['name']=names[api.pop('nameId')]
        correlations={a['correlationId'] for a in apis}
        gpu=[]
        for table,kind in [('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_KERNEL','kernel')]:
            if table not in tables: continue
            for activity in db.execute(f'select * from {table} where start>=? and end<=?',(root['start'],root['end'])):
                if activity['correlationId'] not in correlations: continue
                row={key:activity[key] for key in ('start','end','deviceId','streamId','correlationId')}
                row['kind']=kind
                if kind=='memcpy': row.update(bytes=activity['bytes'],copy_kind=activity['copyKind'])
                else: row['name']=names[activity['demangledName']]
                gpu.append(row)
        assert gpu, 'missing GPU activity for '+root['text']
        requests.append({'label':root['text'],'start':root['start'],'end':root['end'],
            'cuda_runtime_apis':apis,'api_counts':dict(Counter(a['name'] for a in apis)),
            'gpu_activity_not_additive':gpu,'link_bytes':sum(a.get('bytes',0) for a in gpu)})
    groups=defaultdict(list)
    for request in requests: groups[request['label']].append(request)
    return {'sqlite_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'scope':'Separate traced diagnostics. Counts include all CUDA runtime APIs inside each completed-call NVTX range. Raw timestamps are Nsight nanoseconds. GPU intervals overlap host work and are not added to latency.',
            'summary':{label:{'requests':len(rows),
                'mean_api_counts':{name:statistics.mean(row['api_counts'].get(name,0) for row in rows)
                                   for name in sorted({name for row in rows for name in row['api_counts']})},
                'mean_link_bytes':statistics.mean(row['link_bytes'] for row in rows)} for label,rows in groups.items()},
            'requests':requests}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sqlite',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.write_text(json.dumps(analyze(args.sqlite),indent=2)+'\n')
