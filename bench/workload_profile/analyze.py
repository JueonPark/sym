#!/usr/bin/env python3
"""Correlate CUDA activity and partition the caller's profiled wall time.

CUDA API/host phases are disjoint. GPU durations are a separate view and MUST
NOT be added to those host phases (a blocking API may already include them).
Native CPU transform scopes include worker dispatch/wait, not only CPU busy time.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

PHASES=['cpu_transform','staging_management','device_management','cuda_copy_api',
        'cuda_wait_api','cuda_other_api','frontend_prepare','frontend_execute',
        'native_other','outer_host']


def within(child,parent):
    return child['globalTid']==parent['globalTid'] and parent['start']<=child['start']<=child['end']<=parent['end']


def indexed(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['globalTid']].append(row)
    for tid,values in grouped.items():
        values.sort(key=lambda r:r['start'])
        grouped[tid]=([r['start'] for r in values],values)
    return grouped


def children(parent,index):
    starts,values=index.get(parent['globalTid'],([],[]))
    lo=bisect.bisect_left(starts,parent['start']);hi=bisect.bisect_right(starts,parent['end'])
    return [r for r in values[lo:hi] if r['end']<=parent['end']]


def union_ns(intervals):
    end=-1;total=0
    for begin,finish in sorted(intervals):
        total+=max(0,finish-max(begin,end));end=max(end,finish)
    return total


def api_phase(name):
    if re.match(r'cuda(HostAlloc|MallocHost|FreeHost|HostRegister|HostUnregister)',name):return 'staging_management'
    if re.match(r'cuda(Malloc|Free)(_|$)',name):return 'device_management'
    if name.startswith('cudaMemcpy'):return 'cuda_copy_api'
    if 'Synchronize' in name:return 'cuda_wait_api'
    return 'cuda_other_api'


def scope_phase(name):
    if name=='reloc.typed.host_program':return 10,'cpu_transform'
    if name in ('reloc.staging.allocate','reloc.staging.free','reloc.scratch.host_acquire','reloc.scratch.host_release'):return 11,'staging_management'
    if name in ('reloc.scratch.device_acquire','reloc.scratch.device_release'):return 11,'device_management'
    if name in ('python.prepare_typed_transfer','python.layout.prepare_transfer','python.runtime.bind_symbols','python.runtime.destination_descriptor'):return 5,'frontend_prepare'
    if name=='python.adapter.preflight':return 4,'frontend_prepare'
    if name in ('python.native.execute_dispatch','python.native.execute_transfer'):return 7,'native_other'
    if name in ('python.execute_typed_transfer','python.layout.execute_transfer','python.runtime.verify_result'):return 5,'frontend_execute'
    if name=='python.adapter.execute':return 4,'frontend_execute'
    return None


def exclusive(span,apis,scopes):
    pieces=[(span['start'],span['end'],0,'outer_host')]
    pieces += [(r['start'],r['end'],20,api_phase(r['name'])) for r in apis]
    pieces += [(r['start'],r['end'],*scope_phase(r['name'])) for r in scopes if scope_phase(r['name'])]
    boundaries=sorted({t for row in pieces for t in row[:2]})
    totals=dict.fromkeys(PHASES,0.0);timeline=[]
    for begin,end in zip(boundaries,boundaries[1:]):
        label=max((p for p in pieces if p[0]<=begin and end<=p[1]),key=lambda p:p[2])[3]
        totals[label]+=(end-begin)/1e6
        if timeline and timeline[-1]['phase']==label and timeline[-1]['end_ns']==begin:
            timeline[-1]['end_ns']=end
        else:timeline.append(dict(phase=label,start_ns=begin,end_ns=end))
    assert abs(sum(totals.values())-(span['end']-span['start'])/1e6)<1e-6
    for row in timeline:
        row['start_ms']=(row.pop('start_ns')-span['start'])/1e6
        row['end_ms']=(row.pop('end_ns')-span['start'])/1e6
    return totals,timeline


def analyze(path):
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        strings=dict(db.execute('SELECT id,value FROM StringIds'))
        nvtx=[dict(r) for r in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL')]
        apis=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        kinds={name:[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_'+table)]
               for name,table in [('kernels','KERNEL'),('copies','MEMCPY'),('memsets','MEMSET')]}
    for r in nvtx:r['name']=r['text'] or strings.get(r['textId'],'')
    for r in apis:r['name']=strings[r['nameId']]
    correlations={(r['globalTid']>>24,r['correlationId']):r for r in apis}
    for name,events in kinds.items():
        for r in events:
            r['api']=correlations.get((r['globalPid']>>24,r['correlationId']))
            if name=='kernels':r['name']=strings[r['demangledName']]
            else:r['name']=('H2D' if r['copyKind']==1 else 'D2H' if r['copyKind']==2 else 'D2D') if name=='copies' else 'memset'
    transfers=[r for r in nvtx if r['name'].startswith('transfer/')]
    checks=[r for r in nvtx if r['name'].startswith('verification/')]
    models=[r for r in nvtx if r['name'].startswith('model/')]
    assert transfers and models
    assert all(r['api'] is not None for rows in kinds.values() for r in rows), 'unmapped GPU activity correlation'
    api_index=indexed(apis);scope_index=indexed(nvtx)
    result=dict(case=path.stem,sqlite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),transfers=[],model={})
    for span in transfers:
        _,who,kind,index=span['name'].split('/')
        calls=children(span,api_index)
        scopes=children(span,scope_index)
        launches={(r['globalTid']>>24,r['correlationId']) for r in calls}
        gpu={key:[r for r in values if (r['globalPid']>>24,r['correlationId']) in launches] for key,values in kinds.items()}
        all_gpu=[r for values in gpu.values() for r in values]
        assert all(span['start']<=r['start']<r['end']<=span['end']+1000 for r in all_gpu), 'activity beyond completed transfer'
        phases,timeline=exclusive(span,calls,scopes)
        def aggregate(rows):
            out=defaultdict(lambda:dict(count=0,ms=0.0))
            for r in rows:
                out[r['name']]['count']+=1;out[r['name']]['ms']+=(r['end']-r['start'])/1e6
            return dict(out)
        def event(r):
            return dict(name=r['name'],start_ms=(r['start']-span['start'])/1e6,duration_ms=(r['end']-r['start'])/1e6,
                stream=r.get('streamId'),bytes=r.get('bytes'),correlation=r.get('correlationId'))
        result['transfers'].append(dict(path=who,kind=kind,index=int(index),wall_ms=(span['end']-span['start'])/1e6,
            exclusive_ms=phases,exclusive_timeline=timeline,api_totals=aggregate(calls),inclusive_scopes=aggregate(scopes),
            gpu={key:[event(r) for r in values] for key,values in gpu.items()},
            gpu_busy_ms=union_ns([(r['start'],r['end']) for r in all_gpu])/1e6,
            gpu_copy_ms=sum(r['end']-r['start'] for r in gpu['copies'])/1e6,
            gpu_kernel_ms=sum(r['end']-r['start'] for r in gpu['kernels'])/1e6,
            copy_bytes=sum(r['bytes'] for r in gpu['copies'])))
    for who in ('torch','sym'):
        spans=[r for r in models if r['name'].startswith('model/'+who+'/')]
        def model_only(r):
            api=r['api']
            return api is not None and any(within(api,s) for s in spans) and not any(within(api,s) for s in transfers+checks)
        gpu={key:[r for r in values if model_only(r)] for key,values in kinds.items()}
        keys=['gridX','gridY','gridZ','blockX','blockY','blockZ']
        def signature(r):return (r['name'],*(r[k] for k in keys))
        counts=Counter(signature(r) for r in gpu['kernels'])
        sequence=[signature(r) for r in sorted(gpu['kernels'],key=lambda r:r['api']['start'])]
        result['model'][who]=dict(kernel_count=len(gpu['kernels']),kernel_ms=sum(r['end']-r['start'] for r in gpu['kernels'])/1e6,
            copy_count=len(gpu['copies']),copy_bytes=sum(r['bytes'] for r in gpu['copies']),
            memset_count=len(gpu['memsets']),memset_bytes=sum(r['bytes'] for r in gpu['memsets']),
            kernel_signatures=[dict(signature=list(k),count=v) for k,v in sorted(counts.items())],
            ordered_signature_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest())
    result['model']['kernel_signatures_equal']=result['model']['torch']['kernel_signatures']==result['model']['sym']['kernel_signatures']
    result['model']['kernel_sequence_equal']=result['model']['torch']['ordered_signature_sha256']==result['model']['sym']['ordered_signature_sha256']
    assert result['model']['kernel_signatures_equal'] and result['model']['kernel_sequence_equal']
    for key in ('copy_count','copy_bytes','memset_count','memset_bytes'):
        assert result['model']['torch'][key]==result['model']['sym'][key],key
    result['by_kind']={};result['totals']={}
    for who in ('torch','sym'):
        rows=[r for r in result['transfers'] if r['path']==who and r['index']>0]
        def summarize(selected):
            return dict(calls=len(selected),wall_ms=sum(r['wall_ms'] for r in selected),
                gpu_copy_ms=sum(r['gpu_copy_ms'] for r in selected),gpu_kernel_ms=sum(r['gpu_kernel_ms'] for r in selected),
                gpu_busy_ms=sum(r['gpu_busy_ms'] for r in selected),copy_bytes=sum(r['copy_bytes'] for r in selected),
                exclusive_ms={p:sum(r['exclusive_ms'][p] for r in selected) for p in PHASES})
        result['totals'][who]=summarize(rows)
        for kind in {r['kind'] for r in rows}:
            result['by_kind'].setdefault(kind,{})[who]=summarize([r for r in rows if r['kind']==kind])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    summary={}
    for path in sorted(args.directory.glob('*.sqlite')):
        result=analyze(path)
        path.with_suffix('.analysis.json').write_text(json.dumps(result,indent=2)+'\n')
        summary[path.stem]={k:result[k] for k in ('totals','by_kind','model')}
        print(path.stem,{who:round(r['wall_ms'],3) for who,r in result['totals'].items()},'model signatures identical',flush=True)
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
