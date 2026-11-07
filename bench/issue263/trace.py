#!/usr/bin/env python3
"""Correlate Nsight CUDA activity to launch-thread NVTX requests.

Host and GPU intervals are never added. Event wait dependencies are preserved
as trace evidence; a host interval coincident with GPU idle is not causal savings.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def interval_union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end <= start: continue
        if result and start <= result[-1][1]: result[-1]=(result[-1][0],max(end,result[-1][1]))
        else: result.append((start,end))
    return result


def duration(intervals): return sum(b-a for a,b in interval_union(intervals))


def exclusive_segments(start,end,ranges):
    ranges=[(max(start,a),min(end,b),name) for a,b,name in ranges if b>start and a<end]
    points=sorted({start,end,*[v for a,b,_ in ranges for v in (a,b)]})
    result=[]
    for a,b in zip(points,points[1:]):
        active=[x for x in ranges if x[0]<=a and x[1]>=b]
        category=min(active,key=lambda x:x[1]-x[0])[2] if active else 'other'
        if result and result[-1][2]==category: result[-1]=(result[-1][0],b,category)
        else: result.append((a,b,category))
    return result


def analyze_trace(path):
    conn=sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    def rows(name): return [dict(r) for r in conn.execute('SELECT * FROM '+name)] if name in tables else []
    strings={r['id']:r['value'] for r in rows('StringIds')}
    ranges=rows('NVTX_EVENTS')
    for r in ranges: r['label']=r.get('text') or strings.get(r.get('textId'),'')
    roots=sorted((r for r in ranges if r['label'].startswith('request263/') and r['end'] is not None),key=lambda r:r['start'])
    if not roots: raise ValueError('no request263 NVTX ranges')
    by_tid=defaultdict(list)
    for r in ranges:
        if r['end'] is not None: by_tid[r['globalTid']].append(r)
    apis=[]; correlation=defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        for a in rows(table):
            a['name']=strings.get(a.get('nameId'),'unknown'); a['table']=table
            a['pid']=a['globalTid'] & ~((1<<24)-1)
            correlation[a['pid'],a['correlationId']].append(a); apis.append(a)
    activities=[]
    for kind in ('KERNEL','MEMCPY','MEMSET'):
        for a in rows('CUPTI_ACTIVITY_KIND_'+kind):
            a['kind']=kind; a['name']=strings.get(a.get('shortName'),kind)
            launches=correlation.get((a['globalPid'],a['correlationId']),[])
            # Prefer outer runtime call when the driver is nested in it.
            launch=min(launches,key=lambda r:r['start']) if launches else None
            a['launch_tid']=launch['globalTid'] if launch else None
            a['launch_start']=launch['start'] if launch else None
            a['launch_end']=launch['end'] if launch else None
            a['launch_name']=launch['name'] if launch else None
            activities.append(a)
    if not activities: raise ValueError('no CUDA kernel/copy/memset evidence')
    sync=rows('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')
    enum={r['id']:r['label'] for r in rows('ENUM_CUPTI_SYNC_TYPE')}
    event_records=defaultdict(list)
    for e in rows('CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
        candidates=correlation.get((e['globalPid'],e['correlationId']),[])
        if candidates:
            a=min(candidates,key=lambda r:r['start'])
            event_records[e['globalPid'],e['contextId'],e['eventId']].append(dict(e,api_start=a['start'],api_end=a['end'],api_name=a['name']))
    output=[]; assigned=set()
    for root in roots:
        start,end,tid=root['start'],root['end'],root['globalTid']
        label=root['label'].removeprefix('request263/')
        inside=[r for r in by_tid[tid] if start<=r['start'] and r['end']<=end]
        phases=[(r['start'],r['end'],r['label'].split('/')[1]) for r in inside if r['label'].startswith('phase263/')]
        segments=exclusive_segments(start,end,phases)
        exclusive=defaultdict(int); idle_by_phase=defaultdict(int)
        pid=tid & ~((1<<24)-1)
        device_intervals=interval_union([(max(start,a['start']),min(end,a['end'])) for a in activities
                                        if a['globalPid']==pid and a['end']>start and a['start']<end])
        for a,b,category in segments:
            exclusive[category]+=b-a
            idle_by_phase[category]+=b-a-duration([(max(a,c),min(b,d)) for c,d in device_intervals if d>a and c<b])
        own=[]
        groups=[r for r in inside if r['label'].startswith('group263/')]
        for index,a in enumerate(activities):
            if a['launch_tid']!=tid or not start<=a['launch_start']<end: continue
            if a['end']>end: raise ValueError(f'CUDA activity after completed request completion: {label}')
            matching=[g for g in groups if g['start']<=a['launch_start'] and a['launch_end']<=g['end']]
            group=min(matching,key=lambda g:g['end']-g['start'])['label'] if matching else None
            own.append(dict(a,group=group)); assigned.add(index)
        dependencies=[]
        for s in sync:
            launch=correlation.get((s['globalPid'],s['correlationId']),[])
            launch=next((a for a in launch if a['globalTid']==tid and start<=a['start']<end),None)
            if launch is None: continue
            records=[e for e in event_records.get((s['globalPid'],s['contextId'],s['eventId']),[]) if e['api_start']<=launch['start']]
            record=max(records,key=lambda e:e['api_start']) if records else None
            dependencies.append(dict(s,sync_label=enum.get(s['syncType'],'unknown'),wait_api=launch,record=record))
        output.append(dict(label=label,start_ns=start,end_ns=end,tid=tid,elapsed_ns=end-start,
            exclusive_ns=dict(exclusive),gpu_idle_by_host_phase_ns=dict(idle_by_phase),
            assigned_gpu_ns=duration([(a['start'],a['end']) for a in own]),device_busy_ns=duration(device_intervals),
            assigned_gpu_by_kind_ns={kind:duration([(a['start'],a['end']) for a in own if a['kind']==kind]) for kind in ('KERNEL','MEMCPY','MEMSET')},
            groups=[dict(label=g['label'],start=g['start'],end=g['end']) for g in groups],
            phases=[dict(start=a,end=b,category=c) for a,b,c in segments],activities=own,dependencies=dependencies))
    conn.close()
    # Unassigned activity includes setup/oracles outside measured requests. Count
    # separately activity overlapping a request, which can also be another thread.
    active_windows=interval_union([(r['start_ns'],r['end_ns']) for r in output])
    unassigned=[a for i,a in enumerate(activities) if i not in assigned and any(a['start']<e and a['end']>s for s,e in active_windows)]
    return dict(sqlite=str(path),requests=output,total_cuda_activities=len(activities),
        unassigned_overlapping_cuda_activities=len(unassigned),unassigned_examples=unassigned[:20],
        limitations=['Host-exclusive and GPU times overlap and must not be added.',
          'GPU-idle intersections are descriptive; only unprofiled interventions estimate removable latency.',
          'Device busy includes concurrent requests on the same process/device; assigned GPU uses launch-thread correlation.',
          'Event record-to-wait evidence uses the latest preceding record for the same process/context/event; stream ordering remains visible in raw activity.',
          'Native CPU NVTX is disabled in the frozen runtime. Native submission includes binding/runtime work that this build cannot subdivide.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sqlite',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(analyze_trace(a.sqlite),indent=2)+'\n')


if __name__=='__main__': main()
