#!/usr/bin/env python3
"""Validate the frozen matrix and aggregate independent process pairs."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from bench.issue263.paired import distribution, paired_summary
from bench.issue263.support import digest,save
from bench.issue263.trace import analyze_trace


def pair_rows(baseline,candidate,metric,expected):
    def index(rows):
        result={r['round']:r for r in rows}
        if len(result)!=len(rows) or set(result)!=set(range(1,expected+1)):
            raise ValueError('missing/duplicate process round')
        return result
    a,b=index(baseline),index(candidate)
    return [dict(round=r,baseline=a[r][metric],candidate=b[r][metric]) for r in sorted(a)]


def lifecycle_metrics(x):
    times=[r['completed_ms'] for r in x['requests']];p=x['phases']
    initial=[r['completed_ms'] for r in x['requests'] if r['phase']=='initial']
    shape=[r['completed_ms'] for r in x['requests'] if r['phase']=='shape_change']
    scale=[r['completed_ms'] for r in x['requests'] if r['phase']=='scale_change']
    setup=x['launch_to_pre_cuda_ms']+sum(p[k] for k in ('cuda_init_ms','fixture_ms','snapshot_total_ms','request_owner_ms'))
    warm=distribution(initial[1:]);retirement=p['owner_retirement_ms']+p['caller_release_and_cache_clear_ms']
    return dict(setup_ms=setup,first_ms=times[0],warm_p50_ms=warm['p50'],warm_p95_ms=warm['p95'],warm_n=warm['n'],
        shape_first_ms=shape[0],shape_warm_p50_ms=statistics.median(shape[1:]),
        scale_first_ms=scale[0],scale_warm_p50_ms=statistics.median(scale[1:]),
        scale_reprepare_ms=p['scale_reprepare_ms'],retirement_ms=retirement,
        full120_accounted_ms=setup+sum(times)+p['shape_update_ms']+p['scale_reprepare_ms']+retirement)


def burst_metrics(x):
    requests=[r for b in x['batches'] for r in b['requests']]
    result={'requests_per_second':1000*len(requests)/sum(b['completed_ms'] for b in x['batches']),
            'mean_burst_ms':statistics.mean(b['completed_ms'] for b in x['batches'])}
    for key in ('service_ms','executor_execution_ms','sojourn_ms','queue_wait_ms'):
        dist=distribution([r[key] for r in requests])
        result.update({key.removesuffix('_ms')+'_p50_ms':dist['p50'],key.removesuffix('_ms')+'_p95_ms':dist['p95']})
    return result


def collect(directory,*,partial=False):
    matrix=json.loads((directory/'matrix.json').read_text());records=[];missing=[];input_hashes={}
    for job in matrix:
        path=directory/(job['name']+'.json')
        if not path.exists():missing.append(job['name']);continue
        x=json.loads(path.read_text())
        if not x.get('complete') or not x.get('isolation',{}).get('valid'):
            missing.append(job['name']);continue
        assert x['campaign_job']==job and not x['metadata']['baseline_hash_mismatches'],job['name']
        audit=json.loads((directory/x['isolation']['audit']).read_text())
        assert audit['returncode']==0 and not audit['observations'],job['name']
        c=x['configuration'];kind=job['kind']
        base=dict(file=path.name,kind=kind,round=job['round'],variant=job['variant'],instrument=job['mode'],profile=job['profile'],accepted=job['accepted'])
        if kind=='models':
            for case,row in x['cases'].items():
                assert row['warm_compile_callbacks']==0
                assert row['native_after_close']['retained_bytes']==row['native_after_close']['live_bytes']==0
                for arm in row['arms'].values():
                    assert arm['checks']==c['samples']==len(arm['samples_ms'])
                    d=distribution(arm['samples_ms'])
                    records.append(dict(base,case=case,compute=arm['factors']['compute'],runtime=arm['factors']['runtime'],
                        p50_ms=d['p50'],p95_ms=d['p95'],samples=d['n'],first_ms=arm['first_ms'],
                        allocation_diagnostic=arm['allocation_diagnostic']))
        else:
            memory=directory/x['physical_memory']['file'];assert digest(memory)==x['physical_memory']['sha256']
            assert x['held_output_checks_after_close']==(4 if kind=='lifecycle' else 8)
            closed=x['native_after_close'];closed=closed if isinstance(closed,list) else [closed]
            for item in closed:
                if item is not None:
                    assert all(item['typed'][k]==0 for k in ('retained_bytes','live_bytes','streams','background_workers','active_contexts'))
            if kind=='lifecycle':
                assert not x['cuda_initialized_before_explicit_init']
                assert len(x['requests'])==c['requests']+20
                assert all(r['correct'] for r in x['requests'])
                if job['accepted']:assert c['requests']==100
                if c['runtime']=='sym':assert x['before_retirement']['snapshots']['revisions']==x['initial_stats']['snapshots']['revisions']+1
                records.append(dict(base,case=c['case'],runtime=c['runtime'],compute=c['compute'],**lifecycle_metrics(x),
                    phases=x['phases'],native_after_close=x['native_after_close'],memory=x['physical_memory']))
            else:
                assert x['physical_meter_valid'] and x['checks']==8*c['batches']
                assert all(r['correct'] for b in x['batches'] for r in b['requests'])
                assert len(set(x['input_sha256']))==8
                if c['case'] in input_hashes:assert input_hashes[c['case']]==x['input_sha256']
                else:input_hashes[c['case']]=x['input_sha256']
                assert (1008<<20)<=x['budget']['after']['free']<=(1024<<20)
                assert x['protocol']['torch_threads_per_caller']*c['callers']==8
                if job['accepted']:assert c['batches']==30
                deltas={}
                if c['runtime']=='sym':
                    for key in ('device_allocations','host_allocations','context_creations'):
                        deltas[key]=sum(b['transfer']['resources'][key]-a['transfer']['resources'][key]
                            for a,b in zip(x['ready']['workers'],x['warm_after']['workers']))
                        assert deltas[key]==0
                records.append(dict(base,case=c['case'],runtime=c['runtime'],callers=c['callers'],**burst_metrics(x),
                    warm_native_deltas=deltas,torch_ready=x['ready']['torch'],torch_warm=x['warm_after']['torch'],
                    observed_warm_app_increment_bytes=x['observed_warm_app_increment_peak'],memory=x['physical_memory']))
    if missing and not partial:raise ValueError('incomplete matrix: '+', '.join(missing))
    return records,missing


def comparisons(records):
    output=[]
    def select(**filters):return [r for r in records if all(r.get(k)==v for k,v in filters.items())]
    def compare(name,a,b,metrics,expected,**labels):
        if len(a)!=expected or len(b)!=expected:return
        output.append(dict(comparison=name,**labels,metrics={m:paired_summary(pair_rows(a,b,m,expected)) for m in metrics}))
    cases=('llm_decode','llm_prefill','moe_sparse','moe_dense')
    for case in cases:
        for compute in ('eager','inductor'):
            key=dict(kind='models',case=case,compute=compute)
            baseline=select(**key,runtime='sym',variant='baseline',instrument='off')
            for v in ('baseline','no_report','no_check','both'):
                candidate=select(**key,runtime='sym',variant=v,instrument='off')
                control=select(**key,runtime='torch',variant=v,instrument='off')
                compare('sym_reduction_to_torch_parity',candidate,control,['p50_ms','p95_ms'],5,**key,variant=v)
                if v!='baseline':compare('intervention_reduction',baseline,candidate,['p50_ms','p95_ms'],5,**key,variant=v)
            for runtime in ('sym','torch'):
                a=select(**key,runtime=runtime,variant='baseline',instrument='off')
                for mode,n in (('counters',5),('nvtx',3)):
                    b=select(**key,runtime=runtime,variant='baseline',instrument=mode)
                    compare('instrumentation_reduction_negative_is_overhead',[r for r in a if r['round']<=n],b,['p50_ms','p95_ms'],n,**key,runtime=runtime,instrument=mode)
    for kind in ('lifecycle','concurrency'):
        for case in ('llm_decode','moe_sparse'):
            for condition in (('eager','inductor') if kind=='lifecycle' else (1,2,4)):
                key=dict(kind=kind,case=case,**({'compute':condition} if kind=='lifecycle' else {'callers':condition}))
                metrics=(['full120_accounted_ms','setup_ms','first_ms','warm_p50_ms','warm_p95_ms','shape_first_ms','scale_first_ms','scale_reprepare_ms']
                    if kind=='lifecycle' else ['mean_burst_ms','requests_per_second','service_p50_ms','service_p95_ms','sojourn_p50_ms','sojourn_p95_ms'])
                a=select(**key,runtime='sym',variant='baseline',instrument='off')
                b=select(**key,runtime='sym',variant='no_report',instrument='off')
                control=select(**key,runtime='torch',variant='baseline',instrument='off')
                compare('sym_reduction_to_torch_parity',a,control,metrics,3,**key,variant='baseline')
                compare('sym_reduction_to_torch_parity',b,control,metrics,3,**key,variant='no_report')
                compare('intervention_reduction',a,b,metrics,3,**key,variant='no_report')
                for runtime in ('sym','torch'):
                    off=select(**key,runtime=runtime,variant='baseline',instrument='off')
                    counters=select(**key,runtime=runtime,variant='baseline',instrument='counters')
                    compare('instrumentation_reduction_negative_is_overhead',off,counters,metrics,3,**key,runtime=runtime,instrument='counters')
    return output


def trace_summary(path,out):
    result=analyze_trace(path)
    with gzip.open(out,'wt') as f:json.dump(result,f,separators=(',',':'))
    groups=defaultdict(list)
    for r in result['requests']:
        parts=r['label'].split('/')
        key='/'.join(parts[:-1])
        groups[key].append(r)
    summaries=[]
    for key,rows in groups.items():
        categories={c for r in rows for c in r['exclusive_ns']}
        first=rows[0]
        signatures=Counter((a['kind'],a.get('copyKind'),a['name']) for a in first['activities'])
        witnesses=[]
        for d in first['dependencies']:
            record=d['record']
            if record is None or record['streamId']==d['streamId']:continue
            producer=[a for a in first['activities'] if a['streamId']==record['streamId'] and a['launch_start']<=record['api_start']]
            consumer=[a for a in first['activities'] if a['streamId']==d['streamId'] and a['start']>=d['end'] and a['kind']=='KERNEL']
            if not producer or not consumer:continue
            last=max(producer,key=lambda a:a['launch_start']);following=min(consumer,key=lambda a:a['start'])
            if last['group'] is None:continue
            witnesses.append(dict(event_id=d['eventId'],producer_stream=record['streamId'],consumer_stream=d['streamId'],
                producer_last_activity=last,record=record,wait=d,consumer_next_kernel=following))
            if len(witnesses)==2:break
        summaries.append(dict(label=key,requests=len(rows),elapsed_ms=distribution([r['elapsed_ns']/1e6 for r in rows]),
            assigned_gpu_ms=distribution([r['assigned_gpu_ns']/1e6 for r in rows]),
            device_busy_ms=distribution([r['device_busy_ns']/1e6 for r in rows]),
            host_exclusive_median_ms={c:statistics.median(r['exclusive_ns'].get(c,0)/1e6 for r in rows) for c in categories},
            gpu_idle_by_phase_median_ms={c:statistics.median(r['gpu_idle_by_host_phase_ns'].get(c,0)/1e6 for r in rows) for c in categories},
            group_counts=sorted({len(r['groups']) for r in rows}),
            activity_counts=sorted({len(r['activities']) for r in rows}),
            event_dependency_count=sum(d['record'] is not None for r in rows for d in r['dependencies']),
            representative_request=first['label'],
            representative_activity_signature=[dict(kind=k[0],copy_kind=k[1],name=k[2],count=v) for k,v in sorted(signatures.items())],
            representative_copy_bytes=sum(a.get('bytes',0) for a in first['activities'] if a['kind']=='MEMCPY'),
            dependency_witnesses=witnesses))
    return dict(file=path.name,analysis=out.name,unassigned_overlapping_cuda_activities=result['unassigned_overlapping_cuda_activities'],
                summaries=summaries,limitations=result['limitations'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true');p.add_argument('--traces',action='store_true')
    args=p.parse_args();rows,missing=collect(args.directory,partial=args.partial)
    result=dict(complete=not missing,missing=missing,process_rows=rows,comparisons=comparisons(rows),
        definitions=dict(percentiles='linear interpolation at (n-1)*q',uncertainty='10,000 percentile bootstrap resamples of independent process pairs, seed 263; exploratory with 3 or 5 rounds',
          effects='positive reduction means candidate has a smaller metric. For requests_per_second, negate reduction and reverse CI endpoints to express throughput increase.',
          lifecycle='original accounted envelope: launch/setup + completed requests + updates + retirement; excludes oracle validation',
          throughput='requests per summed completed burst duration; not sustained serving',
          instrumentation='negative reduction means overhead; profiled samples have fewer requests and are diagnostic only'))
    if args.traces:
        result['traces']=[trace_summary(path,path.with_suffix('.trace.json.gz')) for path in sorted(args.directory.glob('*.sqlite'))]
    save(args.output,result)
    print('records',len(rows),'comparisons',len(result['comparisons']),'missing',len(missing))


if __name__=='__main__':main()
