"""Strict CPU-only qualification and explicit per-round comparisons."""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bench.issue269 import contract
from bench.issue268.analyze import (require,finite,membership,validate_audit,
    memory_peak,phase_peaks,percentile)


def validate_record(row,job):
    require(row['job']==job,'job changed')
    require(row['protocol']==contract.protocol(job),'controls changed')
    require(row['environment']==contract.expected_environment(job),'environment changed')
    require(row['outcome'] in ('success','oom'),'unknown outcome')
    if row['outcome']=='oom':
        require(bool(row.get('error')) and row.get('phase') in
                ('fixture','prepare','first','warmup','warm'),'OOM provenance')
        return
    require(row['cache_requests']==8 and row['cache_checks']==8*4*contract.CASES[job['case']][1]*2,'decoder cache audit')
    require(row['warmups']==5,'warmups changed')
    require(row['checks']==30 and row['retained_checks']==30 and row['held_after_close']==1,'correctness counts')
    for key in ('completed_ms','submission_ms'):
        require(len(row[key])==30,'sample membership')
        for value in row[key]:finite(value,key)
    for key in ('preparation_ms','first_ms','warmup_ms','retirement_ms','compile_ms',
                'warm_start_ms','warm_end_ms'):finite(row[key],key)
    require(row['compile_before']==row['compile_after'],'warm compilation')
    require(row['warm_start_ms']<row['warm_end_ms'],'warm interval')
    require(len(row['input_sha256'])==8,'input membership')
    require(all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in row['input_sha256']), 'input digest')
    phases=row['phases']
    require([p['name'] for p in phases]==['fixture','prepare','first','warmup','warm','cache_audit','retirement'],'phase membership')
    previous=0.
    for p in phases:
        finite(p['start_ms'],'phase start');finite(p['end_ms'],'phase end')
        require(previous<=p['start_ms']<p['end_ms'],'phase ordering');previous=p['end_ms']
        if p['name']=='warm': require((p['start_ms'],p['end_ms'])==(row['warm_start_ms'],row['warm_end_ms']),'warm phase')
    memory=row['torch_memory']
    require(memory['max_allocated']>=memory['allocated']>=0 and
            memory['max_reserved']>=memory['reserved']>=memory['allocated'],'Torch memory bounds')
    before,after=row['replay_before'],row['replay_after']
    require(len(before)==len(after)==(3 if job['replay'] else 0),'replay membership')
    for a,b in zip(before,after):
        require(a['captures']==b['captures'] and b['frozen'],'warm capture/freeze')
        require(0<=b['retained_bytes']<=b['max_bytes']==256<<20,'replay admission')
    if job['variant']!='lowered':require(not any(r['selected'] for r in row['lowering']),'control lowered')
    elif job['case']=='prefill256':require(not any(r['selected'] for r in row['lowering']),'prefill unexpectedly restored')
    else:require(any(r['selected'] for r in row['lowering']),'candidate never selected')


def summarize(row,physical_peak):
    out=row['job']|dict(outcome=row['outcome'])
    if row['outcome']!='success':return out|dict(error=row['error'],phase=row['phase'])
    warm=statistics.median(row['completed_ms'])
    setup=math.fsum(row[k] for k in ('preparation_ms','first_ms','warmup_ms'))
    return out|dict(p50_ms=warm,p95_ms=percentile(row['completed_ms'],.95),
        submit_p50_ms=statistics.median(row['submission_ms']),submit_p95_ms=percentile(row['submission_ms'],.95),
        preparation_ms=row['preparation_ms'],first_ms=row['first_ms'],warmup_ms=row['warmup_ms'],
        compile_ms=row['compile_ms'],retirement_ms=row['retirement_ms'],setup_ms=setup,
        total_ms={str(n):setup+n*warm for n in (1,100,1000)},physical_peak_bytes=physical_peak,
        torch_memory=row['torch_memory'],lowering=row['lowering'],transfer_after=row['transfer_after'],
        warm_replays=sum(b['replays']-a['replays'] for a,b in zip(row['replay_before'],row['replay_after'])),
        retained_replay_bytes=sum(r['retained_bytes'] for r in row['replay_after']),
        capture_ms=math.fsum(r.get('capture_ms',0) for r in row['replay_after']))


def validate_micro(record,job):
    require(record['job']==job and record['environment']==contract.expected_environment(job),'micro environment/job')
    require(record['protocol']==contract.protocol(job),'micro controls')
    expected={(case['name'],mode) for case in contract.MICROS for mode in contract.VARIANTS}
    membership([(r['case']['name'],r['variant']) for r in record['rows']],expected)
    for row in record['rows']:
        require(row['case'] in contract.MICROS,'micro case')
        require(row['outcome'] in ('success','oom'),'micro outcome')
        if row['case']['budget_mib']:
            budget=row['budget'];target=row['case']['budget_mib']<<20
            require(budget['requested_free_budget']==target and target-(16<<20)<=budget['after']['free']<=target,'micro physical budget')
        if row['outcome']=='oom':
            require(bool(row.get('error')) and row.get('phase') in ('prepare','first','warmup','warm'),'micro OOM')
            continue
        require(row['checks']==100 and row['retained_checks']==100 and row['held_after_close']==1,'micro checks')
        require(row['warmups']==5 and row['compile_before']==row['compile_after'],'micro warmup/compile')
        for key in ('completed_ms','submission_ms','consumer_ms','transfer_ms'):
            require(len(row[key])==100,'micro sample membership')
            for value in row[key]:finite(value,key)
        require(len(row['input_sha256'])==4,'micro input membership')
        selected=[r for r in row['lowering'] if r['selected']]
        expected=2 if row['variant']=='lowered' and not row['case']['visible'] else 0
        require(len(selected)==expected,'micro eligibility changed')
        require(row['warm_start_ms']<row['warm_end_ms'],'micro warm interval')
        for key in ('preparation_ms','first_ms','warmup_ms','retirement_ms'):finite(row[key],key)
        if row['case']['budget_mib']:
            budget=row['budget'];target=row['case']['budget_mib']<<20
            require(budget['requested_free_budget']==target and target-(16<<20)<=budget['after']['free']<=target,'micro physical budget')


def validate_files(root,row):
    files=list(row.get('compiler_graphs',[]))
    if row.get('profile'):files.append(row['profile'])
    for item in files:
        require(contract.sha(root/item['file'])==item['sha256'],'generated evidence checksum')


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text())
    require(manifest['qualification'],'pilot is not qualification')
    require(manifest['jobs']==contract.jobs(),'matrix definition changed')
    membership(list(manifest['records']),[j['id'] for j in contract.jobs()])
    results=[];supplements=[];inputs={}
    for job in contract.jobs():
        saved=manifest['records'][job['id']]
        def read(name,hash_name):
            p=root/saved[name];require(contract.sha(p)==saved[hash_name],'record checksum')
            return json.loads(p.read_text())
        record=read('path','sha256');audit=read('audit_path','audit_sha256');validate_audit(audit)
        require(audit['job']==job,'audit job')
        require(record['provenance']==manifest['provenance'],'provenance mismatch')
        p=(root/saved['path']).parent/record['memory']['file']
        require(contract.sha(p)==record['memory']['sha256'],'memory checksum')
        samples=json.loads(gzip.decompress(p.read_bytes()))['samples'];require(bool(samples),'missing physical trace')
        if job['kind']=='micro':
            validate_micro(record,job)
            for row in record['rows']:
                validate_files((root/saved['path']).parent,row)
                key=('micro',row['case']['name'])
                if 'input_sha256' in row:require(row['input_sha256']==inputs.setdefault(key,row['input_sha256']),'unmatched micro inputs')
                peak=memory_peak(samples,row['warm_start_ms'],row['warm_end_ms']) if row['outcome']=='success' else None
                supplements.append(dict(job=job,row=row,physical_peak_bytes=peak))
            continue
        validate_record(record,job)
        validate_files((root/saved['path']).parent,record)
        if 'input_sha256' in record:
            require(record['input_sha256']==inputs.setdefault(job['case'],record['input_sha256']),'unmatched inputs')
        peak=memory_peak(samples,record['warm_start_ms'],record['warm_end_ms']) if record['outcome']=='success' else None
        results.append(summarize(record,peak)|dict(phase_memory=phase_peaks(samples,record['phases']),
            whole_process_peak_bytes=max(s[1] for s in samples),whole_process_rss_peak_bytes=max(s[2] for s in samples)))
    for row in manifest['excluded']:
        for name,digest in row['files'].items():require(contract.sha(root/row['path']/name)==digest,'excluded checksum')
    return results,supplements


def comparisons(rows):
    index={(r['case'],r['compute'],r['round'],r['replay'],r['runtime'],r['variant']):r for r in rows}
    results=[]
    for row in rows:
        if row['outcome']!='success' or row['variant']!='lowered':continue
        for variant in ('native','materialized'):
            ref=index[(row['case'],row['compute'],row['round'],row['replay'],row['runtime'],variant)]
            if ref['outcome']!='success':continue
            delta=ref['p50_ms']-row['p50_ms'];intercept=row['setup_ms']-ref['setup_ms']
            results.append({k:row[k] for k in ('case','compute','round','replay','runtime')}|dict(
                against=variant,speedup=ref['p50_ms']/row['p50_ms'],break_even=math.ceil(max(0,intercept)/delta) if delta>0 else None))
    targets=[]
    for case in contract.CASES:
        for compute in ('eager','inductor'):
            for replay in (False,True):
                if replay and case!='decode8':continue
                for variant in contract.VARIANTS:
                    rounds=[]
                    for round_ in (1,2,3):
                        candidate=index[(case,compute,round_,replay,'sym',variant)]
                        controls=[index[(case,c,round_,replay,'torch',variant)] for c in ('eager','inductor')]
                        rounds.append(candidate['outcome']=='success' and all(r['outcome']=='success' and candidate['p50_ms']<=.95*r['p50_ms'] for r in controls))
                    targets.append(dict(case=case,compute=compute,replay=replay,variant=variant,round_passes=rounds,qualified=all(rounds)))
    return results,targets


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows,micro=analyze(args.root);pairs,targets=comparisons(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in (('measurements.json',rows),('supplements.json',micro),('comparisons.json',pairs),('parent-targets.json',targets)):
        contract.save(args.output/name,value)
    print('Validated',len(rows),'model arms and',len(micro),'supplement rows')


if __name__=='__main__':main()
