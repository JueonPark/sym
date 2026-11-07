#!/usr/bin/env python3
"""Serialized, resumable campaign. Reject contamination and stale job definitions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from bench.issue227.run import audit
from bench.issue263.support import digest,save


def matrix():
    jobs=[]
    def add(name,kind,extra,round=1,variant='baseline',mode='off',profile=False,accepted=True):
        jobs.append(dict(name=name,kind=kind,extra=extra,round=round,variant=variant,mode=mode,profile=profile,accepted=accepted))
    for r in range(1,6):
        variants=['baseline','no_report','no_check','both']
        variants=variants[r%4:]+variants[:r%4]
        for v in variants:add(f'e2-r{r}-{v}','models',[],r,v)
        add(f'e2-r{r}-counters','models',[],r,mode='counters',accepted=False)
    for r in range(1,4):
        add(f'e2-r{r}-nvtx','models',['--samples','3'],r,mode='nvtx',profile=True,accepted=False)
        arms=[('torch','baseline'),('sym','baseline'),('sym','no_report')]
        arms=arms[r%3:]+arms[:r%3]
        for case in ('llm_decode','moe_sparse'):
            for compute in ('eager','inductor'):
                for runtime,v in arms:
                    add(f'life-r{r}-{case}-{compute}-{runtime}-{v}','lifecycle',
                        ['--case',case,'--compute',compute,'--runtime',runtime],r,v)
            for callers in (1,2,4):
                for runtime,v in arms:
                    add(f'burst-r{r}-{case}-c{callers}-{runtime}-{v}','concurrency',
                        ['--case',case,'--callers',str(callers),'--runtime',runtime],r,v)
        for runtime in ('sym','torch'):
            add(f'life-r{r}-counter-{runtime}','lifecycle',
                ['--case','llm_decode','--compute','eager','--runtime',runtime],r,mode='counters',accepted=False)
            add(f'burst-r{r}-counter-{runtime}','concurrency',
                ['--case','llm_decode','--callers','4','--runtime',runtime],r,mode='counters',accepted=False)
    for case in ('llm_decode','moe_sparse'):
        for runtime in ('sym','torch'):
            for compute in ('eager','inductor'):
                add(f'life-trace-{case}-{compute}-{runtime}','lifecycle',
                    ['--case',case,'--compute',compute,'--runtime',runtime,'--requests','3'],mode='nvtx',profile=True,accepted=False)
            for callers in (1,2,4):
                add(f'burst-trace-{case}-c{callers}-{runtime}','concurrency',
                    ['--case',case,'--callers',str(callers),'--runtime',runtime,'--batches','3'],mode='nvtx',profile=True,accepted=False)
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('artifact','build','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--nsys',required=True)
    p.add_argument('--suite',choices=('all','models','lifecycle','concurrency'),default='all')
    p.add_argument('--affinity',default='4-7,20-23');p.add_argument('--device',default='0')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    jobs=matrix();manifest=args.output/'matrix.json'
    if manifest.exists():assert json.loads(manifest.read_text())==jobs,'matrix changed after freeze'
    else:save(manifest,jobs)
    env=dict(os.environ,PYTHONPATH=str(args.build/'python'),LD_LIBRARY_PATH=str(args.build/'libreloc'),
        CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OMP_WAIT_POLICY='PASSIVE',
        GOMP_SPINCOUNT='0',TORCHINDUCTOR_COMPILE_THREADS='1')
    source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for job in jobs:
        if args.suite!='all' and job['kind']!=args.suite:continue
        name=job['name'];output=args.output/(name+'.json');audit_path=args.output/(name+'.audit.json')
        if output.exists():
            previous=json.loads(output.read_text())
            assert previous.get('complete') and previous.get('campaign_job')==job and previous.get('isolation',{}).get('valid'),f'incomplete existing job {name}'
            print(name,'already verified',flush=True);continue
        foreign=audit(set())
        if foreign:raise RuntimeError(f'foreign workload before {name}: {foreign}')
        cache=args.output/(name+'-cache')
        if cache.exists():raise FileExistsError(f'unfinished cache {cache}')
        variables=env|dict(TORCHINDUCTOR_CACHE_DIR=str(cache),EXPERIMENT_LAUNCH_NS=str(time.time_ns()))
        command=['taskset','-c',args.affinity,args.python,str(REPO/'bench/issue263'/f'{job["kind"]}.py'),
            '--artifact',str(args.artifact),'--build',str(args.build),'--output',str(output),
            '--round',str(job['round']),'--variant',job['variant'],'--instrument',job['mode'],*job['extra']]
        if job['profile']:
            command=[args.nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                     '--force-overwrite=false','-o',str(args.output/name),*command]
        metadata=dict(job=job,command=command,harness_sha256=source,revision=revision,started_ns=time.time_ns(),observations=[])
        save(audit_path,metadata)
        with (args.output/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=REPO,env=variables,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                foreign=audit({process.pid})
                if foreign:metadata['observations'].append(dict(time_ns=time.time_ns(),foreign=foreign))
                time.sleep(3)
        metadata.update(returncode=process.returncode,finished_ns=time.time_ns())
        save(audit_path,metadata)
        if process.returncode or metadata['observations']:raise RuntimeError(f'failed/contaminated job {name}; see {audit_path}')
        result=json.loads(output.read_text());assert result['complete'],name
        result.update(campaign_job=job,isolation=dict(valid=True,audit=audit_path.name,interval_seconds=3),runner_revision=revision)
        save(output,result)
        if job['profile']:
            with (args.output/(name+'.export.log')).open('w') as log:
                subprocess.run([args.nsys,'export','--type','sqlite','--output',str(args.output/(name+'.sqlite')),
                    str(args.output/(name+'.nsys-rep'))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,'complete',round((metadata['finished_ns']-metadata['started_ns'])/1e9,1),'s',flush=True)


if __name__=='__main__':main()
