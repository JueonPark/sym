#!/usr/bin/env python3
"""Serialized acceptance runs with process-tree-aware interference checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BUILDERS={'ninja','cmake','cc1plus','clang','clang++','g++','scons','ld.lld','tar','gzip','pigz','zstd','xz'}


def audit(allowed):
    rows=[line.split(None,3) for line in subprocess.check_output(
        ['ps','-eo','pid=,ppid=,stat=,comm='],text=True).splitlines()]
    own=set(allowed)
    for _ in range(10):
        own.update(int(pid) for pid,ppid,state,comm in rows if int(ppid) in own)
    busy=[(int(pid),comm) for pid,ppid,state,comm in rows
          if comm in BUILDERS and not state.startswith(('T','Z')) and int(pid) not in own]
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    return busy+[(int(pid),'GPU compute') for pid in gpu if pid.strip().isdigit() and int(pid) not in own]


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=os.environ.copy()
    env.update(PYTHONPATH=str(args.build/'python'),SYM_RELOC_EXPORT=str(args.build/'sym/tools/sym-reloc-export'),
        SYM_OPT=str(args.build/'sym/tools/sym-opt'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',TORCHINDUCTOR_COMPILE_THREADS='1',SYM_RUNTIME_REVISION=revision)
    def run(name,script,extra,overrides=None):
        output=args.output/(name+'.json')
        cache=args.output/(name+'-cache')
        assert not output.exists() and not cache.exists(),'acceptance output already exists'
        assert not audit(set()),'foreign workload before run'
        variables=env|dict(TORCHINDUCTOR_CACHE_DIR=str(cache))|(overrides or {})
        with (args.output/(name+'.log')).open('w') as log:
            start=time.monotonic()
            process=subprocess.Popen(['taskset','-c',args.affinity,args.python,
                str(ROOT/'bench/issue227'/script),'--output',str(output),*extra],
                cwd=ROOT,env=variables,stdout=log,stderr=subprocess.STDOUT)
            observed=[]
            while process.poll() is None:
                observed+=audit({process.pid})
                time.sleep(3)
        print(name,'exit',process.returncode,'seconds',round(time.monotonic()-start),'foreign',observed,flush=True)
        if process.returncode:raise SystemExit(process.returncode)
        result=json.loads(output.read_text())
        result['isolation']=dict(foreign_busy_processes=sorted(set(observed)),interval_seconds=3,
            monitor='process-tree-aware build/packaging and GPU processes')
        result['source_revision']=revision
        output.write_text(json.dumps(result,separators=(',',':'))+'\n')
        if observed:raise SystemExit('contaminated run')
    for i in range(1,4):
        run(f'round{i}','placement.py',['--round',str(i)])
        profile=str(args.output/f'round{i}.profile.json')
        order=['torch','inductor','sym_inductor','sym_placement_inductor']
        order=order[i%4:]+order[:i%4]
        run(f'decode{i}','decode.py',['--cases','llm_decode','--paths',*order],
            dict(SYM_PLACEMENT_PROFILE=profile))
        for scenario in ('cpu_busy','gpu_busy'):
            run(f'{scenario}{i}','context.py',['--profile',profile,'--scenario',scenario])
    profile=str(args.output/'round1.profile.json')
    run('foreign_hardware','context.py',['--profile',profile,'--scenario','hardware','--device','cuda:1'])
    run('foreign_overlap','context.py',['--profile',profile,'--scenario','overlap'])
    run('hardware2','placement.py',['--round','1','--device','cuda:1','--threads','1'],
        dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',default='/tmp/sym-210-venv/bin/python')
    parser.add_argument('--affinity',default='4-7,20-23')
    main(parser.parse_args())
