"""Serial fresh-process arms, complete frozen membership and contamination audit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bench.issue269 import contract
from bench.issue266.isolation import audit


def quiet(wait, pilot):
    since=None; observations=0
    while True:
        foreign=audit({os.getpid()})
        if foreign:
            since=None; observations=0
            if not wait: raise RuntimeError(f'foreign activity: {foreign}')
        else:
            if since is None: since=time.monotonic()
            observations+=1
        if since is not None and time.monotonic()-since >= (0 if pilot else 60):
            return dict(quiet_seconds=time.monotonic()-since,quiet_observations=observations)
        if foreign: print('waiting for idle',foreign,flush=True)
        time.sleep(3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','artifact','compiler','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wait-for-idle',action='store_true'); parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--jobs',nargs='+')
    args=parser.parse_args()
    if args.jobs and not args.pilot: parser.error('subsets are pilot-only')
    if args.profile and not args.pilot: parser.error('profiling is diagnostic only')
    selected=[j for j in contract.jobs() if not args.jobs or j['id'] in args.jobs]
    if args.jobs and len(selected)!=len(set(args.jobs)): parser.error('invalid jobs')
    args.output.mkdir(parents=True,exist_ok=True)
    current=dict(schema=1,qualification=not args.pilot,jobs=selected,
                 provenance=contract.provenance(args.runtime,args.artifact,args.compiler))
    path=args.output/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in current.items()): raise ValueError('frozen campaign changed')
    else:
        manifest=current|dict(records={},excluded=[]); contract.save(path,manifest)
    previous_cell=None; preflight=None
    for job in selected:
        if job['id'] in manifest['records']:
            row=manifest['records'][job['id']]
            assert contract.sha(args.output/row['path'])==row['sha256']
            assert contract.sha(args.output/row['audit_path'])==row['audit_sha256']
            continue
        cell=job['cell']
        while True:
            if cell!=previous_cell or preflight is None:
                preflight=quiet(args.wait_for_idle,args.pilot); previous_cell=cell
            target=args.output/job['id']; target.mkdir()
            env=os.environ|dict(PYTHONPATH=str(args.runtime/'python'),LD_LIBRARY_PATH=str(args.runtime/'libreloc'),
                SYM_RELOC_EXPORT=str(args.compiler),CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS=str(8//job['callers']),MKL_NUM_THREADS=str(8//job['callers']),
                OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
                TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(target/'inductor-cache'),
                CXX='/usr/bin/g++',CC='/usr/bin/gcc')
            command=['taskset','-c','4-7,20-23',sys.executable,str(Path(__file__).with_name('worker.py')),
                     '--job',job['id'],'--runtime',str(args.runtime),'--artifact',str(args.artifact),
                     '--output',str(target/'record.json')]+(['--pilot'] if args.pilot else [])
            if args.profile: command.append('--profile')
            start=time.monotonic(); seen=[]; observations=0
            with (target/'worker.log').open('w') as log:
                child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while child.poll() is None:
                        seen.extend(audit({os.getpid(),child.pid})); observations+=1; time.sleep(3)
                finally:
                    if child.poll() is None: os.killpg(child.pid,signal.SIGTERM); child.wait()
            checked=dict(job=job,command=command,exit_code=child.returncode,preflight=preflight,
                         observations=observations,foreign_busy_processes=sorted(set(seen)),
                         wall_seconds=time.monotonic()-start)
            contract.save(target/'audit.json',checked)
            if child.returncode: raise RuntimeError(f'worker failed; retained evidence at {target}')
            if seen:
                excluded=args.output/'excluded'/f"{job['id']}-{len(manifest['excluded'])+1}"
                excluded.parent.mkdir(exist_ok=True); shutil.move(target,excluded)
                manifest['excluded'].append(dict(job=job,reason='foreign activity',path=str(excluded.relative_to(args.output)),
                    files={str(p.relative_to(excluded)):contract.sha(p) for p in excluded.rglob('*') if p.is_file()}))
                contract.save(path,manifest); preflight=None
                if not args.wait_for_idle: raise RuntimeError('contaminated attempt excluded')
                continue
            if contract.provenance(args.runtime,args.artifact,args.compiler)!=manifest['provenance']:
                raise RuntimeError('sources changed during measurement')
            manifest['records'][job['id']]=dict(path=str((target/'record.json').relative_to(args.output)),
                sha256=contract.sha(target/'record.json'),audit_path=str((target/'audit.json').relative_to(args.output)),
                audit_sha256=contract.sha(target/'audit.json'))
            contract.save(path,manifest)
            print(job['id'],'complete',round(checked['wall_seconds'],1),'seconds',flush=True)
            break


if __name__=='__main__': main()
