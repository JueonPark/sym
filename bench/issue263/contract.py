"""Frozen launch settings and per-process qualification; no benchmark imports."""
import json
import os
from pathlib import Path
import subprocess

from bench.issue263.support import (BASELINE, REPO, baseline_sources, digest, save,
                                    verify_files, verify_repository)

MEASURED = ('support.py','instrument.py','interventions.py','models.py','e5.py','lifecycle.py','concurrency.py')
PLATFORM = ('python','torch','torch_cuda','gpu')


def cpus(value):
    result=set()
    for item in value.split(','):
        ends=[int(x) for x in item.split('-')]
        if len(ends)==1:result.add(ends[0])
        elif len(ends)==2 and ends[0]<=ends[1]:result.update(range(ends[0],ends[1]+1))
        else:raise ValueError('invalid affinity')
    return sorted(result)


def requested_contract(args,env):
    # Keep the venv's executable path: resolving its symlink changes the venv.
    launch={k:os.path.abspath(getattr(args,k)) for k in ('artifact','build','python','nsys')}
    launch.update(affinity=cpus(args.affinity),device=args.device)
    for key in ('artifact','build'):
        if not Path(launch[key]).is_dir():raise ValueError('missing '+key)
    tools={key:digest(launch[key]) for key in ('python','nsys')}
    frozen=baseline_sources();artifact=Path(launch['artifact']);build=Path(launch['build'])
    inventory={x['path']:x['sha256'] for x in json.loads((artifact/'SHA256-MANIFEST.json').read_text())['files']}
    names=['02_fp16_attribution/benchmark.py','02_fp16_attribution/controls.py',
           '05_lifecycle_concurrency/workload.py','00_protocol/observed_memory.py']
    names += [str(p.relative_to(artifact)) for p in (artifact/'02_fp16_attribution').glob('*.so')]
    if len(names)==4:raise ValueError('missing native control helper')
    artifact_hashes=verify_files(artifact,{p:inventory[p] for p in names})
    verify_files(artifact,{'05_lifecycle_concurrency/'+k+'.py':v for k,v in frozen['frozen_runners'].items()})
    original=json.loads((artifact/'00_protocol/final-runtime-verification.json').read_text())
    runtime={x['path'].split('/build/perf/',1)[1]:x['sha256'] for x in original['files']}
    runtime.update(frozen['extra_runtime_sha256'])
    runtime=verify_files(build,runtime)
    probe='import json,sys,torch; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))'
    platform=json.loads(subprocess.check_output(['taskset','-c',args.affinity,launch['python'],'-c',probe],env=env,text=True))
    return dict(schema=1,launch=launch,tools_sha256=tools,platform=platform,
                environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_HOME','LD_LIBRARY_PATH',
                    'OMP_NUM_THREADS','MKL_NUM_THREADS','OMP_WAIT_POLICY','GOMP_SPINCOUNT','TORCHINDUCTOR_COMPILE_THREADS')},
                runtime_sha256=runtime,artifact_sha256=artifact_hashes,repository_sha256=verify_repository(),
                measurement_sha256={k:digest(REPO/'bench/issue263'/k) for k in MEASURED},
                frozen_runners=frozen['frozen_runners'])


def freeze_contract(directory,contract):
    path=directory/'execution-contract.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('execution contract changed; use a new output directory')
    else:
        existing=[p for p in directory.glob('*.json') if p.name!='matrix.json']
        if existing:raise ValueError('existing results lack an execution contract; require an explicit historical audit')
        save(path,contract)


def validate_record(contract,x,job,audit,*,historical=False):
    def check(ok,description):
        if not ok:raise ValueError(job['name']+': '+description+' mismatch')
    c=x['configuration'];m=x['metadata'];launch=contract['launch']
    for key in ('artifact','build'):check(c[key]==launch[key],key)
    for key in PLATFORM:check(m[key]==contract['platform'][key],key)
    check(m['affinity']==launch['affinity'],'affinity')
    check(m['torch_threads']==8//(c['callers'] if job['kind']=='concurrency' else 1),'threads')
    check(m['interop_threads']==1,'interop threads')
    for key,value in contract['environment'].items():check(m['environment'].get(key)==value,'environment '+key)
    expected_cache=str(Path(c['output']).parent/(job['name']+'-cache'))
    check(m['environment']['TORCHINDUCTOR_CACHE_DIR']==expected_cache,'cache')
    for key in ('round','variant'):check(c[key]==job[key],key)
    check(c['instrument']==job['mode'],'instrument')
    for key in ('runtime_sha256','artifact_sha256'):check(m[key]==contract[key],key)
    check({key:m['harness_sha256'][key] for key in contract['measurement_sha256']}==contract['measurement_sha256'],'measurement source')
    if not historical:check(m['repository_sha256']==contract['repository_sha256'],'repository source')
    if job['kind']!='models':check(x['frozen_runner_sha256']==contract['frozen_runners'][job['kind']],'frozen runner')
    command=audit['command'];i=command.index('taskset')
    check(command[i+1]=='-c' and cpus(command[i+2])==launch['affinity'],'launch affinity')
    check(command[i+3]==launch['python'],'interpreter')
    if 'python_executable' in m:check(m['python_executable']==launch['python'],'recorded interpreter')
    if job['profile']:check(command[0]==launch['nsys'],'profiler')
    for key in ('artifact','build'):
        check(command[command.index('--'+key)+1]==launch[key],'launch '+key)


def load_contract(directory):
    path=directory/'execution-contract.json';historical=not path.exists()
    if historical:
        sidecar=json.loads((directory/'historical-qualification.json').read_text())
        if sidecar['kind']!='post-campaign-audit' or sidecar['baseline']!=BASELINE:
            raise ValueError('unrecognized historical qualification')
        verify_files(directory,sidecar['bound_files_sha256'])
        names={job['name']+'.json' for job in json.loads((directory/'matrix.json').read_text())}
        if not names <= sidecar['bound_files_sha256'].keys():raise ValueError('unbound historical result')
        contract=sidecar['contract']
    else:contract=json.loads(path.read_text())
    frozen=baseline_sources()
    if contract['repository_sha256']!=frozen['repository_sha256'] or contract['frozen_runners']!=frozen['frozen_runners']:
        raise ValueError('baseline workload source mismatch')
    return contract,historical
