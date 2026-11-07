#!/usr/bin/env python3
"""Serialized, resumable runs with frozen inputs, policies and contamination audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.issue227.run import audit
from bench.issue264.analyze import validate_record, validate_workload
from bench.issue264.support import POLICIES, code_hashes, digest, measurement_hashes, runtime_hashes, save


def matrix(suite, candidate):
    jobs = []
    if suite == 'groups':
        policies = list(POLICIES)
        for r in range(1, 6):
            for p in policies[r % 5:] + policies[:r % 5]:
                jobs.append(dict(name=f'groups-r{r}-{p}', kind='groups', round=r, policy=p,
                                 samples=300, submit_samples=100, warmup=5))
    else:
        policies = ['baseline', 'legacy-eager', candidate]
        for r in range(1, 6):
            for p in policies[r % 3:] + policies[:r % 3]:
                jobs.append(dict(name=f'e2-r{r}-{p}', kind='models', round=r, policy=p, samples=60, warmup=3))
        arms = [('torch', 'baseline'), ('sym', 'baseline'), ('sym', 'legacy-eager'), ('sym', candidate)]
        for r in range(1, 4):
            order = arms[r % 4:] + arms[:r % 4]
            for case in ('llm_decode', 'moe_sparse'):
                for compute in ('eager', 'inductor'):
                    for runtime, p in order:
                        jobs.append(dict(name=f'life-r{r}-{case}-{compute}-{runtime}-{p}', kind='lifecycle',
                            round=r, case=case, compute=compute, runtime=runtime, policy=p, requests=100))
                for callers in (1, 2, 4):
                    for runtime, p in order:
                        jobs.append(dict(name=f'burst-r{r}-{case}-c{callers}-{runtime}-{p}', kind='concurrency',
                            round=r, case=case, callers=callers, runtime=runtime, policy=p, batches=30))
    return jobs


def clean_environment():
    keep = ('HOME', 'USER', 'LOGNAME', 'PATH', 'SHELL', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR')
    return {key: os.environ[key] for key in keep if key in os.environ}


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('frozen contract changed: ' + str(path))
    else:
        save(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('artifact', 'harness', 'baseline-build', 'candidate-build', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--suite', choices=('groups', 'acceptance'), required=True)
    parser.add_argument('--candidate', choices=('legacy-native', 'prepared-eager', 'prepared-native'), default='prepared-native')
    parser.add_argument('--affinity', default='4-7,20-23')
    parser.add_argument('--device', default='0')
    parser.add_argument('--only', help='optional job-name prefix; full matrix remains frozen')
    args = parser.parse_args()
    for key in ('artifact', 'harness', 'baseline_build', 'candidate_build', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    args.python = os.path.abspath(args.python)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = matrix(args.suite, args.candidate)
    freeze(args.output / 'matrix.json', jobs)
    env = clean_environment() | dict(CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1')
    probe = json.loads(subprocess.check_output(['taskset', '-c', args.affinity, args.python, '-c',
        'import json,os,torch; print(json.dumps(dict(affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,torch_cuda=torch.version.cuda)))'],
        env=env, text=True))
    probe['gpu'] = subprocess.check_output(['nvidia-smi', '--id=' + args.device, '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    files = ['02_fp16_attribution/benchmark.py', '02_fp16_attribution/controls.py',
             '05_lifecycle_concurrency/workload.py', '00_protocol/observed_memory.py']
    files += [str(p.relative_to(args.artifact)) for p in (args.artifact / '02_fp16_attribution').glob('*.so')]
    frozen_runners = json.loads((args.harness / 'bench/issue263/baseline-sources.json').read_text())['frozen_runners']
    contract = dict(schema=1, frozen_runners=frozen_runners, runtime_sha256={p: runtime_hashes(args.baseline_build if p == 'baseline' else args.candidate_build)
                                            for p in {job['policy'] for job in jobs}},
        harness_sha256=measurement_hashes(), dependency_sha256=code_hashes(args.harness / 'bench/issue263'),
        artifact_sha256={p: digest(args.artifact / p) for p in files}, environment=probe,
        configuration={k: str(v) for k, v in vars(args).items() if k != 'only'})
    freeze(args.output / 'contract.json', contract)
    for job in jobs:
        if args.only and not job['name'].startswith(args.only): continue
        name = job['name']
        output, audit_path = args.output / (name + '.json'), args.output / (name + '.audit.json')
        if output.exists():
            previous, previous_audit = json.loads(output.read_text()), json.loads(audit_path.read_text())
            validate_record(contract, previous, job, previous_audit)
            validate_workload(previous, job)
            print(name, 'already verified', flush=True)
            continue
        foreign = audit(set())
        if foreign: raise RuntimeError(f'foreign workload before {name}: {foreign}')
        build = args.baseline_build if job['policy'] == 'baseline' else args.candidate_build
        script = job['kind'] if job['kind'] in ('groups', 'models') else 'e5'
        cache = args.output / (name + '-cache')
        if cache.exists(): raise FileExistsError('unfinished cache: ' + str(cache))
        variables = env | dict(PYTHONPATH=str(build / 'python'), LD_LIBRARY_PATH=str(build / 'libreloc'),
            SYM_RELOC_EXPORT=str(build / 'sym/tools/sym-reloc-export'), SYM_OPT=str(build / 'sym/tools/sym-opt'),
            TORCHINDUCTOR_CACHE_DIR=str(cache), EXPERIMENT_LAUNCH_NS=str(time.time_ns()))
        command = ['taskset', '-c', args.affinity, args.python, str(REPO / f'bench/issue264/{script}.py'),
            '--artifact', str(args.artifact), '--harness', str(args.harness), '--build', str(build),
            '--output', str(output), '--round', str(job['round']), '--policy', job['policy']]
        if script == 'e5': command += ['--kind', job['kind']]
        for key in ('samples', 'submit_samples', 'warmup', 'case', 'runtime', 'compute', 'callers', 'requests', 'batches'):
            if key in job: command += ['--' + key.replace('_', '-'), str(job[key])]
        record = dict(job=job, command=command, environment=variables, started_ns=time.time_ns(), observations=[])
        save(audit_path, record)
        with (args.output / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=REPO, env=variables, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                foreign = audit({process.pid})
                if foreign: record['observations'].append(dict(time_ns=time.time_ns(), foreign=foreign))
                time.sleep(3)
        record.update(returncode=process.returncode, finished_ns=time.time_ns())
        save(audit_path, record)
        if process.returncode or record['observations']: raise RuntimeError(f'failed/contaminated job {name}; see {audit_path}')
        result = json.loads(output.read_text())
        result.update(job=job, isolation=dict(valid=True, audit=audit_path.name, interval_seconds=3))
        validate_record(contract, result, job, record)
        validate_workload(result, job)
        save(output, result)
        print(name, 'complete', round((record['finished_ns'] - record['started_ns']) / 1e9, 1), 's', flush=True)


if __name__ == '__main__': main()
