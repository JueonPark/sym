#!/usr/bin/env python3
"""Serialized fresh-process qualification with immutable contracts and audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.issue227.run import audit
from bench.issue265.analyze import validate_campaign, validate_record
from bench.issue265.support import (E3_ARMS, E3_CASES, E3_SEQUENCE, HELDOUT_CASES, HELDOUT_SEQUENCE,
    POLICIES, SELECTION_CASES, SELECTION_SEQUENCE, artifact_hashes, harness_hashes, runtime_hashes, save)


def matrix(suite, rounds, selected):
    if suite == 'selection':
        cases, arms, sequence = SELECTION_CASES, ('torch', 'sym_auto_eager', *POLICIES), SELECTION_SEQUENCE
    elif suite == 'e3':
        cases, arms, sequence = E3_CASES, E3_ARMS, E3_SEQUENCE
    else:
        cases, arms, sequence = HELDOUT_CASES, ('torch', 'sym_auto_eager', 'sym_prepared_eager', 'always_device'), HELDOUT_SEQUENCE
    jobs = []
    for round in range(1, rounds + 1):
        order = arms[round % len(arms):] + arms[:round % len(arms)]
        for case in cases:
            for arm in order:
                prepared = arm.startswith('sym_prepared') or arm in POLICIES
                policy = POLICIES[arm] if arm in POLICIES else POLICIES[selected] if prepared else None
                jobs.append(dict(name=f'{suite}-r{round}-{case}-{arm}', suite=suite, round=round, case=case, arm=arm,
                                 runtime='candidate' if prepared else 'baseline', policy=policy))
    return jobs, sequence


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('frozen contract changed: ' + str(path))
    else: save(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('artifact', 'baseline-build', 'candidate-build', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--suite', choices=('selection', 'e3', 'heldout'), required=True)
    parser.add_argument('--selected-policy', choices=POLICIES, default='admit_64k')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--affinity', default='4-7,20-23')
    parser.add_argument('--device', default='0')
    parser.add_argument('--only', help='optional job-name substring; complete matrix remains frozen')
    args = parser.parse_args()
    for key in ('artifact', 'baseline_build', 'candidate_build', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    args.python = os.path.abspath(args.python)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs, sequence = matrix(args.suite, args.rounds, args.selected_policy)
    keep = ('HOME', 'USER', 'LOGNAME', 'PATH', 'SHELL', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR')
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1')
    probe = json.loads(subprocess.check_output(['taskset', '-c', args.affinity, args.python, '-c',
        "import sys,json,torch; sys.path.insert(0,sys.argv[1]); torch.set_num_threads(8); torch.set_num_interop_threads(1); from bench.issue265.support import environment; print(json.dumps(environment()))", str(ROOT)], env=env, text=True))
    contract = dict(schema=1, suite=args.suite, jobs=jobs, sequence=list(sequence), epochs=args.epochs, rounds=args.rounds,
        environment=probe, harness_sha256=harness_hashes(), artifact_sha256=artifact_hashes(args.artifact),
        runtime_sha256={name: runtime_hashes(build) for name, build in
                       (('baseline', args.baseline_build), ('candidate', args.candidate_build))},
        configuration={key: str(value) for key, value in vars(args).items() if key != 'only'})
    freeze(args.output / 'contract.json', contract)
    for job in jobs:
        name = job['name']
        if args.only and args.only not in name: continue
        output, audit_path = args.output / (name + '.json'), args.output / (name + '.audit.json')
        if output.exists():
            row = json.loads(output.read_text())
            row['audit'] = json.loads(audit_path.read_text())
            validate_record(contract, row, job)
            print(name, 'already complete', flush=True)
            continue
        foreign = audit(set())
        if foreign: raise RuntimeError(f'foreign work before {name}: {foreign}')
        build = args.candidate_build if job['runtime'] == 'candidate' else args.baseline_build
        cache = args.output / (name + '-cache')
        if cache.exists(): raise FileExistsError('unfinished cache: ' + str(cache))
        variables = env | dict(PYTHONPATH=str(build / 'python'), LD_LIBRARY_PATH=str(build / 'libreloc'),
            SYM_RELOC_EXPORT=str(build / 'sym/tools/sym-reloc-export'), SYM_OPT=str(build / 'sym/tools/sym-opt'),
            TORCHINDUCTOR_CACHE_DIR=str(cache), EXPERIMENT_LAUNCH_NS=str(time.time_ns()))
        command = ['taskset', '-c', args.affinity, args.python, str(ROOT / 'bench/issue265/worker.py'),
            '--artifact', str(args.artifact), '--build', str(build), '--contract', str(args.output / 'contract.json'),
            '--output', str(output), '--job', name]
        record = dict(job=job, command=command, environment=variables, started_ns=time.time_ns(), observations=[])
        save(audit_path, record)
        with (args.output / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=variables, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                foreign = audit({process.pid})
                if foreign: record['observations'].append(dict(time_ns=time.time_ns(), foreign=foreign))
                time.sleep(2)
        record.update(returncode=process.returncode, finished_ns=time.time_ns())
        save(audit_path, record)
        print(name, 'exit', process.returncode, 'seconds', round((record['finished_ns'] - record['started_ns']) / 1e9, 1), flush=True)
        if process.returncode or record['observations']: raise RuntimeError('failed/contaminated job: ' + name)
    if args.only: return
    records = []
    for job in jobs:
        row = json.loads((args.output / (job['name'] + '.json')).read_text())
        row['audit'] = json.loads((args.output / (job['name'] + '.audit.json')).read_text())
        records.append(row)
    validate_campaign(contract, records)
    print('Complete validated matrix:', len(records), flush=True)


if __name__ == '__main__': main()
