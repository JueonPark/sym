#!/usr/bin/env python3
"""Three independent, serialized baseline/candidate rounds; no shared GPU load."""
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


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for round_number in range(1, 4):
        builds = [('baseline', args.baseline), ('candidate', args.candidate)]
        if round_number % 2 == 0:
            builds.reverse()
        for label, build in builds:
            for threads in (8, 1, 4):
                name = f'{label}-r{round_number}-t{threads}'
                output = args.output / (name + '.json')
                assert not output.exists(), f'{output} already exists'
                assert not audit(set()), 'foreign build or GPU workload before run'
                env = dict(os.environ, PYTHONPATH=f'{build}/python:{args.candidate}/python',
                    LD_LIBRARY_PATH=str(build / 'libreloc'),
                    SYM_RELOC_EXPORT=str(build / 'sym/tools/sym-reloc-export'),
                    SYM_OPT=str(build / 'sym/tools/sym-opt'),
                    SYM_RUNTIME_REVISION=args.baseline_revision if label == 'baseline' else revision,
                    OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                    OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1',
                    TORCHINDUCTOR_CACHE_DIR=str(args.output / f'compile-cache-r{round_number}'))
                command = ['taskset', '-c', args.affinity, args.python,
                    str(ROOT / 'bench/issue226/kernels.py'), '--output', str(output),
                    '--threads', str(threads), '--round', str(round_number)]
                if threads != 8:
                    command += ['--sweep', '--cases', 'cast_small', 'cast_medium', 'cast_large', 'chain_vector']
                observed = []
                start = time.monotonic()
                with (args.output / (name + '.log')).open('w') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        observed += audit({child.pid})
                        time.sleep(3)
                print(name, 'exit', child.returncode, 'seconds', round(time.monotonic()-start),
                      'foreign', observed, flush=True)
                if child.returncode:
                    raise SystemExit(child.returncode)
                result = json.loads(output.read_text())
                result.update(label=label, source_revision=revision, isolation=dict(
                    foreign_busy_processes=sorted(set(observed)), interval_seconds=3,
                    monitor='process-tree-aware build/packaging and GPU processes'))
                output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
                if observed:
                    raise SystemExit('contaminated run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default='/tmp/sym-210-venv/bin/python')
    parser.add_argument('--affinity', default='4-7,20-23')
    main(parser.parse_args())
