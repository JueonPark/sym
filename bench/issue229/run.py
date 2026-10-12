#!/usr/bin/env python3
"""Serialize three independent matched baseline/candidate rounds."""
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
    for build in (args.baseline, args.candidate):
        assert 'RELOC_ENABLE_NVTX:BOOL=OFF' in (build/'CMakeCache.txt').read_text()
    for r in range(1, 4):
        builds = [('baseline', args.baseline), ('candidate', args.candidate)]
        if r % 2 == 0:
            builds.reverse()
        for label, build in builds:
            for suite in args.suites:
                name = f'{suite}-{label}-r{r}'
                output = args.output/(name+'.json')
                cache = args.output/(name+'-cache')
                assert not output.exists() and not cache.exists(), name
                assert not audit(set()), 'foreign build or GPU work'
                env = dict(os.environ, PYTHONPATH=str(build/'python'), LD_LIBRARY_PATH=str(build/'libreloc'),
                    SYM_RELOC_EXPORT=str(build/'sym/tools/sym-reloc-export'), SYM_OPT=str(build/'sym/tools/sym-opt'),
                    SYM_RUNTIME_REVISION=args.baseline_revision if label == 'baseline' else revision,
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0',
                    TORCHINDUCTOR_COMPILE_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(cache))
                if suite == 'rawmodels':
                    script = ROOT/'bench/issue223/model_prefetch.py'
                    paths = ['sym_prefetch', 'torch_prefetch', 'inductor_prefetch']
                    paths = paths[r%3:] + paths[:r%3]
                    extra = ['--cases', 'llm_decode', 'moe_sparse', '--paths', *paths]
                else:
                    script = ROOT/f'bench/issue229/{suite}.py'
                    extra = ['--round', str(r)]
                    if suite == 'transfers' and label == 'candidate':
                        extra += ['--kernels']
                command = ['taskset', '-c', args.affinity, args.python, str(script), '--output', str(output), *extra]
                observed = []
                start = time.monotonic()
                with (args.output/(name+'.log')).open('w') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        observed += audit({child.pid})
                        time.sleep(3)
                print(name, 'exit', child.returncode, 'seconds', round(time.monotonic()-start), 'foreign', observed, flush=True)
                if child.returncode:
                    raise SystemExit(child.returncode)
                result = json.loads(output.read_text())
                result.update(label=label, source_revision=revision, round=r, suite=suite, command=command,
                    isolation=dict(foreign_busy_processes=sorted(set(observed)), interval_seconds=3))
                output.write_text(json.dumps(result, separators=(',', ':'))+'\n')
                if observed:
                    raise SystemExit('contaminated measurement')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline-revision', default='8342acc3e407f67477fa7181bad4d2719727a5d0')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suites', nargs='+', choices=['transfers', 'models', 'rawmodels'], default=['transfers', 'models', 'rawmodels'])
    p.add_argument('--python', default='/tmp/sym-210-venv/bin/python')
    p.add_argument('--affinity', default='4-7,20-23')
    main(p.parse_args())
