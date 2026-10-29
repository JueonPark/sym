#!/usr/bin/env python3
"""Run bench/beat_torch/compare.py in fresh processes with rotated path order.

Mirrors bench/post_roadmap/run.py's environment (affinity, thread and
Inductor settings, foreign-work audit) without the placement calibration.
"""
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

PATHS = ['torch', 'inductor', 'sym_inductor', 'sym_fast', 'sym_fast_inductor', 'torch_whole']


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cmake = (args.build / 'CMakeCache.txt').read_text()
    assert 'RELOC_ENABLE_NVTX:BOOL=OFF' in cmake and 'CMAKE_BUILD_TYPE:STRING=Release' in cmake
    env = dict(os.environ, PYTHONPATH=str(args.build / 'python'), LD_LIBRARY_PATH=str(args.build / 'libreloc'),
               SYM_RELOC_EXPORT=str(args.build / 'sym/tools/sym-reloc-export'),
               SYM_OPT=str(args.build / 'sym/tools/sym-opt'), SYM_RUNTIME_REVISION=revision,
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0',
               TORCHINDUCTOR_COMPILE_THREADS='1')
    paths = args.paths or PATHS
    for r in range(1, args.rounds + 1):
        name = f'workloads-r{r}'
        output, cache = args.output / f'{name}.json', args.output / f'{name}-cache'
        assert not output.exists() and not cache.exists(), name
        assert not audit(set()), 'foreign build or GPU work'
        order = paths[r % len(paths):] + paths[:r % len(paths)]
        command = ['taskset', '-c', args.affinity, args.python, str(ROOT / 'bench/beat_torch/compare.py'),
                   '--output', str(output), '--samples', str(args.samples), '--warmup', str(args.warmup),
                   '--paths', *order]
        if args.cases:
            command += ['--cases', *args.cases]
        started = time.time()
        observed = []
        with open(args.output / f'{name}.log', 'w') as log:
            process = subprocess.Popen(command, env=env | dict(TORCHINDUCTOR_CACHE_DIR=str(cache)),
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                observed += audit({process.pid})
                time.sleep(3)
        if process.returncode:
            raise SystemExit(f'{name} failed with {process.returncode}')
        result = json.loads(output.read_text())
        result.update(source_revision=revision, command=command, process_wall_seconds=time.time() - started,
                      isolation=dict(foreign_busy_processes=sorted(set(observed)), interval_seconds=3))
        output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
        print(name, 'done', f'{time.time() - started:.0f}s', flush=True)
        if observed:
            raise SystemExit('contaminated measurement')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default='/tmp/sym-210-venv/bin/python')
    p.add_argument('--affinity', default='4-7,20-23')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--paths', nargs='*')
    p.add_argument('--cases', nargs='*')
    main(p.parse_args())
