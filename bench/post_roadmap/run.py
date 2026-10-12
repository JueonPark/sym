#!/usr/bin/env python3
"""Run offline calibration and all workloads serially in three fresh processes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.issue227.run import audit

PATHS = ['torch', 'inductor', 'sym', 'sym_inductor', 'sym_placement_inductor']


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cmake = (args.build / 'CMakeCache.txt').read_text()
    assert 'RELOC_ENABLE_NVTX:BOOL=OFF' in cmake
    assert 'CMAKE_BUILD_TYPE:STRING=Release' in cmake
    assert 'CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG' in cmake
    flags = [line for line in cmake.splitlines() if line.startswith((
        'CMAKE_BUILD_TYPE:', 'CMAKE_CUDA_COMPILER:', 'CMAKE_CUDA_HOST_COMPILER:',
        'CMAKE_CUDA_ARCHITECTURES:', 'CMAKE_CUDA_FLAGS_RELEASE:',
        'RELOC_ENABLE_CUDA:', 'RELOC_ENABLE_NVTX:'))]
    env = dict(os.environ, PYTHONPATH=str(args.build / 'python'),
        LD_LIBRARY_PATH=str(args.build / 'libreloc'),
        SYM_RELOC_EXPORT=str(args.build / 'sym/tools/sym-reloc-export'),
        SYM_OPT=str(args.build / 'sym/tools/sym-opt'), SYM_RUNTIME_REVISION=revision,
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE',
        GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1')

    def run(name, script, extra, overrides=None):
        output = args.output / (name + '.json')
        cache = args.output / (name + '-cache')
        assert not output.exists() and not cache.exists(), name
        assert not audit(set()), 'foreign build or GPU work'
        variables = env | dict(TORCHINDUCTOR_CACHE_DIR=str(cache)) | (overrides or {})
        command = ['taskset', '-c', args.affinity, args.python, str(ROOT / script),
                   '--output', str(output), *extra]
        observed = []
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        with (args.output / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=variables,
                                     stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                observed += audit({child.pid})
                time.sleep(3)
        elapsed = time.monotonic() - start
        print(name, 'exit', child.returncode, 'seconds', round(elapsed),
              'foreign', observed, flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)
        result = json.loads(output.read_text())
        result.update(source_revision=revision, build_flags=flags, command=command,
            started_utc=started, process_wall_seconds=elapsed,
            isolation=dict(foreign_busy_processes=sorted(set(observed)),
                           interval_seconds=3))
        output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
        if observed:
            raise SystemExit('contaminated measurement')

    for r in range(1, args.rounds + 1):
        run(f'calibration-r{r}', 'bench/issue227/placement.py',
            ['--round', str(r), '--samples', str(args.samples)])
        order = PATHS[r % len(PATHS):] + PATHS[:r % len(PATHS)]
        run(f'workloads-r{r}', 'bench/post_roadmap/compare.py',
            ['--samples', str(args.samples), '--warmup', str(args.warmup), '--paths', *order],
            dict(SYM_PLACEMENT_PROFILE=str(args.output / f'calibration-r{r}.profile.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default='/tmp/sym-210-venv/bin/python')
    p.add_argument('--affinity', default='4-7,20-23')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    main(p.parse_args())
