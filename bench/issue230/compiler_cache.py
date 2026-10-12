#!/usr/bin/env python3
"""Fresh-process compiler-only preparation; one client amortizes binary hashing."""
import time
START = time.perf_counter()
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--count', type=int)
parser.add_argument('--cache', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.count is None:
    import os
    import subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bench.issue227.run import audit
    args.output.mkdir(parents=True, exist_ok=True)
    for round_index in range(1, 4):
        for count in (1, 8, 32):
            cache = args.output / f'r{round_index}-n{count}-cache'
            for mode in ('disabled', 'seed', 'hit'):
                output = args.output / f'r{round_index}-n{count}-{mode}.json'
                assert not output.exists() and not audit(set())
                command = ['taskset', '-c', '4-7,20-23', sys.executable, __file__,
                           '--count', str(count), '--output', str(output)]
                if mode != 'disabled':
                    command += ['--cache', str(cache)]
                begin = time.perf_counter()
                process = subprocess.Popen(command)
                observed = []
                while process.poll() is None:
                    observed.extend(audit({process.pid}))
                    time.sleep(.2)
                assert process.returncode == 0 and not observed
                result = json.loads(output.read_text())
                result['launch_to_script_ms'] = (result.pop('script_start_perf') - begin) * 1000
                result['foreign_processes'] = observed
                output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
                print(output.stem, result['prepare_ms'], flush=True)
    raise SystemExit(0)
from reloc_torch import CompilerClient
from reloc_torch.recipe import Recipe, TensorSpec, Cast, Transpose
from reloc_torch.symbolic import Const, dense_strides

recipes = []
for index in range(args.count):
    shape = (Const(index+1), Const(128))
    dest = tuple(reversed(shape))
    spec = lambda s, d: TensorSpec(s, dense_strides(s), Const(0), d)
    recipes.append(Recipe(spec(shape, 'float32'), (Transpose((1, 0)), Cast('float16', 'ieee_rne')),
                          spec(dest, 'float16'), 'h2d'))
start = time.perf_counter()
compiler = CompilerClient.from_environment(cache_dir=args.cache)
artifacts = [compiler.compile(recipe) for recipe in recipes]
prepare_ms = (time.perf_counter()-start)*1000
assert 'torch' not in sys.modules
args.output.write_text(json.dumps(dict(count=args.count, script_start_perf=START, prepare_ms=prepare_ms,
    from_script_ms=(time.perf_counter()-START)*1000, cache=compiler.cache_stats(),
    checks=all(a.recipe == r for a,r in zip(artifacts,recipes)))))
