#!/usr/bin/env python3
"""Three serial independent rounds; fresh Inductor cache and rotating path order."""
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

PATHS = ['sym_lazy', 'sym_prepared', 'sym_cached', 'torch_lazy', 'torch_prepared',
         'inductor_lazy', 'inductor_prepared']


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    def run(name, path, case, cache, initialized, prime=False):
        output = args.output / (name + '.json')
        assert not output.exists()
        assert not audit(set()), 'foreign workload before run'
        env = os.environ | dict(TORCHINDUCTOR_CACHE_DIR=str(args.output / (name+'-inductor')))
        command = ['taskset', '-c', '4-7,20-23', sys.executable, str(Path(__file__).with_name('startup.py')),
                   '--case', case, '--path', path, '--cache', str(cache), '--output', str(output)]
        if initialized:
            command.append('--initialized')
        if prime:
            command.append('--prime')
        with (args.output / (name + '.log')).open('w') as log:
            begin = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            observed = []
            while process.poll() is None:
                observed.extend(audit({process.pid}))
                time.sleep(1)
        if process.returncode:
            print((args.output / (name+'.log')).read_text())
            raise SystemExit(process.returncode)
        result = json.loads(output.read_text())
        result['isolation'] = dict(foreign_processes=observed, poll_seconds=1)
        result['process_wall_ms'] = (time.perf_counter() - begin) * 1000
        result['launch_to_script_ms'] = (result.pop('script_start_perf') - begin) * 1000
        output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
        print(name, 'first_ms', round(result.get('samples_ms', [0])[0], 3), flush=True)
        assert not observed, 'contaminated run'
    for round in range(1, 4):
        paths = PATHS[round:] + PATHS[:round]
        for case in ['small', 'dlrm', 'kv']:
            for initialized in [False, True]:
                prefix = f'r{round}-{case}-'+('initialized' if initialized else 'cold')
                cache = args.output / (prefix + '-artifact-cache')
                run(prefix+'-prime', 'sym_cached', case, cache, initialized, prime=True)
                for path in paths:
                    run(prefix+'-'+path, path, case, cache, initialized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
