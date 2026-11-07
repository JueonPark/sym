"""Serialize complete matrices with frozen hashes and interference audits."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue227.run import audit
from bench.issue266.support import ROOT, matrix, provenance, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'build', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--pilot-regex', help='explicitly excluded pilot subset; cannot produce acceptance analysis')
    args = parser.parse_args()
    args.artifact, args.build, args.output = args.artifact.resolve(), args.build.resolve(), args.output.resolve()
    jobs = matrix()
    if args.pilot_regex: jobs = [j for j in jobs if re.search(args.pilot_regex, j['id'])]
    assert jobs
    frozen = provenance(args.artifact, args.build)
    manifest = dict(acceptance=not bool(args.pilot_regex), jobs=jobs, provenance=frozen,
        artifact=str(args.artifact), build=str(args.build), python=args.python,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'stale resume manifest'
    else:
        save(path, manifest)
    env = os.environ | dict(PYTHONPATH=str(args.build / 'python') + ':' + str(ROOT),
        LD_LIBRARY_PATH=str(args.build / 'libreloc'), CUDA_VISIBLE_DEVICES='0',
        SYM_OPT=str(args.build / 'sym/tools/sym-opt'), SYM_RELOC_EXPORT=str(args.build / 'sym/tools/sym-reloc-export'),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    for index, job in enumerate(jobs):
        record_path = args.output / (job['id'] + '.record.json')
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            assert previous['job'] == job and previous['provenance'] == frozen and previous['complete']
            assert not previous['isolation']['foreign_busy_processes']
            continue
        assert not audit({os.getpid()}), 'foreign workload before measurement'
        start = time.monotonic()
        with (args.output / (job['id'] + '.log')).open('w') as log:
            process = subprocess.Popen(['taskset', '-c', '4-7,20-23', args.python,
                str(ROOT / 'bench/issue266/worker.py'), '--manifest', str(path), '--job', job['id']],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            observed = []
            while process.poll() is None:
                observed += audit({os.getpid(), process.pid})
                time.sleep(2)
        elapsed = time.monotonic() - start
        save(args.output / (job['id'] + '.run.json'), dict(exit_code=process.returncode,
            elapsed_seconds=elapsed, foreign_busy_processes=observed))
        print(index + 1, '/', len(jobs), job['id'], 'exit', process.returncode, round(elapsed, 1), 'seconds', flush=True)
        if process.returncode: raise SystemExit(process.returncode)
        record = json.loads(record_path.read_text())
        record['isolation'] = dict(foreign_busy_processes=observed, interval_seconds=2)
        save(record_path, record)
        assert not observed, 'contaminated measurement'
    assert provenance(args.artifact, args.build) == frozen, 'sources/runtime/inputs changed during campaign'
    save(args.output / 'completion.json', dict(jobs=len(jobs), provenance=frozen, complete=True))


if __name__ == '__main__': main()
