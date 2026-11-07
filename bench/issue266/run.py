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
from bench.issue266.isolation import audit
from bench.issue266.support import ROOT, digest, matrix, provenance, save


def exclude_contaminated(directory, job):
    """Preserve one recorded, externally contaminated attempt; never omit OOMs."""
    directory = Path(directory)
    prefix = job['id']
    execution = json.loads((directory / (prefix + '.run.json')).read_text())
    record = json.loads((directory / (prefix + '.record.json')).read_text())
    busy = execution['foreign_busy_processes']
    if (execution['exit_code'] != 0 or not busy or record['job'] != job or
            not record.get('complete') or record['isolation']['foreign_busy_processes'] != busy):
        raise ValueError('only a completed, explicitly contaminated attempt may be excluded')
    parent = directory / 'excluded' / prefix
    attempt = parent / ('attempt-' + str(len(list(parent.glob('attempt-*'))) + 1))
    attempt.mkdir(parents=True, exist_ok=False)
    files = sorted(directory.glob(prefix + '.*'))
    inventory = {p.name: digest(p) for p in files}
    save(attempt / 'exclusion.json', dict(job=job, reason='foreign workload during measurement',
        foreign_busy_processes=busy, files=inventory))
    for path in files: path.replace(attempt / path.name)
    return attempt


def wait_idle(directory, allowed):
    quiet = None
    print('Waiting for 60 consecutive seconds without foreign CPU/build/GPU work', flush=True)
    while True:
        busy = audit(allowed)
        now = time.monotonic()
        quiet = None if busy else (now if quiet is None else quiet)
        elapsed = 0 if quiet is None else now - quiet
        with (directory / 'isolation-pauses.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix_time=time.time(), foreign_busy_processes=busy,
                                         quiet_seconds=elapsed)) + '\n')
        if elapsed >= 60: return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'build', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--pilot-regex', help='explicitly excluded pilot subset; cannot produce acceptance analysis')
    parser.add_argument('--wait-for-idle', action='store_true',
        help='wait for idle host and preserve/retry only explicitly contaminated completed attempts')
    args = parser.parse_args()
    args.artifact, args.build, args.output = args.artifact.resolve(), args.build.resolve(), args.output.resolve()
    jobs = matrix()
    if args.pilot_regex: jobs = [j for j in jobs if re.search(args.pilot_regex, j['id'])]
    assert jobs
    frozen = provenance(args.artifact, args.build)
    manifest = dict(acceptance=not bool(args.pilot_regex), jobs=jobs, provenance=frozen,
        artifact=str(args.artifact), build=str(args.build), python=args.python,
        interference_retry='preserve-and-retry' if args.wait_for_idle else 'stop',
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
    if args.wait_for_idle: wait_idle(args.output, {os.getpid()})
    for index, job in enumerate(jobs):
        record_path = args.output / (job['id'] + '.record.json')
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            assert previous['job'] == job and previous['provenance'] == frozen and previous['complete']
            if previous['isolation']['foreign_busy_processes'] and args.wait_for_idle:
                exclude_contaminated(args.output, job)
            else:
                assert not previous['isolation']['foreign_busy_processes']
                continue
        while True:
            if audit({os.getpid()}):
                assert args.wait_for_idle, 'foreign workload before measurement'
                wait_idle(args.output, {os.getpid()})
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
            if observed and args.wait_for_idle:
                print('Preserved contaminated attempt:', exclude_contaminated(args.output, job), flush=True)
                wait_idle(args.output, {os.getpid()})
                continue
            assert not observed, 'contaminated measurement'
            break
    assert provenance(args.artifact, args.build) == frozen, 'sources/runtime/inputs changed during campaign'
    save(args.output / 'completion.json', dict(jobs=len(jobs), provenance=frozen, complete=True))


if __name__ == '__main__': main()
