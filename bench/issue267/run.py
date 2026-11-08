"""Serialize frozen qualification jobs and retain excluded contaminated attempts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from bench.issue266.isolation import audit
import contract
from evidence import provenance, sha


def wait_for_idle(wait):
    quiet_since = None
    while True:
        foreign = audit({os.getpid()})
        if foreign:
            quiet_since = None
            if not wait:
                raise RuntimeError(f'foreign activity: {foreign}')
        elif quiet_since is None:
            quiet_since = time.monotonic()
        if quiet_since is not None and (not wait or time.monotonic() - quiet_since >= 60):
            return
        print('waiting for quiet host', foreign, flush=True)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-idle', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--jobs', nargs='+')
    args = parser.parse_args()
    if args.jobs and not args.pilot:
        parser.error('a subset is permitted only for an excluded pilot')
    roots = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    selected = [j for j in contract.jobs() if not args.jobs or j['id'] in args.jobs]
    if args.jobs and len(selected) != len(set(args.jobs)):
        parser.error('unknown or duplicate job')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    environment = json.loads(subprocess.check_output([sys.executable, '-c',
        'import json,torch; from bench.issue267.evidence import environment; print(json.dumps(environment(torch)))'],
        cwd=REPO, text=True))
    current = dict(schema=1, qualification=not args.pilot, jobs=selected,
                   environment=environment,
                   provenance={name: provenance(REPO, root, args.compiler) for name, root in roots.items()})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert all(manifest[k] == v for k, v in current.items()), 'frozen campaign changed'
    else:
        manifest = current | {'records': {}, 'excluded': []}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for job in selected:
        if job['id'] in manifest['records']:
            saved = manifest['records'][job['id']]
            assert sha(args.output / saved['path']) == saved['sha256']
            assert sha(args.output / saved['audit_path']) == saved['audit_sha256']
            continue
        while True:
            wait_for_idle(args.wait_for_idle)
            runtime = roots[job['runtime']]
            target = args.output / job['id']
            target.mkdir()  # Refuse an unrecorded partial run; preserve it for inspection.
            output = target / 'record.json'
            cache = target / 'inductor-cache'
            env = os.environ | dict(
                PYTHONPATH=str(runtime / 'python') + ':' + str(runtime / 'helper'),
                LD_LIBRARY_PATH=str(runtime / 'libreloc'), SYM_RELOC_EXPORT=str(args.compiler),
                CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS=str(job['threads']),
                MKL_NUM_THREADS=str(job['threads']), OPENBLAS_NUM_THREADS=str(job['threads']),
                OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1',
                TORCHINDUCTOR_CACHE_DIR=str(cache), CXX='/usr/bin/g++', CC='/usr/bin/gcc')
            command = ['taskset', '-c', '4-7,20-23', sys.executable, str(ROOT / 'worker.py'),
                       '--job', job['id'], '--runtime-root', str(runtime), '--output', str(output)]
            start = time.monotonic()
            seen = []
            observations = 0
            with (target / 'worker.log').open('w') as log:
                child = subprocess.Popen(command, env=env, cwd=REPO, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while child.poll() is None:
                        seen.extend(audit({os.getpid(), child.pid}))
                        observations += 1
                        time.sleep(3)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
            checked = dict(job=job, command=command, exit_code=child.returncode,
                           observations=observations, audit_interval_seconds=3,
                           foreign_busy_processes=sorted(set(seen)),
                           wall_seconds=time.monotonic() - start)
            audit_path = target / 'audit.json'
            audit_path.write_text(json.dumps(checked, indent=2) + '\n')
            if child.returncode:
                raise RuntimeError(f'worker error; evidence preserved: {target}')
            if seen:
                excluded = args.output / 'excluded' / f"{job['id']}-{len(manifest['excluded']) + 1}"
                excluded.parent.mkdir(exist_ok=True)
                shutil.move(target, excluded)
                manifest['excluded'].append(dict(job=job, reason='foreign activity',
                    path=str(excluded.relative_to(args.output)),
                    files={str(p.relative_to(excluded)): sha(p) for p in excluded.rglob('*') if p.is_file()}))
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                if not args.wait_for_idle:
                    raise RuntimeError('contaminated attempt excluded; use --wait-for-idle to retry')
                continue
            manifest['records'][job['id']] = dict(path=str(output.relative_to(args.output)),
                sha256=sha(output), audit_path=str(audit_path.relative_to(args.output)),
                audit_sha256=sha(audit_path))
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            print(job['id'], 'complete', round(checked['wall_seconds'], 2), 'seconds', flush=True)
            break


if __name__ == '__main__':
    main()
