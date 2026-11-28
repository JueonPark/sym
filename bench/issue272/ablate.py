"""Matched automatic-path ablations on the immutable E3 request trace."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from bench.issue227.run import audit
from bench.issue265.analyze import coverage, paired, validate_campaign, validate_record
from bench.issue265.run import freeze
from bench.issue265.support import (E3_CASES, E3_SEQUENCE, POLICIES, ROOT,
    artifact_hashes, digest, harness_hashes, runtime_hashes, save)

EPOCHS = 5
ARMS = ('cache_off', 'groups_off', 'all_on', 'specialization_off')
CONTRASTS = (
    ('shape_cache', 'cache_off', 'groups_off'),
    ('prepared_group', 'groups_off', 'all_on'),
    ('kernel_specialization', 'specialization_off', 'all_on'),
)
TIMINGS = ('cold_first_ms', 'first_cycle_ms', 'full_ms', 'repeated_ms',
           'repeated_p50_ms', 'accounted_full_ms')


def matrix(rounds):
    if type(rounds) is not int or rounds < 1:
        raise ValueError('rounds must be positive')
    policies = {
        arm: POLICIES['always_cpu'] | {
            'cache_specializations': arm != 'cache_off',
            'reuse_groups': arm not in ('cache_off', 'groups_off'),
        } for arm in ARMS
    }
    jobs = []
    for round_number in range(1, rounds + 1):
        offset = (round_number - 1) % len(ARMS)
        for case in E3_CASES:
            for arm in ARMS[offset:] + ARMS[:offset]:
                jobs.append(dict(name=f'e3-r{round_number}-{case}-{arm}', case=case, arm=arm,
                    round=round_number, policy=policies[arm].copy(),
                    runtime='generic' if arm == 'specialization_off' else 'specialized'))
    return jobs


def prepared_stats(stats):
    return (stats.get('backend') or {}).get('prepared_execution') or {}


def deltas(current, previous):
    """Keep cumulative counter deltas; cache size and slot counts are gauges."""
    gauges = {'policy', 'cache_entries', 'retained_parameter_bytes', 'live_slots', 'cached_handles'}
    result = {}
    for key, value in current.items():
        if key in gauges:
            continue
        if type(value) is int:
            result[key] = value - previous.get(key, 0)
        elif isinstance(value, dict):
            result[key] = deltas(value, previous.get(key, {}))
    return result


def summarize_process(row):
    job = row['job']
    requests = row['requests']
    times = [step['completed_ms'] for step in requests]
    phases = row['phases_ms']
    for name in ('import_and_configure', 'cuda_init', 'owner_setup', 'owner_retirement'):
        if not math.isfinite(phases[name]) or phases[name] < 0:
            raise ValueError(f'{job["name"]}: invalid {name} time')
    setup = sum(phases[name] for name in ('import_and_configure', 'cuda_init', 'owner_setup'))
    details = []
    previous = {}
    for step in requests:
        prepared = prepared_stats(step['stats'])
        group = prepared.get('native', {}).get('prepared_groups', {})
        if not job['policy']['cache_specializations'] and any(
                prepared.get(name, 0) for name in ('cache_hits', 'cache_entries', 'retained_parameter_bytes')):
            raise ValueError(f'{job["name"]}: retained specialization in cache-off arm')
        if not job['policy']['reuse_groups'] and any(
                group.get(name, 0) for name in ('handles_created', 'cached_handles', 'live_slots', 'rebinds')):
            raise ValueError(f'{job["name"]}: retained prepared group in group-off arm')
        details.append({key: step[key] for key in ('step', 'n', 'shapes', 'strides', 'completed_ms')} |
            {'prepared_delta': deltas(prepared, previous)})
        previous = prepared
    if row['final_stats'] != requests[-1]['stats']:
        raise ValueError(f'{job["name"]}: final stats differ from last request')
    cycle = len(E3_SEQUENCE)
    repeated = times[cycle:]
    stats = row['final_stats']
    return dict(job=job, requests=details, cold_first_ms=setup + times[0], first_ms=times[0],
        first_cycle_ms=sum(times[:cycle]), full_ms=sum(times), repeated_ms=sum(repeated),
        repeated_p50_ms=statistics.median(repeated),
        accounted_full_ms=setup + sum(times) + phases['owner_retirement'],
        phases_ms=phases, memory=row['memory'], final_stats=stats,
        coverage=coverage(stats.get('backend') or {}, len(times)),
        warm_dynamo_callbacks=stats['dynamo_callbacks'] - requests[cycle - 1]['stats']['dynamo_callbacks'])


def validate_pair(before, after):
    for left, right in zip(before['requests'], after['requests']):
        if (left['shapes'], left['strides']) != (right['shapes'], right['strides']):
            raise ValueError(f'{after["job"]["name"]}: operand metadata differs at request {right["step"]}')
        left_stats, right_stats = left['stats'], right['stats']
        left_native = prepared_stats(left_stats).get('native', {})
        right_native = prepared_stats(right_stats).get('native', {})
        if (any(left_native.get(key) != right_native.get(key)
                for key in ('dispatches', 'logical_transfers', 'payload_bytes')) or
                left_stats['dynamo_callbacks'] != right_stats['dynamo_callbacks'] or
                any((left_stats['backend'] or {}).get(key) != (right_stats['backend'] or {}).get(key)
                    for key in ('runtime_executions', 'fallbacks', 'exclusions'))):
            raise ValueError(f'{after["job"]["name"]}: execution path differs at request {right["step"]}')


def score(contract, records):
    rounds = contract['rounds']
    if (contract['jobs'] != matrix(rounds) or contract['sequence'] != list(E3_SEQUENCE)
            or contract['epochs'] != EPOCHS):
        raise ValueError('contract differs from the registered matrix or original E3 trace')
    validate_campaign(contract, records)
    raw = {(r['job']['case'], r['job']['arm'], r['job']['round']): r for r in records}
    summaries = {key: summarize_process(row) for key, row in raw.items()}
    comparisons = []
    for case in E3_CASES:
        for factor, before, after in CONTRASTS:
            pairs = [((case, before, r), (case, after, r)) for r in range(1, rounds + 1)]
            for left, right in pairs:
                validate_pair(raw[left], raw[right])
            metrics = {}
            for metric in TIMINGS:
                a = [summaries[left][metric] for left, _ in pairs]
                b = [summaries[right][metric] for _, right in pairs]
                if rounds > 1:
                    metrics[metric] = paired(a, b)
                else:
                    reduction = 100 * (a[0] - b[0]) / a[0]
                    metrics[metric] = dict(process_pairs=1, reduction_percent=reduction,
                        ci95_percent=None, paired_reductions_percent=[reduction],
                        baseline_ms=a, candidate_ms=b)
            comparisons.append(dict(case=case, factor=factor, control=before, enabled=after, metrics=metrics))
    return dict(complete=True, rounds=rounds, jobs=len(records),
        exact_outputs=sum(len(row['requests']) for row in records),
        summaries=list(summaries.values()), comparisons=comparisons,
        interpretation=('Repeated calls include any remaining compile/preparation. '
            'Contrasts hold the other factors fixed as specified by the paired arms. '
            'The kernel-build contrast alone does not prove that a specialized kernel fired.'))


def verify_builds(builds):
    hashes = {name: runtime_hashes(path) for name, path in builds.items()}
    specialized, generic = hashes['specialized'], hashes['generic']
    if not specialized or specialized.keys() != generic.keys():
        raise ValueError('both builds must contain the same runtime files')
    changes = [name for name in specialized if specialized[name] != generic[name]]
    if changes != ['libreloc/libreloc_runtime.so']:
        raise ValueError('only libreloc_runtime.so may differ between kernel controls')
    if not any(name.endswith('/automatic.py') for name in specialized):
        raise ValueError('builds must contain the automatic frontend')
    return hashes


def load_record(directory, job):
    name = job['name']
    row = json.loads((directory / f'{name}.json').read_text())
    row['audit'] = json.loads((directory / f'{name}.audit.json').read_text())
    return row


def launch(directory, contract, job, build, artifact, command_prefix, environment):
    name = job['name']
    output = directory / f'{name}.json'
    audit_path = directory / f'{name}.audit.json'
    if output.exists():
        validate_record(contract, load_record(directory, job), job)
        print(name, 'already complete', flush=True)
        return
    cache = directory / f'{name}-cache'
    if cache.exists() or audit_path.exists():
        raise FileExistsError(f'unfinished process for {name}')
    foreign = audit(set())
    if foreign:
        raise RuntimeError(f'foreign work before {name}: {foreign}')
    variables = environment | dict(PYTHONPATH=str(build / 'python'),
        LD_LIBRARY_PATH=str(build / 'libreloc'),
        SYM_RELOC_EXPORT=str(build / 'sym/tools/sym-reloc-export'),
        SYM_OPT=str(build / 'sym/tools/sym-opt'), TORCHINDUCTOR_CACHE_DIR=str(cache))
    command = [*command_prefix, str(ROOT / 'bench/issue265/worker.py'),
        '--artifact', str(artifact), '--build', str(build),
        '--contract', str(directory / 'contract.json'), '--output', str(output), '--job', name]
    record = dict(job=job, command=command, started_ns=time.time_ns(), observations=[])
    save(audit_path, record)
    with (directory / f'{name}.log').open('w') as log:
        variables['EXPERIMENT_LAUNCH_NS'] = str(time.time_ns())
        with subprocess.Popen(command, cwd=ROOT, env=variables, stdout=log, stderr=subprocess.STDOUT) as process:
            while process.poll() is None:
                foreign = audit({process.pid})
                if foreign:
                    record['observations'].append(dict(time_ns=time.time_ns(), foreign=foreign))
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
    record.update(returncode=process.returncode, finished_ns=time.time_ns())
    save(audit_path, record)
    if process.returncode or record['observations']:
        raise RuntimeError(f'failed or contaminated process {name}. See {audit_path}')
    validate_record(contract, load_record(directory, job), job)


def run(args):
    jobs = matrix(args.rounds)
    directory = args.output.resolve()
    builds = {'specialized': args.specialized_build.resolve(), 'generic': args.generic_build.resolve()}
    hashes = verify_builds(builds)
    artifact = args.artifact.resolve()
    keep = ('HOME', 'USER', 'LOGNAME', 'PATH', 'SHELL', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR')
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    environment.update(CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', TORCHINDUCTOR_COMPILE_THREADS='1')
    command_prefix = ['taskset', '-c', args.affinity, os.path.abspath(args.python)]
    probe = (
        'import json, torch; from bench.issue265.support import environment; '
        'torch.set_num_threads(8); torch.set_num_interop_threads(1); print(json.dumps(environment()))'
    )
    observed = json.loads(subprocess.check_output([*command_prefix, '-c', probe],
        cwd=ROOT, env=environment, text=True))
    contract = dict(schema=1, jobs=jobs, rounds=args.rounds, epochs=EPOCHS, sequence=list(E3_SEQUENCE),
        environment=observed, harness_sha256=harness_hashes(), artifact_sha256=artifact_hashes(artifact),
        runtime_sha256=hashes, ablation_sha256=digest(__file__),
        configuration=dict(affinity=args.affinity, device=args.device, python=command_prefix[-1],
            artifact=str(artifact), builds={key: str(path) for key, path in builds.items()},
            transfer_options=dict(gather_threads=8, n_buffers=2, n_streams=1, pinning='pinned')))
    freeze(directory / 'contract.json', contract)
    for job in jobs:
        launch(directory, contract, job, builds[job['runtime']], artifact, command_prefix, environment)
    score_directory(directory, directory / 'analysis.json')


def score_directory(directory, output):
    contract = json.loads((directory / 'contract.json').read_text())
    result = score(contract, [load_record(directory, job) for job in contract['jobs']])
    save(output, result)
    print(f'Validated {result["jobs"]} independent processes and {result["exact_outputs"]} exact outputs.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    runner = subcommands.add_parser('run', help='run and score the serialized GPU campaign')
    for key in ('artifact', 'specialized-build', 'generic-build', 'output'):
        runner.add_argument('--' + key, type=Path, required=True)
    runner.add_argument('--python', required=True)
    runner.add_argument('--rounds', type=int, default=3)
    runner.add_argument('--affinity', default='0-7')
    runner.add_argument('--device', default='0')
    scorer = subcommands.add_parser('score', help='validate and score saved records without CUDA')
    scorer.add_argument('directory', type=Path)
    scorer.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        score_directory(args.directory, args.output or args.directory / 'analysis.json')


if __name__ == '__main__':
    main()
