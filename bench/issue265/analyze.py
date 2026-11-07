"""Standard-library-only validation and process-paired analysis for #265."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def paired(baseline, candidate):
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError('need matched independent process rounds')
    if any(not math.isfinite(x) or x <= 0 for x in baseline + candidate):
        raise ValueError('timings must be positive finite values')
    effects = [100 * (a - b) / a for a, b in zip(baseline, candidate)]
    rng = random.Random(265)
    samples = [statistics.mean(rng.choices(effects, k=len(effects))) for _ in range(10000)]
    return dict(process_pairs=len(effects), reduction_percent=statistics.mean(effects),
        ci95_percent=[percentile(samples, .025), percentile(samples, .975)],
        paired_reductions_percent=effects, baseline_ms=baseline, candidate_ms=candidate,
        estimator='mean process-paired percentage reduction; process bootstrap', seed=265, resamples=10000)


def coverage(stats, requests):
    """Request, runtime, fallback and capture counts have different denominators."""
    return dict(requests=requests, runtime_executions=stats.get('runtime_executions', 0),
        fallback_events=sum(stats.get('fallbacks', {}).values()), fallbacks=stats.get('fallbacks', {}),
        capture_exclusion_events=sum(stats.get('exclusions', {}).values()), exclusions=stats.get('exclusions', {}),
        prepared_execution=stats.get('prepared_execution'),
        interpretation='capture exclusions count regions, not requests; no sum is an eligibility percentage')


def validate_campaign(contract, records):
    def require(condition, message):
        if not condition: raise ValueError(message)
    jobs = {j['name']: j for j in contract['jobs']}
    require(len(jobs) == len(contract['jobs']), 'duplicate contracted job')
    require(len(records) == len(jobs) and {r['job']['name'] for r in records} == set(jobs), 'missing or duplicate process')
    groups = defaultdict(set)
    for job in jobs.values(): groups[(job['case'], job['arm'])].add(job['round'])
    require(all(rounds == set(range(1, contract['rounds'] + 1)) for rounds in groups.values()), 'missing contracted round')
    for row in records:
        validate_record(contract, row, jobs[row['job']['name']])


def validate_record(contract, row, job):
    def require(condition, message):
        if not condition: raise ValueError(message)
    sequence = contract['sequence'] * contract['epochs']
    require(row['job'] == job and row.get('complete') is True, 'incomplete or changed job')
    require(row.get('policy') == job['policy'], 'policy changed')
    metadata = row['metadata']
    for key in ('harness_sha256', 'artifact_sha256', 'environment'):
        require(metadata.get(key) == contract[key], key + ' changed')
    require(metadata.get('runtime_sha256') == contract['runtime_sha256'][job['runtime']], 'runtime changed')
    audit = row['audit']
    require(audit.get('job') == job and audit.get('returncode') == 0 and audit.get('observations') == [], 'invalid audit')
    requests = row['requests']
    require(len(requests) == len(sequence), 'incomplete sequence')
    for i, (request, n) in enumerate(zip(requests, sequence)):
        require(request.get('step') == i and request.get('n') == n, 'sequence changed')
        require(request.get('correct') is True and request.get('held_correct') is True, 'oracle/held output failure')
        ms = request['completed_ms']
        require(isinstance(ms, (float, int)) and math.isfinite(ms) and ms > 0, 'invalid timing')
    require(row['held_checks_after_close'] == 1, 'held output not checked after close')
    resources = row['resources_after_close']
    if resources is not None:
        for kind in (resources, resources.get('typed', {})):
            require(kind.get('live_bytes', 0) == kind.get('retained_bytes', 0) == 0, 'native retirement leaked')


def summarize(contract, records):
    validate_campaign(contract, records)
    cycle = len(contract['sequence'])
    summaries = []
    for row in records:
        times = [r['completed_ms'] for r in row['requests']]
        phases = row['phases_ms']
        setup = sum(phases[name] for name in ('import_and_configure', 'cuda_init', 'owner_setup'))
        stats = row['final_stats']
        summaries.append(dict(job=row['job'], first_ms=times[0], first_cycle_ms=sum(times[:cycle]),
            full_ms=sum(times), repeated_ms=sum(times[cycle:]),
            cold_first_ms=setup + times[0], accounted_full_ms=setup + sum(times) + phases['owner_retirement'],
            phases_ms=phases, memory=row['memory'], final_stats=stats,
            coverage=coverage(stats.get('backend') or {}, len(times)),
            warm_dynamo_callbacks=stats['dynamo_callbacks'] - row['requests'][cycle - 1]['stats']['dynamo_callbacks']))
    by_case = defaultdict(dict)
    for row in summaries:
        job = row['job']
        by_case[job['case']].setdefault(job['arm'], {})[job['round']] = row
    comparisons = []
    for case, arms in sorted(by_case.items()):
        controls = [name for name in ('torch', 'inductor', 'sym_auto_eager', 'sym_auto_inductor', 'sym_fast_cache') if name in arms]
        candidates = [name for name in arms if name not in ('torch', 'inductor', 'sym_auto_eager', 'sym_auto_inductor')]
        for control in controls:
            for candidate in candidates:
                if control == candidate: continue
                metrics = {}
                for metric in ('cold_first_ms', 'first_cycle_ms', 'full_ms', 'repeated_ms', 'accounted_full_ms'):
                    a = [arms[control][r][metric] for r in range(1, contract['rounds'] + 1)]
                    b = [arms[candidate][r][metric] for r in range(1, contract['rounds'] + 1)]
                    metrics[metric] = paired(a, b)
                comparisons.append(dict(case=case, baseline=control, candidate=candidate, metrics=metrics))
    return dict(complete=True, jobs=len(records), rounds=contract['rounds'], summaries=summaries, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.directory / 'contract.json').read_text())
    records = []
    for job in contract['jobs']:
        row = json.loads((args.directory / (job['name'] + '.json')).read_text())
        row['audit'] = json.loads((args.directory / (job['name'] + '.audit.json')).read_text())
        records.append(row)
    result = summarize(contract, records)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Validated', len(records), 'independent processes;', len(result['comparisons']), 'paired comparisons')


if __name__ == '__main__': main()
