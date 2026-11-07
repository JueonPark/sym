"""CPU-only validation and process-paired estimates for issue 264."""
import math
import random
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('expected nonempty finite samples')
    pos = (len(values) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def distribution(values):
    return dict(n=len(values), p50=percentile(values, 50), p95=percentile(values, 95),
                mean=statistics.mean(values))


def paired(baseline, candidate, metric, rounds):
    def index(rows):
        indexed = {row['round']: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(range(1, rounds + 1)):
            raise ValueError('missing or duplicate process round')
        return indexed
    a, b = index(baseline), index(candidate)
    pairs = [dict(round=i, baseline=a[i][metric], candidate=b[i][metric]) for i in sorted(a)]
    if any(not math.isfinite(x) or x <= 0 for p in pairs for x in (p['baseline'], p['candidate'])):
        raise ValueError('timings must be finite and positive')
    effects = [100 * (p['baseline'] - p['candidate']) / p['baseline'] for p in pairs]
    rng = random.Random(264)
    boot = [statistics.mean(rng.choices(effects, k=len(effects))) for _ in range(10000)]
    return dict(process_pairs=len(pairs), reduction_percent=statistics.mean(effects),
                ci95_percent=[percentile(boot, 2.5), percentile(boot, 97.5)], pairs=pairs,
                paired_reductions_percent=effects, seed=264, resamples=10000,
                estimator='mean process-paired percentage reduction; process bootstrap')


def break_even(base_setup, new_setup, base_first, new_first, base_warm, new_warm):
    saving = base_warm - new_warm
    if saving <= 0:
        return None
    premium = new_setup + new_first - base_setup - base_first
    return 1 + max(0, math.ceil(premium / saving))


def validate_record(contract, data, job, audit):
    def require(value, reason):
        if not value:
            raise ValueError(reason)
    require(data.get('complete') is True, 'incomplete record')
    require(data.get('job') == job, 'job mismatch')
    require(data.get('optimization', {}).get('policy') == job['policy'], 'policy mismatch')
    selected = job['policy']
    expected = (None, None) if selected == 'baseline' else (selected.startswith('prepared-'), selected.split('-')[1])
    owners = data['optimization'].get('owners', [])
    count = 3 if job['kind'] == 'groups' else (4 if job['kind'] == 'models' else
            (job.get('callers', 1) if job.get('runtime') == 'sym' else 0))
    require(len(owners) == count and all((o.get('prepared_groups'), o.get('report_mode')) == expected for o in owners),
            'owner constructor policy mismatch')
    if job['kind'] in ('lifecycle', 'concurrency'):
        require(data.get('frozen_runner_sha256') == contract['frozen_runners'][job['kind']], 'frozen runner changed')
    require(data.get('isolation', {}).get('valid') is True, 'invalid isolation')
    require(audit.get('job') == job and audit.get('returncode') == 0 and audit.get('observations') == [],
            'invalid audit')
    metadata = data['metadata']
    require(metadata.get('runtime_sha256') == contract['runtime_sha256'][job['policy']], 'runtime changed')
    for key in ('harness_sha256', 'dependency_sha256', 'artifact_sha256'):
        require(metadata.get(key) == contract[key], key + ' changed')
    for key, value in contract['environment'].items():
        require(metadata.get(key) == value, key + ' changed')


def summarize_groups(data, *, round, policy):
    result = []
    for case, row in data['cases'].items():
        completed, submit = distribution(row['samples_ms']), distribution(row['submit_samples_ms'])
        result.append(dict(kind='groups', case=case, round=round, policy=policy,
            setup_ms=row['setup_ms'], first_ms=row['first_ms'], p50_ms=completed['p50'], p95_ms=completed['p95'],
            samples=completed['n'], submit_p50_ms=submit['p50'], submit_p95_ms=submit['p95'],
            explicit_prepare_ms=row['explicit_prepare_ms']))
    return result


def validate_workload(data, job):
    def require(value, message):
        if not value: raise ValueError(message)
    def no_growth(before, after):
        require(all(before[k] == after[k] for k in ('context_creations', 'host_allocations', 'device_allocations')),
                'warm native resource growth')
    def closed(row):
        require(row['retained_bytes'] == row['live_bytes'] == 0, 'native retirement leaked')
    if job['kind'] == 'groups':
        require(set(data['cases']) == {'weights_small', 'weights_decode', 'kv_restore'}, 'group cases missing')
        for row in data['cases'].values():
            require(len(row['samples_ms']) == job['samples'] and len(row['submit_samples_ms']) == job['submit_samples'],
                    'group samples missing')
            require(row['correctness'] and row['held_output_checks_after_close'] == 2 * row['members'], 'group oracle')
            no_growth(row['before_warm']['transfer']['resources'], row['after_warm']['transfer']['resources'])
            closed(row['native_after_close'])
    elif job['kind'] == 'models':
        require(set(data['cases']) == {'llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense'}, 'model cases missing')
        for row in data['cases'].values():
            require(row['warm_compile_callbacks'] == 0, 'warm recompile')
            closed(row['native_after_close'])
            require({(a['factors']['runtime'], a['factors']['compute']) for a in row['arms'].values()} ==
                    {('torch', 'eager'), ('sym', 'eager'), ('torch', 'inductor'), ('sym', 'inductor')}, 'model arms missing')
            for arm in row['arms'].values():
                require(len(arm['samples_ms']) == arm['checks'] == job['samples'], 'model samples missing')
                d = arm['allocation_diagnostic']
                if d['before']['native'] is not None: no_growth(d['before']['native'], d['after']['native'])
    elif job['kind'] == 'lifecycle':
        require(len(data['requests']) == job['requests'] + 20, 'lifecycle requests missing')
        require(all(r['correct'] for r in data['requests']), 'lifecycle oracle')
        require(data['held_output_checks_after_close'] == 4, 'lifecycle held outputs')
        if data['native_after_close'] is not None: closed(data['native_after_close']['typed'])
    elif job['kind'] == 'concurrency':
        require(len(data['batches']) == job['batches'] and data['checks'] == job['batches'] * 8, 'burst requests missing')
        require(data['held_output_checks_after_close'] == 8 and data['physical_meter_valid'], 'burst oracle or meter')
        for row in data['native_after_close']:
            if row is not None: closed(row['typed'])
        for a, b in zip(data['ready']['workers'], data['warm_after']['workers']):
            require(a['compile'] == b['compile'], 'warm recompile')
            if job['runtime'] == 'sym': no_growth(a['transfer']['resources'], b['transfer']['resources'])
    else:
        raise ValueError('unknown workload')
