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


def verify_trace(directory, data):
    from pathlib import Path
    import hashlib
    info = data['physical_memory']
    root = Path(directory).resolve()
    path = (root / info['file']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('missing or invalid physical memory trace')
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError('physical memory trace digest mismatch')


def summarize_lifecycle(data):
    phases = data['phases']
    initial = [r['completed_ms'] for r in data['requests'] if r['phase'] == 'initial']
    shape = [r['completed_ms'] for r in data['requests'] if r['phase'] == 'shape_change']
    scale = [r['completed_ms'] for r in data['requests'] if r['phase'] == 'scale_change']
    setup = data['launch_to_pre_cuda_ms'] + sum(phases[k] for k in
            ('cuda_init_ms', 'fixture_ms', 'snapshot_total_ms', 'request_owner_ms'))
    retirement = phases['owner_retirement_ms'] + phases['caller_release_and_cache_clear_ms']
    return dict(setup_ms=setup, first_ms=initial[0], p50_ms=percentile(initial[1:], 50),
                p95_ms=percentile(initial[1:], 95), samples=len(initial)-1,
                shape_first_ms=shape[0], shape_p50_ms=percentile(shape[1:], 50),
                scale_first_ms=scale[0], scale_p50_ms=percentile(scale[1:], 50),
                scale_reprepare_ms=phases['scale_reprepare_ms'], retirement_ms=retirement,
                full_accounted_ms=setup + sum(r['completed_ms'] for r in data['requests']) +
                    phases['shape_update_ms'] + phases['scale_reprepare_ms'] + retirement,
                observed_deployment_device_peak=data['observed_deployment_device_peak'])


def summarize_burst(data):
    requests = [r for batch in data['batches'] for r in batch['requests']]
    result = dict(samples=len(requests), requests_per_second=1000 * len(requests) /
                  sum(b['completed_ms'] for b in data['batches']),
                  mean_burst_ms=statistics.mean(b['completed_ms'] for b in data['batches']),
                  observed_warm_app_increment_peak=data['observed_warm_app_increment_peak'])
    for key in ('service_ms', 'executor_execution_ms', 'sojourn_ms', 'queue_wait_ms'):
        for quantile in (50, 95):
            result[key.removesuffix('_ms') + f'_p{quantile}_ms'] = percentile([r[key] for r in requests], quantile)
    result['p50_ms'], result['p95_ms'] = result['service_p50_ms'], result['service_p95_ms']
    return result


def collect(directory, *, partial=False):
    import json
    from pathlib import Path
    directory = Path(directory)
    contract = json.loads((directory / 'contract.json').read_text())
    jobs = json.loads((directory / 'matrix.json').read_text())
    records, missing = [], []
    for job in jobs:
        path = directory / (job['name'] + '.json')
        if not path.exists():
            missing.append(job['name'])
            continue
        data = json.loads(path.read_text())
        if partial and (not data.get('complete') or 'job' not in data):
            missing.append(job['name'])
            continue
        audit = json.loads((directory / data['isolation']['audit']).read_text())
        validate_record(contract, data, job, audit)
        validate_workload(data, job)
        base = dict(file=path.name, kind=job['kind'], round=job['round'], policy=job['policy'])
        if job['kind'] == 'groups':
            for row in summarize_groups(data, round=job['round'], policy=job['policy']):
                raw = data['cases'][row['case']]
                before, after = raw['diagnostics']['before'], raw['diagnostics']['after']
                prepared_a, prepared_b = before.get('prepared_groups', {}), after.get('prepared_groups', {})
                counts = {k: prepared_b.get(k, 0) - prepared_a.get(k, 0)
                          for k in ('group_constructions', 'view_descriptors', 'rebinds')}
                counts['group_constructions'] += raw['diagnostics']['legacy_group_constructions']
                counts['view_descriptors'] += raw['diagnostics']['legacy_view_descriptors']
                counts['report_materializations'] = (after.get('report_materializations', after['groups']) -
                                                       before.get('report_materializations', before['groups']))
                row.update(file=path.name, diagnostics=counts,
                    rss_bytes=raw['after_warm']['process']['rss_bytes'],
                    rss_growth_bytes=raw['after_warm']['process']['rss_bytes'] - raw['before_first']['process']['rss_bytes'],
                    native_retained_bytes=raw['after_warm']['transfer']['resources']['retained_bytes'],
                    live_slots=raw['after_warm']['transfer'].get('prepared_groups', {}).get('live_slots', 0),
                    source_bytes=raw['source_bytes'])
                records.append(row)
        elif job['kind'] == 'models':
            for case, row in data['cases'].items():
                for arm in row['arms'].values():
                    stats = distribution(arm['samples_ms'])
                    records.append(dict(base, case=case, compute=arm['factors']['compute'], runtime=arm['factors']['runtime'],
                        p50_ms=stats['p50'], p95_ms=stats['p95'], samples=stats['n'], first_ms=arm['first_ms'],
                        torch_snapshot_ms=row['torch_snapshot_ms'], sym_snapshot_ms=row['sym_snapshot_ms'],
                        allocation_diagnostic=arm['allocation_diagnostic']))
        else:
            verify_trace(directory, data)
            row = summarize_lifecycle(data) if job['kind'] == 'lifecycle' else summarize_burst(data)
            records.append(dict(base, case=job['case'], runtime=job['runtime'],
                                compute=job.get('compute', 'eager'), callers=job.get('callers', 1), **row))
    if missing and not partial: raise ValueError('missing jobs: ' + ', '.join(missing))
    return dict(records=records, missing=missing, contract=contract, jobs=len(jobs))


def comparisons(records):
    result = []
    def select(**criteria):
        return [r for r in records if all(r.get(k) == v for k, v in criteria.items())]
    def compare(label, a, b, metrics, rounds, **labels):
        if len(a) != rounds or len(b) != rounds: return
        row = dict(comparison=label, **labels, metrics={k: paired(a, b, k, rounds) for k in metrics})
        if all(all(k in r for k in ('setup_ms', 'first_ms', 'p50_ms')) for r in a + b):
            left, right = {r['round']: r for r in a}, {r['round']: r for r in b}
            row['estimated_break_even_requests'] = [dict(round=i, requests=break_even(
                left[i]['setup_ms'], right[i]['setup_ms'], left[i]['first_ms'], right[i]['first_ms'],
                left[i]['p50_ms'], right[i]['p50_ms'])) for i in sorted(left)]
        result.append(row)
    for kind in sorted({r['kind'] for r in records}):
        keys = ('case',) if kind == 'groups' else (('case', 'compute') if kind == 'models' else ('case', 'compute', 'callers'))
        combinations = sorted({tuple(r[k] for k in keys) for r in records if r['kind'] == kind})
        for values in combinations:
            key = dict(zip(keys, values)) | {'kind': kind}
            rows = select(**key)
            policies = sorted({r['policy'] for r in rows})
            rounds = 5 if kind in ('groups', 'models') else 3
            metrics = ['p50_ms', 'p95_ms']
            if kind == 'groups': metrics += ['submit_p50_ms', 'submit_p95_ms', 'setup_ms', 'first_ms']
            if kind == 'lifecycle': metrics += ['first_ms', 'full_accounted_ms', 'scale_reprepare_ms', 'shape_first_ms', 'scale_first_ms']
            if kind == 'concurrency': metrics += ['sojourn_p95_ms', 'mean_burst_ms']
            runtime = {} if kind == 'groups' else {'runtime': 'sym'}
            baseline = select(**key, **runtime, policy='baseline')
            legacy = select(**key, **runtime, policy='legacy-eager')
            for p in policies:
                candidate = select(**key, **runtime, policy=p)
                if p != 'baseline': compare('vs_frozen_sym', baseline, candidate, metrics, rounds, **key, policy=p)
                if p not in ('baseline', 'legacy-eager'):
                    compare('vs_new_legacy_sym', legacy, candidate, metrics, rounds, **key, policy=p)
                if kind != 'groups':
                    control = select(**key, runtime='torch', policy=p if kind == 'models' else 'baseline')
                    compare('sym_vs_torch', control, candidate, ['p50_ms', 'p95_ms'], rounds, **key, policy=p)
    return result


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = collect(args.directory, partial=args.partial)
    result['comparisons'] = comparisons(result['records'])
    result['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(len(result['records']), 'rows;', len(result['comparisons']), 'comparisons;', len(result['missing']), 'missing jobs')


if __name__ == '__main__': main()
