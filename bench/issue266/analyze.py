"""Strict standard-library reconstruction of the complete campaign."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue266.support import digest, environment_contract, executed_configuration, matrix, policy_configuration, save


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_records(manifest, records, *, acceptance=True):
    try:
        return _validate_records(manifest, records, acceptance)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('malformed campaign record: ' + str(error)) from error


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _validate_records(manifest, records, acceptance):
    jobs = manifest['jobs']
    if acceptance:
        require(manifest.get('acceptance') and jobs == matrix(), 'altered acceptance matrix')
    else:
        allowed = {j['id']: j for j in matrix()}
        require(manifest.get('acceptance') is False and jobs and
                all(j == allowed.get(j['id']) for j in jobs), 'invalid excluded pilot matrix')
    expected = {j['id']: j for j in jobs}
    inputs_by_case = {}
    require(Counter(r['job']['id'] for r in records) == Counter(expected.keys()), 'missing/duplicate/extra jobs')
    for record in records:
        job, result = record['job'], record['result']
        require(job == expected[job['id']], 'changed job settings')
        require(record['provenance'] == manifest['provenance'], 'stale provenance')
        require(all(record['environment'].get(k) == v for k, v in environment_contract(job).items()),
                'unqualified environment')
        policy = policy_configuration(job)
        require(all(result['optimization'].get(k) == v for k, v in policy.items()), 'unexecuted policy')
        expected_configuration = executed_configuration(job, policy)
        actual_configuration = {k: v for k, v in result['configuration'].items() if k != 'output'}
        require(actual_configuration == expected_configuration, 'wrong executed workload configuration')
        require(record.get('complete') and not record['isolation']['foreign_busy_processes'], 'incomplete/contaminated')
        require(result.get('physical_meter_valid'), 'invalid physical meter')
        budget = job['budget_mib'] << 20
        require(result['budget']['requested_free_budget'] == budget and
                budget - (16 << 20) <= result['budget']['after']['free'] <= budget, 'invalid physical reservation')
        outcome = result['outcome']
        require(outcome in ('success', 'oom', 'admission'), 'invalid execution outcome')
        if outcome != 'success':
            require(bool(result.get('oom_phase') if outcome == 'oom' else result.get('admission_phase')),
                    'negative result needs a classified policy phase')
            phase = result.get('oom_phase', result.get('admission_phase'))
            allowed = {'preparation_start', 'residency_start', 'first_request', 'warmup_request', 'timed_request'}
            require(phase in (allowed if job['kind'] == 'e4' else {'worker_or_request'}),
                    'fixture/oracle failure is not a policy outcome')
            continue
        require(not result.get('oom_phase') and not result.get('admission_phase') and not result.get('error'), 'false success')
        require(finite(result['optimization']['planning_ms']), 'missing planning cost')
        count = 20 if job['kind'] == 'e4' else 240 if job['kind'] == 'e5' else 100
        require(result.get('checks') == count, 'missing output checks')
        if job['kind'] == 'e5':
            require(result.get('held_output_checks_after_close') == 8, 'missing held-output retirement check')
            require(len(result['batches']) == 30, 'invalid burst count')
            for index, batch in enumerate(result['batches']):
                require(batch['batch'] == index and finite(batch['completed_ms'], positive=True), 'invalid burst timing')
                require([q['ordinal'] for q in batch['requests']] == list(range(8)), 'invalid request matrix')
                for ordinal, request in enumerate(batch['requests']):
                    require(request['correct'] and request['batch'] == index and request['input_index'] == ordinal and
                            request['caller'] == ordinal % job['callers'], 'changed request identity')
                    require(all(finite(request[k], positive=True) for k in
                        ('service_ms', 'executor_execution_ms', 'sojourn_ms')) and
                        finite(request['queue_wait_ms']), 'invalid request timing')
            inputs = result['input_sha256']
            require(len(inputs) == 8 and inputs == inputs_by_case.setdefault(job['case'], inputs),
                    'changed paired inputs')
            retired = result['native_after_close']
            require(len(retired) == job['callers'], 'missing worker retirement')
            if job['runtime'] == 'sym':
                require(all(s['typed']['closed'] and not s['typed']['quarantined'] and
                    all(s['typed'][key] == 0 for key in ('live_bytes', 'retained_bytes', 'active_contexts'))
                    for s in retired), 'native storage survived retirement')
            require(result['retained_check_calls'] == 240, 'missing interburst retained checks')
            require(len(result['ready']['workers']) == job['callers'] and
                    len(result['warm_after']['workers']) == job['callers'], 'missing readiness evidence')
            require(finite(result['snapshot_preparation_ms']) and
                    finite(result['optimization']['owner_preparation_ms']), 'missing snapshot preparation cost')
            costs = result['worker_costs']
            require(len(costs) == job['callers'] and [c['caller'] for c in costs] == list(range(job['callers'])),
                    'missing worker lifecycle costs')
            for cost in costs:
                require(all(finite(cost[k], positive=True) for k in
                    ('preparation_ms', 'first_request_ms', 'retirement_ms')), 'invalid worker lifecycle timing')
                phases = [cost[k] for k in ('prepare_start_ms', 'prepare_end_ms', 'first_start_ms',
                    'first_end_ms', 'warmup_end_ms', 'close_start_ms', 'close_end_ms')]
                require(all(finite(v) for v in phases) and phases == sorted(phases), 'invalid lifecycle phase order')
                require(cost['warmup_end_ms'] <= result['warm_start_ms'] and
                        cost['close_start_ms'] >= result['warm_end_ms'], 'lifecycle phases overlap warm bursts')
                require(abs(cost['preparation_ms'] - (phases[1] - phases[0])) < 1e-6 and
                        abs(cost['retirement_ms'] - (phases[-1] - phases[-2])) < 1e-6, 'inconsistent lifecycle duration')
                require(len(cost['warmup_samples_ms']) == 24 // job['callers'] and
                        all(finite(v, positive=True) for v in cost['warmup_samples_ms']), 'missing first/warmup costs')
                if policy['packed']:
                    arena = cost['arena_after_close']
                    require(arena['closed'] and arena['allocated_bytes'] == 0 and arena['quarantined_bytes'] == 0,
                            'output arena survived retirement')
        else:
            require(len(result['samples_ms']) == count and all(finite(v, positive=True) for v in result['samples_ms']),
                    'invalid completed samples')
            require('torch' in result['warm_before'] and 'resources' in result['warm_before'] and
                    'torch' in result['warm_after'] and 'resources' in result['warm_after'], 'missing readiness evidence')
            if job['kind'] == 'groups':
                require(result.get('held_output_checks_after_close') == 1, 'missing held group after retirement')
                require(all(finite(result[k], positive=True) for k in ('preparation_ms', 'first_ms', 'retirement_ms')),
                        'missing group lifecycle costs')
            else:
                require(all(finite(result[k], positive=k != 'residency_preload_ms') for k in
                    ('snapshot_preparation_ms', 'first_request_ms', 'residency_preload_ms')), 'missing E4 setup costs')
    return records


def validate_trace(job, result, trace):
    try:
        return _validate_trace(job, result, trace)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('malformed physical trace: ' + str(error)) from error


def _validate_trace(job, result, trace):
    require(trace['columns'] == ['elapsed_ms', 'device_used_bytes', 'process_rss_bytes'], 'wrong physical columns')
    rows = trace['samples']
    require(rows and all(len(r) == 3 and all(finite(v) for v in r) for r in rows), 'invalid physical sample')
    require(all(a[0] < b[0] for a, b in zip(rows, rows[1:])), 'unordered physical samples')
    require(trace['observed_device_used_peak'] == max(r[1] for r in rows) and
            trace['observed_process_rss_peak'] == max(r[2] for r in rows), 'incorrect trace extrema')
    baseline = result['nvml_after_reservation']['used']
    require(finite(baseline, positive=True) and trace['observed_device_used_peak'] >= baseline - (16 << 20),
            'physical meter does not observe reservation')
    if job['kind'] == 'e4':
        phases = result['phases']
        boundaries = [phases[k] for k in ('fixture_start', 'oracle_start', 'preparation_start')]
        require(all(finite(v) for v in boundaries) and boundaries == sorted(boundaries), 'invalid deployment phases')
        deployment = [r for r in rows if r[0] >= phases['fixture_start'] and
                      (r[0] < phases['oracle_start'] or r[0] >= phases['preparation_start'])]
        require(deployment, 'unobserved deployment interval')
        peak = max(r[1] for r in deployment)
        require(result['observed_deployment_device_used_peak'] == peak and
                result['observed_application_device_peak_above_initial_context'] == peak - baseline,
                'incorrect deployment peak')
        if result['outcome'] != 'success': return
        start, end = phases['warm_start'], phases['warm_end']
    else:
        if result['outcome'] != 'success': return
        start, end = result['warm_start_ms'], result['warm_end_ms']
    require(finite(start) and finite(end) and start < end, 'invalid warm interval')
    warm = [r for r in rows if start <= r[0] <= end]
    require(warm, 'unobserved warm interval')
    peak = max(r[1] for r in warm)
    recorded = result['observed_warm_device_used_peak'] if job['kind'] == 'e4' else result['observed_warm_app_increment_peak']
    require(recorded == (peak if job['kind'] == 'e4' else peak - baseline), 'incorrect warm peak')


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(len(values) * fraction + .999999) - 1))]


def scoped_metrics(job, result):
    """Separate allocator/owner scopes; never synthesize a physical peak."""
    mib = 1 << 20
    before = result['ready'] if job['kind'] == 'e5' else result['warm_before']
    after = result['warm_after']
    def owners(snapshot):
        if job['kind'] == 'e5': return [w['transfer'] for w in snapshot['workers']]
        resources = snapshot['resources']
        return [resources['native'] if job['runtime'] == 'sym' else resources]
    def counter(snapshot, key, *, arena=False):
        total = 0
        for owner in owners(snapshot):
            if arena:
                item = (owner.get('weight_storage') or {}).get('output_arena') if job['runtime'] == 'sym' else owner.get('output_arena')
            else:
                item = owner['resources'] if job['runtime'] == 'sym' else None
            total += item[key] if item is not None else 0
        return total
    metrics = dict(planning_ms=result['optimization']['planning_ms'])
    for key in ('allocated', 'reserved', 'max_allocated', 'max_reserved'):
        metrics['torch_' + key + '_mib'] = after['torch'][key] / mib
    metrics['torch_pinned_active_mib'] = after['torch']['pinned_host']['active_bytes.current'] / mib
    for key in ('device', 'host'):
        metrics['native_' + key + '_retained_mib'] = counter(after, key + '_bytes') / mib
    for key in ('device_allocations', 'context_creations'):
        metrics['native_warm_' + key] = counter(after, key) - counter(before, key)
    for key in ('allocated', 'live'):
        metrics['arena_' + key + '_mib'] = counter(after, key + '_bytes', arena=True) / mib
    for key in ('allocations', 'reuses', 'completion_barriers', 'refusals'):
        metrics['arena_warm_' + key] = counter(after, key, arena=True) - counter(before, key, arena=True)
    if job['kind'] == 'e5':
        metrics['snapshot_pinned_mib'] = after['snapshots']['pinned_bytes'] / mib
        metrics['owner_preparation_ms'] = result['optimization']['owner_preparation_ms']
        costs = result['worker_costs']
        metrics['worker_preparation_sum_ms'] = sum(c['preparation_ms'] for c in costs)
        metrics['worker_first_max_ms'] = max(c['first_request_ms'] for c in costs)
        metrics['worker_retirement_sum_ms'] = sum(c['retirement_ms'] for c in costs)
        requests = [q for b in result['batches'] for q in b['requests']]
        for key in ('service', 'queue_wait'):
            metrics[key + '_p95_ms'] = percentile([q[key + '_ms'] for q in requests], .95)
    elif job['kind'] == 'e4':
        metrics['first_request_ms'] = result['first_request_ms']
        metrics['resident_weights_mib'] = result['resident_weight_bytes'] / mib
        metrics['snapshot_pinned_mib'] = result['owned_pinned_checkpoint_bytes'] / mib
    else:
        metrics['first_request_ms'] = result['first_ms']
        metrics['retirement_ms'] = result['retirement_ms']
        # Sym snapshots expose their own owner counter; the Torch control's
        # pinned allocations are reported by the process-wide Torch scope.
        if job['runtime'] == 'sym':
            metrics['snapshot_pinned_mib'] = after['resources']['snapshots']['pinned_bytes'] / mib
    if job['kind'] != 'e4':
        for key in ('allocated', 'reserved'):
            metrics['torch_after_close_' + key + '_mib'] = result['torch_with_outputs_after_close'][key] / mib
    return metrics


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        key = tuple((k, v) for k, v in record['job'].items() if k not in ('id', 'round'))
        groups[key].append(record)
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(key)
        row['outcomes'] = dict(Counter(r['result']['outcome'] for r in values))
        successes = [r['result'] for r in values if r['result']['outcome'] == 'success']
        row['rounds'] = len(values)
        if successes:
            timings = [[b['completed_ms'] / 8 for b in r['batches']] if row['kind'] == 'e5'
                       else r['samples_ms'] for r in successes]
            row['median_ms'] = statistics.median(statistics.median(v) for v in timings)
            row['round_medians_ms'] = [statistics.median(v) for v in timings]
            row['p95_ms'] = statistics.median(percentile(
                [q['sojourn_ms'] for b in r['batches'] for q in b['requests']] if row['kind'] == 'e5'
                else r['samples_ms'], .95) for r in successes)
            field = 'observed_application_device_peak_above_initial_context' if row['kind'] == 'e4' else 'observed_warm_app_increment_peak'
            row['physical_peak_mib'] = statistics.median(r[field] / (1 << 20) for r in successes)
            row['physical_window'] = 'deployment, excluding oracle' if row['kind'] == 'e4' else 'warm loop including checks'
            field = 'preparation_ms' if row['kind'] == 'groups' else 'snapshot_preparation_ms'
            row['preparation_ms'] = statistics.median(r[field] for r in successes)
            row['preload_ms'] = statistics.median(r.get('residency_preload_ms', 0) for r in successes)
            scoped = [scoped_metrics(row, r) for r in successes]
            row.update({k: statistics.median(v[k] for v in scoped) for k in scoped[0]})
        rows.append(row)
    return rows


def markdown_tables(analysis):
    """Render every predeclared arm, including unsuccessful arms, in order."""
    lines = ['# Complete issue 266 tables', '',
        'Accepted campaign: **' + str(analysis['accepted']).lower() + '**. '
        'All predeclared arms are retained. Times and memory are medians across '
        'successful rounds only; outcomes show the denominator. A dash means '
        'not applicable or no successful measurement, never zero.', '',
        'E4 latency is a completed request; E5 latency is completed burst time '
        'divided by eight requests and p95 is request sojourn. Group latency '
        'is completed iterator delivery, excluding initial iterator submission. '
        'P95 is the median of the round p95s, not a pooled percentile.', '',
        'Physical peaks are sampled NVML increments above the reservation/context '
        'baseline: E4 deployment excluding the oracle; E5/groups the warm loop '
        'including validation gaps. MiB means 2^20 bytes. Native device/host '
        'retention sums callers; arena storage is already inside Torch allocation. '
        'These scopes must not be added to estimate a physical peak.', '']
    def number(value):
        if value is None: return '—'
        if type(value) is int: return str(value)
        return f'{value:.4f}'.rstrip('0').rstrip('.')
    def label(row):
        if row['kind'] == 'e4':
            return f"L{row['layers']}/S{row['context']}/B{row['budget_mib']} {row['placement']} {row['runtime']} {row['policy']}"
        if row['kind'] == 'e5':
            return f"{row['case']} C{row['callers']} {row['runtime']} {row['policy']}"
        return f"{row['size']} {row['runtime']} {row['policy']}"
    def table(rows, columns):
        lines.append('| Arm | ' + ' | '.join(title for title, _ in columns) + ' |')
        lines.append('| --- | ' + ' | '.join('---:' for _ in columns) + ' |')
        for row in rows:
            cells = []
            for _, key in columns:
                if key == 'outcomes':
                    cells.append('/'.join(str(row['outcomes'].get(k, 0)) for k in ('success', 'oom', 'admission')))
                else: cells.append(number(row.get(key)))
            lines.append('| ' + label(row) + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    for kind in ('e4', 'e5', 'groups'):
        rows = [r for r in analysis['rows'] if r['kind'] == kind]
        lines += ['## ' + kind.upper() + ': completed timing and physical memory', '']
        table(rows, [('Success/OOM/admission', 'outcomes'), ('Median ms', 'median_ms'),
                     ('P95 ms', 'p95_ms'), ('Physical MiB', 'physical_peak_mib')])
        lines += ['### Preparation and retirement', '']
        columns = [('Plan ms', 'planning_ms'), ('Snapshot/preparation ms', 'preparation_ms')]
        if kind == 'e5':
            columns += [('Owner ms', 'owner_preparation_ms'), ('Workers prepare sum ms', 'worker_preparation_sum_ms'),
                        ('First max ms', 'worker_first_max_ms'), ('Workers retire sum ms', 'worker_retirement_sum_ms'),
                        ('Service p95 ms', 'service_p95_ms'), ('Queue p95 ms', 'queue_wait_p95_ms')]
        else:
            if kind == 'e4': columns.append(('Preload ms', 'preload_ms'))
            columns.append(('First ms', 'first_request_ms'))
            if kind == 'groups': columns.append(('Retire ms', 'retirement_ms'))
        table(rows, columns)
        lines += ['### Retained owner and allocator scopes at warm end', '']
        table(rows, [('Torch allocated MiB', 'torch_allocated_mib'), ('Torch reserved MiB', 'torch_reserved_mib'),
            ('Native device MiB', 'native_device_retained_mib'), ('Native host MiB', 'native_host_retained_mib'),
            ('Arena allocated MiB', 'arena_allocated_mib'), ('Arena live MiB', 'arena_live_mib'),
            ('Torch pinned active MiB', 'torch_pinned_active_mib')])
        lines += ['### Warm allocation and reuse counter deltas', '']
        table(rows, [('Native device allocations', 'native_warm_device_allocations'),
            ('Native context creations', 'native_warm_context_creations'),
            ('Arena allocations', 'arena_warm_allocations'), ('Arena reuses', 'arena_warm_reuses'),
            ('Arena barriers', 'arena_warm_completion_barriers'), ('Arena refused attempts', 'arena_warm_refusals')])
    lines += ['Arena refused attempts include speculative prefetch backpressure '
        'that subsequently progressed; they are not terminal admission outcomes. '
        'Full per-round samples, worker lifecycle timings, allocator peaks, snapshot '
        'counters and post-close counters remain in the JSON records and analysis.', '']
    return '\n'.join(lines)


def analyze(directory, *, acceptance=True):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    require(completion['complete'] and completion['jobs'] == len(manifest['jobs']) and
            completion['provenance'] == manifest['provenance'], 'missing final provenance verification')
    paths = sorted(directory.glob('*.record.json'))
    records = [json.loads(p.read_text()) for p in paths]
    validate_records(manifest, records, acceptance=acceptance)
    for record in records:
        physical = record['result']['physical_memory']
        path = directory / physical['file']
        require(digest(path) == physical['sha256'], 'physical trace checksum mismatch')
        with gzip.open(path, 'rt') as stream: trace = json.load(stream)
        validate_trace(record['job'], record['result'], trace)
    result = dict(accepted=acceptance, jobs=len(records), outcomes=dict(Counter(r['result']['outcome'] for r in records)),
                  rows=summarize(records))
    save(directory / ('analysis.json' if acceptance else 'analysis-pilot.json'), result)
    (directory / ('tables.md' if acceptance else 'tables-pilot.md')).write_text(markdown_tables(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--pilot', action='store_true', help='validate an explicitly excluded pilot, never accept it')
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, acceptance=not args.pilot), indent=2))
