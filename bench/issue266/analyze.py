"""Strict standard-library reconstruction of the complete campaign."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue266.support import digest, matrix, save


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_records(manifest, records):
    jobs = manifest['jobs']
    require(manifest.get('acceptance') and jobs == matrix(), 'altered acceptance matrix')
    expected = {j['id']: j for j in jobs}
    inputs_by_case = {}
    require(Counter(r['job']['id'] for r in records) == Counter(expected.keys()), 'missing/duplicate/extra jobs')
    for record in records:
        job, result = record['job'], record['result']
        require(job == expected[job['id']], 'changed job settings')
        require(record['provenance'] == manifest['provenance'], 'stale provenance')
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
        require(not result.get('oom_phase') and not result.get('error'), 'false success')
        count = 20 if job['kind'] == 'e4' else 240 if job['kind'] == 'e5' else 100
        require(result.get('checks') == count, 'missing output checks')
        if job['kind'] == 'e5':
            require(result.get('held_output_checks_after_close') == 8, 'missing held-output retirement check')
            require(len(result['batches']) == 30 and all([q['ordinal'] for q in b['requests']] == list(range(8)) and
                all(r['correct'] for r in b['requests']) for b in result['batches']), 'invalid request matrix')
            inputs = result['input_sha256']
            require(len(inputs) == 8 and inputs == inputs_by_case.setdefault(job['case'], inputs),
                    'changed paired inputs')
            retired = result['native_after_close']
            require(len(retired) == job['callers'], 'missing worker retirement')
            if job['runtime'] == 'sym':
                require(all(s['typed']['closed'] and not s['typed']['quarantined'] and
                    all(s['typed'][key] == 0 for key in ('live_bytes', 'retained_bytes', 'active_contexts'))
                    for s in retired), 'native storage survived retirement')
        else:
            require(len(result['samples_ms']) == count and all(v > 0 for v in result['samples_ms']),
                    'invalid completed samples')
    return records


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(len(values) * fraction + .999999) - 1))]


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
            row['physical_peak_mib'] = statistics.median(r.get('observed_warm_app_increment_peak',
                r.get('observed_application_device_peak_above_initial_context', 0)) / (1 << 20) for r in successes)
            row['preparation_ms'] = statistics.median(r.get('snapshot_preparation_ms',
                r.get('preparation_ms', 0)) for r in successes)
            row['preload_ms'] = statistics.median(r.get('residency_preload_ms', 0) for r in successes)
        rows.append(row)
    return rows


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    require(completion['complete'] and completion['jobs'] == len(manifest['jobs']) and
            completion['provenance'] == manifest['provenance'], 'missing final provenance verification')
    paths = sorted(directory.glob('*.record.json'))
    records = [json.loads(p.read_text()) for p in paths]
    validate_records(manifest, records)
    for record in records:
        physical = record['result']['physical_memory']
        path = directory / physical['file']
        require(digest(path) == physical['sha256'], 'physical trace checksum mismatch')
        with gzip.open(path, 'rt') as stream: trace = json.load(stream)
        require(bool(trace['samples']), 'empty physical trace')
    result = dict(jobs=len(records), outcomes=dict(Counter(r['result']['outcome'] for r in records)),
                  rows=summarize(records))
    save(directory / 'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
