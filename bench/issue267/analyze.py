"""Strict saved-data validation and complete tables; standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_checked(path, digest):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, f'checksum: {path}')
    return json.loads(data)


def validate_membership(jobs):
    require(jobs == contract.jobs(), 'campaign membership differs from frozen grid')


def validate_record(record, job, provenance, environment):
    require(record['job'] == job, 'job identity')
    require(record['complete'] is True, 'incomplete process record')
    require(record['environment'] == environment, 'environment mismatch')
    require(record['tf32'] is False, 'TF32 contract')
    require(record['provenance'] == provenance, 'source/runtime provenance')
    require(record['samples'] == contract.SAMPLES and record['warmup'] == contract.WARMUP,
            'sample/warmup contract')
    require(record['affinity'] == list(contract.AFFINITY), 'CPU affinity')
    require(record['torch_threads'] == job['threads'] and record['interop_threads'] == 1,
            'CPU thread contract')
    require(record['emulate_precision_casts'] is True, 'precision casts disabled')
    expected = {contract.case_key(c): c for c in contract.cases(job['suite'])}
    require(set(record['cases']) == set(expected), 'case membership')
    for key, case in expected.items():
        row = record['cases'][key]
        require(row['case'] == case, 'case configuration')
        require(set(row['paths']) == set(contract.arms(case)), 'arm membership')
        modes = contract.modes(job['suite'])
        require(row['checks'] == len(row['paths']) * len(modes) * contract.SAMPLES,
                'timed correctness checks')
        consumer = job['suite'] == 'consumer'
        require(row['consumer_checks'] == (len(row['paths']) * contract.SAMPLES if consumer else 0),
                'consumer correctness checks')
        require(row['consumer_batch'] == (32 if consumer else None), 'consumer batch')
        require(row['h2d_copies_per_call'] == 1, 'H2D contract')
        for name in ('input_sha256', 'oracle_sha256', 'scale_sha256'):
            require(len(row[name]) == 64 and all(c in '0123456789abcdef' for c in row[name]),
                    'input/oracle provenance')
        for arm, results in row['paths'].items():
            for name in ('first_completed_ms', 'warmup_total_ms'):
                require(math.isfinite(results[name]) and results[name] > 0, 'invalid first/warmup cost')
            for name in ('compile_ms', 'bind_ms', 'native_ms', 'storage_ms'):
                value = row['preparation'][arm][name]
                require(math.isfinite(value) and value >= 0, 'invalid preparation cost')
            require(isinstance(row['preparation'][arm]['plan_bytes'], int)
                    and row['preparation'][arm]['plan_bytes'] >= 0, 'invalid plan size')
            for mode in modes:
                samples = results[mode]['samples_ms']
                require(len(samples) == contract.SAMPLES, 'sample membership')
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                            for v in samples), 'invalid timing sample')
        common = row['preparation']['common_allocation_ms']
        require(math.isfinite(common) and common >= 0, 'invalid common allocation cost')
        require(row['runtime_shape_code_cache_bytes'] == 0, 'unexpected shape code cache')
        for name in ('cache_before', 'cache_after'):
            cache = row[name]
            require(set(cache) == {'files', 'bytes', 'generated_files', 'generated_bytes'}, 'cache fields')
            require(all(isinstance(v, int) and v >= 0 for v in cache.values()), 'invalid cache counters')
            require(cache['generated_files'] <= cache['files'] and cache['generated_bytes'] <= cache['bytes'],
                    'inconsistent cache counters')


def total_cost(row, arm, mode, requests):
    require(mode != 'producer' and requests >= 1, 'total requires completed requests')
    prepare = row['preparation'][arm]
    setup = row['preparation']['common_allocation_ms'] + sum(
        prepare[k] for k in ('compile_ms', 'bind_ms', 'native_ms', 'storage_ms'))
    return (setup + row['paths'][arm]['first_completed_ms'] + (requests - 1) *
            statistics.median(row['paths'][arm][mode]['samples_ms']))


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def analyze(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    validate_membership(manifest['jobs'])
    require(manifest['qualification'] is True, 'pilot is not acceptance')
    require(set(manifest['records']) == {j['id'] for j in manifest['jobs']}, 'record membership')
    observations = {}
    checks = 0
    for job in manifest['jobs']:
        saved = manifest['records'][job['id']]
        record = read_checked(directory / saved['path'], saved['sha256'])
        audit = read_checked(directory / saved['audit_path'], saved['audit_sha256'])
        require(audit['exit_code'] == 0 and not audit['foreign_busy_processes'],
                'failed or contaminated job')
        require(audit['job'] == job and audit['observations'] >= 1, 'missing process audit')
        validate_record(record, job, manifest['provenance'][job['runtime']], manifest['environment'])
        for key, row in record['cases'].items():
            checks += row['checks']
            for arm, values in row['paths'].items():
                for mode in contract.modes(job['suite']):
                    index = (job['suite'], key, job['threads'], job['runtime'], arm, mode)
                    observations.setdefault(index, []).append({
                        'round': job['round'], 'p50_ms': statistics.median(values[mode]['samples_ms']),
                        'p95_ms': percentile(values[mode]['samples_ms'], .95),
                        'input_sha256': row['input_sha256'], 'oracle_sha256': row['oracle_sha256'],
                        'scale_sha256': row['scale_sha256'], 'preparation': row['preparation'][arm],
                        'common_allocation_ms': row['preparation']['common_allocation_ms'],
                        'first_completed_ms': values['first_completed_ms'], 'warmup_total_ms': values['warmup_total_ms'],
                        'cache_generated_bytes': row['cache_after']['generated_bytes'] - row['cache_before']['generated_bytes'],
                        'cache_generated_files': row['cache_after']['generated_files'] - row['cache_before']['generated_files'],
                        'total_ms': {str(n): total_cost(row, arm, mode, n) for n in (1, 100, 1000)}
                                    if mode != 'producer' else None})
    rows = []
    for index, values in sorted(observations.items()):
        require(sorted(v['round'] for v in values) == [1, 2, 3], 'round membership')
        require(len({(v['input_sha256'], v['oracle_sha256'], v['scale_sha256']) for v in values}) == 1,
                'different round inputs')
        suite, case, threads, runtime, arm, mode = index
        rows.append(dict(suite=suite, case=case, threads=threads, runtime=runtime, arm=arm,
                         mode=mode, median_ms=statistics.median(v['p50_ms'] for v in values),
                         p95_ms=statistics.median(v['p95_ms'] for v in values), rounds=values,
                         first_completed_ms=statistics.median(v['first_completed_ms'] for v in values),
                         preparation_ms=statistics.median(v['common_allocation_ms'] + sum(v['preparation'][name]
                             for name in ('compile_ms','bind_ms','native_ms','storage_ms')) for v in values),
                         total_ms={str(n): statistics.median(v['total_ms'][str(n)] for v in values)
                                   for n in (1,100,1000)} if mode != 'producer' else None))
    # Both runtime processes must consume identical data and oracles.
    paired = {}
    for row in rows:
        key = (row['suite'], row['case'], row['threads'], row['arm'], row['mode'])
        value = tuple(row['rounds'][0][k] for k in ('input_sha256','oracle_sha256','scale_sha256'))
        require(key not in paired or paired[key] == value, 'runtime input mismatch')
        paired[key] = value
    result = dict(accepted=True, jobs=len(manifest['jobs']), checks=checks, rows=rows)
    (directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Complete issue 267 results', '',
             'Median of three process medians/p95s, in milliseconds. All cases and controls retained.', '',
             '| Suite | Case | Workers | Runtime | Arm | Mode | Median ms | P95 ms |',
             '| --- | --- | ---: | --- | --- | --- | ---: | ---: |']
    lines += [f"| {r['suite']} | {r['case']} | {r['threads']} | {r['runtime']} | {r['arm']} | {r['mode']} | {r['median_ms']:.6f} | {r['p95_ms']:.6f} |"
              for r in rows]
    lines += ['', '## Setup, first completed call and amortized totals', '',
              'Total(N) = shared allocation + arm-specific compile/bind/setup/storage + first completed call + (N−1)×warm median. Warmup runs used to estimate steady state are reported separately in raw data and excluded from this derived total. Medians are computed from complete per-process sums. All times are ms.', '',
              '| Suite | Case | Workers | Runtime | Arm | Preparation | First completed | Total(1) | Total(100) | Total(1000) |',
              '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['suite']} | {r['case']} | {r['threads']} | {r['runtime']} | {r['arm']} | {r['preparation_ms']:.3f} | {r['first_completed_ms']:.3f} | {r['total_ms']['1']:.3f} | {r['total_ms']['100']:.3f} | {r['total_ms']['1000']:.3f} |"
              for r in rows if r['mode'] != 'producer']
    (directory / 'tables.md').write_text('\n'.join(lines) + '\n')
    return dict(accepted=True, jobs=result['jobs'], checks=checks, rows=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
