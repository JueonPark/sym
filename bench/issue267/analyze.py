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


def validate_record(record, job, provenance):
    require(record['job'] == job, 'job identity')
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
        require(row['h2d_copies_per_call'] == 1, 'H2D contract')
        for name in ('input_sha256', 'oracle_sha256'):
            require(len(row[name]) == 64 and all(c in '0123456789abcdef' for c in row[name]),
                    'input/oracle provenance')
        for arm, results in row['paths'].items():
            for mode in modes:
                samples = results[mode]['samples_ms']
                require(len(samples) == contract.SAMPLES, 'sample membership')
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                            for v in samples), 'invalid timing sample')


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
        validate_record(record, job, manifest['provenance'][job['runtime']])
        for key, row in record['cases'].items():
            checks += row['checks']
            for arm, values in row['paths'].items():
                for mode in contract.modes(job['suite']):
                    index = (job['suite'], key, job['threads'], job['runtime'], arm, mode)
                    observations.setdefault(index, []).append({
                        'round': job['round'], 'p50_ms': statistics.median(values[mode]['samples_ms']),
                        'p95_ms': percentile(values[mode]['samples_ms'], .95),
                        'input_sha256': row['input_sha256'], 'oracle_sha256': row['oracle_sha256']})
    rows = []
    for index, values in sorted(observations.items()):
        require(sorted(v['round'] for v in values) == [1, 2, 3], 'round membership')
        require(len({(v['input_sha256'], v['oracle_sha256']) for v in values}) == 1,
                'different round inputs')
        suite, case, threads, runtime, arm, mode = index
        rows.append(dict(suite=suite, case=case, threads=threads, runtime=runtime, arm=arm,
                         mode=mode, median_ms=statistics.median(v['p50_ms'] for v in values),
                         p95_ms=statistics.median(v['p95_ms'] for v in values), rounds=values))
    # Both runtime processes must consume identical data and oracles.
    paired = {}
    for row in rows:
        key = (row['suite'], row['case'], row['threads'], row['arm'], row['mode'])
        value = (row['rounds'][0]['input_sha256'], row['rounds'][0]['oracle_sha256'])
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
    (directory / 'tables.md').write_text('\n'.join(lines) + '\n')
    return dict(accepted=True, jobs=result['jobs'], checks=checks, rows=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
