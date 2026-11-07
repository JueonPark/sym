"""CPU-only rejection tests for qualification and process-paired estimates."""
from copy import deepcopy
import math
import pytest
from bench.issue265.analyze import paired, validate_campaign, coverage


def fixture():
    jobs = [dict(name=f'r{r}-{arm}', round=r, case='cast', arm=arm, runtime=arm,
                 policy=None if arm == 'torch' else {'min_bytes': 0, 'max_entries': 2, 'max_parameter_bytes': 10})
            for r in range(1, 4) for arm in ('torch', 'prepared')]
    contract = dict(jobs=jobs, sequence=[3, 5, 3], epochs=2, rounds=3,
        harness_sha256={'worker.py': 'a'}, artifact_sha256={'benchmark.py': 'b'},
        runtime_sha256={'torch': {'module': 'c'}, 'prepared': {'module': 'd'}},
        environment={'affinity': [4, 5], 'torch': 'test'})
    records = []
    for job in jobs:
        record = dict(job=job, complete=True, metadata=dict(harness_sha256=contract['harness_sha256'],
            artifact_sha256=contract['artifact_sha256'], runtime_sha256=contract['runtime_sha256'][job['runtime']],
            environment=contract['environment']), policy=job['policy'],
            requests=[dict(step=i, n=n, correct=True, held_correct=True, completed_ms=1.,
                stats={'backend': None if job['policy'] is None else {'replaced_regions': 1, 'runtime_executions': i + 1,
                    'prepared_execution': {'policy': job['policy'], 'cache_entries': 1, 'retained_parameter_bytes': 0,
                        'executions': i + 1, 'native': {'logical_transfers': i + 1}}}})
                      for i, n in enumerate(contract['sequence'] * contract['epochs'])],
            held_checks_after_close=1, resources_after_close={'typed': {'retained_bytes': 0, 'live_bytes': 0}},
            audit=dict(job=job, returncode=0, observations=[]))
        records.append(record)
    return contract, records


def test_complete_process_matrix_and_process_pairing():
    contract, records = fixture()
    validate_campaign(contract, records)
    result = paired([10, 20, 40], [5, 10, 20])
    assert result['process_pairs'] == 3
    assert result['reduction_percent'] == result['ci95_percent'][0] == result['ci95_percent'][1] == 50
    with pytest.raises(ValueError):
        paired([1, 2], [1])
    with pytest.raises(ValueError):
        paired([1, math.inf], [1, 2])


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'runtime', 'harness', 'artifact', 'environment',
    'policy', 'sequence', 'oracle', 'held', 'retirement', 'audit', 'nan', 'incomplete'])
def test_rejects_incomplete_or_unmatched_evidence(damage):
    contract, records = fixture()
    records = deepcopy(records)
    if damage == 'missing': records.pop()
    elif damage == 'duplicate': records.append(records[0])
    elif damage in ('runtime', 'harness', 'artifact'):
        records[0]['metadata'][damage + '_sha256'] = {}
    elif damage == 'environment': records[0]['metadata']['environment']['torch'] = 'changed'
    elif damage == 'policy': records[0]['policy'] = {'min_bytes': 99}
    elif damage == 'sequence': records[0]['requests'].pop()
    elif damage == 'oracle': records[0]['requests'][0]['correct'] = False
    elif damage == 'held': records[0]['held_checks_after_close'] = 0
    elif damage == 'retirement': records[0]['resources_after_close']['typed']['live_bytes'] = 1
    elif damage == 'audit': records[0]['audit']['observations'] = ['foreign GPU process']
    elif damage == 'nan': records[0]['requests'][0]['completed_ms'] = math.nan
    elif damage == 'incomplete': records[0]['complete'] = False
    with pytest.raises(ValueError):
        validate_campaign(contract, records)


def test_region_exclusions_are_not_counted_as_fallback_requests():
    stats = {'runtime_executions': 45, 'fallbacks': {'small': 5}, 'exclusions': {'stride': 3},
             'prepared_execution': {'attempts': 50, 'executions': 40, 'exclusions': {'small': 5}}}
    row = coverage(stats, requests=60)
    assert row['runtime_executions'] == 45 and row['fallback_events'] == 5
    assert row['capture_exclusion_events'] == 3
    assert row['requests'] == 60
    assert 'eligible_plus_excluded' not in row


def test_resumption_validates_the_whole_record_not_only_runtime_hashes():
    from bench.issue265.analyze import validate_record
    contract, records = fixture()
    row = records[0]
    validate_record(contract, row, contract['jobs'][0])
    row['requests'][0]['held_correct'] = False
    with pytest.raises(ValueError, match='oracle'):
        validate_record(contract, row, contract['jobs'][0])


@pytest.mark.parametrize('damage', ['actual_policy', 'missing_manager', 'entries', 'bytes', 'execution_counts'])
def test_actual_backend_policy_bounds_and_native_counts_are_verified(damage):
    contract, records = fixture()
    records = deepcopy(records)
    backend = records[1]['requests'][0]['stats']['backend']
    prepared = backend['prepared_execution']
    if damage == 'actual_policy': prepared['policy'] = {'min_bytes': 99}
    elif damage == 'missing_manager': backend['prepared_execution'] = None
    elif damage == 'entries': prepared['cache_entries'] = 3
    elif damage == 'bytes': prepared['retained_parameter_bytes'] = 11
    elif damage == 'execution_counts': prepared['native']['logical_transfers'] = 99
    with pytest.raises(ValueError):
        validate_campaign(contract, records)
