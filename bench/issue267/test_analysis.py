"""Acceptance must reject incomplete or altered measured evidence."""
import copy
import hashlib
import json

import pytest

from bench.issue267 import analyze, contract

ENVIRONMENT = {'torch': 'qualified', 'python': 'qualified', 'gpu': 'qualified', 'platform': 'qualified'}


def fixture_record(suite='main'):
    job = next(j for j in contract.jobs() if j['suite'] == suite)
    record = {'job': job, 'provenance': {'source': {'worker.py': 'abc'},
              'runtime': {'libreloc_runtime.so': 'def'}, 'compiler': 'ghi'},
              'samples': 30, 'warmup': 5, 'affinity': list(contract.AFFINITY),
              'torch_threads': 1, 'interop_threads': 1, 'emulate_precision_casts': True,
              'complete': True, 'environment': dict(ENVIRONMENT), 'tf32': False,
              'cases': {}}
    for case in contract.cases(suite):
        record['cases'][contract.case_key(case)] = {
            'case': case, 'checks': len(contract.arms(case))*len(contract.modes(suite))*30,
            'consumer_checks': len(contract.arms(case))*30 if suite == 'consumer' else 0,
            'consumer_batch': 32 if suite == 'consumer' else None, 'h2d_copies_per_call': 1,
            'paths': {arm: {mode: {'samples_ms': [1.] * 30}
                      for mode in contract.modes(suite)} | {'first_completed_ms': 2.,'warmup_total_ms': 5.}
                      for arm in contract.arms(case)},
            'preparation': {'common_allocation_ms': 1., **{
                arm: {'compile_ms': 1., 'bind_ms': 1., 'native_ms': 1., 'storage_ms': 1.,
                      'plan_bytes': 32} for arm in contract.arms(case)}},
            'input_sha256': '0' * 64, 'oracle_sha256': '1' * 64, 'scale_sha256': '2' * 64,
            'cache_before': dict(files=0,bytes=0,generated_files=0,generated_bytes=0),
            'cache_after': dict(files=1,bytes=10,generated_files=1,generated_bytes=10),
            'runtime_shape_code_cache_bytes': 0,
        }
    return job, record


def test_complete_record_retains_all_arms():
    job, record = fixture_record()
    analyze.validate_record(record, job, record['provenance'], ENVIRONMENT)


@pytest.mark.parametrize('change', ['case', 'arm', 'sample', 'checks', 'h2d',
                                    'setting', 'source', 'runtime', 'nonfinite',
                                    'complete', 'environment', 'tf32', 'scale',
                                    'first', 'cost', 'cache', 'code_cache'])
def test_changed_measured_contract_rejected(change):
    job, record = fixture_record()
    provenance = copy.deepcopy(record['provenance'])
    row = next(iter(record['cases'].values()))
    if change == 'case': record['cases'].pop(next(iter(record['cases'])))
    elif change == 'arm': row['paths'].pop('sym_unfused')
    elif change == 'sample': row['paths']['sym_folded']['producer']['samples_ms'].pop()
    elif change == 'checks': row['checks'] -= 1
    elif change == 'h2d': row['h2d_copies_per_call'] = 2
    elif change == 'setting': record['emulate_precision_casts'] = False
    elif change == 'source': record['provenance']['source']['worker.py'] = 'changed'
    elif change == 'runtime': record['provenance']['runtime']['libreloc_runtime.so'] = 'changed'
    elif change == 'complete': record['complete'] = False
    elif change == 'environment': record['environment']['gpu'] = 'other GPU'
    elif change == 'tf32': record['tf32'] = True
    elif change == 'scale': row['scale_sha256'] = 'absent'
    elif change == 'first': row['paths']['sym_folded']['first_completed_ms'] = float('nan')
    elif change == 'cost': row['preparation']['sym_folded']['compile_ms'] = -1.
    elif change == 'cache': row['cache_after']['generated_bytes'] = -1
    elif change == 'code_cache': row['runtime_shape_code_cache_bytes'] = 999
    else: row['paths']['sym_folded']['producer']['samples_ms'][0] = float('nan')
    with pytest.raises(ValueError):
        analyze.validate_record(record, job, provenance, ENVIRONMENT)


@pytest.mark.parametrize('change', ['checks', 'batch'])
def test_consumer_work_and_checks_are_not_optional(change):
    job, record = fixture_record('consumer')
    row = next(iter(record['cases'].values()))
    if change == 'checks': row['consumer_checks'] = 0
    else: row['consumer_batch'] = 1
    with pytest.raises(ValueError):
        analyze.validate_record(record, job, record['provenance'], ENVIRONMENT)


def test_total_cost_keeps_preparation_and_first_completed_request():
    # Hand-derived: shared allocation2 + compile3 + bind5 + preparation7 +
    # storage11 + first completed13 + 99 warm calls*17 = 1724 ms.
    row = {'preparation': {'common_allocation_ms': 2., 'sym_folded': {
        'compile_ms': 3., 'bind_ms': 5., 'native_ms': 7., 'storage_ms': 11.}},
        'paths': {'sym_folded': {'first_completed_ms': 13., 'warmup_total_ms': 200.,
                                'completed_h2d': {'samples_ms': [17.] * 30}}}}
    assert analyze.total_cost(row, 'sym_folded', 'completed_h2d', 100) == 1724.


def test_campaign_membership_cannot_be_redefined():
    with pytest.raises(ValueError, match='membership'):
        analyze.validate_membership(contract.jobs()[:-1])
    duplicated = contract.jobs()
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match='membership'):
        analyze.validate_membership(duplicated)


def test_checksum_changes_are_rejected(tmp_path):
    path = tmp_path / 'record.json'
    path.write_text('{"value":1}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert analyze.read_checked(path, digest) == {'value': 1}
    path.write_text('{"value":2}\n')
    with pytest.raises(ValueError, match='checksum'):
        analyze.read_checked(path, digest)
