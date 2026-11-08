"""Acceptance must reject incomplete or altered measured evidence."""
import copy
import hashlib
import json

import pytest

from bench.issue267 import analyze, contract


def fixture_record():
    job = contract.jobs()[0]
    record = {'job': job, 'provenance': {'source': {'worker.py': 'abc'},
              'runtime': {'libreloc_runtime.so': 'def'}, 'compiler': 'ghi'},
              'samples': 30, 'warmup': 5, 'affinity': list(contract.AFFINITY),
              'torch_threads': 1, 'interop_threads': 1, 'emulate_precision_casts': True,
              'cases': {}}
    for case in contract.cases('main'):
        record['cases'][contract.case_key(case)] = {
            'case': case, 'checks': 300, 'h2d_copies_per_call': 1,
            'paths': {arm: {mode: {'samples_ms': [1.] * 30}
                      for mode in contract.modes('main')} for arm in contract.arms(case)},
            'preparation': {'compile_ms': 1., 'bind_ms': 1., 'native_ms': 1.},
            'input_sha256': '0' * 64, 'oracle_sha256': '1' * 64,
        }
    return job, record


def test_complete_record_retains_all_arms():
    job, record = fixture_record()
    analyze.validate_record(record, job, record['provenance'])


@pytest.mark.parametrize('change', ['case', 'arm', 'sample', 'checks', 'h2d',
                                    'setting', 'source', 'runtime', 'nonfinite'])
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
    else: row['paths']['sym_folded']['producer']['samples_ms'][0] = float('nan')
    with pytest.raises(ValueError):
        analyze.validate_record(record, job, provenance)


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
