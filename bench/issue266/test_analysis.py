"""CPU rejection tests for scientific acceptance, including negative outcomes."""
import copy

import pytest

from bench.issue266.analyze import validate_records
from bench.issue266.support import matrix


def campaign():
    jobs = matrix()
    manifest = dict(jobs=jobs, provenance=dict(runtime={'bridge': 'abc'}, inputs={'oracle': 'def'},
                                             source={'worker': '123'}), acceptance=True)
    records = []
    for job in jobs:
        result = dict(outcome='success', physical_meter_valid=True,
                      budget=dict(requested_free_budget=job['budget_mib'] << 20,
                                  after=dict(free=(job['budget_mib'] << 20) - (2 << 20))),
                      checks=20 if job['kind'] == 'e4' else 240 if job['kind'] == 'e5' else 100,
                      held_output_checks_after_close=8 if job['kind'] == 'e5' else 1,
                      samples_ms=[1.] * (20 if job['kind'] == 'e4' else 100))
        if job['kind'] == 'e5':
            result['input_sha256'] = [str(i) for i in range(8)]
            result['native_after_close'] = ([dict(typed=dict(closed=True, live_bytes=0,
                retained_bytes=0, active_contexts=0, quarantined=False))] * job['callers']
                if job['runtime'] == 'sym' else [None] * job['callers'])
            result['batches'] = [dict(completed_ms=8., requests=[dict(correct=True, service_ms=1.,
                sojourn_ms=1., ordinal=i) for i in range(8)]) for _ in range(30)]
        records.append(dict(job=job, provenance=copy.deepcopy(manifest['provenance']), result=result,
                            isolation=dict(foreign_busy_processes=[]), complete=True))
    return manifest, records


def test_accepts_complete_matrix_and_keeps_actual_oom():
    manifest, records = campaign()
    failed = next(r for r in records if r['job']['kind'] == 'e4')
    failed['result'].update(outcome='oom', checks=0, samples_ms=[], oom_phase='residency_start')
    valid = validate_records(manifest, records)
    assert len(valid) == len(records) and any(r['result']['outcome'] == 'oom' for r in valid)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'extra', 'hash', 'matrix', 'budget',
                                      'physical', 'contamination', 'checks', 'held', 'false_success',
                                      'oracle_oom', 'request_identity', 'inputs', 'retirement'])
def test_rejects_incomplete_or_invalid_evidence(mutation):
    manifest, records = campaign()
    e5 = next(r for r in records if r['job']['kind'] == 'e5')
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records.append(copy.deepcopy(records[0]))
    elif mutation == 'extra': records[-1]['job']['id'] = 'unknown'
    elif mutation == 'hash': records[0]['provenance']['runtime']['bridge'] = 'stale'
    elif mutation == 'matrix': manifest['jobs'].pop()
    elif mutation == 'budget': records[0]['result']['budget']['after']['free'] += 64 << 20
    elif mutation == 'physical': records[0]['result']['physical_meter_valid'] = False
    elif mutation == 'contamination': records[0]['isolation']['foreign_busy_processes'] = [[123, 'GPU']]
    elif mutation == 'checks': e5['result']['checks'] -= 1
    elif mutation == 'held': e5['result']['held_output_checks_after_close'] = 0
    elif mutation == 'false_success': records[0]['result']['oom_phase'] = 'timed_request'
    elif mutation == 'oracle_oom': records[0]['result'].update(outcome='oom', oom_phase='oracle_start')
    elif mutation == 'request_identity': e5['result']['batches'][0]['requests'][1]['ordinal'] = 0
    elif mutation == 'inputs': e5['result']['input_sha256'][0] = 'changed'
    elif mutation == 'retirement':
        sym = next(r for r in records if r['job']['kind'] == 'e5' and r['job']['runtime'] == 'sym')
        sym['result']['native_after_close'][0]['typed']['live_bytes'] = 4096
    with pytest.raises(ValueError): validate_records(manifest, records)
