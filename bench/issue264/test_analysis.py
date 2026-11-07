"""Reject incomplete/tampered campaigns; pair independent processes, not requests."""
from copy import deepcopy
import math
import pytest

from bench.issue264.analyze import paired, break_even, validate_record, summarize_groups


def test_pairing_uses_round_identity_and_rejects_missing_duplicate_rounds():
    a = [dict(round=i, p50_ms=10.) for i in range(1, 6)]
    b = [dict(round=i, p50_ms=8.) for i in reversed(range(1, 6))]
    result = paired(a, b, 'p50_ms', 5)
    assert result['reduction_percent'] == 20.
    assert result['ci95_percent'] == [20., 20.]
    assert result['process_pairs'] == 5
    for invalid in (b[:-1], b + b[:1], b[:-1] + [dict(round=6, p50_ms=8.)]):
        with pytest.raises(ValueError, match='round'):
            paired(a, invalid, 'p50_ms', 5)
    with pytest.raises(ValueError, match='positive|finite'):
        paired(a, [dict(round=i, p50_ms=math.nan) for i in range(1, 6)], 'p50_ms', 5)


def test_break_even_includes_preparation_and_first_request_and_requires_positive_savings():
    assert break_even(10., 12., 5., 6., 2., 1.) == 4
    assert break_even(10., 12., 5., 6., 1., 1.) is None
    assert break_even(10., 12., 5., 6., 1., 2.) is None
    assert break_even(10., 9., 5., 5., 2., 1.) == 1


def record():
    job = dict(name='g-r1-prepared-native', kind='groups', round=1, policy='prepared-native')
    contract = dict(runtime_sha256={'prepared-native': {'runtime': 'native-hash'}},
                    harness_sha256={'groups.py': 'code-hash'},
                    dependency_sha256={'models.py': 'frozen-hash'},
                    artifact_sha256={'weights': 'data-hash'}, environment={'affinity': [4, 5], 'torch': '2.14.0'})
    data = dict(complete=True, job=job, optimization={'policy': 'prepared-native', 'owners': [dict(prepared_groups=True, report_mode='native')] * 3},
                metadata=dict(runtime_sha256={'runtime': 'native-hash'}, harness_sha256={'groups.py': 'code-hash'},
                    dependency_sha256={'models.py': 'frozen-hash'}, artifact_sha256={'weights': 'data-hash'},
                    affinity=[4, 5], torch='2.14.0'),
                isolation=dict(valid=True, audit='test.audit.json'))
    audit = dict(returncode=0, observations=[], job=job)
    return contract, data, job, audit


def test_provenance_accepts_only_complete_matching_inputs_and_clean_audits():
    contract, data, job, audit = record()
    validate_record(contract, data, job, audit)
    for path, value in [(('complete',), False), (('metadata', 'runtime_sha256'), {}),
                        (('metadata', 'harness_sha256'), {}), (('metadata', 'dependency_sha256'), {}),
                        (('metadata', 'artifact_sha256'), {}), (('metadata', 'affinity'), [4]),
                        (('metadata', 'torch'), 'different'), (('optimization', 'policy'), 'legacy-eager'),
                        (('isolation', 'valid'), False)]:
        broken = deepcopy(data)
        target = broken
        for key in path[:-1]: target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError): validate_record(contract, broken, job, audit)
    for mutation in (dict(returncode=1), dict(observations=[{'foreign': 'GPU'}]), dict(job={})):
        with pytest.raises(ValueError): validate_record(contract, data, job, audit | mutation)


def test_group_summary_does_not_substitute_submission_for_completed_latency():
    data = {'cases': {'small': dict(setup_ms=5., first_ms=4., samples_ms=[3., 1., 2.],
                                  submit_samples_ms=[.1, .2, .3], explicit_prepare_ms=.5)}}
    result = summarize_groups(data, round=2, policy='prepared-native')[0]
    assert result['p50_ms'] == 2. and result['submit_p50_ms'] == .2
    assert result['setup_ms'] == 5. and result['first_ms'] == 4.
    assert result['round'] == 2


def test_workload_validation_rejects_truncated_counts_and_resource_growth():
    from bench.issue264.analyze import validate_workload
    job = dict(kind='groups', samples=3, submit_samples=2)
    row = dict(samples_ms=[1., 2., 3.], submit_samples_ms=[.1, .2], correctness=True,
               members=2, held_output_checks_after_close=4, native_after_close={'retained_bytes': 0, 'live_bytes': 0},
               before_warm={'transfer': {'resources': {'context_creations': 1, 'host_allocations': 1, 'device_allocations': 1}}},
               after_warm={'transfer': {'resources': {'context_creations': 1, 'host_allocations': 1, 'device_allocations': 1}}})
    data = dict(cases={key: deepcopy(row) for key in ('weights_small', 'weights_decode', 'kv_restore')})
    validate_workload(data, job)
    broken = deepcopy(data); broken['cases']['weights_small']['samples_ms'].pop()
    with pytest.raises(ValueError): validate_workload(broken, job)
    broken = deepcopy(data); broken['cases']['kv_restore']['after_warm']['transfer']['resources']['host_allocations'] = 2
    with pytest.raises(ValueError): validate_workload(broken, job)


def test_claimed_policy_must_match_actual_owner_constructor_settings():
    contract, data, job, audit = record()
    data['optimization']['owners'] = [dict(prepared_groups=False, report_mode='eager')]
    with pytest.raises(ValueError, match='constructor|owner'):
        validate_record(contract, data, job, audit)
