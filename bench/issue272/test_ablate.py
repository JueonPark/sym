from copy import deepcopy

import pytest

from bench.issue265.support import E3_CASES, E3_SEQUENCE


def test_matrix_turns_one_factor_off_at_a_time():
    from bench.issue272.ablate import matrix

    jobs = matrix(3)
    assert len(jobs) == 4 * len(E3_CASES) * 3
    assert len({job['name'] for job in jobs}) == len(jobs)
    assert {job['case'] for job in jobs} == set(E3_CASES)
    assert {job['round'] for job in jobs} == {1, 2, 3}
    arms = {job['arm']: (job['policy'], job['runtime']) for job in jobs}
    policy, runtime = arms['all_on']
    assert policy['min_bytes'] == policy['min_reuses'] == 0
    assert policy['row_policy'] == 'cpu'
    assert arms['cache_off'] == (policy | {'cache_specializations': False, 'reuse_groups': False}, runtime)
    assert arms['groups_off'] == (policy | {'reuse_groups': False}, runtime)
    assert arms['specialization_off'] == (policy, 'generic')


def fixture():
    from bench.issue272.ablate import matrix

    jobs = matrix(1)
    contract = dict(rounds=1, jobs=jobs, sequence=list(E3_SEQUENCE), epochs=5,
        harness_sha256={}, artifact_sha256={}, environment={},
        runtime_sha256={'specialized': {'native.so': 'on'}, 'generic': {'native.so': 'off'}})
    records = []
    for job in jobs:
        requests = []
        ms = 1 if job['arm'] == 'all_on' else 2
        policy = job['policy']
        for step, n in enumerate(E3_SEQUENCE * 5):
            executed = step + 1
            prepared = dict(policy=policy, attempts=executed, executions=executed,
                preparations=executed if not policy['cache_specializations'] else 1,
                cache_entries=int(policy['cache_specializations']), retained_parameter_bytes=0,
                cache_hits=step if policy['cache_specializations'] else 0,
                native=dict(logical_transfers=executed, dispatches={'cpu_reference': executed},
                    prepared_groups={'handles_created': int(policy['reuse_groups'])}))
            stats = dict(dynamo_callbacks=1, backend=dict(runtime_executions=executed,
                prepared_execution=prepared))
            requests.append(dict(step=step, n=n, completed_ms=ms, correct=True, held_correct=True,
                shapes=[[n]], strides=[[1]], stats=stats))
        records.append(dict(job=job, complete=True, policy=policy, requests=requests,
            final_stats=requests[-1]['stats'], held_checks_after_close=1, resources_after_close=None,
            phases_ms=dict(import_and_configure=1, cuda_init=2, owner_setup=3, owner_retirement=4),
            metadata=dict(harness_sha256={}, artifact_sha256={}, environment={},
                runtime_sha256=contract['runtime_sha256'][job['runtime']]),
            audit=dict(job=job, returncode=0, observations=[]), memory={}))
    return contract, records


def test_score_charges_lifecycle_once_and_retains_per_request_deltas():
    from bench.issue272.ablate import score

    contract, records = fixture()
    result = score(contract, records)
    assert result['jobs'] == 16
    assert result['exact_outputs'] == 960
    assert len(result['comparisons']) == 12
    row = next(row for row in result['summaries'] if row['job']['arm'] == 'all_on')
    assert row['cold_first_ms'] == 7
    assert row['first_cycle_ms'] == 12
    assert row['full_ms'] == 60
    assert row['repeated_ms'] == 48
    assert row['accounted_full_ms'] == 70
    assert row['requests'][1]['prepared_delta']['cache_hits'] == 1
    assert row['requests'][1]['prepared_delta']['native']['dispatches']['cpu_reference'] == 1
    comparison = result['comparisons'][0]['metrics']['full_ms']
    assert comparison['reduction_percent'] == 0
    assert comparison['ci95_percent'] is None


@pytest.mark.parametrize('change,reason', [
    ('short_matrix', 'registered matrix'),
    ('native_row', 'execution path differs'),
    ('cache_off_hit', 'retained specialization'),
    ('group_off_build', 'retained prepared group'),
    ('held_failure', 'oracle/held output failure'),
    ('other_stride', 'operand metadata differs'),
])
def test_score_rejects_incomplete_or_unmatched_evidence(change, reason):
    from bench.issue272.ablate import score

    contract, records = fixture()
    row = next(row for row in records if row['job']['arm'] == 'cache_off')
    step = row['requests'][3]
    prepared = step['stats']['backend']['prepared_execution']
    if change == 'short_matrix':
        contract['jobs'] = contract['jobs'][:-1]
        records.pop()
    elif change == 'native_row':
        prepared['native']['dispatches'] = {'cuda_relocate_f32': 4}
    elif change == 'cache_off_hit':
        prepared['cache_hits'] = 1
    elif change == 'group_off_build':
        group_control = next(row for row in records if row['job']['arm'] == 'groups_off')
        group_control['requests'][3]['stats']['backend']['prepared_execution']['native']['prepared_groups']['handles_created'] = 1
    elif change == 'held_failure':
        step['held_correct'] = False
    elif change == 'other_stride':
        step['strides'] = [[2]]
    with pytest.raises(ValueError, match=reason):
        score(contract, records)


def test_score_checks_exact_runtime_artifacts():
    from bench.issue272.ablate import score

    contract, records = fixture()
    records[0]['metadata'] = deepcopy(records[0]['metadata'])
    records[0]['metadata']['runtime_sha256'] = {'native.so': 'different'}
    with pytest.raises(ValueError, match='runtime changed'):
        score(contract, records)
