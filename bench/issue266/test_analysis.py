"""CPU rejection tests for scientific acceptance, including negative outcomes."""
import copy

import pytest

from bench.issue266.analyze import validate_records
from bench.issue266.support import environment_contract, executed_configuration, matrix, policy_configuration


def campaign():
    jobs = matrix()
    manifest = dict(jobs=jobs, provenance=dict(runtime={'bridge': 'abc'}, inputs={'oracle': 'def'},
                                             source={'worker': '123'}), acceptance=True)
    records = []
    for job in jobs:
        policy = policy_configuration(job)
        result = dict(outcome='success', physical_meter_valid=True,
                      configuration=executed_configuration(job, policy),
                      optimization=dict(policy, planning_ms=.01, owner_preparation_ms=1.),
                      budget=dict(requested_free_budget=job['budget_mib'] << 20,
                                  after=dict(free=(job['budget_mib'] << 20) - (2 << 20))),
                      checks=20 if job['kind'] == 'e4' else 240 if job['kind'] == 'e5' else 100,
                      held_output_checks_after_close=8 if job['kind'] == 'e5' else 1,
                      preparation_ms=1., first_ms=1., retirement_ms=1., snapshot_preparation_ms=1.,
                      first_request_ms=1., residency_preload_ms=1.,
                      warm_before=dict(torch={}, resources={}), warm_after=dict(torch={}, resources={}),
                      samples_ms=[1.] * (20 if job['kind'] == 'e4' else 100))
        if job['kind'] == 'e5':
            result['ready'] = dict(workers=[{}] * job['callers'])
            result['warm_after']['workers'] = [{}] * job['callers']
            result['retained_check_calls'] = 240
            result.update(warm_start_ms=31., warm_end_ms=39.)
            result['worker_costs'] = [dict(caller=i, preparation_ms=1., first_request_ms=2., retirement_ms=1.,
                warmup_samples_ms=[1.] * (24 // job['callers']), prepare_start_ms=1., prepare_end_ms=2.,
                first_start_ms=3., first_end_ms=5., warmup_end_ms=30., close_start_ms=40., close_end_ms=41.,
                arena_after_close=dict(closed=True, allocated_bytes=0, quarantined_bytes=0))
                for i in range(job['callers'])]
            result['input_sha256'] = [str(i) for i in range(8)]
            result['native_after_close'] = ([dict(typed=dict(closed=True, live_bytes=0,
                retained_bytes=0, active_contexts=0, quarantined=False))] * job['callers']
                if job['runtime'] == 'sym' else [None] * job['callers'])
            result['batches'] = [dict(batch=b, completed_ms=8., requests=[dict(correct=True, service_ms=1.,
                executor_execution_ms=1., queue_wait_ms=0., batch=b, input_index=i, caller=i % job['callers'],
                sojourn_ms=1., ordinal=i) for i in range(8)]) for b in range(30)]
        records.append(dict(job=job, provenance=copy.deepcopy(manifest['provenance']), result=result,
                            environment=environment_contract(job),
                            isolation=dict(foreign_busy_processes=[]), complete=True))
    return manifest, records


def test_accepts_complete_matrix_and_keeps_actual_oom():
    manifest, records = campaign()
    failed = next(r for r in records if r['job']['kind'] == 'e4')
    failed['result'].update(outcome='oom', checks=0, samples_ms=[], oom_phase='residency_start')
    valid = validate_records(manifest, records)
    assert len(valid) == len(records) and any(r['result']['outcome'] == 'oom' for r in valid)


def test_pilot_requires_an_explicit_nonacceptance_analysis():
    manifest, records = campaign()
    manifest.update(acceptance=False, jobs=manifest['jobs'][:2])
    with pytest.raises(ValueError): validate_records(manifest, records[:2])
    assert len(validate_records(manifest, records[:2], acceptance=False)) == 2


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'extra', 'hash', 'matrix', 'budget',
                                      'physical', 'contamination', 'checks', 'held', 'false_success',
                                      'oracle_oom', 'request_identity', 'inputs', 'retirement',
                                      'group_held', 'negative_time', 'nan_time', 'environment',
                                      'executed', 'policy', 'false_admission', 'worker_costs'])
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
    elif mutation == 'group_held':
        next(r for r in records if r['job']['kind'] == 'groups')['result']['held_output_checks_after_close'] = 0
    elif mutation == 'negative_time': e5['result']['batches'][0]['completed_ms'] = -1.
    elif mutation == 'nan_time': e5['result']['batches'][0]['requests'][0]['sojourn_ms'] = float('nan')
    elif mutation == 'environment': records[0]['environment']['gpu'] = 'wrong GPU'
    elif mutation == 'executed': records[0]['result']['configuration']['layers'] = 999
    elif mutation == 'policy': records[0]['result']['optimization']['slots'] = 8
    elif mutation == 'false_admission': records[0]['result']['admission_phase'] = 'timed_request'
    elif mutation == 'worker_costs': e5['result'].pop('worker_costs')
    with pytest.raises(ValueError): validate_records(manifest, records)


@pytest.mark.parametrize('mutation', ['none', 'empty_window', 'unordered', 'nan', 'wrong_peak'])
def test_trace_windows_require_observations_and_reconstruct_the_peak(mutation):
    from bench.issue266.analyze import validate_trace
    mib = 1 << 20
    result = dict(outcome='success', nvml_after_reservation=dict(used=100 * mib),
                  warm_start_ms=1., warm_end_ms=4., observed_warm_app_increment_peak=20 * mib,
                  physical_meter_valid=True)
    trace = dict(columns=['elapsed_ms', 'device_used_bytes', 'process_rss_bytes'],
                 samples=[[.5, 100*mib, mib], [2., 120*mib, mib], [3., 110*mib, mib], [5., 100*mib, mib]],
                 observed_device_used_peak=120*mib, observed_process_rss_peak=mib)
    if mutation == 'empty_window': result.update(warm_start_ms=6., warm_end_ms=7.)
    elif mutation == 'unordered': trace['samples'].reverse()
    elif mutation == 'nan': trace['samples'][1][1] = float('nan')
    elif mutation == 'wrong_peak': result['observed_warm_app_increment_peak'] = 0
    if mutation == 'none': validate_trace(dict(kind='groups'), result, trace)
    else:
        with pytest.raises(ValueError): validate_trace(dict(kind='groups'), result, trace)


def test_scoped_metrics_sum_callers_without_double_counting_torch_storage():
    from bench.issue266.analyze import scoped_metrics
    mib = 1 << 20
    def snapshot(count, retained):
        return dict(torch=dict(allocated=100*mib, reserved=200*mib,
                    max_allocated=120*mib, max_reserved=220*mib,
                    pinned_host={'active_bytes.current': 30*mib}),
            snapshots=dict(pinned_bytes=20*mib), workers=[dict(transfer=dict(
                resources=dict(device_bytes=retained*mib, host_bytes=mib,
                    device_allocations=count, context_creations=1),
                weight_storage=dict(output_arena=dict(allocated_bytes=24*mib,
                    live_bytes=8*mib, allocations=3, reuses=count,
                    completion_barriers=count, refusals=0)))) for _ in range(2)])
    result = dict(ready=snapshot(4, 8), warm_after=snapshot(7, 12),
                  optimization=dict(planning_ms=.2, owner_preparation_ms=.5),
                  snapshot_preparation_ms=3.,
                  worker_costs=[dict(preparation_ms=p, first_request_ms=f, retirement_ms=c)
                                for p, f, c in [(1., 10., .1), (2., 20., .2)]],
                  torch_with_outputs_after_close=dict(allocated=8*mib, reserved=200*mib),
                  batches=[dict(requests=[dict(service_ms=2., queue_wait_ms=3.)])])
    metrics = scoped_metrics(dict(kind='e5', runtime='sym'), result)
    assert metrics['native_device_retained_mib'] == 24
    assert metrics['native_host_retained_mib'] == 2
    assert metrics['native_warm_device_allocations'] == 6
    assert metrics['native_warm_context_creations'] == 0
    assert metrics['arena_allocated_mib'] == 48
    assert metrics['arena_warm_reuses'] == 6
    assert metrics['torch_reserved_mib'] == 200
    assert metrics['snapshot_pinned_mib'] == 20  # Shared, not multiplied by callers.
    assert metrics['worker_preparation_sum_ms'] == 3
    assert metrics['worker_first_max_ms'] == 20
    assert metrics['worker_retirement_sum_ms'] == pytest.approx(.3)
    assert 'physical_peak_mib' not in metrics  # Only the sampled trace supplies this.
