"""CPU checks for the executable catalog planner."""
import copy
import itertools
import math

import pytest

from bench.issue271.planner import choose_plan


def event(name, us, *resources, deps=()):
    return dict(id=name, us=us, resources=list(resources), deps=list(deps))


def candidate(name, events, **fields):
    return dict(name=name, qualified=True, events=events, setup_us=0,
                memory=dict(resident_bytes=0, retained_bytes=0, context_bytes=0,
                            warm_floor_bytes=0, setup_peak_bytes=0, intervals=[])) | fields


def test_link_gpu_overlap_and_exclusive_lane_order():
    events = [event('copy0', 3, 'link'), event('work', 4, 'gpu'),
              event('copy1', 2, 'link'), event('wait', 1, 'sync', deps=['work', 'copy1'])]
    result = choose_plan([candidate('pipeline', events)], budget_bytes=0)
    assert result['prediction']['warm_us'] == 6
    assert result['prediction']['schedule'] == {
        'copy0': dict(start_us=0, end_us=3), 'work': dict(start_us=0, end_us=4),
        'copy1': dict(start_us=3, end_us=5), 'wait': dict(start_us=5, end_us=6)}


def test_slowdown_changes_only_the_overlapping_part_and_can_change_selection():
    events = [event('copy', 2, 'link'), event('run', 5, 'gpu'),
              event('join', 1, 'sync', deps=['run', 'copy'])]
    pipelined = candidate('pipeline', events, overlap_slowdowns=[
        dict(resources=['link', 'gpu'], factor=2)])
    reference = candidate('reference', [event('run', 7, 'cpu')])
    prediction = choose_plan([pipelined], budget_bytes=0)['prediction']
    assert prediction['schedule']['copy']['end_us'] == 4
    assert prediction['schedule']['run']['end_us'] == 7
    assert prediction['warm_us'] == 8
    assert choose_plan([pipelined, reference], budget_bytes=0)['selected'] == 'reference'


def test_transfer_slowdown_extends_temporary_lifetime_into_later_allocation():
    events = [event('copy', 4, 'link'), event('pack', 2, 'cpu'),
              event('run', 2, 'gpu', deps=['pack']), event('timer', 4, 'sync'),
              event('aux', 1, 'cpu', deps=['timer'])]
    plan = candidate('prefetch', events)
    plan['memory']['intervals'] = [dict(begin='copy', end='copy', bytes=4),
                                   dict(begin='aux', end='aux', bytes=3)]
    assert choose_plan([plan], budget_bytes=4)['prediction']['peak_bytes'] == 4
    plan['overlap_slowdowns'] = [dict(resources=['link', 'gpu'], factor=2)]
    assert choose_plan([plan], budget_bytes=4)['fallback']
    pred = choose_plan([plan], budget_bytes=7)['prediction']
    assert (pred['warm_us'], pred['peak_bytes']) == (6, 7)


def test_decimal_calibrations_can_finish_within_one_float_tick():
    events = [event('copy', 3.1, 'link'), event('gpu0', .2, 'gpu'), event('gpu1', 2.9, 'gpu')]
    plan = candidate('decimal', events, overlap_slowdowns=[dict(resources=['link', 'gpu'], factor=1.3)])
    pred = choose_plan([plan], budget_bytes=0)['prediction']
    assert pred is not None
    assert pred['warm_us'] == pytest.approx(4.03)


def test_positive_tiny_event_cannot_disappear_from_capacity_timeline():
    events = [event('long', 1e16, 'gpu'), event('short', .001, 'gpu')]
    plan = candidate('tiny', events)
    plan['memory']['intervals'] = [dict(begin='short', end='short', bytes=4)]
    assert choose_plan([plan], budget_bytes=3)['fallback']
    pred = choose_plan([plan], budget_bytes=4)['prediction']
    assert pred['schedule']['short']['end_us'] > pred['schedule']['short']['start_us']


def exhaustive_earliest(events):
    """Enumerate integer schedules under fixed submission order and compute earliest starts."""
    horizon = sum(e['us'] for e in events)
    ids = [e['id'] for e in events]
    legal = []
    for starts in itertools.product(*(range(horizon - e['us'] + 1) for e in events)):
        ends = [s + e['us'] for s, e in zip(starts, events)]
        if all(ends[i] <= starts[j]
               for j, e in enumerate(events) for i in range(j)
               if ids[i] in e['deps'] or set(events[i]['resources']) & set(e['resources'])):
            legal.append(starts)
    earliest = min(legal)
    assert all(all(x <= y for x, y in zip(earliest, later)) for later in legal)
    return {e['id']: (s, s + e['us']) for e, s in zip(events, earliest)}


def test_joint_search_matches_independent_discrete_schedule_and_byte_oracle():
    plans = []
    reference = {}
    for placement, serial, reuse in itertools.product(['host', 'device'], [False, True], [False, True]):
        events = [event('pack', 2 if placement == 'host' else 1, 'cpu'),
                  event('copy', 2 if placement == 'host' else 1, 'link', deps=['pack']),
                  event('prev', 2, 'gpu', deps=['copy'] if serial else []),
                  event('next', 1 if placement == 'host' else 2, 'gpu', deps=['copy'])]
        name = f'{placement}-{int(serial)}-{int(reuse)}'
        plan = candidate(name, events)
        plan['memory'].update(resident_bytes=2, retained_bytes=int(reuse), context_bytes=1,
                              setup_peak_bytes=4 if reuse else 2, warm_floor_bytes=3,
                              intervals=[dict(begin='copy', end='next', bytes=1 if reuse else 3),
                                         dict(begin='prev', end='prev', bytes=2)])
        plans.append(plan)
        schedule = exhaustive_earliest(events)
        warm = max(end for _, end in schedule.values())
        peak = max(3 + int(reuse) + (1 if reuse else 3) *
                   (schedule['copy'][0] <= tick < schedule['next'][1]) +
                   2 * (schedule['prev'][0] <= tick < schedule['prev'][1]) for tick in range(warm))
        reference[name] = (warm, peak, name)
        pred = choose_plan([plan], budget_bytes=20)['prediction']
        assert pred['schedule'] == {k: dict(start_us=v[0], end_us=v[1]) for k, v in schedule.items()}
        assert (pred['warm_us'], pred['peak_bytes']) == (warm, peak)
    for budget in range(3, 10):
        feasible = [value for value in reference.values() if value[1] <= budget]
        expected = min(feasible)[2] if feasible else None
        for ordered in (plans, list(reversed(plans))):
            result = choose_plan(ordered, budget_bytes=budget)
            assert result['selected'] == expected
            assert result['fallback'] == (expected is None)


def test_setup_and_warm_qualification_peaks_and_reuse_at_completion():
    plan = candidate('peaks', [event('a', 2, 'link'), event('b', 2, 'link')])
    plan['memory'].update(resident_bytes=1, retained_bytes=2, context_bytes=3,
                          intervals=[dict(begin=e, end=e, bytes=4) for e in ['a', 'b']])
    for field, measured in [('warm_floor_bytes', 12), ('setup_peak_bytes', 13)]:
        plan['memory'][field] = measured
        assert choose_plan([plan], budget_bytes=measured - 1)['fallback']
        assert choose_plan([plan], budget_bytes=measured)['prediction']['peak_bytes'] == measured


def test_lifecycle_selects_amortized_setup_separately_from_completed_warm_latency():
    resident = candidate('resident', [event('compute', 1, 'gpu')], setup_us=10)
    streaming = candidate('stream', [event('compute', 3, 'gpu')], setup_us=1)
    plans = [streaming, resident]
    assert choose_plan(plans, budget_bytes=0)['selected'] == 'resident'
    short = choose_plan(plans, budget_bytes=0, objective='lifecycle', reuses=2)
    assert short['selected'] == 'stream'
    assert short['prediction']['lifecycle_us'] == 7
    assert short['prediction']['amortized_us'] == 3.5
    assert choose_plan(plans, budget_bytes=0, objective='lifecycle', reuses=12)['selected'] == 'resident'


def test_slo_scores_explicit_arrivals_and_rejects_high_backlog():
    # Faster requests qualify at the intermediate rate. An offered burst queues for too long.
    loads = [dict(arrivals_us=[0, 10, 20, 30]), dict(arrivals_us=[0, 4, 8, 12]),
             dict(arrivals_us=[0, 1, 2, 3])]
    plan = candidate('fcfs', [event('service', 4, 'gpu')], loads=loads)
    result = choose_plan([plan], objective='slo', budget_bytes=0, slo_us=6)
    pred = result['prediction']
    assert (pred['load_index'], pred['p95_us'], pred['throughput_rps']) == (1, 4, 250_000)
    plan['loads'] = [loads[0]]
    pred = choose_plan([plan], objective='slo', budget_bytes=0, slo_us=6)['prediction']
    assert pred['throughput_rps'] == pytest.approx(4e6 / 34)
    plan['loads'] = [loads[2]]
    assert choose_plan([plan], objective='slo', budget_bytes=0, slo_us=6)['fallback']


def test_slo_can_score_explicit_calibration_completions_with_nearest_rank_p95():
    arrivals = list(range(0, 80, 4))
    completions = [a + 3 for a in arrivals]
    completions[-1] += 6
    plan = candidate('calibration', [event('service', 1, 'gpu')],
                     loads=[dict(arrivals_us=arrivals, completions_us=completions)])
    pred = choose_plan([plan], budget_bytes=0, objective='slo', slo_us=3)['prediction']
    assert pred['p95_us'] == 3
    assert pred['throughput_rps'] == pytest.approx(20e6 / 85)


def test_calibration_completions_must_follow_single_executor_fcfs_order():
    p = candidate('observed', [event('x', 1, 'cpu')])
    p['loads'] = [dict(arrivals_us=[0, 1, 2], completions_us=[3, 6, 5])]
    assert choose_plan([p], budget_bytes=0, objective='slo', slo_us=10)['fallback']


@pytest.mark.parametrize('patch', [
    {'qualified': False}, {'qualified': 1}, {'setup_us': -1.0}, {'setup_us': math.nan},
    {'events': []}, {'events': [event('x', math.inf, 'gpu')]},
    {'events': [event('x', 0, 'gpu')]}, {'events': [event('x', 1)]},
    {'events': [event('x', 1, 'gpu', deps=['x'])]},
    {'events': [event('x', 1, 'gpu'), event('x', 1, 'link')]},
    {'overlap_slowdowns': [dict(resources=['link', 'gpu'], factor=.8)]}, {'memory': {}},
])
def test_invalid_rows_never_beat_qualified_reference(patch):
    unsafe = candidate('unsafe', [event('x', 1, 'gpu')]) | patch
    ref = candidate('safe', [event('x', 100, 'cpu')])
    result = choose_plan([unsafe, ref], budget_bytes=5)
    assert result['selected'] == 'safe'
    assert result['rejected']['unsafe']


@pytest.mark.parametrize('patch', [
    {'context_bytes': -1}, {'warm_floor_bytes': math.nan}, {'resident_bytes': True},
    {'intervals': [dict(begin='x', end='missing', bytes=1)]},
    {'intervals': [dict(begin='y', end='x', bytes=1)]}])
def test_missing_or_invalid_byte_evidence_causes_fallback(patch):
    p = candidate('unsafe', [event('x', 1, 'link'), event('y', 1, 'link')])
    p['memory'].update(patch)
    out = choose_plan([p], budget_bytes=5)
    assert out['fallback'] and out['selected'] is None and out['prediction'] is None


@pytest.mark.parametrize('loads', [[], [{}], [dict(arrivals_us=[float('nan')])],
                                   [dict(arrivals_us=[2, 1])], [dict(arrivals_us=[1, 2], completions_us=[3])],
                                   [dict(arrivals_us=[1, 2], completions_us=[2, 1])]])
def test_invalid_or_absent_load_coverage_fails_closed(loads):
    p = candidate('bad', [event('x', 1, 'cpu')], loads=loads)
    assert choose_plan([p], objective='slo', budget_bytes=0, slo_us=4)['fallback']


def test_catalog_bound_and_duplicate_names_cannot_return_ambiguous_or_partial_winner():
    p = candidate('first', [event('x', 1, 'cpu')])
    assert choose_plan([p, copy.deepcopy(p)], budget_bytes=0)['fallback']
    q = p | {'name': 'second'}
    result = choose_plan([p, q], max_candidates=1, budget_bytes=0)
    assert result['fallback'] and result['examined'] == 0
    assert result['planning_us'] >= 0


def test_joint_search_prunes_a_dominated_lane_work_lower_bound():
    quick = candidate('a-quick', [event('run', 2, 'cpu')])
    slow = candidate('b-slow', [event('one', 2, 'gpu'), event('two', 2, 'gpu')])
    result = choose_plan([slow, quick], budget_bytes=0)
    assert result['selected'] == 'a-quick'
    assert 'b-slow' in result['pruned']
