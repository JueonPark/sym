import pytest

from bench.issue272.witness import (
    STAGE_BOUNDARIES,
    STAGE_SLOTS,
    checked_output,
    stage_changes,
    validate_execution,
    validate_stage_runs,
)


def test_output_comparison_detects_signed_zero_and_checks_bits():
    torch = pytest.importorskip('torch')
    expected = torch.tensor([-0.0, 3.0], dtype=torch.float32)
    assert checked_output(expected.clone(), expected)['actual_sha256'] == (
        checked_output(expected, expected)['expected_sha256']
    )
    with pytest.raises(AssertionError, match='output bits'):
        checked_output(torch.tensor([0.0, 3.0]), expected)


def test_output_comparison_rejects_layout_change():
    torch = pytest.importorskip('torch')
    expected = torch.arange(9).view(3, 3)
    actual = expected.t().contiguous().t()
    with pytest.raises(AssertionError, match='output metadata'):
        checked_output(actual, expected)


def test_native_execution_requires_matching_completed_row():
    delta = {
        'fallbacks': {}, 'runtime_executions': 1, 'executions': 1,
        'preparations': 1, 'cache_hits': 0, 'groups': 1,
        'group_constructions': 1, 'rebinds': 0,
        'dispatches': {'cuda_relocate_f32': 1},
    }
    selection = {'implementation': 'cuda_relocate_f32', 'wire_boundary': 0}
    report = {'logical_transfers': 1, 'items': [dict(selection, executed=True)]}
    validate_execution('prepare', delta, {'selection': selection, 'report': report})
    with pytest.raises(AssertionError, match='completed dispatch'):
        validate_execution('prepare', delta, {'selection': selection, 'report': {
            'logical_transfers': 1, 'items': [dict(selection, wire_boundary=1, executed=True)],
        }})


def test_fallback_requires_one_saved_region_call_and_no_native_work():
    delta = {
        'fallbacks': {'prepared_small': 1}, 'runtime_executions': 0, 'executions': 0,
        'preparations': 0, 'cache_hits': 0, 'groups': 0, 'group_constructions': 0, 'rebinds': 0,
        'dispatches': {},
    }
    validate_execution('small', delta, None)
    with pytest.raises(AssertionError, match='fallback reason'):
        validate_execution('small', dict(delta, fallbacks={'prepared_small': 2}), None)


def test_stage_observation_decodes_bounds_and_subtracts_previous_requests():
    previous = [(0, 0)] * STAGE_SLOTS
    slot = (2 * STAGE_BOUNDARIES + 1) * STAGE_BOUNDARIES + 3
    previous[slot] = (9, 207)
    current = previous.copy()
    current[slot] = (12, 270)
    assert stage_changes(previous, current) == [
        {'kind': 'register_dequant_cast_chain', 'start_stage': 1, 'stop_stage': 3, 'calls': 3, 'elements': 63},
    ]


def test_observed_stage_range_must_cover_every_element_of_the_cpu_request():
    item = {'wire_boundary': 2}
    stage = {'kind': 'register_cast_chain', 'start_stage': 0, 'stop_stage': 2, 'calls': 2, 'elements': 32}
    row = {'native': {'report': {'items': [item]}}, 'source': {'shape': [2, 16]}}
    validate_stage_runs('ordered_cast', 'cpu', dict(row, observed_stage_runs=[stage]))
    with pytest.raises(AssertionError, match='full source'):
        validate_stage_runs('ordered_cast', 'cpu', dict(row, observed_stage_runs=[dict(stage, elements=31)]))
