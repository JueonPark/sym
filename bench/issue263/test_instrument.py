"""Behavioral gates for attribution: nested time and thread/request isolation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from bench.issue263.instrument import Instrumentation


def test_nested_phases_partition_elapsed_without_double_counting():
    clock = SimpleNamespace(now=0)
    meter = Instrumentation('counters', clock=lambda: clock.now)
    with meter.request('request-a'):
        clock.now = 2
        with meter.phase('binding'):
            clock.now = 5
        clock.now = 8
        with meter.phase('native'):
            clock.now = 10
            with meter.phase('wait'):
                clock.now = 12
            clock.now = 14
        clock.now = 20
    row, = meter.records
    assert row['elapsed_ns'] == 20
    assert row['exclusive_ns'] == {'other': 11, 'binding': 3, 'native': 4, 'wait': 2}
    assert row['calls'] == {'binding': 1, 'native': 1, 'wait': 1}


def test_requests_on_different_threads_do_not_share_active_phase():
    meter = Instrumentation('counters')
    barrier = Barrier(2)
    def run(name):
        with meter.request(name):
            with meter.phase(name):
                barrier.wait(timeout=5)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, ['left', 'right']))
    assert {r['label'] for r in meter.records} == {'left', 'right'}
    for row in meter.records:
        assert row['calls'] == {row['label']: 1}
        assert sum(row['exclusive_ns'].values()) == row['elapsed_ns']


def test_wrapped_callable_restored_after_exception_and_result_unchanged():
    calls = []
    def original(x):
        calls.append(x)
        return x + 7
    obj = SimpleNamespace(run=original)
    with pytest.raises(RuntimeError, match='intentional'):
        with Instrumentation('counters') as meter:
            meter.wrap(obj, 'run', 'work')
            with meter.request('failure'):
                assert obj.run(4) == 11
                raise RuntimeError('intentional')
    assert obj.run is original
    assert calls == [4]
    assert meter.records[0]['error'] == 'RuntimeError'


def test_off_mode_installs_no_wrappers_and_records_no_samples():
    def original():
        return 17
    obj = SimpleNamespace(run=original)
    with Instrumentation('off') as meter:
        meter.wrap(obj, 'run', 'work')
        assert obj.run is original
        with meter.request('not-instrumented'):
            assert obj.run() == 17
    assert meter.records == []


def test_nested_request_is_rejected_without_losing_outer_state():
    meter = Instrumentation('counters')
    with meter.request('outer'):
        with pytest.raises(RuntimeError, match='nested request'):
            with meter.request('inner'):
                pass
        with meter.phase('remaining'):
            pass
    assert len(meter.records) == 1
    assert meter.records[0]['calls'] == {'remaining': 1}
