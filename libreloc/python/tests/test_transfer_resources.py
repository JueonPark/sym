"""Explicit native Python owners; fault/GIL cases run in bounded subprocesses."""
import gc
import os
import pickle
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from conftest import golden_hex

pyreloc = pytest.importorskip("pyreloc")


def buffers(n=64, seed=1, direction="h2d"):
    bound = pyreloc.bind(pyreloc.load_plan(bytes.fromhex(golden_hex("reference"))), {"N": n})
    src = (np.arange(bound.min_src_bytes, dtype=np.uint32) * 131 + seed).astype(np.uint8)
    dst = np.full(bound.total_bytes, 0xCD, dtype=np.uint8)
    expected = np.zeros_like(dst)
    pyreloc.relocate(bound, src.ctypes.data, src.nbytes, expected.ctypes.data, expected.nbytes)
    source = pyreloc.BufferView(src.ctypes.data, src.nbytes, 0, bound.extents,
                               bound.src_strides, bound.element_size, "host")
    target = pyreloc.BufferView(dst.ctypes.data, dst.nbytes, 0,
                               [bound.total_bytes // bound.element_size], [1],
                               bound.element_size, "host")
    return pyreloc.make_transfer(bound, source, target, direction), src, dst, expected


def execute(bundle, cache=None, **options):
    request, src, dst, expected = bundle
    pyreloc.execute_transfer(request, resources=cache, owners=(src, dst), n_buffers=1, **options)
    np.testing.assert_array_equal(dst, expected)
    assert request.consumed


def until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "native state did not advance"
        time.sleep(0.001)


def test_native_cache_import_and_construction_are_torch_free():
    result = subprocess.run([sys.executable, "-c", """
import sys, pyreloc
with pyreloc.TransferResourceCache() as cache:
    assert cache.stats()['contexts'] == 0
    assert not cache.closed
assert cache.closed and 'torch' not in sys.modules
"""], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_native_reuse_processes_current_data_and_releases_owners(direction):
    retained_outputs = []
    references = []
    with pyreloc.TransferResourceCache() as cache:
        for i, n in enumerate([256, 64, 128, 64]):
            bundle = buffers(n, i + 17, direction)
            execute(bundle, cache, gather_threads=3)
            _, src, dst, expected = bundle
            references.append((weakref.ref(src), weakref.ref(dst)))
            retained_outputs.append((dst, expected))
            del bundle, src, dst, expected
            assert all(src_ref() is None for src_ref, _ in references)
            for old, oracle in retained_outputs:
                np.testing.assert_array_equal(old, oracle)
        s = cache.stats()
        assert (s['misses'], s['hits'], s['staging_allocations']) == (1, 3, 1)
        assert (s['stream_creations'], s['worker_creations']) == (2, 2)
        assert s['outstanding_events'] == 0 and s['devices'][0]['kind'] == 'host'
        del old, oracle
        retained_outputs.clear()
        assert all(dst_ref() is None for _, dst_ref in references)
        cache.clear()
        assert cache.stats()['contexts'] == 0 and cache.stats()['generation'] == 1
    cache.close()
    assert cache.stats()['allocated_staging_bytes'] == 0


@pytest.mark.parametrize("options", [
    {"max_contexts": -1}, {"max_retained_bytes": -1}, {"max_contexts_per_device": -1},
    {"max_background_workers": -1}, {"max_live_staging_bytes": -1},
    {"acquire_timeout_ms": -1}, {"max_contexts": 1.5}, {"max_contexts": 2**100},
])
def test_native_limits_reject_invalid_values(options):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        pyreloc.TransferResourceCache(**options)


def test_native_owners_preflight_consumption_and_legacy_ephemeral_calls():
    cache = pyreloc.TransferResourceCache(max_contexts=0)
    bundle = buffers()
    request, src, dst, _ = bundle
    for owners in [None, (), (src,), (src, None), [src, dst]]:
        with pytest.raises(ValueError, match="owners"):
            pyreloc.execute_transfer(request, resources=cache, owners=owners)
        assert not request.consumed
    with pytest.raises(pyreloc.TransferError, match="^resource_limit"):
        execute(bundle, cache)
    assert request.consumed and cache.stats()['contexts'] == 0
    with pytest.raises(pyreloc.TransferError, match="^already_executed"):
        execute(bundle, cache)
    for owning in [False, True]:
        bundle = buffers()
        if owning:
            execute(bundle)
        else:
            request, src, dst, expected = bundle
            pyreloc.execute_transfer(request)
            np.testing.assert_array_equal(dst, expected)
    cache.close()
    with pytest.raises(pyreloc.TransferError, match="^resources_closed"):
        execute(buffers(), cache)
    with pytest.raises(pyreloc.TransferError, match="^resources_closed"):
        cache.__enter__()
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(cache)


def test_argument_callbacks_cannot_reenter_the_same_native_request():
    bundle = buffers()
    request, source, destination, expected = bundle
    with pyreloc.TransferResourceCache() as cache:
        class Owners(tuple):
            def __getitem__(self, index):
                with pytest.raises(pyreloc.TransferError, match='^already_executed'):
                    execute(bundle, cache)
                return super().__getitem__(index)

        pyreloc.execute_transfer(request, resources=cache, owners=Owners((source, destination)))
        np.testing.assert_array_equal(destination, expected)
        assert cache.stats()['requests'] == 1


def scenario_lifecycle(operation):
    from _reloc_transfer_test import Control

    control = Control()
    cache = control.cache(timeout_ms=30 if operation == 'timeout' else None)
    active, waiting = buffers(), buffers(seed=2)
    pool = pyreloc.GatherPool(3)
    with ThreadPoolExecutor(3) as threads:
        first = threads.submit(execute, active, cache)
        try:
            assert control.wait_for_copy()
            assert active[0].consumed
            with pytest.raises(pyreloc.TransferError, match="^already_executed"):
                execute(active, cache)
            second = threads.submit(execute, waiting, cache, gather_pool=pool, gather_threads=999)
            until(lambda: cache.stats()['waiters'] == 1)
            if operation == 'close':
                closed = threads.submit(cache.close)
                until(lambda: cache.closed)
                with pytest.raises(pyreloc.TransferError, match="^resources_closed"):
                    second.result(timeout=5)
                assert not closed.done()
            elif operation == 'timeout':
                with pytest.raises(pyreloc.TransferError, match="^resource_timeout"):
                    second.result(timeout=5)
            elif operation == 'clear':
                cache.clear()
                assert cache.stats()['generation'] == 1
            else:
                pool.close()  # admitted borrowed pools may close while waiting
        finally:
            control.release()
        first.result(timeout=5)
        if operation == 'close':
            closed.result(timeout=5)
        elif operation != 'timeout':
            second.result(timeout=5)
        if operation == 'clear':
            assert cache.stats()['misses'] == 2
    cache.close()
    assert cache.stats()['contexts'] == 0
    assert pool.closed == (operation == 'borrowed_close')
    pool.close()


def scenario_failure(mode):
    from _reloc_transfer_test import Control

    control = Control(mode=mode, gated=mode != 'allocation')
    cache = control.cache()
    references, outcomes = [], []

    def worker():
        request, src, dst, _ = buffers()
        references.extend([weakref.ref(src), weakref.ref(dst)])
        try:
            pyreloc.execute_transfer(request, resources=cache, owners=(src, dst), n_buffers=1)
        except pyreloc.TransferError as error:
            outcomes.append(str(error).split(':')[0])  # discard exception/traceback

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        if mode != 'allocation':
            assert control.wait_for_copy()
            assert all(ref() is not None for ref in references)
        if mode not in {'allocation', 'unknown'}:
            assert thread.is_alive()  # quiescence still needs the pending copy
            assert control.stats()['frees'] == 0
    finally:
        if mode != 'unknown':
            control.release()
    thread.join(timeout=5)
    assert not thread.is_alive()
    gc.collect()
    if mode == 'unknown':
        assert outcomes == ['completion_unknown']
        assert all(ref() is not None for ref in references)
        assert cache.stats()['quarantined'] == 1 and control.stats()['frees'] == 0
        with pytest.raises(pyreloc.TransferError, match="^completion_unknown"):
            cache.close()
        control.release()
        control.drain_unknown()  # test-only finish of the simulated pending DMA
        del cache
        gc.collect()
        assert all(ref() is not None for ref in references)
        assert control.stats()['destroyed'] == 0
        # Exit normally with quarantined Python objects/native workers alive.
        # Their persistent token must not run Python destructors at shutdown.
    else:
        assert outcomes == ['backend_failure']
        assert all(ref() is None for ref in references)
        assert cache.stats()['contexts'] == 0 and control.stats()['destroyed'] == 1
        cache.close()


def scenario_fork():
    cache = pyreloc.TransferResourceCache()
    execute(buffers(), cache, gather_threads=2)
    child = os.fork()
    if child == 0:
        for call in [cache.stats, cache.close, cache.clear, lambda: execute(buffers(), cache)]:
            try:
                call()
            except pyreloc.TransferError as error:
                if not str(error).startswith('process_mismatch:'):
                    os._exit(2)
            else:
                os._exit(3)
        del call, cache
        gc.collect()
        os._exit(0)
    assert os.waitpid(child, 0)[1] == 0
    execute(buffers(seed=3), cache, gather_threads=2)
    cache.close()


@pytest.mark.parametrize('case', ['close', 'clear', 'timeout', 'borrowed_close',
                                  'allocation', 'complete_failure', 'copy', 'throw_copy',
                                  'wait', 'unknown', 'fork'])
def test_native_thread_failure_and_shutdown_cases(case):
    pytest.importorskip('_reloc_transfer_test', reason='requires the build-only fault harness')
    if case == 'fork' and not hasattr(os, 'fork'):
        pytest.skip('requires fork')
    result = subprocess.run([sys.executable, __file__, case], capture_output=True,
                            text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    case = sys.argv[1]
    if case == 'fork':
        scenario_fork()
    elif case in {'allocation', 'complete_failure', 'copy', 'throw_copy', 'wait', 'unknown'}:
        scenario_failure(case)
    else:
        scenario_lifecycle(case)
