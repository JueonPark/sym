"""D1 (issue #65): GatherPool lifecycle + parallel-gather parity through the
pybind surface. Teardown is ASSERTED via the process thread count (/proc,
Linux-only -- which is what CI runs), not assumed."""
import pathlib
import sys

import numpy as np
import pytest

from conftest import golden_hex

pyreloc = pytest.importorskip("pyreloc")


def _thread_count():
    status = pathlib.Path("/proc/self/status").read_text()
    return int(status.split("Threads:")[1].split()[0])


def _bind_reference(n=256, strategy="chunked_pipeline"):
    plan = pyreloc.load_plan(bytes.fromhex(golden_hex("reference")))
    return pyreloc.bind(plan, {"N": n}, strategy=strategy)


def _ptr(arr):
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data, arr.nbytes


def test_gather_pool_thread_count_resolves():
    pool = pyreloc.GatherPool()  # threads=0 -> hardware concurrency
    assert pool.threads >= 1
    pool.close()
    pool = pyreloc.GatherPool(threads=3)
    assert pool.threads == 3
    pool.close()


@pytest.mark.skipif(sys.platform != "linux", reason="/proc thread count")
def test_gather_pool_close_joins_all_threads():
    base = _thread_count()
    pool = pyreloc.GatherPool(threads=4)
    # The pool owns T-1 OS workers; the caller is the T-th worker.
    assert _thread_count() == base + 3
    pool.close()
    assert pool.closed
    assert _thread_count() == base
    pool.close()  # idempotent
    assert _thread_count() == base


@pytest.mark.skipif(sys.platform != "linux", reason="/proc thread count")
def test_gather_pool_context_manager_tears_down():
    base = _thread_count()
    with pyreloc.GatherPool(threads=2) as pool:
        assert not pool.closed
        assert _thread_count() == base + 1
    assert pool.closed
    assert _thread_count() == base


def test_relocate_gather_threads_parity():
    # N=2048 (16 MiB): big enough that the 1 MiB/worker byte floor lets the
    # parallel path actually engage (N=256 would collapse to inline).
    bound = _bind_reference(n=2048)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 255, bound.min_src_bytes, dtype=np.uint8)
    outs = []
    for threads in (1, 2, 0):  # 0 -> hardware concurrency
        dst = np.zeros(bound.total_bytes, dtype=np.uint8)
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst),
                         gather_threads=threads)
        outs.append(dst.tobytes())
    assert outs[0] == outs[1] == outs[2]
    # And byte-identical to the single-thread reference strategy.
    ref_bound = _bind_reference(n=2048, strategy="single_thread_simd")
    ref = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(ref_bound, *_ptr(src), *_ptr(ref))
    assert outs[0] == ref.tobytes()


def test_relocate_with_reused_gather_pool():
    bound = _bind_reference(n=2048)  # see parity test: engages the pool
    rng = np.random.default_rng(1)
    src = rng.integers(0, 255, bound.min_src_bytes, dtype=np.uint8)
    ref = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(bound, *_ptr(src), *_ptr(ref))
    with pyreloc.GatherPool(threads=4) as pool:
        for _ in range(3):  # the SAME pool serves several calls
            dst = np.zeros(bound.total_bytes, dtype=np.uint8)
            pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_pool=pool)
            assert dst.tobytes() == ref.tobytes()


def test_closed_pool_is_rejected():
    bound = _bind_reference()
    src = np.zeros(bound.min_src_bytes, dtype=np.uint8)
    dst = np.zeros(bound.total_bytes, dtype=np.uint8)
    pool = pyreloc.GatherPool(threads=2)
    pool.close()
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_pool=pool)


def test_invalid_thread_counts_rejected():
    bound = _bind_reference()
    src = np.zeros(bound.min_src_bytes, dtype=np.uint8)
    dst = np.zeros(bound.total_bytes, dtype=np.uint8)
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_threads=-1)
    with pytest.raises(ValueError):
        pyreloc.GatherPool(threads=-2)
