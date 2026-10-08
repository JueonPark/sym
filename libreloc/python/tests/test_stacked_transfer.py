"""Torch-free stacked transfer bindings (torch.stack support)."""
import weakref

import numpy as np
import pytest

from conftest import golden_hex

pyreloc = pytest.importorskip("pyreloc")


def bound_golden(name):
    return pyreloc.bind(pyreloc.load_plan(bytes.fromhex(golden_hex(name))), {})


def split(bound, count, seed=3):
    """The plan's dense logical source cut into `count` equal inputs, each its
    own allocation, plus the single-source relocation of the whole."""
    elements = bound.min_src_bytes // bound.element_size
    assert elements % count == 0
    whole = ((np.arange(bound.min_src_bytes, dtype=np.uint32) * 131 + seed) & 0xFF).astype(np.uint8)
    expected = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(bound, whole.ctypes.data, whole.nbytes, expected.ctypes.data, expected.nbytes)
    inputs = [part.copy() for part in np.split(whole, count)]
    segment = elements // count
    views = [pyreloc.BufferView(part.ctypes.data, part.nbytes, 0, [segment], [1],
                                bound.element_size, "host") for part in inputs]
    return inputs, views, expected


def destination(bound):
    dst = np.full(bound.total_bytes, 0xCD, dtype=np.uint8)
    view = pyreloc.BufferView(dst.ctypes.data, dst.nbytes, 0,
                              [bound.total_bytes // bound.element_size], [1],
                              bound.element_size, "host")
    return dst, view


@pytest.mark.parametrize(("name", "count"), [
    ("identity", 1), ("identity", 4), ("identity", 8),
    ("pad", 2), ("pad", 5), ("pad_nonzero", 3),
])
def test_stacked_host_transfer_matches_the_whole_source(name, count):
    bound = bound_golden(name)
    inputs, views, expected = split(bound, count)
    dst, target = destination(bound)
    assert pyreloc.validate_stacked_sources(bound, views, "h2d") == sum(p.nbytes for p in inputs)
    request = pyreloc.make_stacked_transfer(bound, views, target, "h2d")
    assert request.stack_segment_elements == views[0].extents[0]
    assert [v.base for v in request.stack_sources] == [v.base for v in views]
    pyreloc.execute_transfer(request, owners=(tuple(inputs), dst), n_buffers=2, gather_threads=2)
    np.testing.assert_array_equal(dst, expected)
    assert request.consumed


def test_stacked_validation_errors_carry_stable_codes():
    bound = bound_golden("pad")
    inputs, views, _ = split(bound, 5)
    dst, target = destination(bound)
    with pytest.raises(pyreloc.TransferError, match="^plan_mismatch"):
        pyreloc.validate_stacked_sources(bound, views[:4], "h2d")
    with pytest.raises(pyreloc.TransferError, match="^plan_mismatch"):
        pyreloc.validate_stacked_sources(bound, [], "h2d")
    with pytest.raises(pyreloc.TransferError, match="^direction_mismatch"):
        pyreloc.make_stacked_transfer(bound, views, target, "d2h")


def test_stacked_cached_transfer_releases_owners_and_is_single_use():
    bound = bound_golden("pad")
    inputs, views, expected = split(bound, 5)
    dst, target = destination(bound)
    request = pyreloc.make_stacked_transfer(bound, views, target, "h2d")
    references = [weakref.ref(part) for part in inputs]
    with pyreloc.TransferResourceCache() as cache:
        for owners in [None, (tuple(inputs),), (tuple(inputs), None), [tuple(inputs), dst]]:
            with pytest.raises(ValueError, match="owners"):
                pyreloc.execute_transfer(request, resources=cache, owners=owners)
            assert not request.consumed
        del owners
        pyreloc.execute_transfer(request, resources=cache, owners=(tuple(inputs), dst),
                                 n_buffers=2, gather_threads=2)
        np.testing.assert_array_equal(dst, expected)
        assert request.consumed
        with pytest.raises(pyreloc.TransferError, match="^already_executed"):
            pyreloc.execute_transfer(request, resources=cache, owners=(tuple(inputs), dst))
        # Neither the cache nor the consumed request retains an input.
        del inputs
        assert all(ref() is None for ref in references)
        assert cache.stats()['requests'] == 1
