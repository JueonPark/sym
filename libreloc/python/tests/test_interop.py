"""torch_interop.as_ptr: object -> (ptr, nbytes) without importing torch."""
import numpy as np
import pytest

pytest.importorskip("pyreloc")
from pyreloc.torch_interop import as_ptr


def test_numpy_contiguous():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    ptr, nbytes = as_ptr(a)
    assert ptr == a.ctypes.data
    assert nbytes == 48


def test_numpy_non_contiguous_rejected():
    a = np.arange(12, dtype=np.float32).reshape(3, 4).T
    with pytest.raises(ValueError):
        as_ptr(a)


def test_tuple_passthrough():
    assert as_ptr((0x1000, 64)) == (0x1000, 64)


def test_tuple_malformed_rejected():
    with pytest.raises(ValueError):
        as_ptr((1, 2, 3))
    with pytest.raises(ValueError):
        as_ptr(("p", 2))


def test_unsupported_type_rejected():
    with pytest.raises(TypeError):
        as_ptr("not a buffer")


def test_torch_tensor_if_available():
    torch = pytest.importorskip("torch")
    t = torch.arange(6, dtype=torch.float32)
    ptr, nbytes = as_ptr(t)
    assert ptr == t.data_ptr()
    assert nbytes == 24
    with pytest.raises(ValueError):
        as_ptr(t.reshape(2, 3).T)   # non-contiguous
