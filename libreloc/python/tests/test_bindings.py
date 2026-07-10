"""Unit tests for the pyreloc binding surface, against the frozen golden
wire blobs pinned in test/dialect/reloc/serialize.mlir."""
import pytest

from conftest import golden_hex

pyreloc = pytest.importorskip("pyreloc")


def test_load_plan_golden_identity():
    plan = pyreloc.load_plan(bytes.fromhex(golden_hex("identity")))
    assert plan.symbols == []
    assert plan.element_size == 4  # i32
    assert plan.num_axes == 1
    assert plan.no_copy is False
    assert "PlanHandle" in repr(plan)


def test_load_plan_golden_reference_symbols():
    plan = pyreloc.load_plan(bytes.fromhex(golden_hex("reference")))
    assert plan.symbols == ["N"]
    assert plan.num_axes == 4


def test_load_plan_garbage_raises_decode_error():
    with pytest.raises(pyreloc.DecodeError) as exc:
        pyreloc.load_plan(b"not a plan")
    assert str(exc.value)  # carries the C++ diagnostic


def test_load_plan_truncated_raises_decode_error():
    blob = bytes.fromhex(golden_hex("identity"))
    with pytest.raises(pyreloc.DecodeError):
        pyreloc.load_plan(blob[: len(blob) // 2])


def test_decode_error_is_exception_subclass():
    assert issubclass(pyreloc.DecodeError, Exception)


import numpy as np


def _bind_golden(name, symbols, **kw):
    return pyreloc.bind(pyreloc.load_plan(bytes.fromhex(golden_hex(name))),
                        symbols, **kw)


def _ptr(arr):
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data, arr.nbytes


def test_bind_static_identity():
    bound = _bind_golden("identity", {})
    assert bound.extents == [8]
    assert bound.element_size == 4
    assert bound.total_bytes == 32
    assert bound.valid_elements == 8
    assert bound.min_src_bytes == 32
    assert bound.strategy == "single_thread_simd"


def test_bind_reference_ok_and_errors():
    bound = _bind_golden("reference", {"N": 4096})
    assert bound.total_bytes == 4096 * 4096 * 4
    with pytest.raises(pyreloc.BindError):        # divisibility violated
        _bind_golden("reference", {"N": 100})
    with pytest.raises(pyreloc.BindError):        # missing symbol
        _bind_golden("reference", {})
    with pytest.raises(pyreloc.BindError):        # extra symbol
        _bind_golden("reference", {"N": 4096, "M": 1})
    with pytest.raises(pyreloc.BindError) as exc:
        _bind_golden("reference", {"N": 100})
    assert str(exc.value)  # carries the C++ diagnostic


def test_bind_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        _bind_golden("identity", {}, strategy="warp_drive")


def test_relocate_identity_bytes_pass_through():
    bound = _bind_golden("identity", {})
    src = np.arange(8, dtype=np.int32)
    dst = np.zeros(8, dtype=np.int32)
    pyreloc.relocate(bound, *_ptr(src), *_ptr(dst))
    assert dst.tobytes() == src.tobytes()


def test_relocate_pad_fills_and_copies():
    bound = _bind_golden("pad", {})           # [30] f32 -> [32], fill 0.0
    src = np.random.default_rng(0).random(30, dtype=np.float32)
    dst = np.full(32, np.nan, dtype=np.float32)
    pyreloc.relocate(bound, *_ptr(src), *_ptr(dst))
    expected = np.pad(src, (0, 2), constant_values=np.float32(0.0))
    assert dst.tobytes() == expected.tobytes()


def test_relocate_strategies_bit_identical():
    src = np.arange(8, dtype=np.int32)
    outs = []
    for strategy in ("single_thread_simd", "multi_thread_tiled",
                     "chunked_pipeline"):
        bound = _bind_golden("identity", {}, strategy=strategy)
        assert bound.strategy == strategy
        dst = np.zeros(8, dtype=np.int32)
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst))
        outs.append(dst.tobytes())
    assert outs[0] == outs[1] == outs[2]


def test_relocate_inverse_round_trips():
    bound = _bind_golden("pad", {})
    src = np.random.default_rng(1).random(30, dtype=np.float32)
    dst = np.zeros(32, dtype=np.float32)
    pyreloc.relocate(bound, *_ptr(src), *_ptr(dst))
    back = np.zeros(30, dtype=np.float32)
    pyreloc.relocate_inverse(bound, *_ptr(dst), *_ptr(back))
    assert back.tobytes() == src.tobytes()


def test_relocate_validates_buffer_sizes():
    bound = _bind_golden("identity", {})
    src = np.arange(8, dtype=np.int32)
    small = np.zeros(4, dtype=np.int32)
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, *_ptr(src), *_ptr(small))   # dst too small
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, small.ctypes.data, small.nbytes,
                         *_ptr(np.zeros(8, dtype=np.int32)))  # src too small


def test_h2d_d2h_raise_without_cuda():
    if pyreloc.cuda_enabled:
        pytest.skip("CUDA build: covered by test_gpu.py")
    bound = _bind_golden("identity", {})
    buf = np.zeros(8, dtype=np.int32)
    with pytest.raises(RuntimeError):
        pyreloc.h2d(bound, *_ptr(buf), *_ptr(buf))
    with pytest.raises(RuntimeError):
        pyreloc.d2h(bound, *_ptr(buf), *_ptr(buf))
