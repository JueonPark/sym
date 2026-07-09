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
