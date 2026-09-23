"""R3 typed dispatch through the Torch bridge (issue #147).

``prepare_typed_transfer`` / ``execute_typed_transfer`` over real typed
artifacts: parameter validation and snapshots, policy selection, the forced
CPU baseline and every auto/explicit row on a real GPU (``gpu`` marked),
reports, and staleness after parameter mutation. Expected values come from
a NumPy oracle spelling out C1's arithmetic.
"""
import dataclasses
import types

import numpy as np
import pytest
import torch

import pyreloc


def _spec(shape, dtype):
    from reloc_torch.recipe import TensorSpec
    from reloc_torch.symbolic import Const, dense_strides

    shape = tuple(Const(d) if isinstance(d, int) else d for d in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


@pytest.fixture
def quantize_transpose_recipe():
    """f32 [B, 3] -> per-channel int8 (runtime scale over 3 channels) -> [3, B]."""
    from reloc_torch.recipe import BindingParam, Quantize, Recipe, Transpose
    from reloc_torch.symbolic import Const, Symbol

    b = Symbol("s0")
    quantize = Quantize("int8", BindingParam("s", "float32", (Const(3),)), None, 1, "symmetric_rne")
    return Recipe(_spec((b, 3), "float32"), (quantize, Transpose((1, 0))), _spec((3, b), "int8"), "h2d")


@pytest.fixture
def dequantize_recipe():
    """int8 [N] -> f32 with a runtime per-tensor scale and zero point."""
    from reloc_torch.recipe import BindingParam, Dequantize, Recipe
    from reloc_torch.symbolic import Symbol

    n = Symbol("s0")
    dequantize = Dequantize(
        "float32", BindingParam("s", "float32", ()), BindingParam("zp", "int32", ()), None, "affine"
    )
    return Recipe(_spec((n,), "int8"), (dequantize,), _spec((n,), "float32"), "h2d")


def oracle_quantize(x, scale):
    inv = (np.float32(1) / np.asarray(scale, dtype=np.float32)).astype(np.float32)
    t = (np.asarray(x, dtype=np.float32) * inv).astype(np.float32)
    clamped = np.minimum(np.maximum(t, np.float32(-128)), np.float32(127))
    clamped = np.where(np.isnan(t), np.float32(-128), clamped)
    return np.rint(clamped).astype(np.int8)


def oracle_dequantize(q, zero_point, scale):
    d = (np.asarray(q, dtype=np.int32) - np.int32(zero_point)).astype(np.float32)
    return (d * np.float32(scale)).astype(np.float32)


def api():
    from reloc_torch import dispatch

    return dispatch


#===----------------------------------------------------------------------===#
# Preparation (CPU-visible).
#===----------------------------------------------------------------------===#

def test_layout_only_recipes_are_not_typed_dispatches(compiler, identity_recipe):
    from reloc_torch import UnsupportedRecipe

    compiled = compiler.compile(identity_recipe)
    with pytest.raises(UnsupportedRecipe) as failure:
        api().prepare_typed_transfer(compiled, torch.zeros(8), "cuda", parameters={})
    assert failure.value.reason == "not_typed_recipe"


def test_parameters_are_checked_before_the_device_is(compiler, quantize_transpose_recipe, dequantize_recipe):
    from reloc_torch import UnsupportedRecipe

    compiled = compiler.compile(quantize_transpose_recipe)
    source = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        api().prepare_typed_transfer(compiled, source, "cuda", parameters={}, policy="fastest")

    def reason(parameters, recipe=compiled, src=source):
        with pytest.raises(UnsupportedRecipe) as failure:
            api().prepare_typed_transfer(recipe, src, "cuda", parameters=parameters)
        return failure.value.reason

    assert reason({}) == "parameter_binding"
    assert reason({"s": torch.ones(3), "t": torch.ones(3)}) == "parameter_binding"
    assert reason({"s": torch.ones(3, dtype=torch.float16)}) == "parameter_dtype"
    assert reason({"s": torch.ones(2)}) == "bind_error"  # length 2 against the channel extent 3
    assert reason({"s": torch.tensor([1.0, 0.0, 1.0])}) == "bind_error"  # non-positive scale
    assert reason({"s": torch.ones(3)}, src=torch.zeros(4, 3, dtype=torch.float16)) == "source_descriptor"
    dequant = compiler.compile(dequantize_recipe)
    q = torch.zeros(5, dtype=torch.int8)
    assert reason({"s": torch.tensor(0.3), "zp": torch.tensor(200, dtype=torch.int32)}, dequant, q) == "bind_error"
    assert reason({"s": torch.tensor(0.3), "zp": torch.tensor(0.0)}, dequant, q) == "parameter_dtype"
    # Everything above passes only once the parameters are right; on a
    # CUDA-less host the remaining exclusion is the runtime itself.
    if not (pyreloc.cuda_enabled and torch.cuda.is_available()):
        assert reason({"s": torch.ones(3)}) == "cuda_unavailable"
        assert reason({"s": torch.tensor(0.3), "zp": torch.tensor(0, dtype=torch.int32)}, dequant, q) == "cuda_unavailable"


#===----------------------------------------------------------------------===#
# Execution on a real GPU.
#===----------------------------------------------------------------------===#

def _rows(bound, direction):
    return [row["implementation"] for row in pyreloc.query_capability(bound, direction, "cuda")["eligible"]]


@pytest.mark.gpu
def test_forced_baseline_and_every_qualified_row_match_the_oracle(compiler, quantize_transpose_recipe, cuda_device):
    compiled = compiler.compile(quantize_transpose_recipe)
    source = torch.randn(64, 3) * 40
    source[0, 0], source[0, 1], source[0, 2] = float("nan"), float("inf"), -float("inf")
    scales = torch.tensor([1.0, 0.5, 0.25])
    expected = np.ascontiguousarray(oracle_quantize(source.numpy(), scales.numpy()[None, :]).T)
    seen = []
    for policy in ("original_cpu", "auto"):
        request = api().prepare_typed_transfer(compiled, source, cuda_device, parameters={"s": scales}, policy=policy)
        assert request.capability["eligible"]
        result = api().execute_typed_transfer(request)
        assert result.tensor.device.type == "cuda" and result.tensor.dtype == torch.int8
        assert tuple(result.tensor.shape) == (3, 64)
        np.testing.assert_array_equal(result.tensor.cpu().numpy(), expected)
        report = result.report
        assert isinstance(report, types.MappingProxyType)
        assert report["policy"] == policy
        assert report["executed"] is True
        assert (report["source_bytes"], report["destination_bytes"], report["parameter_bytes"]) == (768, 192, 12)
        assert report["payload_bytes_transferred"] >= report["wire_bytes"] > 0
        assert report["artifact_version"] == 1
        seen.append((report["implementation"], report["placement_reason"]))
        with pytest.raises(RuntimeError, match="already executed"):
            api().execute_typed_transfer(request)
    assert seen[0] == ("cpu_reference", "forced")
    assert seen[1] == ("cpu_reference", "no_calibration")
    # Every row the runtime qualified is bit-identical to the reference.
    for row in _rows(request.bound, "h2d"):
        request = api().prepare_typed_transfer(compiled, source, cuda_device, parameters={"s": scales}, implementation=row)
        result = api().execute_typed_transfer(request)
        np.testing.assert_array_equal(result.tensor.cpu().numpy(), expected, err_msg=row)
        assert result.report["implementation"] == row and result.report["policy"] == "explicit"


@pytest.mark.gpu
def test_device_to_host_runs_the_forward_program(compiler, quantize_transpose_recipe, cuda_device):
    recipe = dataclasses.replace(quantize_transpose_recipe, direction="d2h")
    compiled = compiler.compile(recipe)
    source = (torch.randn(16, 3) * 40).to(cuda_device)
    scales = torch.tensor([1.0, 0.5, 0.25])
    expected = np.ascontiguousarray(oracle_quantize(source.cpu().numpy(), scales.numpy()[None, :]).T)
    for row in _rows(pyreloc.bind_typed(pyreloc.load_typed_plan(compiled.plan_bytes), {"s0": 16},
                                        {"s": ("float32", [3], scales.numpy().tobytes())}), "d2h"):
        request = api().prepare_typed_transfer(compiled, source, "cpu", parameters={"s": scales}, implementation=row)
        result = api().execute_typed_transfer(request)
        assert result.tensor.device.type == "cpu" and result.tensor.dtype == torch.int8
        np.testing.assert_array_equal(result.tensor.numpy(), expected, err_msg=row)
        # A forward D2H program: no inverse-layout scatter was applied.
        assert result.report["wire_boundary"] == request.selected["wire_boundary"]


@pytest.mark.gpu
def test_parameter_values_are_snapshotted_and_rechecked(compiler, dequantize_recipe, cuda_device):
    compiled = compiler.compile(dequantize_recipe)
    q = torch.randint(-128, 128, (100,), dtype=torch.int8)
    scale = torch.tensor(0.3)
    zp = torch.tensor(0, dtype=torch.int32)
    request = api().prepare_typed_transfer(compiled, q, cuda_device, parameters={"s": scale, "zp": zp})
    result = api().execute_typed_transfer(request)
    np.testing.assert_array_equal(result.tensor.cpu().numpy().view(np.uint32),
                                  oracle_dequantize(q.numpy(), 0, 0.3).view(np.uint32))
    # In-place mutation between preparation and execution is stale, whatever
    # the tensor object identity says.
    request = api().prepare_typed_transfer(compiled, q, cuda_device, parameters={"s": scale, "zp": zp})
    zp.fill_(5)
    with pytest.raises(RuntimeError, match="stale typed transfer: parameter 'zp'"):
        api().execute_typed_transfer(request)
    assert not request.consumed
    # A fresh preparation sees the new value and produces fresh results.
    request = api().prepare_typed_transfer(compiled, q, cuda_device, parameters={"s": scale, "zp": zp})
    result = api().execute_typed_transfer(request)
    np.testing.assert_array_equal(result.tensor.cpu().numpy().view(np.uint32),
                                  oracle_dequantize(q.numpy(), 5, 0.3).view(np.uint32))
    assert result.report["implementation"] == "cpu_reference"  # nonzero zero point: no CUDA row
    # An invalid updated scale fails preflight, never execution.
    from reloc_torch import UnsupportedRecipe

    scale.fill_(0.0)
    with pytest.raises(UnsupportedRecipe) as failure:
        api().prepare_typed_transfer(compiled, q, cuda_device, parameters={"s": scale, "zp": zp})
    assert failure.value.reason == "bind_error"
    # Device-resident parameters are an explicit exclusion.
    with pytest.raises(UnsupportedRecipe) as failure:
        api().prepare_typed_transfer(compiled, q, cuda_device, parameters={"s": torch.tensor(0.3).to(cuda_device), "zp": zp})
    assert failure.value.reason == "device_parameters_unavailable"
