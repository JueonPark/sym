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


@pytest.fixture
def dequantize_matrix_recipe(dequantize_recipe):
    from reloc_torch.recipe import Transpose
    from reloc_torch.symbolic import Symbol
    n = Symbol("s0")
    return dataclasses.replace(dequantize_recipe,
        source=_spec((n, 2), "int8"),
        destination=_spec((2, n), "float32"),
        operations=(*dequantize_recipe.operations, Transpose((1, 0))))


@pytest.mark.gpu
def test_typed_resources_refresh_inputs_and_parameters_and_bound_retention(
        compiler, dequantize_matrix_recipe, cuda_device):
    import weakref
    from reloc_torch import TransferResources

    compiled = compiler.compile(dequantize_matrix_recipe)
    retained = []
    with TransferResources(max_typed_retained_bytes=8192) as resources:
        for n, scale in [(4096, .25), (2048, .5), (4096, 2.)]:
            src = (torch.arange(n) % 127).to(torch.int8).reshape(-1, 2)
            params = {'s': torch.tensor(scale), 'zp': torch.tensor(0, dtype=torch.int32)}
            request = api().prepare_typed_transfer(compiled, src, cuda_device,
                parameters=params, implementation='cuda_dequant_relocate', threads=1)
            ref = weakref.ref(src)
            result = api().execute_typed_transfer(request, resources=resources)
            expected = src.t().contiguous().float() * scale
            assert torch.equal(result.tensor.cpu(), expected)
            retained.append((result.tensor, expected))
            del src, request, result
            assert ref() is None  # scratch cache never keeps successful inputs
        stats = resources.stats()['typed']
        assert stats['hits'] == 2 and stats['context_creations'] == 1
        assert stats['device_allocations'] == 2  # wire + freshly uploaded scalar
        assert stats['retained_bytes'] <= 8192
        for output, expected in retained:
            assert torch.equal(output.cpu(), expected)  # outputs never recycled
        resources.clear()
        assert resources.stats()['typed']['retained_bytes'] == 0
    assert resources.stats()['typed']['closed']
    assert resources.stats()['typed']['streams'] == 0


@pytest.mark.gpu
def test_typed_zero_retention_live_limit_and_closed_owner(compiler, dequantize_matrix_recipe, cuda_device):
    from reloc_torch import TransferResources
    compiled = compiler.compile(dequantize_matrix_recipe)
    src = torch.arange(64, dtype=torch.int8).reshape(32, 2)
    def prepare():
        return api().prepare_typed_transfer(compiled, src, cuda_device,
            parameters={'s': torch.tensor(.25), 'zp': torch.tensor(0, dtype=torch.int32)},
            implementation='cuda_dequant_relocate', threads=1)
    with TransferResources(max_typed_retained_bytes=0) as owner:
        for _ in range(2):
            assert torch.equal(api().execute_typed_transfer(prepare(), resources=owner).tensor.cpu(), src.t().contiguous().float() * .25)
            assert owner.stats()['typed']['retained_bytes'] == 0
        assert owner.stats()['typed']['device_allocations'] == 4
    with pytest.raises(RuntimeError, match='resources_closed'):
        api().execute_typed_transfer(prepare(), resources=owner)
    with TransferResources(max_typed_live_bytes=32) as limited:
        req = prepare()
        with pytest.raises(RuntimeError, match='limit exceeded'):
            api().execute_typed_transfer(req, resources=limited)
        assert req.consumed
        assert limited.stats()['typed']['retained_bytes'] == 0
        assert not limited.stats()['typed']['quarantined']


@pytest.mark.gpu
def test_typed_concurrent_requests_share_exclusive_resources(compiler, dequantize_matrix_recipe, cuda_device):
    from concurrent.futures import ThreadPoolExecutor
    from reloc_torch import TransferResources
    compiled = compiler.compile(dequantize_matrix_recipe)
    with TransferResources() as resources:
        def transfer(i):
            src = torch.full((4096, 2), i, dtype=torch.int8)
            request = api().prepare_typed_transfer(compiled, src, cuda_device,
                parameters={'s': torch.tensor(float(i)), 'zp': torch.tensor(0, dtype=torch.int32)},
                implementation='cuda_dequant_relocate', threads=1)
            return api().execute_typed_transfer(request, resources=resources).tensor.cpu(), i
        with ThreadPoolExecutor(4) as pool:
            for output, i in pool.map(transfer, range(1, 9)):
                assert torch.equal(output, torch.full((2, 4096), float(i * i)))
        assert resources.stats()['typed']['requests'] == 8


def test_metadata_cache_keeps_guards_and_returns_fresh_bindings(compiler, identity_recipe, monkeypatch):
    from reloc_torch import artifact
    from reloc_torch.symbolic import GuardError
    compiled = compiler.compile(identity_recipe)
    calls = []
    original = artifact.bind_recipe
    def observed(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(artifact, 'bind_recipe', observed)
    src = torch.zeros(16)
    first = compiled.bind_values(src)
    first.clear()
    assert compiled.bind_values(src)  # a fresh mapping, not the previous dict
    assert len(calls) == 1
    assert compiled.decoded_plan is compiled.decoded_plan
    with pytest.raises(GuardError):
        compiled.bind_values(src[::2])
    with pytest.raises(GuardError):
        compiled.bind_values(src.half())
    assert len(calls) == 3


@pytest.mark.gpu
@pytest.mark.parametrize('mode', [1, 2])
def test_typed_completion_faults_retain_exactly_the_required_owners(cuda_device, mode):
    import os
    import subprocess
    import sys
    from pathlib import Path
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    if not shim.exists():
        pytest.skip('test-only CUDA fault shim was not built')
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    scenario = Path(__file__).with_name('typed_fault_scenario.py')
    result = subprocess.run([sys.executable, str(scenario), str(mode)], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
def test_typed_owner_switches_devices_and_restores_callers_device(compiler, dequantize_matrix_recipe):
    if torch.cuda.device_count() < 2:
        pytest.skip('needs two CUDA devices')
    from reloc_torch import TransferResources
    compiled = compiler.compile(dequantize_matrix_recipe)
    src = torch.arange(128, dtype=torch.int8).reshape(64, 2)
    with torch.cuda.device(0), TransferResources() as owner:
        for ordinal in [0, 1, 0]:
            request = api().prepare_typed_transfer(compiled, src, f'cuda:{ordinal}',
                parameters={'s': torch.tensor(.5), 'zp': torch.tensor(0, dtype=torch.int32)},
                implementation='cuda_dequant_relocate', threads=1)
            out = api().execute_typed_transfer(request, resources=owner).tensor
            assert out.device.index == ordinal
            assert torch.cuda.current_device() == 0
            assert torch.equal(out.cpu(), src.t().contiguous().float() * .5)
        assert owner.stats()['typed']['context_creations'] == 3
