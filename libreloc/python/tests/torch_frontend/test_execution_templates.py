"""Cache reuse must not reuse input ownership, validation or execution state."""
import dataclasses
import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import torch

import pyreloc
from reloc_torch import UnsupportedRecipe, dispatch
from reloc_torch.runtime import source_reason


@pytest.fixture
def matrix(compiler):
    from reloc_torch.recipe import BindingParam, Dequantize, Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol, dense_strides

    rows, cols = Symbol("s0"), Symbol("s1")
    def spec(shape, dtype):
        return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    return compiler.compile(Recipe(
        spec((rows, cols), "int8"),
        (Dequantize("float32", BindingParam("scale", "float32", (cols,)), None, 1, "affine"),
         Transpose((1, 0))), spec((cols, rows), "float32"), "h2d"))


def prepare(compiled, src, scale, device="cuda:0", **options):
    return dispatch.prepare_typed_transfer(compiled, src, device,
                                          parameters={"scale": scale}, **options)


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_live_admission_matches_observation_admission(grad_enabled):
    from reloc_torch import compat
    from reloc_torch.eligibility import _metadata_reason

    x = torch.ones(3, 4)
    tensors = [x, x.t(), x[:, ::2], x[1:], x[:0], torch.tensor(1.),
               torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.int8),
               torch.ones(3, dtype=torch.float16), torch.ones(3, requires_grad=True),
               torch.nn.Parameter(torch.ones(3)), torch.ones(2, device="meta"),
               torch.ones(2, 2).to_sparse(), torch.empty_strided((1, 3), (999, 1))]
    with torch.set_grad_enabled(grad_enabled):
        for tensor in tensors:
            meta = compat.tensor_metadata(tensor)
            if meta.requires_grad and not grad_enabled:
                meta = dataclasses.replace(meta, requires_grad=False)
            assert source_reason(tensor) == _metadata_reason(meta)


def test_frontend_scope_rechecks_storage_and_gradient_state(compiler, identity_recipe):
    from reloc_torch.runtime import (
        _validated_binding, _validated_preparation, bind_symbols, destination_descriptor,
    )
    compiled = compiler.compile(identity_recipe)
    src = torch.ones(8)
    bindings = bind_symbols(compiled, src)
    dest = destination_descriptor(compiled, bindings, torch.device("cpu"))
    with _validated_binding(compiled, src, bindings, dest):
        assert _validated_preparation(compiled, src) is not None
        assert _validated_preparation(compiled, src.clone()) is None
        src.requires_grad_(True)
        assert _validated_preparation(compiled, src) is None
        src.requires_grad_(False)
        src.resize_(4)
        assert _validated_preparation(compiled, src) is None
    assert _validated_preparation(compiled, src) is None


def test_layout_cache_evicts_and_never_owns_source(compiler, identity_recipe):
    from reloc_torch.runtime import bind_symbols, bind_plan

    compiled = compiler.compile(identity_recipe)
    for n in range(1, 70):
        src = torch.ones(n)
        bound = bind_plan(compiled, bind_symbols(compiled, src))
        assert bind_plan(compiled, bind_symbols(compiled, src)) is bound
        ref = weakref.ref(src)
        del src
        assert ref() is None
    info = compiled.execution_cache_info()
    assert info["entries"] == 32 and info["evictions"] == 37
    assert info["hits"] == 69
    compiled.clear_execution_cache()
    assert compiled.execution_cache_info()["entries"] == 0


def test_cache_fork_does_not_acquire_inherited_lock():
    if not hasattr(os, "fork"):
        pytest.skip("fork unavailable")
    # Separate CPU-only process so CUDA initialization elsewhere cannot affect it.
    code = '''
import os
from reloc_torch.execution_templates import TemplateCache
cache = TemplateCache()
cache.put(('parent',), object())
cache._lock.acquire()
pid = os.fork()
if pid == 0:
    assert cache.info()['entries'] == 0
    cache.put(('child',), object())
    os._exit(0)
_, status = os.waitpid(pid, 0)
cache._lock.release()
assert status == 0 and cache.info()['entries'] == 1
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


@pytest.mark.gpu
def test_typed_hits_read_fresh_bytes_and_produce_independent_outputs(matrix, cuda_device):
    from reloc_torch import TransferResources

    outputs = []
    with TransferResources() as owner:
        for i, scale_value in enumerate((.5, 2., .5, 2.)):
            source = torch.full((16, 32), i + 1, dtype=torch.int8)
            scale = torch.full((32,), scale_value)
            request = prepare(matrix, source, scale, cuda_device, implementation="cuda_dequant_relocate")
            result = dispatch.execute_typed_transfer(request, resources=owner)
            expected = source.t().contiguous().float() * scale_value
            assert torch.equal(result.tensor.cpu(), expected)
            outputs.append((result.tensor, expected))
            with pytest.raises(RuntimeError, match="already executed"):
                dispatch.execute_typed_transfer(request, resources=owner)
            source_ref, scale_ref = weakref.ref(source), weakref.ref(scale)
            del source, scale, request, result
            gc.collect()
            assert source_ref() is None and scale_ref() is None
        assert matrix.execution_cache_info()["hits"] == 2
        assert matrix.execution_cache_info()["entries"] == 2
        for actual, expected in outputs:
            assert torch.equal(actual.cpu(), expected)
        source, scale = torch.ones(16, 32, dtype=torch.int8), torch.full((32,), .5)
        request = prepare(matrix, source, scale, cuda_device, implementation="cuda_dequant_relocate")
        matrix.clear_execution_cache()
        assert torch.equal(dispatch.execute_typed_transfer(request, resources=owner).tensor.cpu(),
                           source.t().float() * .5)


@pytest.mark.gpu
def test_warm_typed_cache_does_not_bypass_guards_or_leak_mutable_reports(matrix, cuda_device):
    src, scale = torch.ones(8, 32, dtype=torch.int8), torch.ones(32)
    first = prepare(matrix, src, scale, cuda_device, threads=1)
    first.capability["eligible"].clear()
    first.selected["implementation"] = "invalid"
    second = prepare(matrix, src, scale, cuda_device, threads=1)
    assert second.capability["eligible"] and second.selected["implementation"] == "cpu_reference"
    for invalid in (0., float("nan"), float("inf")):
        with pytest.raises(UnsupportedRecipe, match="scale"):
            prepare(matrix, src, torch.full((32,), invalid), cuda_device, threads=1)
    for options, error in [({"threads": 1.0}, TypeError), ({"threads": 0}, ValueError),
                           ({"calibration": {}}, TypeError), ({"implementation": 0}, TypeError)]:
        with pytest.raises(error):
            prepare(matrix, src, scale, cuda_device, **options)
    with pytest.raises(UnsupportedRecipe):
        prepare(matrix, src[:, ::2], scale, cuda_device)
    with pytest.raises(UnsupportedRecipe):
        prepare(matrix, src, scale.half(), cuda_device)
    request = prepare(matrix, src, scale, cuda_device, threads=1)
    scale.fill_(.25)
    with pytest.raises(RuntimeError, match="stale typed transfer"):
        dispatch.execute_typed_transfer(request)
    request = prepare(matrix, src, scale, cuda_device, threads=1)
    src.resize_(4, 64)
    with pytest.raises(RuntimeError, match="stale typed transfer"):
        dispatch.execute_typed_transfer(request)


@pytest.mark.gpu
def test_selection_key_covers_calibration_policy_threads_row_and_device(matrix, cuda_device, tmp_path):
    root = Path(__file__).resolve().parents[4]
    calibration_text = (root / "calibration/epyc7351-2080ti.cal").read_text()
    paths = [tmp_path / "a.cal", tmp_path / "b.cal"]
    paths[0].write_text(calibration_text)
    import re
    paths[1].write_text(re.sub(r"overhead.b_ms\s+\S+", "overhead.b_ms 10000", calibration_text))
    calibrations = [pyreloc.load_calibration(str(p)) for p in paths]
    assert calibrations[0].machine == calibrations[1].machine
    src, scale = torch.ones(16, 32, dtype=torch.int8), torch.ones(32)
    options = [{}, {"threads": 1}, {"policy": "original_cpu"},
               {"implementation": "cuda_dequant_relocate"},
               *[{"calibration": c} for c in calibrations]]
    for settings in options:
        req = prepare(matrix, src, scale, cuda_device, **settings)
        expected = pyreloc.select_dispatch(req.bound, "h2d", "cuda", **settings)
        assert req.selected == expected
        assert prepare(matrix, src, scale, cuda_device, **settings).selected == expected
    assert matrix.execution_cache_info()["entries"] == len(options)
    if torch.cuda.device_count() > 1:
        other = (cuda_device.index + 1) % torch.cuda.device_count()
        req = prepare(matrix, src, scale, f"cuda:{other}")
        assert req.device.index == other
        assert matrix.execution_cache_info()["entries"] == len(options) + 1


@pytest.mark.gpu
def test_parameter_cache_has_entry_and_byte_bounds_and_oversized_bypass(matrix, cuda_device):
    src = torch.ones(2, 4096, dtype=torch.int8)
    for i in range(40):
        prepare(matrix, src, torch.full((4096,), float(i + 1)), cuda_device)
        info = matrix.execution_cache_info()
        assert info["entries"] <= 32
        assert info["parameter_key_bytes"] <= 256 << 10
    assert info["entries"] == 16 and info["evictions"] == 24
    scale = torch.ones(70000)
    request = prepare(matrix, torch.ones(2, 70000, dtype=torch.int8), scale, cuda_device)
    assert matrix.execution_cache_info()["entries"] == 16
    assert matrix.execution_cache_info()["bypasses"] == 1
    ref = weakref.ref(scale)
    del scale, request
    gc.collect()
    assert ref() is None


@pytest.mark.gpu
def test_concurrent_template_hits_keep_requests_streams_and_outputs_independent(matrix, cuda_device):
    from concurrent.futures import ThreadPoolExecutor
    from reloc_torch import TransferResources

    def run(i):
        with torch.cuda.device(cuda_device), TransferResources() as owner:
            stream = torch.cuda.Stream(device=cuda_device)
            with torch.cuda.stream(stream):
                src, scale = torch.full((64, 96), i + 1, dtype=torch.int8), torch.full((96,), .5)
                req = prepare(matrix, src, scale, cuda_device, implementation="cuda_dequant_relocate")
                out = dispatch.execute_typed_transfer(req, resources=owner).tensor
            stream.synchronize()
            return out, src.t().float() * .5
    # Populate once, then share the immutable template across concurrent owners.
    run(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(12)))
    assert matrix.execution_cache_info()["entries"] == 1
    assert matrix.execution_cache_info()["hits"] == 12
    assert len({out.data_ptr() for out, _ in outputs}) == 12
    for out, expected in outputs:
        assert torch.equal(out.cpu(), expected)
