"""Policy transitions must preserve bytes, producer ordering and ownership."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import pyreloc
import pytest
import torch

from reloc_torch import TransferResources, UnsupportedRecipe, dispatch, transport
from reloc_torch.recipe import Cast, Recipe, Transpose
from test_pinning_diagnostics import spec
from test_resource_qualification import assert_drained
from test_transport import transpose

pytestmark = pytest.mark.gpu


def choices(wire):
    # Exercise both cache kinds again after the first allocation of each kind.
    return [("auto", None, "pageable"), ("auto", wire + 1, "pageable"),
            ("auto", wire, "pinned"), ("auto", wire - 1, "pinned"),
            ("pageable", 0, "pageable"), ("pinned", wire + 1, "pinned")]


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
@pytest.mark.parametrize("shape", [(8, 4), (2048, 1024)])
def test_layout_boundaries_streams_and_outputs(compiler, cuda_device, direction, shape):
    compiled = compiler.compile(transpose(direction=direction))
    wire = shape[0] * shape[1] * 4
    owner = TransferResources()
    streams = [torch.cuda.Stream(device=cuda_device) for _ in range(2)]
    held = []
    try:
        for i, (mode, threshold, kind) in enumerate(choices(wire)):
            cpu = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) + i
            with torch.cuda.stream(streams[i % 2]):
                src = cpu if direction == "h2d" else cpu.to(cuda_device)
                src.add_(1)  # D2H must wait for this producer on the caller stream.
                request = transport.prepare_transfer(compiled, src,
                    cuda_device if direction == "h2d" else "cpu")
                out = transport.execute_transfer(request, resources=owner, pinning=mode,
                    min_pinned_bytes=threshold, gather_threads=4)
            expected = (cpu if direction == "h2d" else cpu + 1).t().contiguous()
            row, = request.staging
            assert (row["memory_kind"], row["wire_bytes"]) == (kind, wire)
            assert row["reused"] == (i not in (0, 2))
            assert row["staging_capacity_bytes"] > 0
            with torch.cuda.stream(streams[(i + 1) % 2]):
                assert torch.equal((out + 3).cpu(), expected + 3)
            held.append((out, expected))
        assert len({out.data_ptr() for out, _ in held}) == len(held)
        for out, expected in held:
            assert torch.equal(out.cpu(), expected)
        owner.clear()
        assert_drained(owner.stats())
    finally:
        owner.close()
    assert_drained(owner.stats())


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_typed_cast_gates_actual_wire_bytes(compiler, cuda_device, direction):
    source_dtype, destination_dtype = (("float32", "float16") if direction == "h2d"
                                      else ("float16", "float32"))
    recipe = Recipe(spec((8, 4), source_dtype),
        (Transpose((1, 0)), Cast(destination_dtype, "ieee_rne" if direction == "h2d" else "exact")),
        spec((4, 8), destination_dtype), direction)
    compiled = compiler.compile(recipe)
    # CPU placement sends the narrowed H2D result or receives the narrow D2H input.
    wire = 64
    with TransferResources() as owner:
        held = []
        for i, (mode, threshold, kind) in enumerate(choices(wire)):
            cpu = (torch.arange(32).reshape(8, 4) + i).to(getattr(torch, source_dtype))
            src = cpu if direction == "h2d" else cpu.to(cuda_device)
            request = dispatch.prepare_typed_transfer(compiled, src,
                cuda_device if direction == "h2d" else "cpu", policy="original_cpu")
            result = dispatch.execute_typed_transfer(request, resources=owner, pinning=mode,
                min_pinned_bytes=threshold)
            row, = request.staging
            assert result.report["wire_bytes"] == row["wire_bytes"] == wire
            assert row["memory_kind"] == kind
            assert row["reused"] == (i not in (0, 2))
            held.append((result.tensor, cpu.t().to(getattr(torch, destination_dtype))))
        for out, expected in held:
            assert torch.equal(out.cpu(), expected)
    stats = owner.stats()["typed"]
    assert stats["retained_bytes"] == 0
    assert stats["host_allocations"] + stats["device_allocations"] == stats["frees"]


@pytest.mark.parametrize("retained_limit", [0, 128 << 10])
@pytest.mark.parametrize("mode,kind", [("auto", "pageable"), ("pinned", "pinned"),
                                       ("pageable", "pageable")])
def test_unretainable_layout_and_ephemeral_paths(compiler, cuda_device, retained_limit, mode, kind):
    compiled = compiler.compile(transpose())
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    owner = TransferResources(max_retained_bytes=retained_limit)
    try:
        for resources in (owner, None):
            request = transport.prepare_transfer(compiled, source, cuda_device)
            out = transport.execute_transfer(request, resources=resources, pinning=mode,
                                             min_pinned_bytes=1)
            row, = request.staging
            assert row["memory_kind"] == kind
            assert not row["retention_eligible"]
            assert not row["reused"]
            assert torch.equal(out.cpu(), source.t())
    finally:
        owner.close()
    assert_drained(owner.stats())


def test_zero_extent_rejected_before_allocation(compiler, cuda_device):
    compiled = compiler.compile(transpose())
    with TransferResources() as owner:
        before = owner.stats()
        with pytest.raises(UnsupportedRecipe):
            transport.prepare_transfer(compiled, torch.empty((0, 4)), cuda_device)
        assert owner.stats() == before


def test_concurrent_typed_policy_reports_are_request_local(compiler, cuda_device):
    compiled = compiler.compile(Recipe(spec((32, 4)), (Cast("float16", "ieee_rne"),),
                                       spec((32, 4), "float16"), "h2d"))
    with TransferResources() as owner:
        def work(i):
            mode = "pinned" if i % 2 else "pageable"
            source = torch.full((32, 4), i, dtype=torch.float32)
            request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                                                       policy="original_cpu")
            result = dispatch.execute_typed_transfer(request, resources=owner, pinning=mode)
            assert torch.equal(result.tensor.cpu(), source.half())
            row, = request.staging
            assert row["memory_kind"] == mode
            assert row["reason"] == "forced_" + mode
            return result.tensor
        with ThreadPoolExecutor(4) as workers:
            outputs = list(workers.map(work, range(16)))
        assert len({out.data_ptr() for out in outputs}) == len(outputs)


@pytest.mark.parametrize("path", ["layout", "typed"])
@pytest.mark.parametrize("pinning", ["pinned", "pageable"])
@pytest.mark.parametrize("mode", [1, 2])
def test_completion_faults_in_each_memory_kind(cuda_device, path, pinning, mode):
    shim = Path(pyreloc.__file__).resolve().parent.parent / "libtyped_dispatch_faults.so"
    if not shim.exists():
        pytest.skip("test-only CUDA fault shim was not built")
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    scenario = Path(__file__).with_name("pinning_fault_scenario.py")
    result = subprocess.run([sys.executable, str(scenario), path, pinning, str(mode)],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_fused_weights_do_not_pin_tiny_scale_upload(compiler, cuda_device):
    from reloc_torch.recipe import Dequantize, InlineParam
    shape = (2048, 1024)
    recipe = Recipe(spec(shape, "int8"),
        (Dequantize("float32", InlineParam("float32", (), (0x3f000000,)), None, None, "affine"),
         Transpose((1, 0))), spec(tuple(reversed(shape))), "h2d")
    compiled = compiler.compile(recipe)
    with TransferResources() as owner:
        for i in range(2):
            source = torch.full(shape, i + 3, dtype=torch.int8)
            request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                                                       implementation="cuda_dequant_relocate")
            result = dispatch.execute_typed_transfer(request, resources=owner,
                                                     min_pinned_bytes=64 << 10)
            row, = request.staging
            # Fused ABI broadcasts the scalar to one FP32 scale per input column.
            assert row["wire_bytes"] == shape[1] * 4
            assert row["memory_kind"] == "pageable"
            assert row["reason"] == "below_threshold"
            assert row["reused"] == bool(i)
            assert result.report["wire_bytes"] >= source.numel()
            assert torch.equal(result.tensor.cpu(), source.t().float() * .5)
