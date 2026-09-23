"""R2 transport adapter: preflight without allocation, forward transfers, streams, lifetime."""
import dataclasses
import gc
import subprocess
import sys
import threading
import time
import weakref

import numpy as np
import pytest
import torch


def transport():
    from reloc_torch import transport as module

    return module


def unsupported():
    from reloc_torch import UnsupportedRecipe

    return UnsupportedRecipe


def _recipe(source_shape, operations, destination_shape, dtype="float32", direction="h2d"):
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import dense_strides

    return Recipe(
        TensorSpec(tuple(source_shape), dense_strides(tuple(source_shape)), _const(0), dtype),
        tuple(operations),
        TensorSpec(tuple(destination_shape), dense_strides(tuple(destination_shape)), _const(0), dtype),
        direction,
    )


def _const(value):
    from reloc_torch.symbolic import Const

    return Const(value)


def _symbols(*names):
    from reloc_torch.symbolic import Symbol

    return tuple(Symbol(name) for name in names)


def identity(dtype="float32", direction="h2d", rank=1):
    shape = _symbols(*[f"s{i}" for i in range(rank)])
    return _recipe(shape, (), shape, dtype, direction)


def transpose(dtype="float32", direction="h2d"):
    from reloc_torch.recipe import Transpose

    rows, columns = _symbols("s0", "s1")
    return _recipe((rows, columns), (Transpose((1, 0)),), (columns, rows), dtype, direction)


def permute_120(direction="d2h"):
    from reloc_torch.recipe import Transpose

    a, b, c = _symbols("s0", "s1", "s2")
    return _recipe((a, b, c), (Transpose((1, 2, 0)),), (b, c, a), "float32", direction)


def pad_recipe(dtype="float32", direction="h2d"):
    from reloc_torch.recipe import Fill, Pad, Transpose
    from reloc_torch.symbolic import Const, operation_shape

    rows, columns = _symbols("s0", "s1")
    fill = {"float32": Fill("float32", 0x3F800000), "float16": Fill("float16", 0x3C00), "int8": Fill("int8", 0x7F)}[dtype]
    operations = (Transpose((1, 0)), Pad(0, Const(1), Const(2), fill))
    # Derive the destination exactly as the importer does so the emitted
    # types agree with the compiler's canonical expression order.
    shape = (rows, columns)
    for operation in operations:
        shape = operation_shape(shape, operation)
    return _recipe((rows, columns), operations, shape, dtype, direction)


TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "int8": torch.int8}


def reference(name, x):
    """PyTorch/NumPy oracle for each recipe, computed on the source's device."""
    if name == "identity":
        return x.clone()
    if name == "transpose":
        return x.t().contiguous()
    if name == "split_transpose":
        return x.reshape(-1, 64).t().contiguous()
    if name == "permute_120":
        return x.permute(1, 2, 0).contiguous()
    if name == "pad":
        fill = {torch.float32: 1.0, torch.float16: 1.0, torch.int8: 127}[x.dtype]
        return torch.nn.functional.pad(x.t().contiguous(), (0, 0, 1, 2), value=fill)
    raise KeyError(name)


def _values(shape, dtype, device="cpu"):
    numel = int(np.prod(shape))
    return (torch.arange(numel) % 120 - 60).to(dtype).reshape(shape).to(device)


# ------------------------------------------------------------------ Task 1: preflight


def test_transfer_bindings_import_without_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pyreloc; pyreloc.BufferView; pyreloc.make_transfer; pyreloc.execute_transfer; "
            "pyreloc.validate_transfer_source; pyreloc.TransferError; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_native_capacity_and_overflow_checks_are_reachable_from_python(compiler):
    import pyreloc

    compiled = compiler.compile(transpose())
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    bound = pyreloc.bind(pyreloc.load_plan(compiled.plan_bytes), compiled.bind_values(x))
    storage = x.untyped_storage()
    good = pyreloc.BufferView(storage.data_ptr(), storage.nbytes(), 0, [4, 6], [6, 1], 4, "host")
    assert pyreloc.validate_transfer_source(bound, good, "h2d") == 96
    assert good.span_bytes == 96 and good.kind == "host" and good.device == -1
    short = pyreloc.BufferView(storage.data_ptr(), storage.nbytes() - 1, 0, [4, 6], [6, 1], 4, "host")
    with pytest.raises(pyreloc.TransferError, match="^insufficient_capacity"):
        pyreloc.validate_transfer_source(bound, short, "h2d")
    huge = pyreloc.BufferView(storage.data_ptr(), 2**63, 0, [2**61, 6], [6, 1], 4, "host")
    with pytest.raises(pyreloc.TransferError, match="^integer_overflow"):
        pyreloc.validate_transfer_source(bound, huge, "h2d")
    overlapping = pyreloc.BufferView(storage.data_ptr(), storage.nbytes(), 0, [4, 6], [1, 1], 4, "host")
    with pytest.raises(pyreloc.TransferError, match="^unsupported_layout"):
        pyreloc.validate_transfer_source(bound, overlapping, "h2d")
    wrong = pyreloc.BufferView(storage.data_ptr(), storage.nbytes(), 0, [2, 6], [6, 1], 4, "host")
    with pytest.raises(pyreloc.TransferError, match="^plan_mismatch"):
        pyreloc.validate_transfer_source(bound, wrong, "h2d")
    with pytest.raises(ValueError):
        pyreloc.BufferView(storage.data_ptr(), storage.nbytes(), 0, [4, 6], [6, 1], 4, "gpu")


def test_native_host_to_host_forward_transfer_matches_reference_and_is_single_use(compiler):
    import pyreloc

    compiled = compiler.compile(transpose())
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    bound = pyreloc.bind(pyreloc.load_plan(compiled.plan_bytes), compiled.bind_values(x))
    out = torch.full((6, 4), -1.0)
    src = pyreloc.BufferView(x.untyped_storage().data_ptr(), x.untyped_storage().nbytes(), 0, [4, 6], [6, 1], 4, "host")
    dst = pyreloc.BufferView(out.untyped_storage().data_ptr(), out.untyped_storage().nbytes(), 0, [6, 4], [4, 1], 4, "host")
    for direction in ("h2d", "d2h"):
        out.fill_(-1.0)
        request = pyreloc.make_transfer(bound, src, dst, direction)
        assert request.direction == direction
        assert request.source_span_bytes == 96 and request.destination_bytes == 96
        assert request.consumed is False
        pyreloc.execute_transfer(request, caller_stream=None, n_buffers=2)
        assert request.consumed is True
        assert torch.equal(out, x.t())
        with pytest.raises(pyreloc.TransferError, match="^already_executed"):
            pyreloc.execute_transfer(request)
    with pytest.raises(ValueError):
        pyreloc.execute_transfer(pyreloc.make_transfer(bound, src, dst, "h2d"), n_buffers=0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_dense_zero_offset_sources_pass_metadata_preflight(compiler, dtype):
    module = transport()
    compiled = compiler.compile(identity(dtype))
    x = _values((6,), TORCH_DTYPES[dtype])
    if not (torch.cuda.is_available() and __import__("pyreloc").cuda_enabled):
        with pytest.raises(unsupported()) as failure:
            module.prepare_transfer(compiled, x, torch.device("cuda"))
        assert failure.value.reason == "cuda_unavailable"
        return
    request = module.prepare_transfer(compiled, x, torch.device("cuda"))
    assert request.bindings == {"s0": 6}
    assert request.destination.shape == (6,) and request.destination.strides == (1,)
    assert request.destination.dtype == dtype
    assert request.destination.device.type == "cuda"
    assert request.direction == "h2d"
    assert request.source_span_bytes == 6 * x.element_size()
    assert request.consumed is False and request.request is None


class _Subclass(torch.Tensor):
    pass


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: torch.arange(8, dtype=torch.float32)[2:], "storage_offset"),
        (lambda: torch.arange(12, dtype=torch.float32)[::2], "unsupported_layout"),
        (lambda: torch.arange(12, dtype=torch.float32).as_strided((6,), (0,)), "unsupported_layout"),
        (lambda: torch.empty(0, dtype=torch.float32), "empty_tensor"),
        (lambda: torch.ones((), dtype=torch.float32), "unsupported_rank"),
        (lambda: torch.arange(6, dtype=torch.float64), "unsupported_dtype"),
        (lambda: torch.arange(6, dtype=torch.float32).as_subclass(_Subclass), "tensor_subclass"),
        (lambda: torch.arange(6, dtype=torch.float32, requires_grad=True), "requires_grad"),
        (lambda: torch.arange(12, dtype=torch.float32).reshape(3, 4), "source_descriptor"),
    ],
)
def test_metadata_preflight_rejects_before_any_allocation(compiler, make, reason):
    module = transport()
    compiled = compiler.compile(identity())
    x = make()
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiled, x, torch.device("cuda"))
    assert failure.value.reason == reason


def test_direction_and_blocking_preflight(compiler):
    module = transport()
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiler.compile(identity()), x, torch.device("cpu"))
    assert failure.value.reason == "direction_mismatch"
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiler.compile(identity(direction="d2h")), x, torch.device("cpu"))
    assert failure.value.reason == "direction_mismatch"
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiler.compile(identity()), x, torch.device("cuda"), non_blocking=True)
    assert failure.value.reason == "nonblocking_unavailable"
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiler.compile(identity()), x, torch.device("meta"))
    assert failure.value.reason == "direction_mismatch"


def test_divisibility_guard_uses_the_compiled_logical_descriptor(compiler, split_transpose_recipe):
    module = transport()
    compiled = compiler.compile(split_transpose_recipe)
    with pytest.raises(unsupported()) as failure:
        module.prepare_transfer(compiled, torch.arange(130, dtype=torch.float32), torch.device("cuda"))
    assert failure.value.reason == "divisibility"


def test_capability_identity_is_a_plain_string():
    module = transport()
    assert isinstance(module.CAPABILITY_IDENTITY, str) and module.CAPABILITY_IDENTITY.startswith("blocking-v1")


# ------------------------------------------------------------------ GPU


gpu = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def cuda_transport(cuda_device):
    import pyreloc

    if not pyreloc.cuda_enabled:
        pytest.fail("GPU tests require a pyreloc built with RELOC_ENABLE_CUDA")
    return transport()


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == 0
    assert torch.equal(actual.cpu().view(torch.uint8), expected.cpu().view(torch.uint8))


@gpu
@needs_cuda
def test_forward_d2h(compiler, split_transpose_recipe, cuda_device):
    from reloc_torch.transport import execute_transfer, prepare_transfer

    recipe = dataclasses.replace(split_transpose_recipe, direction="d2h")
    compiled = compiler.compile(recipe)
    source = torch.arange(192, dtype=torch.float32, device=cuda_device)
    request = prepare_transfer(compiled, source, "cpu")
    actual = execute_transfer(request)
    expected = torch.arange(192, dtype=torch.float32).reshape(3, 64).t().contiguous()
    assert actual.device.type == "cpu"
    assert actual.shape == (64, 3)
    assert actual.stride() == (3, 1)
    assert actual.storage_offset() == 0
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@gpu
@needs_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("name", ["identity", "transpose", "pad"])
def test_default_stream_h2d_matches_forward_torch(compiler, cuda_transport, cuda_device, name, dtype, pinned):
    recipe = {"identity": lambda: identity(dtype, rank=2), "transpose": lambda: transpose(dtype), "pad": lambda: pad_recipe(dtype)}[name]()
    compiled = compiler.compile(recipe)
    for shape in ((4, 6), (6, 4)):
        x = _values(shape, TORCH_DTYPES[dtype])
        if pinned:
            x = x.pin_memory()
        request = cuda_transport.prepare_transfer(compiled, x, cuda_device)
        actual = cuda_transport.execute_transfer(request)
        _assert_exact(actual, reference(name, x).to(cuda_device))
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        assert request.consumed and request.request.consumed


@gpu
@needs_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("name", ["identity", "transpose", "pad", "permute_120"])
def test_default_stream_forward_d2h_matches_the_cuda_source_function(compiler, cuda_transport, cuda_device, name, dtype):
    if name == "permute_120" and dtype != "float32":
        pytest.skip("permutation witness is float32")
    recipe = {
        "identity": lambda: identity(dtype, "d2h", rank=2),
        "transpose": lambda: transpose(dtype, "d2h"),
        "pad": lambda: pad_recipe(dtype, "d2h"),
        "permute_120": lambda: permute_120("d2h"),
    }[name]()
    compiled = compiler.compile(recipe)
    shapes = ((2, 3, 4),) if name == "permute_120" else ((4, 6), (6, 4))
    for shape in shapes:
        x = _values(shape, TORCH_DTYPES[dtype], device=cuda_device)
        request = cuda_transport.prepare_transfer(compiled, x, "cpu")
        actual = cuda_transport.execute_transfer(request)
        # Independent CPU construction of the forward result (never a round trip).
        expected = reference(name, x.cpu())
        _assert_exact(actual, expected)


@gpu
@needs_cuda
def test_forward_d2h_permutation_witness_is_not_inverse_scatter(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(permute_120("d2h"))
    x = torch.arange(24, dtype=torch.float32, device=cuda_device).reshape(2, 3, 4)
    actual = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, "cpu"))
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(1, 2, 0).contiguous()
    assert torch.equal(actual, expected)
    assert actual.shape == (3, 4, 2) and actual.stride() == (8, 2, 1)
    # A rectangular permutation is not its own inverse: the inverse relocation
    # of the same plan would produce a different tensor.
    inverse = expected.permute(2, 0, 1).contiguous()
    assert not torch.equal(actual.reshape(-1), inverse.reshape(-1))


@gpu
@needs_cuda
def test_symbolic_sizes_reuse_one_artifact_across_bindings(compiler, cuda_transport, cuda_device, split_transpose_recipe):
    compiled = compiler.compile(split_transpose_recipe)
    for n in (64, 128, 192):
        x = torch.arange(n, dtype=torch.float32)
        actual = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, cuda_device))
        expected = x.reshape(n // 64, 64).t().contiguous().to(cuda_device)
        assert torch.equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.stride() == (n // 64, 1)


@gpu
@needs_cuda
def test_identity_no_copy_plan_still_allocates_and_moves(compiler, cuda_transport, cuda_device):
    import pyreloc

    compiled = compiler.compile(identity())
    assert pyreloc.load_plan(compiled.plan_bytes).no_copy is True
    x = torch.arange(6, dtype=torch.float32)
    actual = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, cuda_device))
    assert actual.device == cuda_device
    assert torch.equal(actual.cpu(), x)


@gpu
@needs_cuda
def test_stale_or_consumed_requests_fail_before_launch(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(identity())
    x = torch.arange(6, dtype=torch.float32)
    request = cuda_transport.prepare_transfer(compiled, x, cuda_device)
    x.resize_(3)
    with pytest.raises(RuntimeError, match="stale"):
        cuda_transport.execute_transfer(request)
    assert request.consumed is False
    y = torch.arange(6, dtype=torch.float32)
    request = cuda_transport.prepare_transfer(compiled, y, cuda_device)
    out = cuda_transport.execute_transfer(request)
    assert torch.equal(out.cpu(), y)
    with pytest.raises(RuntimeError, match="already executed"):
        cuda_transport.execute_transfer(request)


@gpu
@needs_cuda
def test_device_ownership_is_proven_by_pointer_attributes(compiler, cuda_transport, cuda_device):
    import pyreloc

    x = torch.arange(6, dtype=torch.float32, device=cuda_device)
    assert pyreloc.cuda_pointer_device(x.untyped_storage().data_ptr()) == cuda_device.index
    host = torch.arange(6, dtype=torch.float32)
    with pytest.raises(pyreloc.TransferError, match="^invalid_view"):
        pyreloc.cuda_pointer_device(host.untyped_storage().data_ptr())


# ------------------------------------------------------------------ Task 3: streams, lifetime


def _delay_kernel(stream, device, iterations=40):
    """Enqueue enough dependent work on `stream` that unordered readers race."""
    with torch.cuda.stream(stream):
        a = torch.randn(2048, 2048, device=device)
        for _ in range(iterations):
            a = a @ a * 1e-6
    return a


@gpu
@needs_cuda
def test_d2h_orders_after_a_delayed_producer_on_a_nondefault_stream(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(identity("float32", "d2h"))
    stream = torch.cuda.Stream(device=cuda_device)
    for _ in range(3):
        with torch.cuda.stream(stream):
            keep = _delay_kernel(stream, cuda_device)
            src = torch.zeros(1 << 20, dtype=torch.float32, device=cuda_device)
            src.add_(keep.sum() * 0 + 7)
            out = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, src, "cpu"))
        assert torch.equal(out, torch.full((1 << 20,), 7.0))


@gpu
@needs_cuda
def test_h2d_on_a_nondefault_caller_stream_is_consumed_there_and_on_the_default_stream(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(transpose())
    stream = torch.cuda.Stream(device=cuda_device)
    x = torch.arange(1 << 16, dtype=torch.float32).reshape(256, 256)
    with torch.cuda.stream(stream):
        _delay_kernel(stream, cuda_device, iterations=10)
        on_gpu = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, cuda_device))
        doubled = on_gpu * 2
    total = on_gpu.sum()
    assert torch.equal(doubled.cpu(), x.t().contiguous() * 2)
    assert total.item() == x.sum().item()


@gpu
@needs_cuda
def test_repeated_allocate_transfer_free_reuse_keeps_results_and_resources_bounded(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(transpose())
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(cuda_device)
    requests = []
    results = []
    for iteration in range(64):
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6) + iteration
        request = cuda_transport.prepare_transfer(compiled, x, cuda_device)
        results.append((cuda_transport.execute_transfer(request), (x.t().contiguous() + 0).to(cuda_device)))
        requests.append(weakref.ref(request))
        del request, x
        filler = torch.full((4, 6), -1.0)
        del filler
    for actual, expected in results:
        assert torch.equal(actual, expected)
    del results
    gc.collect()
    assert all(reference() is None for reference in requests)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(cuda_device) - baseline < 1 << 20


@gpu
@needs_cuda
def test_execution_errors_propagate_and_do_not_rerun(compiler, cuda_transport, cuda_device, monkeypatch):
    import pyreloc

    compiled = compiler.compile(identity())
    x = torch.arange(6, dtype=torch.float32)
    request = cuda_transport.prepare_transfer(compiled, x, cuda_device)
    calls = []
    real = pyreloc.execute_transfer

    def failing(native, **kwargs):
        calls.append(kwargs["caller_stream"])
        raise pyreloc.TransferError("backend_failure: injected copy failure")

    monkeypatch.setattr(pyreloc, "execute_transfer", failing)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        cuda_transport.execute_transfer(request)
    assert calls == [torch.cuda.current_stream(cuda_device).cuda_stream]
    assert request.consumed is True
    monkeypatch.setattr(pyreloc, "execute_transfer", real)
    fresh = cuda_transport.prepare_transfer(compiled, x, cuda_device)
    assert torch.equal(cuda_transport.execute_transfer(fresh).cpu(), x)


@gpu
@needs_cuda
def test_transfers_target_a_noncurrent_device_when_available(compiler, cuda_transport):
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-GPU current-device mismatch needs two devices")
    compiled = compiler.compile(transpose())
    other = torch.device("cuda", 1)
    with torch.cuda.device(0):
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        actual = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, other))
        assert torch.cuda.current_device() == 0
    assert actual.device == other
    assert torch.equal(actual.cpu(), x.t().contiguous())
    back = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiler.compile(identity("float32", "d2h", rank=2)), actual, "cpu"))
    assert torch.equal(back, x.t().contiguous())


@gpu
@needs_cuda
def test_concurrent_threads_transfer_independently(compiler, cuda_transport, cuda_device):
    compiled = compiler.compile(transpose())
    errors = []

    def worker(seed):
        try:
            for i in range(8):
                x = torch.arange(24, dtype=torch.float32).reshape(4, 6) + seed * 100 + i
                out = cuda_transport.execute_transfer(cuda_transport.prepare_transfer(compiled, x, cuda_device))
                if not torch.equal(out.cpu(), x.t().contiguous()):
                    errors.append((seed, i))
        except Exception as error:  # pragma: no cover - reported below
            errors.append(repr(error))

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
