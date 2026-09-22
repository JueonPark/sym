"""Actual transfers, stream ordering, allocations and exclusions on real CUDA.

Tests that need real H2D/D2H execution consume the R2 transport bridge through
the ``real_runtime`` fixture and skip with an explicit reason while R2 (#146)
is absent; skips are reported, never counted as passes. Exclusion tests run
through the production adapter and hold with or without R2.
"""
import gc
import weakref

import pytest
import torch

from conftest import make_entry


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]

DTYPES = [torch.float32, torch.float16, torch.int8]


def backend_module():
    from reloc_torch import backend

    return backend


def eager_module():
    from reloc_torch import eager

    return eager


@pytest.fixture
def real_backend(compiler, real_runtime):
    backend = backend_module().RelocBackend(compiler=compiler, runtime=real_runtime)
    yield backend
    backend.close()


@pytest.fixture
def production_backend(compiler):
    from reloc_torch.runtime import TransportAdapter

    backend = backend_module().RelocBackend(compiler=compiler, runtime=TransportAdapter())
    yield backend
    backend.close()


def _source(shape, dtype, *, device="cpu", pinned=False):
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.arange(numel) % 120 - 60
    tensor = values.to(dtype).reshape(shape).to(device)
    return tensor.pin_memory() if pinned else tensor


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == 0
    assert torch.equal(actual.cpu().view(torch.uint8), expected.cpu().view(torch.uint8))


H2D_FUNCTIONS = {
    "identity": lambda x: x.to("cuda"),
    "transpose": lambda x: x.to("cuda").transpose(0, 1).contiguous(),
    "reshape_transpose": lambda x: x.reshape(x.shape[0] // 2, 2, x.shape[1]).transpose(0, 1).contiguous().to("cuda"),
    "pad": lambda x: torch.nn.functional.pad(x, (1, 2, 0, 1), value=0).to("cuda"),
}
D2H_FUNCTIONS = {
    "identity": lambda x: x.cpu(),
    "transpose": lambda x: x.transpose(0, 1).contiguous().cpu(),
    "reshape_transpose": lambda x: x.cpu().reshape(x.shape[0] // 2, 2, x.shape[1]).transpose(0, 1).contiguous(),
    "pad": lambda x: torch.nn.functional.pad(x.cpu(), (1, 2, 0, 1), value=0),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pinned", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("name", sorted(H2D_FUNCTIONS))
def test_h2d_layouts_match_pytorch_exactly(real_backend, name, dtype, pinned):
    fn = H2D_FUNCTIONS[name]
    compiled = torch.compile(fn, backend=real_backend, dynamic=True)
    with torch.no_grad():
        for rows in (4, 6):
            x = _source((rows, 6), dtype, pinned=pinned)
            _assert_exact(compiled(x), fn(x))
    assert real_backend.stats()["runtime_executions"] >= 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", sorted(D2H_FUNCTIONS))
def test_forward_d2h_layouts_match_the_cuda_source_function(real_backend, name, dtype):
    fn = D2H_FUNCTIONS[name]
    compiled = torch.compile(fn, backend=real_backend, dynamic=True)
    with torch.no_grad():
        for rows in (4, 6):
            x = _source((rows, 6), dtype, device="cuda")
            _assert_exact(compiled(x), fn(x))
    assert real_backend.stats()["runtime_executions"] >= 1


def test_forward_d2h_permutation_witness(real_backend):
    def fn(x):
        return x.permute(1, 2, 0).contiguous().cpu()

    compiled = torch.compile(fn, backend=real_backend, dynamic=True)
    with torch.no_grad():
        x = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4)
        actual = compiled(x)
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(1, 2, 0).contiguous()
    _assert_exact(actual, expected)
    assert real_backend.stats()["runtime_executions"] >= 1


def test_side_stream_d2h_producer_and_immediate_cpu_consumer(real_backend):
    backend = real_backend
    eager_transfers = eager_module().eager_transfers
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad(), eager_transfers(backend=backend):
        src = torch.arange(4096, device="cuda", dtype=torch.float32)
        src.add_(7)
        out = src.cpu()
    torch.testing.assert_close(out, torch.arange(4096) + 7, rtol=0, atol=0, check_dtype=False)
    assert backend.stats()["runtime_executions"] == 1


def test_h2d_on_nondefault_caller_stream_is_consumed_there_and_on_the_default_stream(real_backend):
    eager_transfers = eager_module().eager_transfers
    stream = torch.cuda.Stream()
    x = torch.arange(1 << 16, dtype=torch.float32)
    with torch.cuda.stream(stream), torch.no_grad(), eager_transfers(backend=real_backend):
        on_gpu = x.to("cuda")
        doubled = on_gpu * 2
    with torch.no_grad():
        summed = on_gpu.sum()
    assert torch.equal(doubled.cpu(), x * 2)
    assert summed.item() == x.sum().item()
    assert real_backend.stats()["runtime_executions"] == 1


def test_repeated_transfers_with_dropped_inputs_and_reallocation(real_backend):
    fn = H2D_FUNCTIONS["transpose"]
    compiled = torch.compile(fn, backend=real_backend, dynamic=True)
    from reloc_torch import runtime as runtime_module

    calls = []
    original_preflight = real_backend.runtime.preflight

    def observing_preflight(*args, **kwargs):
        call = original_preflight(*args, **kwargs)
        calls.append(weakref.ref(call))
        return call

    real_backend.runtime.preflight = observing_preflight
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    results = []
    with torch.no_grad():
        for iteration in range(64):
            x = torch.arange(24, dtype=torch.float32).reshape(4, 6) + iteration
            results.append((compiled(x), x.t().contiguous().cuda()))
            del x
            filler = torch.full((4, 6), -1.0)
            del filler
    for actual, expected in results:
        assert torch.equal(actual, expected)
    del results
    gc.collect()
    torch.cuda.synchronize()
    assert all(reference() is None for reference in calls)
    stats = real_backend.stats()
    assert stats["runtime_executions"] == 64
    assert stats["cache_entries"] <= 1
    assert stats["live_handles"] <= 2
    assert torch.cuda.memory_allocated() - baseline < 1 << 20
    real_backend.close()
    real_backend.close()
    assert real_backend.stats()["closed"] is True


def test_execution_errors_clean_up_and_do_not_retry(real_backend, compiler, identity_recipe):
    from reloc_torch.runtime import ExecutionError, execute_or_fallback

    class FailingExecution:
        capability_identity = real_backend.runtime.capability_identity

        def preflight(self, compiled, src, device, *, non_blocking=False):
            return real_backend.runtime.preflight(compiled, src, device, non_blocking=non_blocking)

        def execute(self, call):
            raise RuntimeError("injected copy failure")

    entry = make_entry(compiler.compile(identity_recipe), FailingExecution(), lambda src, *s: src.cuda())
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(ExecutionError, match="injected copy failure"):
        execute_or_fallback(entry, x, [6], torch.device("cuda"))
    assert entry.fallback_calls == 0
    healthy = make_entry(compiler.compile(identity_recipe), real_backend.runtime, lambda src, *s: src.cuda())
    assert torch.equal(execute_or_fallback(healthy, x, [6], torch.device("cuda")).cpu(), x)


def test_nonblocking_transfers_use_original_pytorch_with_a_reason(production_backend):
    eager_transfers = eager_module().eager_transfers
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6).pin_memory()
    with torch.no_grad(), eager_transfers(backend=production_backend):
        on_gpu = x.to("cuda", non_blocking=True)
        back = on_gpu.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(on_gpu.cpu(), x) and torch.equal(back, x)
    stats = production_backend.stats()
    assert stats["redispatches"]["nonblocking_unavailable"] == 2
    assert stats["runtime_executions"] == 0
    assert stats["plan_compiles"] == 0


class UnavailableRuntime:
    capability_identity = "unavailable-runtime/0"

    def __init__(self):
        self.preflights = 0
        self.executions = 0

    def preflight(self, compiled, src, device, *, non_blocking=False):
        from reloc_torch import UnsupportedRecipe

        self.preflights += 1
        raise UnsupportedRecipe("runtime_unavailable", "no CUDA-capable runtime")

    def execute(self, call):
        self.executions += 1
        raise AssertionError("unreachable")


def test_missing_runtime_capability_falls_back_with_zero_launches(compiler):
    runtime = UnavailableRuntime()
    backend = backend_module().RelocBackend(compiler=compiler, runtime=runtime)
    fn = H2D_FUNCTIONS["transpose"]
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        _assert_exact(compiled(x), fn(x))
    stats = backend.stats()
    assert stats["replaced_regions"] == 1
    assert stats["fallbacks"]["runtime_unavailable"] == 1
    assert stats["runtime_executions"] == 0
    assert runtime.preflights == 1 and runtime.executions == 0
    backend.close()


def test_missing_compiler_capability_keeps_the_original_graph(compiler):
    from reloc_torch import UnsupportedRecipe

    class RejectingCompiler:
        identity = "rejecting-compiler"

        def __init__(self):
            self.calls = 0

        def compile(self, recipe):
            self.calls += 1
            raise UnsupportedRecipe("fold_unsupported", "capability missing")

    rejecting = RejectingCompiler()
    runtime = UnavailableRuntime()
    backend = backend_module().RelocBackend(compiler=rejecting, runtime=runtime)
    fn = H2D_FUNCTIONS["transpose"]
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        _assert_exact(compiled(x), fn(x))
        _assert_exact(compiled(x + 1), fn(x + 1))
    stats = backend.stats()
    assert stats["exclusions"]["fold_unsupported"] >= 1
    assert stats["replaced_regions"] == 0
    assert runtime.preflights == 0
    assert rejecting.calls == 1
    backend.close()


def _region(src):
    # The split-transpose region's own PyTorch computation.
    return src.cpu().reshape(-1, 64).t().contiguous()


@pytest.mark.parametrize(
    ("make", "symbols", "reason", "expected"),
    [
        # Guard failures before the destination is known fall back to whatever
        # the original region does for that input (here PyTorch's own copy).
        (lambda: torch.arange(130, dtype=torch.float32, device="cuda"), [130], "divisibility", lambda x: x.cpu()),
        # Once the destination is known, the fallback's metadata is verified
        # against it, so the original must compute the real region.
        (lambda: torch.arange(128, dtype=torch.float32, device="cuda"), [64], "symbol_mismatch", _region),
        (lambda: torch.empty(0, dtype=torch.float32, device="cuda"), [0], "empty_tensor", lambda x: x.cpu()),
        (lambda: torch.ones((), dtype=torch.float32, device="cuda"), [], "unsupported_rank", lambda x: x.cpu()),
        (lambda: torch.arange(128, dtype=torch.float64, device="cuda"), [128], "unsupported_dtype", lambda x: x.cpu()),
        (lambda: torch.arange(256, dtype=torch.float32, device="cuda")[::2], [128], "unsupported_layout", lambda x: x.cpu()),
        (lambda: torch.arange(130, dtype=torch.float32, device="cuda")[2:], [128], "storage_offset", lambda x: x.cpu()),
    ],
)
def test_invalid_bindings_and_unsupported_sources_fall_back_with_zero_launches(compiler, split_transpose_recipe, make, symbols, reason, expected):
    import dataclasses
    from reloc_torch.runtime import execute_or_fallback

    recipe = dataclasses.replace(split_transpose_recipe, direction="d2h")
    runtime = UnavailableRuntime()
    calls = []

    def original(src, *scalars):
        calls.append(src)
        return expected(src)

    entry = make_entry(compiler.compile(recipe), runtime, original)
    x = make()
    actual = execute_or_fallback(entry, x, symbols, torch.device("cpu"))
    assert torch.equal(actual, expected(x))
    assert entry.diagnostics.fallbacks[reason] == 1
    assert runtime.preflights == 0 and runtime.executions == 0
    assert calls == [x]
