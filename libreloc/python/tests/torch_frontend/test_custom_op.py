"""Custom op registration: explicit schema, exact fake metadata, opcheck, handles."""
import importlib
import inspect

import pytest
import torch

from conftest import CountingRuntime, make_entry


SCHEMA = (
    "reloc_torch::transfer(Tensor src, str handle, SymInt[] symbols, "
    "SymInt[] out_shape, SymInt[] out_strides, Device device) -> Tensor"
)
DEFAULT_OPCHECK_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def ops():
    from reloc_torch import ops as module

    return module


def transfer():
    ops()
    return torch.ops.reloc_torch.transfer.default


@pytest.fixture
def registered(entry):
    from reloc_torch.cache import REGISTRY

    registration = REGISTRY.register(entry)
    yield registration.handle, entry
    registration.release()


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.storage_offset(),
    )


def test_schema_is_explicit_functional_and_registered_once():
    module = ops()
    overload = transfer()
    assert str(overload._schema) == SCHEMA
    assert module.OP is overload
    arguments = overload._schema.arguments
    assert [a.name for a in arguments] == ["src", "handle", "symbols", "out_shape", "out_strides", "device"]
    # torch prints SymInt[] argument types as List[int]; the schema string above
    # is the authoritative record of the explicit symbolic types.
    assert all(str(a.type) == "List[int]" for a in arguments[2:5])
    assert all(a.alias_info is None for a in arguments)
    assert overload._schema.returns[0].alias_info is None
    assert overload._schema.is_mutable is False


def test_real_and_fake_identity_metadata_agree_and_fake_touches_no_counters(registered, counting_runtime):
    from torch._subclasses.fake_tensor import FakeTensorMode

    handle, entry = registered
    x = torch.arange(6, dtype=torch.float32)
    real = transfer()(x, handle, [6], [6], [1], torch.device("cpu"))
    assert torch.equal(real, x)
    assert real.data_ptr() != x.data_ptr()
    assert counting_runtime.executions == 1
    before = entry.diagnostics.snapshot()
    with FakeTensorMode() as mode:
        fake = transfer()(mode.from_tensor(x), handle, [6], [6], [1], torch.device("cpu"))
    assert _metadata(fake) == _metadata(real)
    assert entry.diagnostics.snapshot() == before
    assert counting_runtime.executions == 1


def test_fake_kernel_consults_no_registry_and_supports_symbolic_shapes(monkeypatch, entry):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from reloc_torch import cache

    def forbidden(handle):
        raise AssertionError("fake kernel consulted the handle registry")

    monkeypatch.setattr(cache, "lookup_handle", forbidden)
    monkeypatch.setattr(ops(), "lookup_handle", forbidden, raising=False)
    from torch.fx.experimental.proxy_tensor import make_fx

    def fn(x):
        n = x.shape[0]
        return transfer()(x, "reloc-unregistered", [n], [64, n // 64], [n // 64, 1], torch.device("cpu"))

    gm = make_fx(fn, tracing_mode="symbolic")(torch.arange(128, dtype=torch.float16))
    fake = next(n for n in gm.graph.nodes if n.target is transfer()).meta["val"]
    assert fake.dtype == torch.float16
    assert fake.device.type == "cpu"
    assert str(fake.shape[0]) == "64"
    assert "//" in str(fake.shape[1])
    assert str(fake.stride()[0]) == str(fake.shape[1]) and str(fake.stride()[1]) == "1"
    assert fake.storage_offset() == 0
    with FakeTensorMode() as mode:
        static = transfer()(mode.from_tensor(torch.arange(6, dtype=torch.int8)), "reloc-unregistered", [6], [2, 3], [3, 1], torch.device("cpu"))
    assert _metadata(static) == ((2, 3), (3, 1), torch.int8, "cpu", 0)


def test_symbolic_capture_traces_fake_metadata_and_real_execution_rebinds(compiler, split_transpose_recipe, counting_runtime):
    from torch.fx.experimental.proxy_tensor import make_fx
    from reloc_torch.cache import REGISTRY

    entry = make_entry(compiler.compile(split_transpose_recipe), counting_runtime, lambda src, *s: src.clone())
    registration = REGISTRY.register(entry)
    handle = registration.handle

    def fn(x):
        n = x.shape[0]
        return transfer()(x, handle, [n], [64, n // 64], [n // 64, 1], torch.device("cpu"))

    gm = make_fx(fn, tracing_mode="symbolic")(torch.arange(128, dtype=torch.float32))
    assert counting_runtime.executions == 0
    node = next(n for n in gm.graph.nodes if n.target is transfer())
    value = node.meta["val"]
    assert str(value.shape[0]) == "64" and "64" in str(value.shape[1])
    for n in (128, 192):
        x = torch.arange(n, dtype=torch.float32)
        actual = gm(x)
        expected = x.reshape(n // 64, 64).t().contiguous()
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()
    assert counting_runtime.executions == 2
    registration.release()


def test_opcheck_default_registration_checks_pass_with_inference_tensors(registered):
    handle, entry = registered
    signature = inspect.signature(torch.library.opcheck)
    assert tuple(signature.parameters["test_utils"].default) == DEFAULT_OPCHECK_UTILS
    with torch.inference_mode():
        x = torch.arange(6, dtype=torch.float32)
    assert x.is_inference()
    args = (x, handle, [6], [6], [1], torch.device("cpu"))
    result = torch.library.opcheck(transfer(), args)
    assert result == {name: "SUCCESS" for name in DEFAULT_OPCHECK_UTILS}
    assert torch.equal(transfer()(*args), x)


def test_declared_metadata_mismatch_is_an_error_before_launch(registered, counting_runtime):
    handle, entry = registered
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="declared output metadata"):
        transfer()(x, handle, [6], [3, 2], [2, 1], torch.device("cpu"))
    assert counting_runtime.executions == 0
    assert entry.fallback_calls == 0


def test_unknown_and_closed_handles_fail_without_fabricating_output(entry):
    from reloc_torch.cache import REGISTRY

    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="unknown or closed"):
        transfer()(x, "reloc-not-registered", [6], [6], [1], torch.device("cpu"))
    registration = REGISTRY.register(entry)
    assert torch.equal(transfer()(x, registration.handle, [6], [6], [1], torch.device("cpu")), x)
    entry.close()
    with pytest.raises(RuntimeError, match="unknown or closed"):
        transfer()(x, registration.handle, [6], [6], [1], torch.device("cpu"))
    entry.closed = False
    registration.release()
    with pytest.raises(RuntimeError, match="unknown or closed"):
        transfer()(x, registration.handle, [6], [6], [1], torch.device("cpu"))


def test_gradient_requiring_direct_calls_are_rejected(registered, counting_runtime):
    handle, entry = registered
    x = torch.arange(6, dtype=torch.float32, requires_grad=True)
    with pytest.raises(RuntimeError):
        transfer()(x, handle, [6], [6], [1], torch.device("cpu"))
    assert counting_runtime.executions == 0
    with torch.no_grad():
        result = transfer()(x, handle, [6], [6], [1], torch.device("cpu"))
    assert torch.equal(result, x.detach())
    assert result.requires_grad is False


class AliasingRuntime(CountingRuntime):
    def execute(self, call):
        self.executions += 1
        return call.src


class WrongStrideRuntime(CountingRuntime):
    def execute(self, call):
        self.executions += 1
        return torch.zeros(3, 2)


@pytest.mark.parametrize(("runtime_type", "match"), [(AliasingRuntime, "alias"), (WrongStrideRuntime, "metadata")])
def test_functional_and_metadata_promises_are_verified(compiler, identity_recipe, runtime_type, match):
    from reloc_torch.cache import REGISTRY

    runtime = runtime_type()
    entry = make_entry(compiler.compile(identity_recipe), runtime, lambda src, *s: src.clone())
    registration = REGISTRY.register(entry)
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=match):
        transfer()(x, registration.handle, [6], [6], [1], torch.device("cpu"))
    assert runtime.executions == 1
    assert entry.fallback_calls == 0
    registration.release()


def test_reload_and_duplicate_import_keep_the_op_and_live_handles_working(registered, counting_runtime):
    import reloc_torch.ops as module

    handle, entry = registered
    x = torch.arange(6, dtype=torch.float32)
    before = transfer()
    reloaded = importlib.reload(module)
    import reloc_torch.ops as again

    assert again is reloaded
    overload = torch.ops.reloc_torch.transfer.default
    assert str(overload._schema) == SCHEMA
    assert reloaded.OP is overload
    assert torch.equal(overload(x, handle, [6], [6], [1], torch.device("cpu")), x)
    assert torch.equal(before(x, handle, [6], [6], [1], torch.device("cpu")), x)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = overload(mode.from_tensor(x), handle, [6], [6], [1], torch.device("cpu"))
    assert _metadata(fake) == _metadata(x)
    assert counting_runtime.executions == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_opcheck_passes_with_cuda_inference_source_through_the_production_adapter(compiler, cuda_device):
    import dataclasses
    from reloc_torch.cache import REGISTRY
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.runtime import TransportAdapter
    from reloc_torch.symbolic import Const, Symbol

    descriptor = TensorSpec((Symbol("s0"),), (Const(1),), Const(0), "float32")
    recipe = Recipe(descriptor, (), descriptor, "d2h")
    adapter = TransportAdapter()
    entry = make_entry(compiler.compile(recipe), adapter, lambda src, *s: src.to("cpu"))
    registration = REGISTRY.register(entry)
    with torch.inference_mode():
        x = torch.arange(6, dtype=torch.float32, device=cuda_device)
    args = (x, registration.handle, [6], [6], [1], torch.device("cpu"))
    result = torch.library.opcheck(torch.ops.reloc_torch.transfer.default, args)
    assert result == {name: "SUCCESS" for name in DEFAULT_OPCHECK_UTILS}
    actual = torch.ops.reloc_torch.transfer.default(*args)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, x.cpu())
    snapshot = entry.diagnostics.snapshot()
    if adapter.available:
        assert snapshot["runtime_executions"] >= 1
    else:
        assert snapshot["runtime_executions"] == 0
        assert snapshot["fallbacks"]["runtime_unavailable"] >= 1
    registration.release()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_identity_h2d_allocates_cuda_storage_even_when_no_copy(compiler, identity_recipe, real_runtime, cuda_device):
    import pyreloc
    from reloc_torch.cache import REGISTRY

    compiled = compiler.compile(identity_recipe)
    assert pyreloc.load_plan(compiled.plan_bytes).no_copy is True
    entry = make_entry(compiled, real_runtime, lambda src, *s: src.to(cuda_device))
    registration = REGISTRY.register(entry)
    x = torch.arange(6, dtype=torch.float32)
    actual = torch.ops.reloc_torch.transfer.default(x, registration.handle, [6], [6], [1], cuda_device)
    assert actual.device == cuda_device
    assert torch.equal(actual.cpu(), x)
    assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert entry.diagnostics.snapshot()["runtime_executions"] == 1
    registration.release()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_and_fake_transpose_metadata_agree_on_cuda(compiler, real_runtime, cuda_device):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from reloc_torch.cache import REGISTRY
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol

    rows, columns = Symbol("s0"), Symbol("s1")
    recipe = Recipe(
        TensorSpec((rows, columns), (columns, Const(1)), Const(0), "float32"),
        (Transpose((1, 0)),),
        TensorSpec((columns, rows), (rows, Const(1)), Const(0), "float32"),
        "h2d",
    )
    entry = make_entry(compiler.compile(recipe), real_runtime, lambda src, *s: src.to(cuda_device).t().contiguous())
    registration = REGISTRY.register(entry)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    args = (x, registration.handle, [4, 6], [6, 4], [4, 1], cuda_device)
    real = torch.ops.reloc_torch.transfer.default(*args)
    with FakeTensorMode() as mode:
        fake = torch.ops.reloc_torch.transfer.default(mode.from_tensor(x), *args[1:])
    assert _metadata(real) == _metadata(fake)
    assert real.device.index == fake.device.index == cuda_device.index
    assert torch.equal(real.cpu(), x.t().contiguous())
    assert torch.library.opcheck(torch.ops.reloc_torch.transfer.default, args) == {name: "SUCCESS" for name in DEFAULT_OPCHECK_UTILS}
    registration.release()
