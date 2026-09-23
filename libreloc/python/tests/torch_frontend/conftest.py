import dataclasses

import pytest

from reloc_torch.records import TensorMetadata, TransferRecord


@pytest.fixture
def compiler():
    from reloc_torch import CompilerClient

    try:
        return CompilerClient.from_environment()
    except RuntimeError as error:
        pytest.fail(str(error))


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Each test compiles with its own backend; Dynamo caches per code object
    and stops recompiling after its cache-size limit, so shared test lambdas
    would silently run eager after a few backends."""
    import torch

    torch._dynamo.reset()
    yield


@pytest.fixture(autouse=True)
def isolate_handle_registry():
    """Release handles a failing test left in the process-global registry."""
    from reloc_torch.cache import REGISTRY

    before = set(REGISTRY._entries)
    yield
    for handle in set(REGISTRY._entries) - before:
        REGISTRY._release(handle)


@pytest.fixture
def split_transpose_recipe():
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, FloorDiv, Symbol, dense_strides

    n = Symbol("s0")
    split = (FloorDiv(n, 64), Const(64))
    destination_shape = (Const(64), FloorDiv(n, 64))
    return Recipe(
        TensorSpec((n,), (Const(1),), Const(0), "float32"),
        (Reshape(split), Transpose((1, 0))),
        TensorSpec(
            destination_shape,
            dense_strides(destination_shape),
            Const(0),
            "float32",
        ),
        "h2d",
    )


@pytest.fixture
def identity_recipe():
    """Rank-one symbolic identity: the dense-input control-flow recipe for T3 tests."""
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import Const, Symbol

    descriptor = TensorSpec((Symbol("s0"),), (Const(1),), Const(0), "float32")
    return Recipe(descriptor, (), descriptor, "h2d")


class CountingRuntime:
    """CPU-only test adapter: real guards, real host relocation, counted launches.

    It never simulates CUDA success. Preflight rejects non-CPU destinations,
    nonblocking requests, and every source the shared frontend guards reject,
    so a stride ``(2,)`` view fails with ``unsupported_layout`` before launch.
    """

    capability_identity = "cpu-test-adapter/1"

    def __init__(self, *, fail_execution=None):
        self.executions = 0
        self.preflights = 0
        self.fail_execution = fail_execution

    def preflight(self, compiled, src, device, *, non_blocking=False):
        from reloc_torch.artifact import UnsupportedRecipe
        from reloc_torch.runtime import prepare_host_call

        self.preflights += 1
        if device.type != "cpu":
            raise UnsupportedRecipe("unsupported_device", "CPU-only test adapter")
        return prepare_host_call(compiled, src, device, non_blocking=non_blocking)

    def execute(self, call):
        import torch
        import pyreloc
        from pyreloc.torch_interop import as_ptr

        self.executions += 1
        if self.fail_execution is not None:
            raise self.fail_execution
        destination = call.destination
        out = torch.empty_strided(
            destination.shape, destination.strides, dtype=call.src.dtype, device="cpu"
        )
        pyreloc.relocate(call.bound, *as_ptr(call.src), *as_ptr(out))
        return out


@pytest.fixture
def counting_runtime():
    return CountingRuntime()


def make_entry(compiled, runtime, fallback, *, symbolic_bindings=()):
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.runtime import ExecutionEntry

    return ExecutionEntry(
        compiled=compiled,
        original=fallback,
        runtime=runtime,
        diagnostics=Diagnostics(),
        symbolic_bindings=tuple(symbolic_bindings),
    )


@pytest.fixture
def entry(compiler, identity_recipe, counting_runtime):
    """Identity execution entry whose fallback is a counted ``src.clone()``."""
    calls = []

    def fallback(src, *symbols):
        calls.append(tuple(symbols))
        return src.clone()

    result = make_entry(compiler.compile(identity_recipe), counting_runtime, fallback)
    result.fallback_log = calls
    return result


@pytest.fixture
def cuda_device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def real_runtime():
    """The production R2 transport bridge, or an explicit skip while R2 is absent.

    Missing R2 (#146) is a documented blocked dependency of T3, not a passing
    result: tests that need actual transfers skip with this reason and are
    reported as skips, never as passes.
    """
    from reloc_torch.runtime import TransportAdapter

    adapter = TransportAdapter()
    if not adapter.available:
        pytest.skip(f"R2 transport adapter unavailable: {adapter.unavailable_reason}")
    return adapter


@pytest.fixture
def backend(compiler, real_runtime):
    """RelocBackend over the real R1 exporter and R2 transport (GPU tests)."""
    from reloc_torch import RelocBackend

    result = RelocBackend(compiler=compiler, runtime=real_runtime)
    yield result
    result.close()


@pytest.fixture
def transpose_weight_recipe():
    """f32 (4, 6) -> (6, 4) through one transpose; the T4 weight recipe."""
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, dense_strides

    source = (Const(4), Const(6))
    destination = (Const(6), Const(4))
    return Recipe(
        TensorSpec(source, dense_strides(source), Const(0), "float32"),
        (Transpose((1, 0)),),
        TensorSpec(destination, dense_strides(destination), Const(0), "float32"),
        "h2d",
    )


@pytest.fixture
def h2d_record():
    source = TensorMetadata(
        shape=(4, 6),
        strides=(6, 1),
        storage_offset=0,
        dtype="float32",
        device_type="cpu",
        device_index=None,
        requires_grad=False,
        layout="strided",
        pinned=False,
        is_subclass=False,
        storage_capacity_bytes=96,
    )
    destination = dataclasses.replace(
        source,
        device_type="cuda",
        device_index=0,
    )
    return TransferRecord(
        operator="aten._to_copy.default",
        phase="input",
        source=source,
        destination=destination,
        non_blocking=False,
        mutates=False,
        aliases_source=False,
        layout_history=(),
    )
