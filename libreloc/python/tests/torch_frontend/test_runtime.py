"""Common guarded execution: preflight before launch, one fallback, no retry."""
import threading

import pytest
import torch

from conftest import CountingRuntime, make_entry


def api():
    from reloc_torch import runtime

    return runtime


def test_guard_failure_never_launches(entry, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback

    x = torch.arange(12, dtype=torch.float32)[::2]
    actual = execute_or_fallback(entry, x, [], torch.device("cpu"))
    assert torch.equal(actual, x.clone())
    assert counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks["unsupported_layout"] == 1
    assert entry.fallback_calls == 1


def test_dense_source_executes_once_through_the_adapter(entry, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback

    x = torch.arange(6, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [6], torch.device("cpu"))
    assert torch.equal(actual, x)
    assert actual.data_ptr() != x.data_ptr()
    assert actual.stride() == (1,) and actual.storage_offset() == 0
    assert counting_runtime.executions == 1
    assert entry.fallback_calls == 0
    snapshot = entry.diagnostics.snapshot()
    assert snapshot["runtime_executions"] == 1
    assert snapshot["symbol_binds"] == 1
    assert snapshot["fallbacks"] == {}


def test_nonblocking_falls_back_before_the_adapter_is_consulted(entry, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback

    x = torch.arange(6, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [6], torch.device("cpu"), non_blocking=True)
    assert torch.equal(actual, x)
    assert counting_runtime.preflights == 0
    assert counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks["nonblocking_unavailable"] == 1


def test_explicit_symbols_are_reconciled_with_source_metadata(entry, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback

    x = torch.arange(6, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [7], torch.device("cpu"))
    assert torch.equal(actual, x)
    assert counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks["symbol_mismatch"] == 1
    # The identity fixture's original region has no scalar placeholders.
    assert entry.fallback_log == [()]


def test_fallback_receives_evaluated_scalar_placeholders(compiler, split_transpose_recipe, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback
    from reloc_torch.symbolic import FloorDiv, Symbol

    seen = []

    def original(src, rows, size):
        seen.append((rows, size))
        return src.clone()

    entry = make_entry(
        compiler.compile(split_transpose_recipe),
        counting_runtime,
        original,
        symbolic_bindings=(("floordiv", FloorDiv(Symbol("s0"), 64)), ("s97", Symbol("s0"))),
    )
    x = torch.arange(130, dtype=torch.float32)
    assert torch.equal(execute_or_fallback(entry, x, [130], torch.device("cpu")), x)
    assert seen == [(2, 130)]
    assert entry.diagnostics.fallbacks["divisibility"] == 1
    assert counting_runtime.executions == 0


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (lambda: torch.empty(0, dtype=torch.float32), "empty_tensor"),
        (lambda: torch.ones((), dtype=torch.float32), "unsupported_rank"),
        (lambda: torch.arange(6, dtype=torch.float64), "unsupported_dtype"),
        (lambda: torch.arange(8, dtype=torch.float32)[2:], "storage_offset"),
        (lambda: torch.arange(6, dtype=torch.float32, requires_grad=True), "requires_grad"),
        (lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3), "source_descriptor"),
    ],
)
def test_expected_guard_reasons_fall_back_with_zero_launches(entry, counting_runtime, source, reason):
    from reloc_torch.runtime import execute_or_fallback

    x = source()
    actual = execute_or_fallback(entry, x, None, torch.device("cpu"))
    assert torch.equal(actual, x.clone())
    assert counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks[reason] == 1
    assert entry.fallback_calls == 1


def test_symbolic_recipe_rebinds_across_shapes_and_guards_divisibility(compiler, split_transpose_recipe, counting_runtime):
    from reloc_torch.runtime import execute_or_fallback

    compiled = compiler.compile(split_transpose_recipe)
    entry = make_entry(compiled, counting_runtime, lambda src, *symbols: src.clone())
    for n in (64, 128, 192):
        x = torch.arange(n, dtype=torch.float32)
        actual = execute_or_fallback(entry, x, [n], torch.device("cpu"))
        expected = x.reshape(n // 64, 64).t().contiguous()
        assert torch.equal(actual, expected)
        # Dense strides of the compiled descriptor; PyTorch keeps a stride of
        # 64 on the singleton axis when n == 64, which the importer guards
        # separately (T2 conditional_materialization).
        assert actual.stride() == (n // 64, 1)
        assert actual.storage_offset() == 0
    assert counting_runtime.executions == 3
    x = torch.arange(130, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [130], torch.device("cpu"))
    assert torch.equal(actual, x)
    assert counting_runtime.executions == 3
    assert entry.diagnostics.fallbacks["divisibility"] == 1
    assert entry.fallback_calls == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_layout_only_dtypes_execute_exact_bytes(compiler, counting_runtime, dtype):
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.runtime import execute_or_fallback
    from reloc_torch.symbolic import Const, Symbol

    name = str(dtype).removeprefix("torch.")
    rows, columns = Symbol("s0"), Symbol("s1")
    recipe = Recipe(
        TensorSpec((rows, columns), (columns, Const(1)), Const(0), name),
        (Transpose((1, 0)),),
        TensorSpec((columns, rows), (rows, Const(1)), Const(0), name),
        "h2d",
    )
    entry = make_entry(compiler.compile(recipe), counting_runtime, lambda src, *s: src.t().contiguous())
    x = torch.arange(12, dtype=dtype).reshape(3, 4)
    actual = execute_or_fallback(entry, x, None, torch.device("cpu"))
    expected = x.t().contiguous()
    assert actual.dtype == dtype
    assert actual.view(torch.uint8).tolist() == expected.view(torch.uint8).tolist()
    assert counting_runtime.executions == 1


def test_bind_error_is_translated_before_execution(entry, counting_runtime, monkeypatch):
    import pyreloc
    from reloc_torch.runtime import execute_or_fallback

    def rejecting_bind(plan, symbols, *args, **kwargs):
        raise pyreloc.BindError("deliberate binder rejection")

    monkeypatch.setattr(pyreloc, "bind", rejecting_bind)
    x = torch.arange(6, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [6], torch.device("cpu"))
    assert torch.equal(actual, x)
    assert counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks["bind_error"] == 1
    assert entry.diagnostics.snapshot()["symbol_binds"] == 1


def test_direct_pyreloc_bind_failures_are_unchanged(compiler, split_transpose_recipe):
    import pyreloc

    compiled = compiler.compile(split_transpose_recipe)
    plan = pyreloc.load_plan(compiled.plan_bytes)
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind(plan, {})
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind(plan, {"s0": 130})


def test_execution_errors_surface_with_context_and_never_retry(compiler, identity_recipe):
    from reloc_torch.runtime import ExecutionError, execute_or_fallback

    runtime = CountingRuntime(fail_execution=RuntimeError("device copy failed"))
    entry = make_entry(compiler.compile(identity_recipe), runtime, lambda src, *s: src.clone())
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(ExecutionError) as failure:
        execute_or_fallback(entry, x, [6], torch.device("cpu"))
    assert "h2d" in str(failure.value)
    assert "device copy failed" in str(failure.value)
    assert runtime.executions == 1
    assert entry.fallback_calls == 0
    assert entry.diagnostics.snapshot()["runtime_executions"] == 1
    assert entry.diagnostics.fallbacks == {}


def test_prepared_call_is_single_use(entry, counting_runtime):
    call = entry.runtime.preflight(entry.compiled, torch.arange(6, dtype=torch.float32), torch.device("cpu"))
    assert call.bindings == {"s0": 6}
    assert call.destination.shape == (6,) and call.destination.strides == (1,)
    first = entry.runtime.execute(call)
    call.consume()
    with pytest.raises(RuntimeError, match="already"):
        call.consume()
    assert first.shape == (6,)


def test_prepared_call_recheck_detects_stale_source(entry):
    call = entry.runtime.preflight(entry.compiled, torch.arange(6, dtype=torch.float32), torch.device("cpu"))
    call.src.resize_(3)
    with pytest.raises(RuntimeError, match="stale"):
        call.recheck()


def test_interception_suspension_is_scoped_nested_and_thread_local():
    runtime = api()
    assert runtime.interception_suspended() is False
    seen = {}
    with runtime.suspend_interception():
        assert runtime.interception_suspended() is True
        with runtime.suspend_interception():
            assert runtime.interception_suspended() is True
        assert runtime.interception_suspended() is True

        def probe():
            seen["other_thread"] = runtime.interception_suspended()

        worker = threading.Thread(target=probe)
        worker.start()
        worker.join()
        try:
            with runtime.suspend_interception():
                raise ValueError("boom")
        except ValueError:
            pass
        assert runtime.interception_suspended() is True
    assert runtime.interception_suspended() is False
    assert seen == {"other_thread": False}


def test_fallback_and_execution_run_with_interception_suspended(entry, counting_runtime):
    runtime = api()
    observed = []
    original = entry.original

    def recording(src, *symbols):
        observed.append(runtime.interception_suspended())
        return original(src, *symbols)

    entry.original = recording
    runtime.execute_or_fallback(entry, torch.arange(12, dtype=torch.float32)[::2], None, torch.device("cpu"))
    execute = counting_runtime.execute

    def recording_execute(call):
        observed.append(runtime.interception_suspended())
        return execute(call)

    counting_runtime.execute = recording_execute
    runtime.execute_or_fallback(entry, torch.arange(6, dtype=torch.float32), None, torch.device("cpu"))
    assert observed == [True, True]


def test_transport_adapter_reports_unavailable_r2_as_a_reason(compiler, identity_recipe):
    from reloc_torch.runtime import TransportAdapter, execute_or_fallback

    adapter = TransportAdapter()
    entry = make_entry(compiler.compile(identity_recipe), adapter, lambda src, *s: src.clone())
    x = torch.arange(6, dtype=torch.float32)
    actual = execute_or_fallback(entry, x, [6], torch.device("cpu"))
    assert torch.equal(actual, x)
    if adapter.available:
        pytest.skip("R2 transport is present; this test covers its absence")
    assert entry.diagnostics.fallbacks["runtime_unavailable"] == 1
    assert "transport" in adapter.unavailable_reason


def test_extent_guards_fall_back_for_singleton_bindings_before_the_adapter(compiler, counting_runtime):
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.runtime import ExecutionEntry, execute_or_fallback
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.symbolic import Const, Symbol

    rows, columns = Symbol("s0"), Symbol("s1")
    recipe = Recipe(
        TensorSpec((rows, columns), (columns, Const(1)), Const(0), "float32"),
        (Transpose((1, 0)),),
        TensorSpec((columns, rows), (rows, Const(1)), Const(0), "float32"),
        "h2d",
    )
    entry = ExecutionEntry(
        compiled=compiler.compile(recipe),
        original=lambda src, *s: src.transpose(0, 1).contiguous(),
        runtime=counting_runtime,
        diagnostics=Diagnostics(),
        extent_guards=(rows, columns),
    )
    singleton = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    actual = execute_or_fallback(entry, singleton, None, torch.device("cpu"))
    assert torch.equal(actual, singleton.t())
    assert actual.stride() == singleton.t().contiguous().stride()
    assert counting_runtime.preflights == 0 and counting_runtime.executions == 0
    assert entry.diagnostics.fallbacks["singleton_extent"] == 1
    wide = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    assert torch.equal(execute_or_fallback(entry, wide, None, torch.device("cpu")), wide.t().contiguous())
    assert counting_runtime.executions == 1


def test_transport_adapter_pins_the_r2_request_contract(monkeypatch, compiler, identity_recipe):
    import sys
    import types
    from reloc_torch.runtime import ConcreteDescriptor, PreparedCall, TransportAdapter, execute_or_fallback

    module = types.ModuleType("reloc_torch.transport")
    module.CAPABILITY_IDENTITY = "stub"
    requests = []

    class Loose:
        pass

    class Request:
        def __init__(self, compiled, src, device):
            self.bindings = compiled.bind_values(src)
            self.destination = ConcreteDescriptor(tuple(src.shape), tuple(src.stride()), "float32", torch.device(device))
            self.bound = None

    module.prepare_transfer = lambda compiled, src, device, *, non_blocking=False: (
        Loose() if getattr(module, "loose", False) else Request(compiled, src, device))

    def execute_transfer(request):
        requests.append(request)
        return torch.empty(request.destination.shape).copy_(current_source[0])

    module.execute_transfer = execute_transfer
    monkeypatch.setitem(sys.modules, "reloc_torch.transport", module)
    adapter = TransportAdapter()
    assert adapter.available
    assert adapter.capability_identity == "reloc_torch.transport/stub"
    entry = make_entry(compiler.compile(identity_recipe), adapter, lambda src, *s: src.clone())
    current_source = [torch.arange(6, dtype=torch.float32)]
    actual = execute_or_fallback(entry, current_source[0], [6], torch.device("cpu"))
    assert torch.equal(actual, current_source[0])
    assert isinstance(requests[0], Request)
    assert entry.diagnostics.snapshot()["runtime_executions"] == 1
    module.loose = True
    with pytest.raises(RuntimeError, match="documented attributes"):
        execute_or_fallback(entry, current_source[0], [6], torch.device("cpu"))
    assert entry.fallback_calls == 0


def test_transport_adapter_distinguishes_a_missing_module_from_a_broken_one(monkeypatch):
    import importlib
    from reloc_torch.runtime import TransportAdapter

    real_import_module = importlib.import_module

    def broken_import(name, package=None):
        if name == TransportAdapter.MODULE:
            raise ModuleNotFoundError("No module named 'pyreloc._cuda'", name="pyreloc._cuda")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="pyreloc._cuda"):
        TransportAdapter()
    monkeypatch.undo()
    adapter = TransportAdapter()
    if not adapter.available:
        assert "has not been delivered" in adapter.unavailable_reason
