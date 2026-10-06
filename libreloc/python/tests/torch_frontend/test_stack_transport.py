"""Stacked preflight (torch.stack support): every input is guarded and validated as one
slice of the logical source before anything is allocated or launched."""
import types

import pytest
import torch

import pyreloc
from conftest import stacked_recipe


@pytest.fixture
def compiled(compiler):
    return compiler.compile(stacked_recipe(count=3, dim=1))


def inputs(rows=4, columns=5):
    return [torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) + 100 * i
            for i in range(3)]


def test_host_preflight_binds_once_and_rechecks_every_input(compiled):
    from reloc_torch.runtime import prepare_stacked_host_call

    xs = inputs()
    call = prepare_stacked_host_call(compiled, xs, torch.device("cpu"))
    assert call.sources == tuple(xs) and call.src is xs[0]
    assert call.bindings == {"s0": 4, "s1": 5}
    assert call.destination.shape == (4, 3, 5)
    call.recheck()
    xs[2].unsqueeze_(0)
    with pytest.raises(RuntimeError, match="stale prepared call"):
        call.recheck()


@pytest.mark.parametrize(("make", "reason"), [
    (lambda xs: [xs[0], xs[1]], "stack_count"),
    (lambda xs: [xs[0], xs[1].half(), xs[2]], "stack_dtype_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 6), xs[2]], "stack_shape_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 10)[:, ::2], xs[2]], "unsupported_layout"),
    (lambda xs: [xs[0], torch.ones(22)[2:].view(4, 5), xs[2]], "storage_offset"),
])
def test_host_preflight_rejections_have_stable_reasons(compiled, make, reason):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.runtime import prepare_stacked_host_call

    with pytest.raises(UnsupportedRecipe) as error:
        prepare_stacked_host_call(compiled, make(inputs()), torch.device("cpu"))
    assert error.value.reason == reason


def test_transport_rejects_before_allocation(compiled):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.transport import prepare_stacked_transfer

    xs = inputs()
    for kwargs, device, reason in (
        ({}, "cpu", "direction_mismatch"),
        ({"non_blocking": True}, "cuda", "nonblocking_unavailable"),
    ):
        with pytest.raises(UnsupportedRecipe) as error:
            prepare_stacked_transfer(compiled, xs, device, **kwargs)
        assert error.value.reason == reason
    with pytest.raises(UnsupportedRecipe) as error:
        prepare_stacked_transfer(compiled, xs[:2], "cuda")
    assert error.value.reason == "plan_mismatch"
    if not (pyreloc.cuda_enabled and torch.cuda.is_available()):
        with pytest.raises(UnsupportedRecipe) as error:
            prepare_stacked_transfer(compiled, xs, "cuda")
        assert error.value.reason == "cuda_unavailable"


def test_transport_adapter_routes_stacked_preflight_and_refuses_typed(compiled):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.runtime import TransportAdapter

    adapter = TransportAdapter()
    try:
        with pytest.raises(UnsupportedRecipe) as error:
            adapter.preflight_stacked(types.SimpleNamespace(typed=True), inputs(), torch.device("cuda"))
        assert error.value.reason == "typed_transform_unavailable"
        with pytest.raises(UnsupportedRecipe) as error:
            adapter.preflight_stacked(compiled, inputs(), torch.device("cpu"))
        assert error.value.reason == "direction_mismatch"
    finally:
        adapter.close()


@pytest.mark.parametrize("make", [
    lambda: torch.arange(20.0).reshape(4, 5),
    lambda: torch.ones(22)[2:].view(4, 5),
    lambda: torch.ones(4, 10)[:, ::2],
    lambda: torch.ones(7, dtype=torch.int8),
])
def test_a_view_built_from_a_snapshot_matches_the_live_storage_view(make):
    """Stacked requests build their native views from the storage snapshot
    their recheck compares against; the view equals the live one."""
    from reloc_torch.compat import storage_snapshot
    from reloc_torch.transport import _snapshot_view, _storage_view

    tensor = make()
    live = _storage_view(tensor, "host", -1)
    built = _snapshot_view(storage_snapshot(tensor), tensor.element_size(), "host", -1)
    fields = ("base", "capacity_bytes", "offset_bytes", "extents", "strides", "element_size", "kind", "device")
    assert [getattr(built, name) for name in fields] == [getattr(live, name) for name in fields]


def _count_guards_binds_and_snapshots(monkeypatch):
    from collections import Counter

    from reloc_torch import compat
    from reloc_torch import runtime as runtime_module
    from reloc_torch import transport as transport_module
    from reloc_torch.artifact import CompiledRecipe

    counts = Counter()

    def counted(name, function):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)
        return wrapper

    # Every name a guard or snapshot is reached through, including the
    # module-level aliases runtime and transport hold.
    source_reason = counted("source_reason", runtime_module.source_reason)
    snapshot = counted("storage_snapshot", compat.storage_snapshot)
    monkeypatch.setattr(runtime_module, "source_reason", source_reason)
    monkeypatch.setattr(transport_module, "source_reason", source_reason)
    monkeypatch.setattr(CompiledRecipe, "bind_stacked_values",
                        counted("bind_stacked_values", CompiledRecipe.bind_stacked_values))
    monkeypatch.setattr(compat, "storage_snapshot", snapshot)
    monkeypatch.setattr(runtime_module, "_metadata_snapshot", snapshot)
    return counts


def stacked_inputs(count, rows=4, columns=5):
    return [torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) + 100 * i
            for i in range(count)]


@pytest.mark.parametrize("count", [2, 16])
def test_a_fused_call_guards_binds_and_snapshots_each_request_once(compiler, cuda_device, monkeypatch, count):
    """Through the frontend, the transport takes the frontend's validated
    binding: one full source guard (input 0) and one binding per request.
    Each input's storage is snapshotted three times: by the frontend's
    check (the handoff key), by the transport when it verifies that key
    (the request's initial snapshot, which its native views are built
    from), and in the recheck immediately before execution."""
    from conftest import make_entry
    from reloc_torch.runtime import TransportAdapter, execute_stacked_or_fallback

    adapter = TransportAdapter()
    entry = make_entry(compiler.compile(stacked_recipe(count=count, dim=1)), adapter,
                       lambda *xs: torch.stack(xs, 1).to(cuda_device))
    xs = stacked_inputs(count)
    values = {"s0": 4, "s1": 5}
    counts = _count_guards_binds_and_snapshots(monkeypatch)
    try:
        result = execute_stacked_or_fallback(entry, xs, [values[n] for n in entry.compiled.symbols], cuda_device)
    finally:
        adapter.close()
    assert torch.equal(result.cpu(), torch.stack(xs, 1))
    assert entry.diagnostics.snapshot()["stacked_executions"] == 1
    assert counts == {"source_reason": 1, "bind_stacked_values": 1, "storage_snapshot": 3 * count}


@pytest.mark.parametrize("count", [2, 16])
def test_a_direct_stacked_request_checks_every_input_leanly(compiler, cuda_device, monkeypatch, count):
    """Direct transport callers (no frontend binding) still run every check:
    the full source guard on input 0 and the lean check on the others, one
    binding, and one storage snapshot pair per input."""
    from reloc_torch.transport import execute_transfer, prepare_stacked_transfer

    compiled = compiler.compile(stacked_recipe(count=count, dim=1))
    xs = stacked_inputs(count)
    counts = _count_guards_binds_and_snapshots(monkeypatch)
    out = execute_transfer(prepare_stacked_transfer(compiled, xs, cuda_device))
    assert torch.equal(out.cpu(), torch.stack(xs, 1))
    assert counts == {"source_reason": 1, "bind_stacked_values": 1, "storage_snapshot": 2 * count}


@pytest.mark.parametrize(("make", "reason"), [
    (lambda xs: [xs[0], torch.ones(4, 10)[:, ::2], xs[2]], "unsupported_layout"),
    (lambda xs: [xs[0], xs[1], torch.ones(22)[2:].view(4, 5)], "storage_offset"),
    (lambda xs: [xs[0], xs[1].clone().requires_grad_(), xs[2]], "requires_grad"),
    (lambda xs: [xs[0], xs[1].bfloat16(), xs[2]], "unsupported_dtype"),
    (lambda xs: [xs[0], xs[1].half(), xs[2]], "stack_dtype_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 6), xs[2]], "stack_shape_mismatch"),
])
def test_direct_stacked_request_rejections_have_stable_reasons(compiled, cuda_device, make, reason):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.transport import prepare_stacked_transfer

    with pytest.raises(UnsupportedRecipe) as error:
        prepare_stacked_transfer(compiled, make(inputs()), cuda_device)
    assert error.value.reason == reason


class _MutatingAdapter:
    """Changes an input after the frontend validated it, then preflights."""

    capability_identity = "test/mutating"

    def __init__(self, inner, mutate):
        self._inner = inner
        self._mutate = mutate

    def preflight(self, *args, **kwargs):
        return self._inner.preflight(*args, **kwargs)

    def preflight_stacked(self, compiled, sources, device, **kwargs):
        self._mutate(sources)
        return self._inner.preflight_stacked(compiled, sources, device, **kwargs)

    def execute(self, call):
        return self._inner.execute(call)


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda xs: xs[2].resize_(40), "stack_shape_mismatch"),
    (lambda xs: xs[1].set_(torch.zeros(4, 5)), None),
], ids=["resize_last", "set_middle"])
def test_inputs_changed_after_validation_are_checked_again(compiler, cuda_device, mutate, reason):
    """The frontend's validated binding is keyed on the input objects and
    their storage snapshots: an input changed between the frontend's checks
    and the transport's preflight gets every check again, never the stale
    binding. A changed shape falls back (PyTorch then raises its own error);
    a new storage of the same shape is transferred as it is now."""
    from conftest import make_entry
    from reloc_torch.runtime import TransportAdapter, execute_stacked_or_fallback

    inner = TransportAdapter()
    entry = make_entry(compiler.compile(stacked_recipe(count=3, dim=1)), _MutatingAdapter(inner, mutate),
                       lambda *xs: torch.stack(xs, 1).to(cuda_device))
    xs = inputs()
    symbols = [{"s0": 4, "s1": 5}[n] for n in entry.compiled.symbols]
    try:
        if reason is None:
            result = execute_stacked_or_fallback(entry, xs, symbols, cuda_device)
            assert torch.equal(result.cpu(), torch.stack(xs, 1))
        else:
            with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
                execute_stacked_or_fallback(entry, xs, symbols, cuda_device)
    finally:
        inner.close()
    stats = entry.diagnostics.snapshot()
    assert stats["fallbacks"] == ({} if reason is None else {reason: 1})
    assert stats["stacked_executions"] == (1 if reason is None else 0)


@pytest.mark.parametrize("adopted", [False, True], ids=["taken_by_the_request", "adopted_from_preflight"])
@pytest.mark.parametrize("mutate", [
    lambda xs: xs[2].resize_(40),
    lambda xs: xs[1].set_(torch.zeros(4, 5)),
], ids=["resize_last", "set_middle"])
def test_a_stacked_request_rechecks_every_input_before_execution(compiled, mutate, adopted):
    """The request's recheck covers every stacked input, not only the
    first: a resized or re-pointed later input makes it stale before any
    native view of the old storage can be read."""
    from reloc_torch.compat import storage_snapshot
    from reloc_torch.runtime import bind_plan, bind_stacked_symbols, destination_descriptor
    from reloc_torch.transport import PreparedTransfer, _storage_view, execute_transfer

    xs = inputs()
    bindings = bind_stacked_symbols(compiled, xs)
    views = tuple(_storage_view(x, "host", -1) for x in xs)
    request = PreparedTransfer(
        compiled, xs[0], bindings, bind_plan(compiled, bindings),
        destination_descriptor(compiled, bindings, torch.device("cpu")), "h2d", torch.device("cpu"),
        views[0], 0, stack_sources=tuple(xs), stack_views=views,
        stack_snapshots=tuple(storage_snapshot(x) for x in xs) if adopted else (),
    )
    request.recheck()
    mutate(xs)
    with pytest.raises(RuntimeError, match="stale transfer request"):
        execute_transfer(request)
    assert not request.consumed
