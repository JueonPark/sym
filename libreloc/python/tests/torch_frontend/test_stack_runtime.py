"""Guarded stacked execution (torch.stack support): every rejection runs the original
region once before any launch; launches go through the real stacked gather."""
import pytest
import torch

from conftest import StackedCountingRuntime, make_entry, stacked_recipe


def stacked_entry(compiler, runtime, *, count=3, dim=1, min_stack_bytes=0):
    calls = []

    def original(*args):
        calls.append(args)
        return torch.stack(args[:count], dim)

    entry = make_entry(compiler.compile(stacked_recipe(count=count, dim=dim)), runtime, original)
    entry.min_stack_bytes = min_stack_bytes
    entry.original_calls = calls
    return entry


def inputs(count=3, rows=4, columns=5, dtype=torch.float32):
    return [(torch.arange(rows * columns) + 100 * i).reshape(rows, columns).to(dtype) for i in range(count)]


def run(entry, xs):
    from reloc_torch.runtime import execute_stacked_or_fallback

    values = {"s0": xs[0].shape[0], "s1": xs[0].shape[1]}
    return execute_stacked_or_fallback(entry, xs, [values[n] for n in entry.compiled.symbols], torch.device("cpu"))


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stacked_call_runs_the_native_gather_once(compiler, dim):
    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime, dim=dim)
    xs = inputs()
    result = run(entry, xs)
    expected = torch.stack(xs, dim)
    assert torch.equal(result, expected) and result.stride() == expected.stride()
    assert runtime.executions == 1 and not entry.original_calls
    stats = entry.diagnostics.snapshot()
    assert stats["runtime_executions"] == 1 and stats["stacked_executions"] == 1
    assert stats["fallbacks"] == {}


def test_the_same_tensor_may_fill_several_positions(compiler):
    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    a, b = inputs(count=2)
    assert torch.equal(run(entry, [a, a, b]), torch.stack([a, a, b], 1))
    assert runtime.executions == 1


@pytest.mark.parametrize(("make", "reason"), [
    (lambda xs: [xs[0], xs[1].half(), xs[2]], "stack_dtype_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 6), xs[2]], "stack_shape_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 10)[:, ::2], xs[2]], "unsupported_layout"),
    (lambda xs: [xs[0], torch.ones(22)[2:].view(4, 5), xs[2]], "storage_offset"),
])
def test_rejected_inputs_run_the_original_region_once_without_a_launch(compiler, make, reason):
    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    xs = make(inputs())
    if reason == "stack_shape_mismatch":
        # PyTorch's own error, exactly as without Sym.
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            run(entry, xs)
    else:
        assert torch.equal(run(entry, xs), torch.stack(xs, 1))
    assert runtime.executions == 0 and len(entry.original_calls) == 1
    assert entry.diagnostics.snapshot()["fallbacks"] == {reason: 1}


def test_size_gate_falls_back_below_and_executes_at_the_threshold(compiler):
    runtime = StackedCountingRuntime()
    xs = inputs()
    total = 3 * 4 * 5 * 4
    below = stacked_entry(compiler, runtime, min_stack_bytes=total + 1)
    assert torch.equal(run(below, xs), torch.stack(xs, 1))
    assert runtime.executions == 0
    assert below.diagnostics.snapshot()["fallbacks"] == {"below_stack_threshold": 1}
    at = stacked_entry(compiler, runtime, min_stack_bytes=total)
    assert torch.equal(run(at, xs), torch.stack(xs, 1))
    assert runtime.executions == 1


@pytest.mark.parametrize(("make", "error"), [
    (lambda xs: [xs[0], torch.ones(4, 10)[:, ::2], xs[2]], None),
    (lambda xs: [xs[0], torch.ones(4, 6), xs[2]], "stack expects each tensor to be equal size"),
])
def test_size_gate_runs_before_any_per_input_work(compiler, monkeypatch, make, error):
    """Below the threshold the call falls back from the input count and input
    0's size alone: no input is guarded, snapshotted or bound, so even a
    guard-failing input records below_stack_threshold, and an invalid stack
    raises PyTorch's own error from the original region."""
    from reloc_torch import compat
    from reloc_torch import runtime as runtime_module

    calls = []

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return wrapper

    for name in ("source_reason", "bind_stacked_symbols", "destination_descriptor"):
        monkeypatch.setattr(runtime_module, name, counted(name, getattr(runtime_module, name)))
    monkeypatch.setattr(compat, "storage_snapshot", counted("storage_snapshot", compat.storage_snapshot))
    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime, min_stack_bytes=10 ** 9)
    xs = make(inputs())
    if error is None:
        assert torch.equal(run(entry, xs), torch.stack(xs, 1))
    else:
        with pytest.raises(RuntimeError, match=error):
            run(entry, xs)
    assert calls == []
    assert runtime.preflights == 0 and runtime.executions == 0 and len(entry.original_calls) == 1
    assert entry.diagnostics.snapshot()["fallbacks"] == {"below_stack_threshold": 1}


def test_an_adapter_without_preflight_stacked_falls_back(compiler):
    """A directly built entry over an adapter that implements only the
    required preflight/execute falls back with runtime_unavailable, never
    an AttributeError."""
    from conftest import CountingRuntime

    runtime = CountingRuntime()
    entry = stacked_entry(compiler, runtime)
    xs = inputs()
    result = run(entry, xs)
    expected = torch.stack(xs, 1)
    assert torch.equal(result, expected) and result.stride() == expected.stride()
    assert runtime.preflights == 0 and runtime.executions == 0 and len(entry.original_calls) == 1
    stats = entry.diagnostics.snapshot()
    assert stats["fallbacks"] == {"runtime_unavailable": 1} and stats["stacked_executions"] == 0


def test_a_list_of_the_wrong_length_is_an_error_not_a_fallback(compiler):
    from reloc_torch.runtime import execute_stacked_or_fallback

    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    with pytest.raises(RuntimeError, match="expected 3 stacked sources .* got 2"):
        execute_stacked_or_fallback(entry, inputs(count=2), [4, 5], torch.device("cpu"))
    assert runtime.executions == 0 and not entry.original_calls


def test_supplied_symbols_must_match_the_first_input(compiler):
    from reloc_torch.runtime import execute_stacked_or_fallback

    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    xs = inputs()
    assert torch.equal(execute_stacked_or_fallback(entry, xs, [7, 7], torch.device("cpu")), torch.stack(xs, 1))
    assert entry.diagnostics.snapshot()["fallbacks"] == {"symbol_mismatch": 1}


def test_failures_after_launch_raise_execution_error_without_replay(compiler):
    from reloc_torch.runtime import ExecutionError

    entry = stacked_entry(compiler, StackedCountingRuntime(fail_execution=RuntimeError("injected")))
    with pytest.raises(ExecutionError, match="injected"):
        run(entry, inputs())
    assert not entry.original_calls


def test_rejected_calls_with_no_symbols_replay_derived_scalar_placeholders(compiler):
    """A guarded refactor regression pin: when the caller supplies no symbols
    (``None``) and the entry carries symbolic_bindings, a fallback (here, the
    size gate) must replay the original region once with ``(*sources, values)``
    where ``values`` are derived from sources[0]: logical axis k of the
    stacked [N, *S] source is input axis k - 1."""
    from reloc_torch.runtime import execute_stacked_or_fallback
    from reloc_torch.symbolic import Symbol

    runtime = StackedCountingRuntime()
    calls = []

    def original(*args):
        calls.append(args)
        return torch.stack(args[:3], 1)

    compiled = compiler.compile(stacked_recipe(count=3, dim=1))
    entry = make_entry(
        compiled, runtime, original,
        symbolic_bindings=(("rows", Symbol("s0")), ("cols", Symbol("s1"))),
    )
    entry.min_stack_bytes = 10 ** 9  # far above any real stacked payload here
    xs = inputs()
    result = execute_stacked_or_fallback(entry, xs, None, torch.device("cpu"))
    assert torch.equal(result, torch.stack(xs, 1))
    assert runtime.executions == 0
    assert len(calls) == 1
    assert calls[0][:3] == tuple(xs)
    assert calls[0][3:] == (xs[0].shape[0], xs[0].shape[1])
    assert entry.diagnostics.snapshot()["fallbacks"] == {"below_stack_threshold": 1}


def test_declared_metadata_mismatch_is_an_error_with_zero_replays_and_zero_launches(compiler):
    from reloc_torch.runtime import ConcreteDescriptor, execute_stacked_or_fallback

    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    xs = inputs()
    # The compiled descriptor for count=3, dim=1 is (4, 3, 5); this is a
    # different shape entirely, so it can never match by accident.
    declared = ConcreteDescriptor((3, 4, 5), (20, 5, 1), "float32", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="declared output metadata"):
        execute_stacked_or_fallback(entry, xs, None, torch.device("cpu"), declared=declared)
    assert runtime.executions == 0 and not entry.original_calls


def test_closed_entry_raises_before_anything_else(compiler):
    from reloc_torch.runtime import execute_stacked_or_fallback

    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime)
    entry.close()
    xs = inputs()
    with pytest.raises(RuntimeError, match="closed"):
        execute_stacked_or_fallback(entry, xs, None, torch.device("cpu"))
    assert runtime.executions == 0 and runtime.preflights == 0 and not entry.original_calls


def test_min_stack_bytes_via_the_constructor_gates_as_expected(compiler):
    """min_stack_bytes set through the ExecutionEntry constructor keyword
    (rather than assigned on the instance afterwards) gates identically."""
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.runtime import ExecutionEntry

    runtime = StackedCountingRuntime()
    calls = []

    def original(*args):
        calls.append(args)
        return torch.stack(args[:3], 1)

    xs = inputs()
    total = 3 * 4 * 5 * 4
    below = ExecutionEntry(
        compiled=compiler.compile(stacked_recipe(count=3, dim=1)),
        original=original, runtime=runtime, diagnostics=Diagnostics(),
        min_stack_bytes=total + 1,
    )
    assert torch.equal(run(below, xs), torch.stack(xs, 1))
    assert runtime.executions == 0 and len(calls) == 1
    assert below.diagnostics.snapshot()["fallbacks"] == {"below_stack_threshold": 1}

    at = ExecutionEntry(
        compiled=compiler.compile(stacked_recipe(count=3, dim=1)),
        original=original, runtime=runtime, diagnostics=Diagnostics(),
        min_stack_bytes=total,
    )
    assert torch.equal(run(at, xs), torch.stack(xs, 1))
    assert runtime.executions == 1


class _Subclass(torch.Tensor):
    pass


def _empty_storage(x):
    x = x.clone()
    x.untyped_storage().resize_(0)
    return x


def _second(make, *, first=None):
    def build():
        xs = inputs() if first is None else [first() + i for i in range(3)]
        xs[1] = make(xs[1])
        return xs
    return build


# (inputs, grad mode, whether every input shares input 0's metadata when admitted)
LEAN_CASES = {
    "same_values": (_second(lambda x: x.clone()), True, True),
    "same_object": (lambda: [inputs()[0]] * 3, True, True),
    "parameter": (_second(lambda x: torch.nn.Parameter(x, requires_grad=False)), True, True),
    "requires_grad": (_second(lambda x: x.clone().requires_grad_()), True, None),
    "requires_grad_without_grad_mode": (_second(lambda x: x.clone().requires_grad_()), False, True),
    "subclass": (_second(lambda x: x.as_subclass(_Subclass)), True, None),
    "strided": (_second(lambda x: torch.ones(4, 10)[:, ::2]), True, None),
    "offset": (_second(lambda x: torch.ones(22)[2:].view(4, 5)), True, None),
    "float16": (_second(lambda x: x.half()), True, False),
    "bfloat16": (_second(lambda x: x.bfloat16()), True, None),
    "int32": (_second(lambda x: x.int()), True, None),
    "shape": (_second(lambda x: torch.ones(4, 6)), True, False),
    "rank_zero": (_second(lambda x: torch.tensor(1.0)), True, None),
    "empty": (_second(lambda x: torch.ones(0, 5)), True, None),
    "empty_storage": (_second(_empty_storage), True, None),
    "sparse": (_second(lambda x: x.to_sparse()), True, None),
    "meta": (_second(lambda x: torch.empty(4, 5, device="meta")), True, None),
    "size_one_stride": (_second(lambda x: torch.empty_strided((4, 1, 5), (5, 1, 1)).fill_(1),
                                first=lambda: torch.ones(4, 1, 5)), True, False),
    "last_input_strided": (lambda: inputs()[:2] + [torch.ones(4, 10)[:, ::2]], True, None),
}


@pytest.mark.parametrize("case", LEAN_CASES)
def test_lean_input_check_reports_exactly_the_full_guards_reason(case):
    """Input 0 runs the full source guard; a later input that matches it in
    every property the guard reads is admitted from that match, and any
    other input runs the full guard itself. The outcome is exactly the full
    guard applied to every input in order: same first reason, and nothing
    the full guard rejects is admitted."""
    from reloc_torch import compat
    from reloc_torch.runtime import check_stacked_sources, source_reason

    make, grad, expected_uniform = LEAN_CASES[case]
    xs = make()
    with torch.set_grad_enabled(grad):
        expected = next((r for r in map(source_reason, xs) if r is not None), None)
        reason, snapshots, uniform = check_stacked_sources(xs)
    assert reason == expected
    if expected is None:
        assert snapshots == tuple(compat.storage_snapshot(x) for x in xs)
        assert uniform is expected_uniform
    else:
        assert expected_uniform is None and snapshots is None


def test_uniform_inputs_run_the_full_guard_and_binding_once(compiler, monkeypatch):
    """Sixteen inputs that share input 0's metadata cost one full source
    guard and one binding per call: the adapter's preflight takes the
    frontend's validated binding instead of guarding and binding again."""
    from reloc_torch import runtime as runtime_module
    from reloc_torch.artifact import CompiledRecipe

    calls = []

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(runtime_module, "source_reason", counted("source_reason", runtime_module.source_reason))
    monkeypatch.setattr(CompiledRecipe, "bind_stacked_values",
                        counted("bind_stacked_values", CompiledRecipe.bind_stacked_values))
    runtime = StackedCountingRuntime()
    entry = stacked_entry(compiler, runtime, count=16)
    xs = inputs(count=16)
    result = run(entry, xs)
    assert torch.equal(result, torch.stack(xs, 1)) and runtime.executions == 1
    assert sorted(calls) == ["bind_stacked_values", "source_reason"]


def test_the_validated_binding_applies_only_to_the_same_request(compiler):
    from reloc_torch.runtime import _validated_stacked_binding, check_stacked_sources, validated_stacked_inputs

    compiled = compiler.compile(stacked_recipe(count=3, dim=1))
    other = compiler.compile(stacked_recipe(count=3, dim=0))
    xs = tuple(inputs())
    _, snapshots, uniform = check_stacked_sources(xs)
    bindings = {"s0": 4, "s1": 5}
    assert validated_stacked_inputs(compiled, xs) is None
    with _validated_stacked_binding(compiled, xs, snapshots, uniform, bindings):
        assert validated_stacked_inputs(compiled, xs) == (snapshots, uniform, bindings)
        assert validated_stacked_inputs(compiled, list(xs)) == (snapshots, uniform, bindings)
        assert validated_stacked_inputs(other, xs) is None
        assert validated_stacked_inputs(compiled, (xs[0], xs[1].clone(), xs[2])) is None
        assert validated_stacked_inputs(compiled, xs[:2]) is None
        xs[2].resize_(40)
        assert validated_stacked_inputs(compiled, xs) is None  # keyed on the snapshots too
    assert validated_stacked_inputs(compiled, xs) is None
