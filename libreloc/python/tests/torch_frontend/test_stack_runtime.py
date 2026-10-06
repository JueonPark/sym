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
