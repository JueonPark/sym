"""Stacked graph rewrite (torch.stack support): one stack_transfer replaces stack ->
chain -> transfer; the size gate keeps small stacks on today's path."""
import pytest
import torch

from conftest import StackedCountingRuntime, stacked_recipe


def stacked_op():
    from reloc_torch import ops

    return ops.STACKED_OP


def capture(fn, *shapes):
    from reloc_torch.compat import symbolic_capture

    return symbolic_capture(fn, *(torch.ones(shape) for shape in shapes))


def transfer(x):
    return torch.ops.aten._to_copy.default(x, device=torch.device("cuda:0"))


def backend(compiler, **kwargs):
    from reloc_torch.backend import RelocBackend

    return RelocBackend(compiler=compiler, runtime=StackedCountingRuntime(), **kwargs)


def test_rewrite_emits_stack_transfer_over_the_inputs(compiler):
    b = backend(compiler, min_stack_bytes=0)
    compiled = b(capture(lambda x, y: transfer(torch.stack([x, y], 1)), (3, 4), (3, 4)), None)
    node, = [n for n in compiled.rewritten.graph.nodes if n.target is stacked_op()]
    srcs, handle, symbols, out_shape, out_strides, device = node.args
    assert [s.name for s in srcs] == ["x_1", "y_1"]
    assert all(s.target == torch.ops.aten.sym_size.int and s.args[0].name == "x_1" for s in symbols)
    assert sorted(s.args[1] for s in symbols) == [0, 1]
    assert device == torch.device("cuda", 0)
    targets = {str(n.target) for n in compiled.rewritten.graph.nodes}
    assert "aten.stack.default" not in targets and "aten._to_copy.default" not in targets
    stats = b.stats()
    assert stats["replaced_regions"] == 1 and stats["plan_compiles"] == 1
    b.close()


def test_static_inputs_below_the_threshold_keep_the_original_graph(compiler):
    gm = torch.fx.symbolic_trace(lambda x, y: torch.stack([x, y], 1).to("cuda"))
    example = [torch.ones(4, 5), torch.ones(4, 5)]
    gated = backend(compiler)
    compiled = gated(gm, example)
    assert compiled.rewritten is gm or compiled.rewritten is None
    stats = gated.stats()
    assert stats["exclusions"].get("below_stack_threshold") == 1
    assert stats["plan_compiles"] == 0
    gated.close()
    always = backend(compiler, min_stack_bytes=0)
    assert any(n.target is stacked_op() for n in always(gm, example).rewritten.graph.nodes)
    always.close()


def stacked_region(x, y, z):
    return torch.stack([x, y, z], 1)


def cpu_stacked_importer(gm):
    """Hand-built stacked candidate over a CPU-only stack (control flow tests)."""
    from reloc_torch.fx_import import Candidate, ImportReport

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    stack = next(n.name for n in gm.graph.nodes if n.op == "call_function")
    candidate = Candidate(stack, stack, (stack,), stacked_recipe(count=3, dim=1), (), (), stacked_region,
                          sources=tuple(names))
    return lambda graph, inputs: ImportReport((candidate,), ())


def test_cpu_stacked_region_executes_and_replays_only_itself(compiler):
    from reloc_torch.backend import RelocBackend

    gm = capture(stacked_region, (4, 5), (4, 5), (4, 5))
    runtime = StackedCountingRuntime()
    b = RelocBackend(compiler=compiler, runtime=runtime, importer=cpu_stacked_importer(gm), min_stack_bytes=0)
    compiled = b(gm, None)
    a, c = torch.arange(20.0).reshape(4, 5), torch.arange(20.0, 40.0).reshape(4, 5)
    assert torch.equal(compiled(a, a, c), torch.stack([a, a, c], 1))
    assert runtime.executions == 1
    strided = torch.arange(40.0).reshape(4, 10)[:, ::2]
    assert torch.equal(compiled(a, strided, c), torch.stack([a, strided, c], 1))
    assert runtime.executions == 1
    assert b.stats()["fallbacks"] == {"unsupported_layout": 1}
    other = [torch.arange(42.0).reshape(6, 7) + i for i in range(3)]  # new shape, same artifact
    assert torch.equal(compiled(*other), torch.stack(other, 1))
    assert runtime.executions == 2
    assert b.stats()["plan_compiles"] == 1
    b.close()


def test_an_adapter_without_preflight_stacked_keeps_the_stack_in_pytorch(compiler):
    """preflight_stacked is optional in the RuntimeAdapter protocol: a custom
    adapter that implements only preflight/execute never gets a stacked
    region, so the region runs in PyTorch exactly as before stack fusion."""
    from conftest import CountingRuntime
    from reloc_torch.backend import RelocBackend

    gm = capture(stacked_region, (4, 5), (4, 5), (4, 5))
    runtime = CountingRuntime()
    b = RelocBackend(compiler=compiler, runtime=runtime, importer=cpu_stacked_importer(gm), min_stack_bytes=0)
    compiled = b(gm, None)
    assert compiled.rewritten is gm
    xs = [torch.arange(20.0).reshape(4, 5) + 100 * i for i in range(3)]
    result = compiled(*xs)
    expected = torch.stack(xs, 1)
    assert torch.equal(result, expected) and result.stride() == expected.stride()
    stats = b.stats()
    assert stats["exclusions"] == {"runtime_unavailable": 1}
    assert stats["replaced_regions"] == 0 and stats["plan_compiles"] == 0
    assert stats["stacked_executions"] == 0 and stats["runtime_executions"] == 0 and not stats["fallbacks"]
    assert runtime.preflights == 0 and runtime.executions == 0
    b.close()


def test_symbolic_inputs_are_gated_per_call(compiler):
    from reloc_torch.backend import RelocBackend

    gm = capture(stacked_region, (4, 5), (4, 5), (4, 5))
    runtime = StackedCountingRuntime()
    b = RelocBackend(compiler=compiler, runtime=runtime, importer=cpu_stacked_importer(gm), min_stack_bytes=1 << 30)
    xs = [torch.ones(4, 5) for _ in range(3)]
    assert torch.equal(b(gm, None)(*xs), torch.stack(xs, 1))
    assert runtime.executions == 0
    assert b.stats()["fallbacks"] == {"below_stack_threshold": 1}
    b.close()


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_min_stack_bytes_must_be_a_non_negative_int(compiler, bad):
    with pytest.raises(ValueError, match="min_stack_bytes"):
        backend(compiler, min_stack_bytes=bad)
