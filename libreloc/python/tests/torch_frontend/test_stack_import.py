"""Stacked region import (torch.stack support): torch.stack of graph inputs, the
layout chain after it and one host-to-device transfer form one candidate."""
import pytest
import torch

from reloc_torch.recipe import Transpose
from reloc_torch.symbolic import Const, Symbol, expression


def importer():
    from reloc_torch import fx_import

    return fx_import


def transfer(x, **kwargs):
    return torch.ops.aten._to_copy.default(x, device=torch.device("cuda:0"), **kwargs)


def capture(fn, *inputs):
    from reloc_torch.compat import symbolic_capture

    return symbolic_capture(fn, *(inputs or (torch.ones(3, 4), torch.ones(3, 4))))


@pytest.mark.parametrize(("dim", "operations", "shape"), [
    (0, (), (Const(2), Symbol("s0"), Symbol("s1"))),
    (1, (Transpose((1, 0, 2)),), (Symbol("s0"), Const(2), Symbol("s1"))),
    (-1, (Transpose((1, 2, 0)),), (Symbol("s0"), Symbol("s1"), Const(2))),
])
def test_make_fx_stack_then_transfer_is_one_stacked_candidate(dim, operations, shape):
    report = importer().import_graph(capture(lambda x, y: transfer(torch.stack([x, y], dim))))
    candidate, = report.candidates
    assert candidate.sources == ("x_1", "y_1")
    assert candidate.members[0] == candidate.source
    recipe = candidate.recipe
    assert recipe.stack_inputs == 2
    assert recipe.source.shape == (Const(2), Symbol("s0"), Symbol("s1"))
    assert recipe.operations == operations
    assert recipe.destination.shape == shape
    assert recipe.direction == "h2d"


CHAINS = [
    # (captured region, the same region on the CPU)
    (lambda x, y: transfer(torch.stack([x, y], 0).permute(2, 0, 1).contiguous()),
     lambda x, y: torch.stack([x, y], 0).permute(2, 0, 1).contiguous()),
    # Merges the stack axis into the next one. Stacked or not, the importer
    # rejects a -1 beside a symbolic extent (a dynamic split factor) and the
    # exporter folds no merging reshape after a transpose (a dim > 0 stack).
    (lambda x, y: transfer(torch.stack([x, y], 0).reshape(2 * x.shape[0], x.shape[1])),
     lambda x, y: torch.stack([x, y], 0).reshape(2 * x.shape[0], x.shape[1])),
    (lambda x, y: transfer(torch.ops.aten.constant_pad_nd.default(torch.stack([x, y], 0), [1, 2])),
     lambda x, y: torch.ops.aten.constant_pad_nd.default(torch.stack([x, y], 0), [1, 2])),
    # Pads the stack axis itself: fill slices that belong to no input.
    (lambda x, y: transfer(torch.ops.aten.constant_pad_nd.default(torch.stack([x, y], -1), [1, 2])),
     lambda x, y: torch.ops.aten.constant_pad_nd.default(torch.stack([x, y], -1), [1, 2])),
    (lambda x, y: torch.ops.aten.clone.default(transfer(torch.stack([x, y], 1)).transpose(0, 1),
                                               memory_format=torch.contiguous_format),
     lambda x, y: torch.ops.aten.clone.default(torch.stack([x, y], 1).transpose(0, 1),
                                               memory_format=torch.contiguous_format)),
]


@pytest.mark.parametrize(("region", "cpu"), CHAINS)
def test_layout_chains_after_the_stack_fold_into_one_recipe(compiler, region, cpu):
    candidate, = importer().import_graph(capture(region)).candidates
    recipe = candidate.recipe
    assert recipe.stack_inputs == 2
    expected = cpu(torch.ones(3, 4), torch.ones(3, 4))
    bindings = {"s0": 3, "s1": 4}
    assert tuple(expression(d).evaluate(bindings) for d in recipe.destination.shape) == tuple(expected.shape)
    assert compiler.compile(recipe).recipe.stack_inputs == 2  # the unchanged exporter folds it


@pytest.mark.parametrize("count", [1, 16])
def test_any_number_of_inputs_forms_one_candidate(count):
    xs = [torch.ones(3, 4) for _ in range(count)]
    candidate, = importer().import_graph(capture(lambda *ts: transfer(torch.stack(ts, 1)), *xs)).candidates
    assert len(candidate.sources) == count
    assert candidate.recipe.stack_inputs == count
    assert candidate.recipe.source.shape == (Const(count), Symbol("s0"), Symbol("s1"))


def test_original_callable_takes_one_source_per_position():
    candidate, = importer().import_graph(capture(lambda x, y: transfer(torch.stack([x, x, y], 1)))).candidates
    assert candidate.sources == ("x_1", "x_1", "y_1")
    placeholders = [n for n in candidate.original.graph.nodes if n.op == "placeholder"]
    assert [n.name for n in placeholders] == ["src0", "src1", "src2"]
    stack = next(n for n in candidate.original.graph.nodes if n.target == torch.ops.aten.stack.default)
    assert list(stack.args[0]) == placeholders


def test_original_reads_an_input_size_from_its_first_position():
    def region(x, y):
        return transfer(torch.stack([y, x, x], 1).reshape(x.shape[0], 3 * x.shape[1]))

    candidate, = importer().import_graph(capture(region)).candidates
    sizes = [n for n in candidate.original.graph.nodes if n.target == torch.ops.aten.sym_size.int]
    assert sizes and all(n.args[0].name == "src1" for n in sizes)


def _mutate_after_stack(x, y):
    stacked = torch.stack([x, y], 0)
    x.add_(1)
    return transfer(stacked)


def _escaping_stack(x, y):
    stacked = torch.stack([x, y], 0)
    return transfer(stacked), stacked


EXCLUSIONS = [
    (lambda x, y: transfer(torch.stack([x, y], 0)), (torch.ones(3, 4), torch.ones(3, 4, dtype=torch.float16)),
     "stack_dtype_mismatch"),
    (lambda x, y: transfer(torch.stack([x, y + 1], 0)), (), "stack_input_not_root"),
    (lambda x, y: transfer(torch.stack([x, y], 0), dtype=torch.float16), (), "typed_transform_unavailable"),
    (lambda x, y: transfer(torch.stack([x, y], 0)).cpu(), (), "multiple_transfers"),
    (_mutate_after_stack, (), "mutation"),
    (_escaping_stack, (), "escaping_intermediate"),
]


@pytest.mark.parametrize(("fn", "inputs", "reason"), EXCLUSIONS)
def test_stacked_exclusions_keep_the_original_region(fn, inputs, reason):
    report = importer().import_graph(capture(fn, *inputs))
    assert not any(c.sources for c in report.candidates)
    assert reason in {e.reason for e in report.exclusions}


def test_out_argument_is_rejected():
    def fn(x, y, out):
        return torch.stack([x, y], 0, out=out).to("cuda")

    report = importer().import_graph(torch.fx.symbolic_trace(fn),
                                     (torch.ones(3, 4), torch.ones(3, 4), torch.empty(2, 3, 4)))
    assert not report.candidates
    assert "stack_out_argument" in {e.reason for e in report.exclusions}


def test_keyword_dim_of_an_aten_stack_is_honored():
    def fn(x, y):
        return torch.ops.aten.stack.default([x, y], dim=1).to("cuda")

    # Two 2 x 2 inputs: stacking at dim 0 or dim 1 gives the same shape.
    report = importer().import_graph(torch.fx.symbolic_trace(fn), (torch.ones(2, 2), torch.ones(2, 2)))
    candidate, = report.candidates
    assert candidate.recipe.operations == (Transpose((1, 0, 2)),)


@pytest.mark.parametrize("stack", [torch.stack, torch.ops.aten.stack.default], ids=["torch", "aten"])
def test_keyword_tensors_form_imports_and_rebuilds_over_its_own_placeholders(stack):
    def fn(x, y):
        return stack(tensors=[x, y], dim=1).to("cuda")

    gm = torch.fx.symbolic_trace(fn)
    users = [(n, tuple(n.users)) for n in gm.graph.nodes]
    candidate, = importer().import_graph(gm, (torch.ones(2, 3), torch.ones(2, 3))).candidates
    assert [(n, tuple(n.users)) for n in gm.graph.nodes] == users  # the caller's graph is unchanged
    assert candidate.sources == ("x", "y")
    assert candidate.recipe.operations == (Transpose((1, 0, 2)),)
    original = candidate.original.graph
    placeholders = [n for n in original.nodes if n.op == "placeholder"]
    rebuilt = next(n for n in original.nodes if n.target is stack)
    assert (rebuilt.args, dict(rebuilt.kwargs)) == ((), {"tensors": placeholders, "dim": 1})
    assert all(i.graph is original for n in original.nodes for i in n.all_input_nodes)


def test_dynamo_torch_stack_call_is_canonicalized_and_imported():
    graphs = []

    def record(gm, example_inputs):
        graphs.append((gm, example_inputs))
        return gm.forward

    torch.compile(lambda x, y: torch.stack([x, y], dim=1), backend=record, dynamic=True, fullgraph=True)(
        torch.ones(4, 6), torch.ones(4, 6))
    gm, example_inputs = graphs[0]
    stack = next(n for n in gm.graph.nodes if n.target is torch.stack)
    output = next(n for n in gm.graph.nodes if n.op == "output")
    with gm.graph.inserting_before(output):
        tail = gm.graph.call_method("to", (output.args[0][0], "cuda"))
    output.args = ((tail,),)
    gm.recompile()
    candidate, = importer().import_graph(gm, example_inputs).candidates
    assert candidate.source == stack.name
    assert len(candidate.sources) == 2
    assert candidate.recipe.stack_inputs == 2
    assert candidate.recipe.operations == (Transpose((1, 0, 2)),)


def test_dynamo_padded_stack_chain_imports_as_a_stacked_candidate():
    """Pins the exact chain the GPU acceptance test exercises
    (torch.nn.functional.pad(torch.stack([a, b], 1), (1, 2)).to("cuda")); it must
    import as one stacked candidate, not fall back via destination_layout. A
    stacked chain's extra Const stack-count axis multiplies a padded
    symbolic dimension's compound (Const + Symbol) extent into its
    destination stride; SymPy's real traced value for that stride arrives
    already distributed over the Add while dense_strides keeps ours
    factored, so a stack-scoped algebraic comparison is required here."""
    graphs = []

    def record(gm, example_inputs):
        graphs.append((gm, example_inputs))
        return gm.forward

    torch.compile(lambda a, b: torch.nn.functional.pad(torch.stack([a, b], 1), (1, 2)),
                 backend=record, dynamic=True, fullgraph=True)(torch.ones(48, 80), torch.ones(48, 80))
    gm, example_inputs = graphs[0]
    stack = next(n for n in gm.graph.nodes if n.target is torch.stack)
    output = next(n for n in gm.graph.nodes if n.op == "output")
    with gm.graph.inserting_before(output):
        tail = gm.graph.call_method("to", (output.args[0][0], "cuda"))
    output.args = ((tail,),)
    gm.recompile()
    report = importer().import_graph(gm, example_inputs)
    assert not report.exclusions
    candidate, = report.candidates
    assert candidate.source == stack.name
    assert len(candidate.sources) == 2
    assert candidate.recipe.stack_inputs == 2
    assert candidate.recipe.direction == "h2d"


def test_dynamo_closure_captured_dim_imports_with_the_right_move_axis_operation():
    """Pins 196e29a: Dynamo's unspecialized-int tracing can lift a plain
    closure-captured dim into a graph node (an `l_dim_` SymInt graph input)
    even though torch.compile sees the same concrete value on every call --
    exactly what happens when dim is an outer (e.g. parametrized test)
    function's argument, as in test_stack_transfers_gpu.py. The importer
    must resolve it to a constant and pick the matching move-axis
    operation, not reject it as unsupported_symbolic_expr."""
    def make_fn(dim):
        def fn(x, y):
            return torch.stack([x, y], dim)
        return fn

    graphs = []

    def record(gm, example_inputs):
        graphs.append((gm, example_inputs))
        return gm.forward

    torch.compile(make_fn(1), backend=record, dynamic=True, fullgraph=True)(torch.ones(4, 6), torch.ones(4, 6))
    gm, example_inputs = graphs[0]
    stack = next(n for n in gm.graph.nodes if n.target is torch.stack)
    # Confirms this really exercises the Node-lifted dim 196e29a fixed (a
    # closure read), not a plain literal embedded in the callee's bytecode.
    assert hasattr(stack.args[1], "op")
    output = next(n for n in gm.graph.nodes if n.op == "output")
    with gm.graph.inserting_before(output):
        tail = gm.graph.call_method("to", (output.args[0][0], "cuda"))
    output.args = ((tail,),)
    gm.recompile()
    report = importer().import_graph(gm, example_inputs)
    assert not report.exclusions
    candidate, = report.candidates
    assert candidate.recipe.stack_inputs == 2
    assert candidate.recipe.operations == (Transpose((1, 0, 2)),)  # dim=1's move-axis permutation


def test_dynamo_genuinely_dynamic_dim_is_excluded_as_unsupported_symbolic_expr():
    """Pins 196e29a's boundary: a dim that is actually shape-dependent (not
    merely unspecialized-but-constant) must still be rejected -- it cannot
    fold to a plain int, so it can never select a concrete move-axis
    permutation. The importer reports unsupported_symbolic_expr (the same
    reason a non-constant dim already gave before 196e29a; that commit only
    stopped rejecting a dim that folds to a constant)."""
    def fn(x, y):
        dim = x.shape[0] % 2  # a genuine Mod(Symbol, 2): never a Const
        return torch.stack([x, y], dim)

    graphs = []

    def record(gm, example_inputs):
        graphs.append((gm, example_inputs))
        return gm.forward

    torch.compile(fn, backend=record, dynamic=True, fullgraph=True)(torch.ones(4, 6), torch.ones(4, 6))
    gm, example_inputs = graphs[0]
    output = next(n for n in gm.graph.nodes if n.op == "output")
    with gm.graph.inserting_before(output):
        tail = gm.graph.call_method("to", (output.args[0][0], "cuda"))
    output.args = ((tail,),)
    gm.recompile()
    report = importer().import_graph(gm, example_inputs)
    assert not report.candidates
    assert {e.reason for e in report.exclusions} == {"unsupported_symbolic_expr"}


def test_copy_then_stack_stays_n_independent_identity_candidates():
    report = importer().import_graph(capture(lambda x, y: torch.stack([transfer(x), transfer(y)], 1)))
    assert len(report.candidates) == 2
    assert all(not c.sources and c.recipe.stack_inputs == 0 for c in report.candidates)


def test_a_pure_stack_inside_a_single_source_region_is_no_side_effect():
    def fn(x, a, b):
        t = x.t()
        s = torch.stack([a, b])
        return transfer(torch.ops.aten.clone.default(t, memory_format=torch.contiguous_format)), s

    report = importer().import_graph(capture(fn, torch.ones(3, 4), torch.ones(2, 5), torch.ones(2, 5)))
    assert not report.exclusions
    candidate, = report.candidates
    assert (candidate.source, candidate.sources) == ("x_1", ())
    recipe = candidate.recipe
    assert recipe.stack_inputs == 0 and recipe.direction == "h2d"
    assert recipe.operations == (Transpose((1, 0)),)
    assert recipe.destination.shape == (Symbol("s1"), Symbol("s0"))
