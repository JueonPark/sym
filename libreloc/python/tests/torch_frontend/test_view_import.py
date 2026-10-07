"""Issue #210: audited view spellings in transfer regions, with exact metadata."""
import pytest
import torch

from reloc_torch import compat, import_graph
from reloc_torch.fx_import import normalize_graph
from reloc_torch.recipe import Reshape


def transfer(x):
    return torch.ops.aten._to_copy.default(x, device=torch.device('cuda:0'))


VIEWS = [
    pytest.param(lambda x: x.unsqueeze(0), id='unsqueeze-method-leading'),
    pytest.param(lambda x: x.unsqueeze(dim=-1), id='unsqueeze-method-trailing'),
    pytest.param(lambda x: torch.unsqueeze(input=x, dim=-3), id='unsqueeze-function-keywords'),
    pytest.param(lambda x: torch.ops.aten.unsqueeze.default(self=x, dim=2), id='unsqueeze-aten-keywords'),
    pytest.param(lambda x: x.squeeze(), id='squeeze-method-all'),
    pytest.param(lambda x: x.squeeze(-2), id='squeeze-method-negative'),
    pytest.param(lambda x: x.squeeze(1), id='squeeze-method-noop'),
    pytest.param(lambda x: x.squeeze((0, -2)), id='squeeze-method-tuple'),
    pytest.param(lambda x: x.squeeze(()), id='squeeze-method-empty-tuple'),
    pytest.param(lambda x: torch.squeeze(input=x), id='squeeze-function-all'),
    pytest.param(lambda x: torch.squeeze(input=x, dim=(0, 2)), id='squeeze-function-tuple'),
    pytest.param(lambda x: torch.ops.aten.squeeze.default(x), id='squeeze-aten-all'),
    pytest.param(lambda x: torch.ops.aten.squeeze.dim(x, -4), id='squeeze-aten-dim'),
    pytest.param(lambda x: torch.ops.aten.squeeze.dims(x, [0, 2]), id='squeeze-aten-dims'),
    pytest.param(lambda x: x.flatten(), id='flatten-method-all'),
    pytest.param(lambda x: x.flatten(start_dim=1, end_dim=-2), id='flatten-method-range'),
    pytest.param(lambda x: x.flatten(2, 2), id='flatten-method-noop'),
    pytest.param(lambda x: torch.flatten(input=x, start_dim=-3), id='flatten-function-keywords'),
    pytest.param(lambda x: torch.ops.aten.flatten.using_ints(x, 1, 3), id='flatten-aten'),
]


@pytest.mark.parametrize('view', VIEWS)
@pytest.mark.parametrize('after_transfer', [False, True])
def test_raw_views_normalize_to_exact_aten_metadata_and_reshape(view, after_transfer):
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    fn = (lambda x: view(transfer(x))) if after_transfer else (lambda x: transfer(view(x)))
    gm = torch.fx.symbolic_trace(fn)
    before = str(gm.graph), [dict(n.meta) for n in gm.graph.nodes]
    normalized = normalize_graph(gm, (source,))
    assert not any('reloc_reason' in n.meta for n in normalized.graph.nodes)
    report = import_graph(gm, (source,))
    assert not report.exclusions
    candidate, = report.candidates
    assert len(candidate.members) == 2
    assert isinstance(candidate.recipe.operations[0], Reshape)
    expected = view(source)
    output = candidate.recipe.destination
    assert tuple(d.evaluate({}) for d in output.shape) == expected.shape
    assert tuple(s.evaluate({}) for s in output.strides) == expected.stride()
    assert output.offset.evaluate({}) == expected.storage_offset()
    # The original callable retains the user's spelling for guarded fallback.
    assert [n.target for n in candidate.original.graph.nodes if n.op.startswith('call_')] == [
        n.target for n in gm.graph.nodes if n.op.startswith('call_')]
    assert (str(gm.graph), [dict(n.meta) for n in gm.graph.nodes]) == before


@pytest.mark.parametrize('view', [
    lambda x: x.unsqueeze(-1).squeeze(-1),
    lambda x: x.squeeze((0, 2)).unsqueeze(1),
    lambda x: x.flatten(1, -1),
])
def test_symbolic_views_compile_and_execute_at_multiple_sizes(compiler, view):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    gm = compat.symbolic_capture(lambda x: transfer(view(x)), torch.ones(1, 3, 1, 4))
    report = import_graph(gm)
    assert not report.exclusions
    candidate, = report.candidates
    compiled = compiler.compile(candidate.recipe)
    plan = pyreloc.load_plan(compiled.plan_bytes)
    for rows, cols in ((3, 4), (5, 7)):
        source = torch.arange(rows * cols, dtype=torch.float32).reshape(1, rows, 1, cols)
        expected = view(source)
        actual = torch.empty_strided(expected.shape, expected.stride())
        bound = pyreloc.bind(plan, compiled.bind_values(source))
        pyreloc.relocate(bound, *as_ptr(source), *as_ptr(actual))
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()


@pytest.mark.parametrize('view', [
    lambda x: x.transpose(1, 3).flatten(1),
    lambda x: x.transpose(1, 3).unsqueeze(2).squeeze((0, 2, 3)).contiguous(),
])
@pytest.mark.parametrize('symbolic', [False, True])
def test_noncontiguous_intermediates_keep_metadata_and_compiler_gate(compiler, counting_runtime, view, symbolic):
    from reloc_torch import RelocBackend

    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    fn = lambda x: transfer(view(x))
    gm = compat.symbolic_capture(fn, source) if symbolic else torch.fx.symbolic_trace(fn)
    report = import_graph(gm, (source,))
    assert not report.exclusions
    candidate, = report.candidates
    expected = view(source)
    bindings = {s.name: source.shape[s.axis] for s in candidate.symbol_sources}
    assert tuple(d.evaluate(bindings) for d in candidate.recipe.destination.shape) == expected.shape
    assert tuple(s.evaluate(bindings) for s in candidate.recipe.destination.strides) == expected.stride()
    # The current compiler cannot merge these transposed source axes. Import
    # preserves the valid recipe, and the existing compiler gate retains Torch.
    backend = RelocBackend(compiler=compiler, runtime=counting_runtime)
    try:
        result = backend(gm, (source,))
        assert str(result.rewritten.graph) == str(gm.graph)
        assert backend.stats()['exclusions']['fold_unsupported'] == 1
        assert backend.stats()['replaced_regions'] == 0
    finally:
        backend.close()


@pytest.mark.parametrize(('view', 'shape', 'reason'), [
    (lambda x: x.squeeze(), (1, 1), 'rank_zero'),
    (lambda x: x.unsqueeze(0), (), 'rank_zero'),
    (lambda x: x.flatten(), (0, 3), 'empty_tensor'),
    # This view is contiguous in Torch, but its singleton stride is not dense.
    # contiguous() is a no-op: replacing it by a dense result would be wrong.
    (lambda x: x.t().unsqueeze(0).contiguous(), (1, 3), 'destination_layout'),
    (lambda x: x.t().flatten(0, 0), (3, 4), 'destination_layout'),
])
def test_unsupported_rank_empty_and_noncanonical_strides_fall_back(view, shape, reason):
    # Put the view after the transfer so _to_copy cannot canonicalize its stride.
    gm = torch.fx.symbolic_trace(lambda x: view(transfer(x)))
    report = import_graph(gm, (torch.ones(shape),))
    assert not report.candidates
    assert reason in {e.reason for e in report.exclusions}


@pytest.mark.parametrize('view', [
    lambda x: x.unsqueeze(-5),
    lambda x: x.squeeze((0, -3)),  # duplicate after negative-dimension normalization
    lambda x: x.flatten(2, 1),
    lambda x: x.flatten(0, 4),
])
def test_invalid_dimensions_keep_original_exception(view):
    gm = torch.fx.symbolic_trace(lambda x: transfer(view(x)))
    before = str(gm.graph)
    source = torch.ones(1, 3, 4)
    report = import_graph(gm, (source,))
    assert not report.candidates
    assert report.exclusions
    assert str(gm.graph) == before
    with pytest.raises((IndexError, RuntimeError)):
        gm(source)


@pytest.mark.parametrize('view', [lambda x: x.unsqueeze(0), lambda x: x.squeeze(0), lambda x: x.flatten()])
def test_view_alias_mutation_and_escaping_intermediates_are_rejected(view):
    def mutation(x):
        alias = view(x)
        result = transfer(x)
        alias.add_(1)
        return result

    def escaping(x):
        alias = view(x)
        return transfer(alias), alias

    for fn, reason in ((mutation, 'mutation'), (escaping, 'escaping_intermediate')):
        gm = torch.fx.symbolic_trace(fn)
        report = import_graph(gm, (torch.ones(1, 3, 4),))
        assert not report.candidates
        assert reason in {e.reason for e in report.exclusions}


def test_symbolic_noop_squeeze_records_guard_and_falls_back_when_rank_changes(compiler, counting_runtime):
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.runtime import ExecutionEntry, execute_or_fallback
    from reloc_torch.symbolic import Symbol

    gm = compat.symbolic_capture(lambda x: transfer(x.squeeze(0)), torch.ones(3, 4))
    candidate, = import_graph(gm).candidates
    assert candidate.extent_guards == (Symbol('s0'),)
    entry = ExecutionEntry(
        compiled=compiler.compile(candidate.recipe), original=lambda x: x.squeeze(0).clone(),
        runtime=counting_runtime, diagnostics=Diagnostics(), extent_guards=candidate.extent_guards)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(execute_or_fallback(entry, source, None, 'cpu'), source)
    assert counting_runtime.executions == 1
    singleton = source[:1].clone()
    actual = execute_or_fallback(entry, singleton, None, 'cpu')
    assert torch.equal(actual, singleton.squeeze(0))
    assert actual.shape == (4,)
    assert counting_runtime.executions == 1
    assert entry.diagnostics.fallbacks['singleton_extent'] == 1


def test_squeeze_of_conditional_singleton_is_not_specialized_from_a_hint():
    gm = compat.symbolic_capture(
        lambda x: transfer(x.reshape(x.shape[0] // 4, 4).squeeze(0)), torch.ones(4))
    report = import_graph(gm)
    assert not report.candidates
    assert 'conditional_squeeze' in {e.reason for e in report.exclusions}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.int8])
@pytest.mark.parametrize('view', [lambda x: x.unsqueeze(1), lambda x: x.squeeze(0), lambda x: x.flatten(1)])
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.gpu
def test_compiled_views_transfer_values_and_exact_metadata(backend, cuda_device, dtype, view, direction):
    target = cuda_device if direction == 'h2d' else torch.device('cpu')
    source_device = 'cpu' if direction == 'h2d' else cuda_device

    def fn(x):
        return view(x.to(target))

    compiled = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)
    for rows in (3, 5):
        source = torch.arange(rows * 4, dtype=dtype, device=source_device).reshape(1, rows, 4)
        expected, actual = fn(source), compiled(source)
        assert torch.equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.device == expected.device
    assert backend.stats()['runtime_executions'] == 2
    assert backend.stats()['replaced_regions'] >= 1


@pytest.mark.gpu
def test_compiled_symbolic_squeeze_recompiles_for_singleton_rank(backend, cuda_device):
    def fn(x):
        return x.to(cuda_device).squeeze(0)

    compiled = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)
    for rows in (3, 5, 1, 4):
        source = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6)
        actual, expected = compiled(source), fn(source)
        assert torch.equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
    assert backend.stats()['runtime_executions'] == 4


@pytest.mark.gpu
def test_compiled_views_fuse_with_cast(backend, cuda_device):
    def fn(x):
        return x.squeeze(0).flatten(1).unsqueeze(-1).to(cuda_device, dtype=torch.float16)

    source = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    compiled = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)
    actual, expected = compiled(source), fn(source)
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert backend.stats()['runtime_executions'] == 1
