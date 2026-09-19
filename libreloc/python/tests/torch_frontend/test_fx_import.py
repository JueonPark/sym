"""Importer boundary tests: each fixture checks semantics or a safety exclusion."""
import importlib.util
import operator

import pytest
import torch


def importer():
    assert importlib.util.find_spec('reloc_torch.fx_import') is not None, 'FX importer missing'
    from reloc_torch import fx_import
    return fx_import


def capture(fn, shape=(3, 4), **options):
    # CPU wheels cannot fake Tensor.to('cuda'); canonical ATen can be captured.
    from reloc_torch.compat import symbolic_capture
    return symbolic_capture(fn, torch.ones(shape, **options))


def transfer(x, **kwargs):
    return torch.ops.aten._to_copy.default(x, device=torch.device('cuda:0'), **kwargs)


def snapshot(gm):
    return str(gm.graph), [(n.name, tuple(u.name for u in n.users), dict(n.meta)) for n in gm.graph.nodes]


@pytest.mark.parametrize('fn', [
    lambda x: transfer(x.transpose(0, 1).contiguous()),
    lambda x: torch.ops.aten.clone.default(transfer(x).transpose(0, 1), memory_format=torch.contiguous_format),
])
def test_symbolic_regions_preserve_source_graph_and_dense_output(fn):
    api = importer()
    from reloc_torch.recipe import Transpose
    from reloc_torch.symbolic import Const, Symbol
    gm = capture(fn)
    before = snapshot(gm)
    report = api.import_graph(gm)
    assert snapshot(gm) == before
    assert not report.exclusions
    candidate, = report.candidates
    assert candidate.recipe.operations == (Transpose((1, 0)),)
    assert candidate.recipe.source.shape == (Symbol('s0'), Symbol('s1'))
    assert candidate.recipe.destination.shape == (Symbol('s1'), Symbol('s0'))
    assert candidate.recipe.destination.strides == (Symbol('s0'), Const(1))
    assert candidate.recipe.direction == 'h2d'
    assert len([n for n in candidate.original.graph.nodes if n.target == torch.ops.aten._to_copy.default]) == 1
    assert candidate.source == 'x_1'
    assert candidate.tail == candidate.members[-1]


@pytest.mark.parametrize('fn', [
    lambda x: x.transpose(-1, -2).contiguous().to('cuda'),
    lambda x: x.to(device='cuda').permute(1, 0).contiguous(),
    lambda x: x.reshape(-1, 2).cuda().contiguous(),
    lambda x: torch.transpose(x, 0, 1).contiguous().to('cuda'),
])
def test_raw_normalization_uses_fake_only_and_retains_original(fn):
    api = importer()
    gm = torch.fx.symbolic_trace(fn)
    before = snapshot(gm)
    normalized = api.normalize_graph(gm, (torch.ones(3, 4),))
    assert snapshot(gm) == before
    assert normalized is not gm
    assert all(n.op != 'call_method' for n in normalized.graph.nodes)
    report = api.import_graph(gm, (torch.ones(3, 4),))
    assert len(report.candidates) == 1, report.exclusions
    assert snapshot(gm) == before
    assert any(n.op == 'call_method' for n in report.candidates[0].original.graph.nodes)


@pytest.mark.parametrize(('fn', 'reason'), [
    (lambda x: transfer(x.transpose(0, 1)), 'destination_layout'),
    (lambda x: transfer(x, non_blocking=True), 'nonblocking_unavailable'),
    (lambda x: transfer(x, dtype=torch.float16), 'typed_transform_unavailable'),
    (lambda x: transfer(transfer(x).cpu()), 'multiple_transfers'),
    (lambda x: (transfer(x.transpose(0, 1).contiguous()), x.transpose(0, 1)), None),
])
def test_precise_exclusions_or_independent_view(fn, reason):
    api = importer()
    gm = capture(fn)
    before = snapshot(gm)
    report = api.import_graph(gm)
    assert snapshot(gm) == before
    if reason is None:
        assert len(report.candidates) == 1
    else:
        assert not report.candidates
        assert reason in {e.reason for e in report.exclusions}


def test_escaping_intermediate_rejects_entire_region():
    api = importer()
    def fn(x):
        y = x.transpose(0, 1)
        return transfer(y.contiguous()), y
    gm = capture(fn)
    report = api.import_graph(gm)
    assert not report.candidates
    assert 'escaping_intermediate' in {e.reason for e in report.exclusions}


def test_alias_write_rejects_even_when_layout_chain_is_single_use():
    api = importer()
    def fn(x):
        alias = x.view(-1)
        y = x.transpose(0, 1).contiguous()
        alias.add_(1)
        return transfer(y)
    report = api.import_graph(capture(fn))
    assert not report.candidates
    assert 'mutation' in {e.reason for e in report.exclusions}


def test_unknown_effect_is_not_executed_by_normalization():
    api = importer()
    calls = []
    def effect(x):
        calls.append('called')
        return x
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    y = graph.call_method('transpose', (x, 0, 1))
    graph.call_function(effect, (x,))
    z = graph.call_method('contiguous', (y,))
    graph.output(graph.call_method('to', (z, 'cuda')))
    gm = torch.fx.GraphModule({}, graph)
    report = api.import_graph(gm, (torch.ones(3, 4),))
    assert calls == []
    assert not report.candidates
    assert {'unrecognized_fx_target', 'unknown_side_effect'} <= {e.reason for e in report.exclusions}


@pytest.mark.parametrize(('value', 'reason'), [
    (torch.ones(3, 4).transpose(0, 1), 'source_layout'),
    (torch.ones(4, 4)[1:], 'source_layout'),
    (torch.ones(3, 4, requires_grad=True), 'requires_grad'),
    (torch.ones(0, 4), 'empty_tensor'),
    (torch.ones(()), 'rank_zero'),
])
def test_root_contract(value, reason):
    api = importer()
    gm = torch.fx.symbolic_trace(lambda x: x.to('cuda'))
    report = api.import_graph(gm, (value,))
    assert not report.candidates
    assert reason in {e.reason for e in report.exclusions}


def test_pad_reversed_axes_and_exact_fill_bits():
    api = importer()
    from reloc_torch.recipe import Fill, Pad
    from reloc_torch.symbolic import Const
    gm = capture(lambda x: transfer(torch.nn.functional.pad(x, (1, 2, 3, 4), value=-0.0)))
    candidate, = api.import_graph(gm).candidates
    assert candidate.recipe.operations == (
        Pad(1, Const(1), Const(2), Fill('float32', 0x80000000)),
        Pad(0, Const(3), Const(4), Fill('float32', 0x80000000)),
    )


def test_symbolic_scalar_dependencies_and_reshape_remain_exact():
    api = importer()
    from reloc_torch.recipe import Reshape, Transpose
    from reloc_torch.symbolic import Const, FloorDiv, Symbol
    gm = capture(lambda x: transfer(x.reshape(x.shape[0] // 64, 64).transpose(0, 1).contiguous()), (128,))
    candidate, = api.import_graph(gm).candidates
    assert candidate.recipe.operations == (Reshape((FloorDiv(Symbol('s0'), 64), Const(64))), Transpose((1, 0)))
    assert [s.axis for s in candidate.symbol_sources] == [0]
    assert 'sym_size' in str(candidate.original.graph)
    assert 'floordiv' in str(candidate.original.graph)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('fn', [
    lambda x: x.transpose(0, 1).contiguous().to('cuda'),
    lambda x: x.to('cuda').transpose(0, 1).contiguous(),
    lambda x: x.reshape(x.shape[0] // 2, 2).to('cuda', copy=True),
])
def test_real_original_callable_matches_raw_and_aten(fn):
    api = importer()
    from torch.fx.experimental.proxy_tensor import make_fx
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) if 'transpose' in fn.__code__.co_names else torch.arange(12, dtype=torch.float32)
    for gm in (torch.fx.symbolic_trace(fn), make_fx(fn, tracing_mode='symbolic')(x)):
        candidate, = api.import_graph(gm, (x,)).candidates
        actual = candidate.original(x)
        expected = fn(x)
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset() == 0
        assert actual.device == expected.device


def test_raw_dynamo_conditional_materialization_is_not_assumed_dense_for_singletons():
    api = importer()
    graphs = []
    def backend(gm, inputs):
        graphs.append((gm, inputs))
        return gm.forward
    # CPU execution supplies real raw Dynamo metadata without needing a GPU.
    def fn(x):
        return x.reshape(-1, 64).transpose(0, 1).contiguous()
    torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)(torch.ones(128))
    gm, inputs = graphs[0]
    graph = gm.graph
    output = next(n for n in graph.nodes if n.op == 'output')
    old_tail = output.args[0][0]
    with graph.inserting_before(output):
        tail = graph.call_method('to', (old_tail, 'cuda'))
    output.args = ((tail,),)
    gm.recompile()
    before = snapshot(gm)
    normalized = api.normalize_graph(gm, inputs)
    contiguous = next(n for n in normalized.graph.nodes if n.name == old_tail.name)
    assert contiguous.target == torch.ops.aten.contiguous.default
    report = api.import_graph(gm, inputs)
    assert not report.candidates
    assert 'conditional_materialization' in {e.reason for e in report.exclusions}
    assert snapshot(gm) == before


def test_explicit_clone_symbolic_split_has_dense_singleton_guarded_descriptor():
    api = importer()
    from reloc_torch.recipe import TensorSpec
    from reloc_torch.symbolic import bind_recipe
    gm = capture(lambda x: transfer(torch.ops.aten.clone.default(x.reshape(-1, 64).transpose(0, 1), memory_format=torch.contiguous_format)), (128,))
    candidate, = api.import_graph(gm).candidates
    for size, expected in [(64, (1, 1)), (128, (2, 1)), (320, (5, 1))]:
        bindings = bind_recipe(candidate.recipe, candidate.symbol_sources, TensorSpec((size,), (1,), 0, 'float32'))
        assert tuple(d.evaluate(bindings) for d in candidate.recipe.destination.strides) == expected


def test_raw_static_singleton_contiguous_does_not_get_relabelled():
    api = importer()
    gm = torch.fx.symbolic_trace(lambda x: x.reshape(-1, 64).transpose(0, 1).contiguous().to('cuda'))
    report = api.import_graph(gm, (torch.ones(64),))
    assert not report.candidates
    assert 'destination_layout' in {e.reason for e in report.exclusions}


def test_subclasses_reject_before_dispatch():
    api = importer()
    class Subclass(torch.Tensor):
        pass
    gm = torch.fx.symbolic_trace(lambda x: x.to('cuda'))
    report = api.import_graph(gm, (torch.ones(3, 4).as_subclass(Subclass),))
    assert not report.candidates
    assert 'tensor_subclass' in {e.reason for e in report.exclusions}


def test_branched_transfer_chain_rejects_every_transfer():
    api = importer()
    def fn(x):
        y = transfer(x)
        return y, torch.ops.aten._to_copy.default(y, device=torch.device('cpu')), torch.ops.aten._to_copy.default(y, device=torch.device('cpu'))
    report = api.import_graph(capture(fn))
    assert not report.candidates
    assert len([e for e in report.exclusions if e.reason == 'multiple_transfers']) == 3


def test_known_alias_ancestor_write_is_rejected():
    api = importer()
    def fn(x):
        root = x.detach()
        y = transfer(root)
        x.add_(1)
        return y
    gm = capture(fn)
    # detach is outside the selected layout vocabulary; its alias schema still
    # participates in safety analysis even though normalization excludes it.
    report = api.import_graph(gm)
    assert not report.candidates
    assert 'mutation' in {e.reason for e in report.exclusions}


def test_d2h_symbolic_fake_capture_on_cpu_wheel():
    api = importer()
    from reloc_torch.compat import symbolic_capture
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(4, 6, device='cuda')
        gm = symbolic_capture(lambda y: torch.ops.aten._to_copy.default(y, device=torch.device('cpu')), x)
    candidate, = api.import_graph(gm).candidates
    assert candidate.recipe.direction == 'd2h'
    assert candidate.recipe.operations == ()


@pytest.mark.parametrize(('fn', 'reason'), [
    (lambda x: torch.nn.functional.pad(x, (-1, 0)).to('cuda'), 'unsupported_padding'),
    (lambda x: torch.nn.functional.pad(x, (1, 1), mode='reflect').to('cuda'), 'unsupported_padding'),
    (lambda x: x.to('cuda', memory_format=torch.channels_last), 'unsupported_memory_format'),
    (lambda x: x.to('cpu', copy=True), 'same_device_transfer'),
])
def test_unsupported_options_are_preserved_and_excluded(fn, reason):
    api = importer()
    gm = torch.fx.symbolic_trace(fn)
    before = snapshot(gm)
    report = api.import_graph(gm, (torch.ones(2, 3, 4, 5),))
    assert not report.candidates
    assert reason in {e.reason for e in report.exclusions}
    assert snapshot(gm) == before


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_original_dynamo_scalar_placeholder_order_and_exceptions():
    api = importer()
    graphs = []
    def backend(gm, inputs):
        graphs.append((gm, inputs))
        return gm.forward
    def fn(x):
        return x.reshape(x.shape[0] // 2, 2).to('cuda', copy=True)
    expected = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)(torch.arange(12, dtype=torch.float32))
    gm, inputs = graphs[0]
    candidate, = api.import_graph(gm, inputs).candidates
    assert len(candidate.symbolic_bindings) == 1
    assert torch.equal(candidate.original(torch.arange(12, dtype=torch.float32), 12), expected)
    with pytest.raises(RuntimeError):
        candidate.original(torch.arange(13, dtype=torch.float32), 13)


def test_keyword_function_forms_normalize_without_changing_values():
    api = importer()
    gm = torch.fx.symbolic_trace(lambda x: torch.reshape(input=torch.transpose(input=x, dim0=-1, dim1=-2), shape=(12,)).to('cuda'))
    report = api.import_graph(gm, (torch.ones(3, 4),))
    assert len(report.candidates) == 1, report.exclusions


def test_sparse_root_excludes_without_fake_conversion_error():
    api = importer()
    gm = torch.fx.symbolic_trace(lambda x: x.to('cuda'))
    report = api.import_graph(gm, (torch.eye(3).to_sparse(),))
    assert not report.candidates
    assert 'unsupported_layout' in {e.reason for e in report.exclusions}


def test_capture_metadata_is_not_enough_to_replay_python_numeric_overloads():
    api = importer()
    calls = []
    class Scalar:
        def __add__(self, value):
            calls.append(value)
            return 1
    graph = torch.fx.Graph()
    x = graph.placeholder('x')
    scalar = graph.placeholder('scalar')
    graph.call_function(operator.add, (scalar, 1))
    graph.output(graph.call_method('to', (x, 'cuda')))
    report = api.import_graph(torch.fx.GraphModule({}, graph), (torch.ones(3), Scalar()))
    assert calls == []
    assert not report.candidates
    assert 'unknown_side_effect' in {e.reason for e in report.exclusions}


def test_positional_to_copy_contract_is_not_lost():
    api = importer()
    gm = torch.fx.symbolic_trace(lambda x: x.to('cuda', torch.float32, False, True))
    report = api.import_graph(gm, (torch.ones(3, 4),))
    assert len(report.candidates) == 1, report.exclusions
    transfer_node = next(n for n in report.candidates[0].original.graph.nodes if n.op == 'call_method')
    assert transfer_node.args[1:] == ('cuda', torch.float32, False, True)


def test_import_symbolic_metadata_never_reads_concrete_tensor_or_shape_hints(monkeypatch):
    api = importer()
    from torch.fx.experimental.sym_node import SymNode
    gm = capture(lambda x: transfer(x.reshape(x.shape[0] // 64, 64).transpose(0, 1).contiguous()), (128,))
    def forbidden(*args, **kwargs):
        raise AssertionError('concretization attempted')
    with monkeypatch.context() as m:
        m.setattr(torch.SymInt, '__int__', forbidden)
        m.setattr(torch.SymInt, '__bool__', forbidden)
        m.setattr(SymNode, 'hint', property(forbidden))
        m.setattr(torch.Tensor, 'item', forbidden)
        report = api.import_graph(gm)
    assert len(report.candidates) == 1, report.exclusions


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_explicit_clone_singleton_original_matches_guarded_recipe():
    api = importer()
    from reloc_torch.recipe import TensorSpec
    from reloc_torch.symbolic import bind_recipe
    gm = capture(lambda x: transfer(torch.ops.aten.clone.default(x.reshape(-1, 64).transpose(0, 1), memory_format=torch.contiguous_format)), (128,))
    candidate, = api.import_graph(gm).candidates
    for size in (64, 128, 320):
        x = torch.arange(size, dtype=torch.float32)
        actual = candidate.original(x)
        expected = x.reshape(-1, 64).transpose(0, 1).clone(memory_format=torch.contiguous_format).cuda()
        bindings = bind_recipe(candidate.recipe, candidate.symbol_sources, TensorSpec((size,), (1,), 0, 'float32'))
        assert torch.equal(actual, expected)
        assert actual.stride() == tuple(d.evaluate(bindings) for d in candidate.recipe.destination.strides)


def test_package_importer_entrypoints_are_lazy_and_torch_free():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', "import sys; from reloc_torch import import_graph, normalize_graph; assert callable(import_graph) and callable(normalize_graph); assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_quantized_root_excludes_before_unsupported_fake_conversion():
    api = importer()
    with pytest.warns(UserWarning, match='deprecated'):
        value = torch.quantize_per_tensor(torch.ones(3), 0.1, 0, torch.qint8)
    gm = torch.fx.symbolic_trace(lambda x: x.to('cuda'))
    report = api.import_graph(gm, (value,))
    assert not report.candidates
    assert 'unsupported_layout' in {e.reason for e in report.exclusions}


@pytest.mark.parametrize('write_before_transfer', [False, True])
@pytest.mark.parametrize('transfer_from_alias', [False, True])
def test_unsafe_view_alias_write_rejects_region(write_before_transfer, transfer_from_alias):
    api = importer()
    def fn(x):
        alias = torch.ops.aten._unsafe_view.default(x, [12])
        if write_before_transfer:
            alias.add_(1)
        result = transfer(alias if transfer_from_alias else x)
        if not write_before_transfer:
            alias.add_(1)
        return result
    gm = capture(fn)
    before = snapshot(gm)
    report = api.import_graph(gm)
    assert not report.candidates
    assert 'mutation' in {e.reason for e in report.exclusions}
    assert snapshot(gm) == before


def test_unsafe_view_inventory_preserves_alias_when_fake_snapshots_differ():
    from reloc_torch import graph_inventory
    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(lambda x: torch.ops.aten._unsafe_view.default(x, [12]))(torch.ones(3, 4))
    record = next(r for r in graph_inventory(gm) if r.target == 'aten._unsafe_view.default')
    assert record.alias_semantics == 'aliases'
