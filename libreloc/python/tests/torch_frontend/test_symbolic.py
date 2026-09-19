"""Frontend symbolic contracts: hints must never become recipe constants."""
import dataclasses
import importlib
import os
import subprocess

import pytest
import torch


def api():
    # A missing feature is an explicit failing assertion during the RED phase.
    assert importlib.util.find_spec('reloc_torch.symbolic'), 'symbolic foundation missing'
    return importlib.import_module('reloc_torch.symbolic')


def capture(size):
    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(lambda x: x.reshape(-1, 64).transpose(0, 1), tracing_mode='symbolic')(torch.ones(size))
    return [n.meta['val'] for n in gm.graph.nodes if n.op == 'placeholder'][0]


def split_recipe(root):
    s = api()
    from reloc_torch.compat import SymbolicContext
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    context = SymbolicContext.from_tensor(root)
    source = context.tensor_spec(root)
    shape = s.infer_reshape(source.shape, (-1, 64))
    result_shape = (s.Const(64), shape[0])
    recipe = Recipe(source, (Reshape(shape), Transpose((1, 0))),
                    TensorSpec(result_shape, s.dense_strides(result_shape), s.Const(0), 'float32'), 'h2d')
    return recipe, context


def test_capture_hints_do_not_change_recipe_or_provenance(monkeypatch):
    roots = [capture(128), capture(320)]
    s = api()
    def forbidden(*args, **kwargs):
        raise AssertionError('symbolic value was concretized')
    # Capture itself uses these APIs; guard only our conversion boundary.
    monkeypatch.setattr(torch.SymInt, '__int__', forbidden)
    monkeypatch.setattr(torch.SymInt, '__bool__', forbidden)
    monkeypatch.setattr(torch.Tensor, 'item', forbidden)
    from torch.fx.experimental.sym_node import SymNode
    monkeypatch.setattr(SymNode, 'hint', property(forbidden))
    left, lc = split_recipe(roots[0])
    right, rc = split_recipe(roots[1])
    assert left.canonical_identity == right.canonical_identity
    assert left.source.shape == (s.Symbol('s0'),)
    assert left.destination.shape == (s.Const(64), s.FloorDiv(s.Symbol('s0'), 64))
    assert lc.sources == rc.sources == (s.SymbolSource('s0', 0, ()),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        left.direction = 'd2h'


def test_expression_vocabulary_and_exact_integer_evaluation():
    s = api()
    n = s.Symbol('s0')
    e = s.Add(s.Mul(n, s.Const(-3)), s.Mod(s.FloorDiv(n, 7), 5))
    assert e.evaluate({'s0': 10**30 + 1}) == -3000000000000000000000000000001
    with pytest.raises(ValueError):
        s.FloorDiv(n, 0)
    with pytest.raises(ValueError):
        s.Mod(n, n)
    assert s.dense_strides((n, s.Const(64), s.Const(2))) == (s.Const(128), s.Const(2), s.Const(1))


def test_unknown_unbacked_and_noninteger_expressions_rejected():
    s = api()
    import sympy
    from reloc_torch.compat import SymbolicContext
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    ctx = SymbolicContext.from_tensor(capture(128))
    unknown = ShapeEnv().create_unbacked_symint()
    for value in (unknown, sympy.sin(sympy.Symbol('s0')), sympy.Rational(1, 2), 1.5, True):
        with pytest.raises(s.UnsupportedSymbolicExpr, match='unsupported_symbolic_expr'):
            ctx.expression(value)
    n = capture(128).shape[0]
    with pytest.raises(s.UnsupportedSymbolicExpr):
        ctx.expression(n)  # Same printed name from a different environment is not provenance.


def test_symbolic_conversion_structural_arithmetic_and_divisor_rejection():
    s = api()
    from reloc_torch.compat import SymbolicContext
    root = capture(128)
    ctx = SymbolicContext.from_tensor(root)
    n = root.shape[0]
    assert ctx.expression(n - 3).evaluate({'s0': 129}) == 126
    assert ctx.expression((n * 3 + 2) % 7).evaluate({'s0': 129}) == 4
    with pytest.raises(s.UnsupportedSymbolicExpr):
        ctx.expression(n // (n + 1))


def test_reshape_rejects_multiple_inference_and_dynamic_factor():
    s = api()
    for target in ((-1, -1), (-1, s.Symbol('s1')), (0, -1)):
        with pytest.raises(s.UnsupportedSymbolicExpr):
            s.infer_reshape((s.Symbol('s0'),), target)


def test_dense_root_proof_does_not_accept_noncontiguous_or_offset():
    s = api()
    from reloc_torch.compat import SymbolicContext
    for root in (torch.ones(4, 6).t(), torch.ones(8)[1:]):
        with pytest.raises(s.UnsupportedSymbolicExpr, match='source_layout'):
            SymbolicContext.from_tensor(root).tensor_spec(root)


def test_binding_guards_validate_descriptors_divisibility_and_i64():
    s = api()
    from reloc_torch.recipe import TensorSpec
    recipe, ctx = split_recipe(capture(128))
    concrete = TensorSpec((128,), (1,), 0, 'float32')
    assert s.bind_recipe(recipe, ctx.sources, concrete) == {'s0': 128}
    for desc, reason in ((dataclasses.replace(concrete, shape=(130,)), 'divisibility'),
                         (dataclasses.replace(concrete, shape=(0,)), 'positive_extent'),
                         (dataclasses.replace(concrete, strides=(2,)), 'source_descriptor'),
                         (dataclasses.replace(concrete, offset=1), 'source_descriptor'),
                         (dataclasses.replace(concrete, dtype='float16'), 'source_descriptor'),
                         (dataclasses.replace(concrete, shape=(2**63,)), 'integer_overflow'),
                         (dataclasses.replace(concrete, shape=(2**61,)), 'integer_overflow')):
        with pytest.raises(s.GuardError, match=reason):
            s.bind_recipe(recipe, ctx.sources, desc)


def test_repeated_symbol_equality_and_output_element_count_guards():
    s = api()
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec
    n = s.Symbol('s0')
    src = TensorSpec((n, n), (n, s.Const(1)), s.Const(0), 'float32')
    recipe = Recipe(src, (), src, 'h2d')
    sources = (s.SymbolSource('s0', 0, (1,)),)
    assert s.bind_recipe(recipe, sources, TensorSpec((4, 4), (4, 1), 0, 'float32')) == {'s0': 4}
    with pytest.raises(s.GuardError, match='repeated_symbol'):
        s.bind_recipe(recipe, sources, TensorSpec((4, 5), (5, 1), 0, 'float32'))
    bad = Recipe(src, (Reshape((s.Const(3),)),), TensorSpec((s.Const(3),), (s.Const(1),), s.Const(0), 'float32'), 'h2d')
    with pytest.raises(s.GuardError, match='element_count'):
        s.bind_recipe(bad, sources, TensorSpec((4, 4), (4, 1), 0, 'float32'))


def test_pad_bits_identity_and_negative_padding_guard():
    s = api()
    from reloc_torch.recipe import Fill, Pad, Recipe, TensorSpec
    src = TensorSpec((s.Const(4),), (s.Const(1),), s.Const(0), 'float32')
    dst = dataclasses.replace(src, shape=(s.Const(6),))
    a = Recipe(src, (Pad(0, s.Const(1), s.Const(1), Fill('float32', 0)),), dst, 'h2d')
    b = dataclasses.replace(a, operations=(Pad(0, s.Const(1), s.Const(1), Fill('float32', 0x80000000)),))
    assert a.canonical_identity != b.canonical_identity
    assert a.canonical_identity != dataclasses.replace(a, direction='d2h').canonical_identity
    assert s.bind_recipe(a, (), TensorSpec((4,), (1,), 0, 'float32')) == {}
    bad = dataclasses.replace(a, operations=(Pad(0, s.Const(-1), s.Const(3), Fill('float32', 0)),))
    with pytest.raises(s.GuardError, match='padding'):
        s.bind_recipe(bad, (), TensorSpec((4,), (1,), 0, 'float32'))


def test_emitted_split_transpose_is_compiler_verified():
    api()
    from reloc_torch.mlir_emit import emit_mlir
    recipe, _ = split_recipe(capture(128))
    result = subprocess.run([os.environ['SYM_OPT'], '--reloc-fold'], input=emit_mlir(recipe), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('reloc.plan_result') == 1
    assert 'divisible(s0, 64)' in result.stdout
    assert '!sym.tensor<[64, s0 floordiv 64], f32>' in result.stdout


def test_identity_and_exact_pad_fill_compile():
    s = api()
    from reloc_torch.recipe import Fill, Pad, Recipe, TensorSpec
    from reloc_torch.mlir_emit import emit_mlir
    for dtype, bits in (('float32', 0x80000000), ('float16', 0x7e01), ('int8', 0xfe)):
        src = TensorSpec((s.Const(4),), (s.Const(1),), s.Const(0), dtype)
        dst = dataclasses.replace(src, shape=(s.Const(6),))
        for recipe in (Recipe(src, (), src, 'h2d'), Recipe(src, (Pad(0, s.Const(1), s.Const(1), Fill(dtype, bits)),), dst, 'h2d')):
            result = subprocess.run([os.environ['SYM_OPT'], '--reloc-fold'], input=emit_mlir(recipe), text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            assert result.stdout.count('reloc.plan_result') == 1


def test_dynamic_dynamo_capture_keeps_one_recipe_across_hints():
    captured = []
    def backend(gm, inputs):
        root = next(n.meta['example_value'] for n in gm.graph.nodes
                    if n.op == 'placeholder' and isinstance(n.meta.get('example_value'), torch.Tensor))
        captured.append(split_recipe(root))
        return gm.forward
    def run(x):
        return x.reshape(x.shape[0] // 64, 64).transpose(0, 1)
    compiled = torch.compile(run, backend=backend, dynamic=True, fullgraph=True)
    for size in (128, 320):
        x = torch.arange(size, dtype=torch.float32)
        assert torch.equal(compiled(x), run(x))
    assert len(captured) == 1
    recipe, ctx = captured[0]
    s = api()
    from reloc_torch.recipe import TensorSpec
    assert s.bind_recipe(recipe, ctx.sources, TensorSpec((320,), (1,), 0, 'float32')) == {'s0': 320}


def test_multiple_source_axes_have_canonical_order_and_repeated_provenance():
    from torch.fx.experimental.proxy_tensor import make_fx
    from reloc_torch.compat import SymbolicContext
    s = api()
    roots = []
    for shape in ((3, 7, 7), (5, 11, 11)):
        gm = make_fx(lambda x: x, tracing_mode='symbolic')(torch.ones(shape))
        roots.append(next(n.meta['val'] for n in gm.graph.nodes if n.op == 'placeholder'))
    contexts = [SymbolicContext.from_tensor(root) for root in roots]
    specs = [ctx.tensor_spec(root) for ctx, root in zip(contexts, roots)]
    assert specs[0] == specs[1]
    assert contexts[0].sources == (s.SymbolSource('s0', 0), s.SymbolSource('s1', 1, (2,)))
    assert specs[0].strides == (s.Mul(s.Symbol('s1'), s.Symbol('s1')), s.Symbol('s1'), s.Const(1))


def test_emitter_rejects_unrepresented_root_layout():
    s = api()
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.mlir_emit import emit_mlir
    src = TensorSpec((s.Const(4),), (s.Const(2),), s.Const(0), 'float32')
    dst = dataclasses.replace(src, strides=(s.Const(1),))
    with pytest.raises(s.UnsupportedSymbolicExpr, match='source_layout'):
        emit_mlir(Recipe(src, (), dst, 'h2d'))


def test_symbol_names_are_independent_of_capture_generated_names():
    from torch.fx.experimental.proxy_tensor import make_fx
    first, _ = split_recipe(capture(128))
    gm = make_fx(lambda unused, x: x.reshape(-1, 64).transpose(0, 1), tracing_mode='symbolic')(torch.ones(7), torch.ones(320))
    root = [n.meta['val'] for n in gm.graph.nodes if n.op == 'placeholder'][1]
    second, _ = split_recipe(root)
    assert first.canonical_identity == second.canonical_identity


def test_i64_checks_expression_intermediates_and_constant_divisors():
    s = api()
    with pytest.raises(s.GuardError, match='integer_overflow'):
        s.Add(s.Mul(s.Symbol('s0'), s.Const(4)), s.Const(-(2**63))).evaluate({'s0': 2**61}, checked=True)
    with pytest.raises(s.GuardError, match='integer_overflow'):
        s.FloorDiv(s.Const(128), 2**64).evaluate({}, checked=True)


def test_dense_stride_proof_uses_canonical_source_order_not_sympy_order():
    from torch.fx.experimental.proxy_tensor import make_fx
    from reloc_torch.compat import SymbolicContext
    specs = []
    for shape in ((3, 5, 7, 11), (13, 17, 19, 23)):
        gm = make_fx(lambda x: x, tracing_mode='symbolic')(torch.ones(shape))
        root = next(n.meta['val'] for n in gm.graph.nodes if n.op == 'placeholder')
        specs.append(SymbolicContext.from_tensor(root).tensor_spec(root))
    assert specs[0] == specs[1]
    assert specs[0].strides[0].evaluate({'s0': 3, 's1': 5, 's2': 7, 's3': 11}) == 385


def test_portable_recipe_recovers_symbol_sources_without_capture_context():
    s = api()
    assert hasattr(s, 'symbol_sources'), 'portable source provenance helper missing'
    assert s.symbol_sources((s.Symbol('s0'), s.Const(4), s.Symbol('s1'), s.Symbol('s0'))) == (
        s.SymbolSource('s0', 0, (3,)), s.SymbolSource('s1', 2))
    assert s.symbol_sources((s.Const(4),)) == ()
    with pytest.raises(s.UnsupportedSymbolicExpr):
        s.symbol_sources((s.Mul(s.Const(64), s.Symbol('s0')),))
