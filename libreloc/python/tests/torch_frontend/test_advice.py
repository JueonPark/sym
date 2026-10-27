"""Cost advice: manifest admission, identity, pricing, selection and the direct path."""
import copy
import dataclasses

import pytest
import torch

from reloc_torch.advice import Advice, parse_advice
from reloc_torch.artifact import CompiledRecipe
from reloc_torch.recipe import BindingParam, Cast, Dequantize, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides
from test_fast import dlrm_recipe, gather_recipe


def spec(shape, dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


def weight_recipe(dtype='float32'):
    rows, cols = Symbol('out'), Symbol('in')
    stages = (Dequantize('float32', BindingParam('scale', 'float32', (rows,)), None, 0, 'affine'),)
    if dtype == 'float16':
        stages += (Cast('float16', 'ieee_rne'),)
    return Recipe(spec((rows, cols), 'int8'), stages, spec((rows, cols), dtype), 'h2d')


def kv_recipe():
    s = Symbol('s0')
    return Recipe(spec((Const(16), s, Const(64)), 'float32'), (Transpose((1, 0, 2)),),
                  spec((s, Const(16), Const(64)), 'float32'), 'd2h')


def test_admission_parses_the_advice_block_of_every_plan_kind(compiler):
    for recipe, kind, direct in ((kv_recipe(), 'layout', False), (dlrm_recipe(), 'stages', False),
                                 (weight_recipe('float16'), 'none', True), (gather_recipe(), 'indexed_rows', True)):
        compiled = compiler.compile(recipe)
        advice = compiled.advice
        assert isinstance(advice, Advice) and advice.version == 1
        assert advice.producer_kind == kind
        assert [p.name for p in advice.paths] == (['direct'] if direct else []) + ['pipelined', 'grouped']
        assert compiled.compiler.interface_version == 2
        # The portable serialization keeps the block.
        assert CompiledRecipe.from_bytes(compiled.to_bytes()).advice == advice
    weights = compiler.compile(weight_recipe('float16'))
    assert weights.advice.receive_kernel == 'dequant_s8_f16' and weights.advice.paths[0].wire == 'source'
    assert compiler.compile(weight_recipe('float32')).advice.receive_kernel == 'dequant_s8_f32'
    gather = compiler.compile(gather_recipe())
    assert gather.advice.pattern == 'indexed_rows' and gather.advice.producer_kernel == 'gather_f16'


def test_advice_rejects_malformed_blocks(compiler):
    compiled = compiler.compile(dlrm_recipe())
    good = compiled.manifest['advice']
    symbols = compiled.symbols
    assert parse_advice(None, symbols) is None
    assert parse_advice(good, symbols) == compiled.advice
    for mutate in (lambda a: a['paths'][0].update(name='teleport'),
                   lambda a: a['paths'][0].update(intercept='overhead.dispatch.stages_ms; rm -rf'),
                   lambda a: a.update(stray=1),
                   lambda a: a['paths'][0].update(terms=[['warp', 'wire']]),
                   lambda a: a['paths'][0].update(form='serial'),
                   lambda a: a.update(paths=list(reversed(a['paths']))),
                   lambda a: a.update(pattern='diagonal'),
                   lambda a: a['producer'].update(kind='none')):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(RuntimeError, match='advice'):
            parse_advice(bad, symbols)


def test_advice_rejects_unknown_symbol(compiler):
    compiled = compiler.compile(dlrm_recipe())
    bad = copy.deepcopy(compiled.manifest['advice'])
    bad['source_bytes'] = ['mul', ['symbol', 'ghost'], ['const', 4]]
    with pytest.raises(RuntimeError, match='symbol'):
        parse_advice(bad, compiled.symbols)
    tampered = dataclasses.replace(compiled, manifest=dict(compiled.manifest, advice=bad))
    with pytest.raises(RuntimeError, match='symbol'):
        CompiledRecipe.from_bytes(tampered.to_bytes())
