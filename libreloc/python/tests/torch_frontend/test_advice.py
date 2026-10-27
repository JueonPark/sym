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


def test_price_advice_binding_and_calibration_accessors(tmp_path):
    import pyreloc
    path = tmp_path / 'tiny.cal'
    path.write_text('# costmodel calibration v0\npcie.h2d_gbps 10\nhbm.bw_gbps 100\nhbm.m.blocked 2\n'
                    'overhead.dispatch.layout_ms 0.3\npipeline.chunks_per_buffer 4\n')
    cal = pyreloc.load_calibration(str(path))
    assert cal.has('pcie.h2d_gbps') and not cal.has('overhead.group_ms')
    assert cal.value('hbm.m.blocked') == 2
    with pytest.raises(KeyError):
        cal.value('overhead.group_ms')
    costs = pyreloc.price_advice(cal, [
        dict(name='pipelined', form='overlapped', intercept='overhead.dispatch.layout_ms', member_intercept=None,
             chunks='pipeline.chunks_per_buffer', terms=[[1e8, 'pcie.h2d_gbps', None], [1e8, 'hbm.bw_gbps', 'hbm.m.blocked']]),
        dict(name='grouped', form='grouped', intercept='overhead.group_ms', member_intercept='overhead.group_member_ms',
             chunks=None, terms=[[1e8, 'pcie.h2d_gbps', None]])], 3)
    assert costs[0] == dict(name='pipelined', ms=pytest.approx(0.3 + 10 + 2 / 4), missing_key='')
    assert costs[1] == dict(name='grouped', ms=None, missing_key='overhead.group_ms')
    with pytest.raises(ValueError, match='advice_invalid'):
        pyreloc.price_advice(cal, [dict(name='x', form='serial', intercept='overhead.dispatch.layout_ms',
                                        member_intercept=None, chunks=None, terms=[[-1, 'pcie.h2d_gbps', None]])])


SYNTH = """# costmodel calibration v0
pcie.h2d_gbps 10
hbm.bw_gbps 100
hbm.m.contiguous 1
hbm.m.blocked 2
hbm.m.indexed_rows 1
recv.m.dequant_s8_f16 1.5
recv.m.dequant_s8_f32 1.2
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.blocked.gather_f32_gbps 5
cpu.t8.blocked.convert_f32_f16_gbps 20
cpu.t8.contiguous.pooled_sum_f16_gbps 20
cpu.t8.indexed_rows.gather_f16_gbps 8
pipeline.chunks_per_buffer 4
overhead.direct.pooled_ms 0.05
overhead.dispatch.pooled_ms 0.3
overhead.direct.indexed_rows_ms 0.05
overhead.dispatch.indexed_rows_ms 0.3
overhead.direct.dequant_ms 0.02
overhead.dispatch.dequant_ms 0.25
overhead.dispatch.layout_ms 0.2
overhead.dispatch.stages_ms 0.2
overhead.group_ms 0.2
overhead.group_member_ms 0.01
"""


@pytest.fixture
def calibration(tmp_path):
    import pyreloc
    path = tmp_path / 'synth.cal'
    path.write_text(SYNTH)
    return pyreloc.load_calibration(str(path))


def test_classify_matches_the_runtime_rule():
    from reloc_torch.advice import classify
    assert classify(1024, 1024) == 'contiguous'
    assert classify(1, 1024) == 'single_element'
    assert classify(64, 1024) == 'blocked'
    assert classify(63, 1024) == 'tiled'


def test_resolve_prices_and_selects_for_a_single_call(compiler, calibration):
    from reloc_torch.advice import Decision, resolve, select
    compiled = compiler.compile(gather_recipe())
    resolved = resolve(compiled.advice, {'s0': 5000, 'k': 1000}, 8)
    assert resolved.pattern == 'indexed_rows' and resolved.result_bytes == 1000 * 128 * 2
    assert resolved.source_bytes == 1000 * 128 * 4 and resolved.direct
    assert [p['name'] for p in resolved.paths] == ['direct', 'pipelined', 'grouped']
    assert resolved.path('direct')['terms'] == [[1000 * 128 * 4, 'cpu.t8.indexed_rows.gather_f16_gbps', None],
                                                [1000 * 128 * 2, 'pcie.h2d_gbps', None],
                                                [1000 * 128 * 2, 'hbm.bw_gbps', 'hbm.m.indexed_rows']]
    decision = select(calibration, resolved, direct_available=True)
    assert isinstance(decision, Decision)
    # Small transfer: the dispatch intercept dwarfs what chunk overlap saves.
    assert decision.path == 'direct' and decision.reason == 'advice_prefers_direct'
    assert set(decision.predicted) == {'direct', 'pipelined', 'grouped'}
    assert decision.predicted['direct'] < decision.predicted['pipelined']
    # Without the native helper the cheapest path cannot run.
    fallback = select(calibration, resolved, direct_available=False)
    assert fallback.path == 'pipelined' and fallback.reason == 'advice_no_native_helper'
    # A large transfer: chunk overlap pays for the dispatcher.
    big = resolve(compiled.advice, {'s0': 5000, 'k': 4_000_000}, 8)
    assert select(calibration, big, direct_available=True).reason == 'advice_prefers_pipelined'


def test_resolve_classifies_symbolic_patterns_at_bind(compiler):
    from reloc_torch.advice import resolve
    weights = compiler.compile(weight_recipe('float16'))
    assert weights.advice.pattern == 'symbolic'
    resolved = resolve(weights.advice, {'out': 3, 'in': 5}, 8)
    assert resolved.pattern == 'contiguous' and resolved.source_bytes == 15 and resolved.result_bytes == 30
    assert resolved.path('direct')['terms'] == [[15, 'pcie.h2d_gbps', None], [15, 'hbm.bw_gbps', 'recv.m.dequant_s8_f16']]
    kv = resolve(compiler.compile(kv_recipe()).advice, {'s0': 12}, 8)
    assert kv.pattern == 'blocked' and kv.path('pipelined')['terms'][0][1] == 'cpu.t8.blocked.gather_f32_gbps'
    with pytest.raises(RuntimeError, match='advice_invalid'):
        resolve(weights.advice, {'out': 3}, 8)


def test_select_reports_absent_disabled_and_missing(compiler, calibration):
    from reloc_torch.advice import resolve, select
    compiled = compiler.compile(gather_recipe())
    resolved = resolve(compiled.advice, {'s0': 50, 'k': 10}, 8)
    assert select(None, resolved, direct_available=True).reason == 'no_calibration'
    assert select(calibration, None, direct_available=True).reason == 'advice_absent'
    assert select(calibration, resolved, direct_available=True, advice_enabled=False).reason == 'advice_disabled'
    four = resolve(compiled.advice, {'s0': 50, 'k': 10}, 4)
    missing = select(calibration, four, direct_available=True)
    assert missing.path == 'pipelined' and missing.reason == 'advice_missing_keys'
    assert missing.predicted['direct'] is None


def test_select_group_keeps_the_callers_group(compiler, calibration, tmp_path):
    import pyreloc
    from reloc_torch.advice import resolve, select_group
    kv = compiler.compile(kv_recipe())
    members = [resolve(kv.advice, {'s0': 12}, 8) for _ in range(16)]
    assert all(not m.direct for m in members)
    decision = select_group(calibration, members, direct_available=True)
    assert decision.path == 'grouped' and decision.reason == 'advice_prefers_grouped'
    assert decision.predicted['direct'] is None and decision.predicted['grouped'] < decision.predicted['pipelined']
    # An absurd per-member intercept: split is priced cheaper, the group is
    # still what runs, and the reason says so.
    path = tmp_path / 'split.cal'
    path.write_text(SYNTH.replace('overhead.group_member_ms 0.01', 'overhead.group_member_ms 5'))
    split = select_group(pyreloc.load_calibration(str(path)), members, direct_available=True)
    assert split.path == 'grouped' and split.reason == 'advice_prefers_split'
    # Four weights: every member admits the direct kernels, which beat the
    # group's intercept at this size.
    weights = compiler.compile(weight_recipe('float16'))
    four = [resolve(weights.advice, {'out': 3072, 'in': 1024}, 8) for _ in range(4)]
    direct = select_group(calibration, four, direct_available=True)
    assert direct.path == 'direct' and direct.reason == 'advice_prefers_direct'
    helperless = select_group(calibration, four, direct_available=False)
    assert helperless.path == 'grouped' and helperless.reason == 'advice_no_native_helper'
    # A mixed group (weights and a layout member) prices but never goes direct.
    mixed = select_group(calibration, four + [members[0]], direct_available=True)
    assert mixed.predicted['direct'] is None and mixed.path == 'grouped'


def test_price_rejects_nonpositive_rate(compiler, tmp_path):
    import pyreloc
    from reloc_torch.advice import resolve, select
    path = tmp_path / 'zero.cal'
    path.write_text(SYNTH.replace('cpu.t8.indexed_rows.gather_f16_gbps 8', 'cpu.t8.indexed_rows.gather_f16_gbps 0'))
    zero = pyreloc.load_calibration(str(path))
    resolved = resolve(compiler.compile(gather_recipe()).advice, {'s0': 50, 'k': 10}, 8)
    with pytest.raises(RuntimeError, match='advice_invalid'):
        select(zero, resolved, direct_available=True)


def test_pooled_advice_is_the_pooled_programs_structure():
    from reloc_torch.advice import pooled_advice, resolve
    advice = pooled_advice()
    assert advice.producer_kind == 'pooled' and advice.producer_kernel == 'pooled_sum_f16'
    assert [p.name for p in advice.paths] == ['direct', 'pipelined', 'grouped']
    resolved = resolve(advice, {'bags': 512, 'tables': 26, 'dim': 64}, 8)
    assert resolved.pattern == 'contiguous' and resolved.result_bytes == 512 * 26 * 64 * 2
    direct = resolved.path('direct')
    assert direct['intercept'] == 'overhead.direct.pooled_ms'
    # The producer is priced on the wire: the pooled rate is wire-normalized.
    assert direct['terms'][0] == [512 * 26 * 64 * 2, 'cpu.t8.contiguous.pooled_sum_f16_gbps', None]


def test_load_calibration_resolves_paths_objects_and_the_environment(calibration, tmp_path, monkeypatch):
    from reloc_torch.advice import load_calibration
    assert load_calibration(calibration) is calibration
    path = tmp_path / 'env.cal'
    path.write_text(SYNTH)
    assert load_calibration(path).has('overhead.group_ms')
    monkeypatch.setenv('RELOC_CALIBRATION', str(path))
    assert load_calibration(None).has('overhead.group_ms')
    monkeypatch.delenv('RELOC_CALIBRATION')
    assert load_calibration(None) is None


# -- the fast lane ------------------------------------------------------------

@pytest.fixture
def advised(compiler, cuda_device, calibration):
    from reloc_torch.fast import FastTransfers
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=8, calibration=calibration) as owner:
        yield owner


def bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def bag_operands(tables, bags, rows, gen):
    indices, offsets = [], []
    for _ in range(tables):
        lengths = torch.randint(1, 5, (bags,), generator=gen)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, rows, (int(lengths.sum()),), generator=gen))
    return indices, offsets


@pytest.mark.gpu
def test_owner_resolves_calibration_from_argument_env_or_nothing(compiler, cuda_device, calibration, tmp_path,
                                                                 monkeypatch):
    from reloc_torch.fast import FastTransfers
    with FastTransfers(cuda_device, compiler=compiler, calibration=calibration) as owner:
        assert owner.calibration is calibration and owner.advice
    path = tmp_path / 'env.cal'
    path.write_text(SYNTH)
    monkeypatch.setenv('RELOC_CALIBRATION', str(path))
    with FastTransfers(cuda_device, compiler=compiler) as owner:
        assert owner.calibration is not None and owner.calibration.has('overhead.group_ms')
    monkeypatch.delenv('RELOC_CALIBRATION')
    source, index = torch.randn(50, 128), torch.randint(50, (10,))
    with FastTransfers(cuda_device, compiler=compiler) as owner:
        assert owner.calibration is None
        owner.indexed(gather_recipe()).execute(source, index)
        assert owner.stats()['advice'] == {'no_calibration': 1}
    with FastTransfers(cuda_device, compiler=compiler, calibration=calibration, advice=False) as owner:
        owner.indexed(gather_recipe()).execute(source, index)
        assert owner.stats()['advice'] == {'advice_disabled': 1}
    with pytest.raises(ValueError):
        FastTransfers(cuda_device, compiler=compiler, direct_wires=0)


@pytest.mark.gpu
@pytest.mark.parametrize('rows,count', [(50, 10), (5000, 4096)])
def test_indexed_direct_path_is_bit_exact_and_reported(advised, cuda_device, rows, count):
    program = advised.indexed(gather_recipe())
    source, index = torch.randn(rows, 128), torch.randint(rows, (count,))
    expected = source.index_select(0, index).to(cuda_device, dtype=torch.float16)
    out = program.execute(source, index)
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(expected))
    out2 = program.submit(source, index).wait_stream()[0]
    torch.cuda.synchronize()
    assert torch.equal(bits(out2), bits(expected))
    stats = advised.stats()
    assert stats['advice'] == {'advice_prefers_direct': 2} and stats['dispatches'] == {'direct': 2}
    assert stats['direct_wire_bytes'] == count * 128 * 2
    index[0] = rows
    with pytest.raises(Exception, match='index_out_of_range'):
        program.execute(source, index)
    index[0] = 0
    # The failed call returned its wire: the next call is direct again.
    assert torch.equal(bits(program.execute(source, index)),
                       bits(source.index_select(0, index).to(cuda_device, dtype=torch.float16)))
    assert advised.stats()['dispatches'] == {'direct': 3}


@pytest.mark.gpu
@pytest.mark.parametrize('bags', [1, 41, 512])
def test_pooled_direct_path_is_bit_exact_and_reported(advised, cuda_device, bags):
    gen = torch.Generator().manual_seed(11)
    tables = [torch.randn(700, 64, generator=gen) for _ in range(26)]
    indices, offsets = bag_operands(26, bags, 700, gen)
    reference = torch.stack([torch.nn.functional.embedding_bag(i, t, o, mode='sum')
                             for t, i, o in zip(tables, indices, offsets)]).transpose(0, 1).half().to(cuda_device)
    program = advised.pooled(tables)
    out = program.execute(indices, offsets)
    out2 = program.submit(indices, offsets).wait_stream()[0]
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(reference)) and torch.equal(bits(out2), bits(reference))
    assert advised.stats()['advice'] == {'advice_prefers_direct': 2}
    # pipeline=False stays the dispatcher (an attribution call, undecided).
    whole = program.execute(indices, offsets, pipeline=False)
    assert torch.equal(bits(whole), bits(reference))
    stats = advised.stats()
    assert stats['dispatches'] == {'direct': 2, 'cpu_reference': 1} and stats['advice'] == {'advice_prefers_direct': 2}
    with pytest.raises(ValueError):
        program.execute(indices[:-1], offsets[:-1])


@pytest.mark.gpu
def test_direct_ring_waits_for_the_event_before_reuse(advised, cuda_device):
    program = advised.indexed(gather_recipe())
    source = torch.randn(4000, 128)
    gen = torch.Generator().manual_seed(5)
    indices = [torch.randint(4000, (3000,), generator=gen) for _ in range(4)]
    stream = torch.cuda.Stream(device=cuda_device)
    outputs = []
    with torch.cuda.stream(stream):
        # Keep the stream busy: the copies of the first two calls cannot have
        # completed when the third asks for a wire, so both wires are in flight.
        torch.cuda._sleep(300_000_000)
        for index in indices[:3]:
            outputs.append(program.execute(source, index))
        assert advised.stats()['advice'] == {'advice_prefers_direct': 2, 'direct_ring_exhausted': 1}
        outputs.append(program.execute(source, indices[3]))
    stream.synchronize()
    # Distinct indices per call: a wire reused before its copy ran would
    # corrupt an earlier output.
    for index, out in zip(indices, outputs):
        assert torch.equal(bits(out), bits(source.index_select(0, index).to(cuda_device, dtype=torch.float16)))
    stats = advised.stats()
    assert sum(stats['advice'].values()) == 4 and stats['advice']['direct_ring_exhausted'] >= 1
    assert stats['dispatches']['direct'] + stats['dispatches']['cpu_reference'] == 4
    assert stats['direct_wire_bytes'] == 2 * 3000 * 128 * 2


@pytest.mark.gpu
def test_missing_keys_fall_back_to_the_dispatcher(compiler, cuda_device, tmp_path):
    """A calibration without the advice keys (an older .cal, an uncalibrated box)."""
    import pyreloc
    from reloc_torch.fast import FastTransfers
    path = tmp_path / 'old.cal'
    path.write_text('\n'.join(line for line in SYNTH.splitlines() if not line.startswith('overhead.')) + '\n')
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=8,
                       calibration=pyreloc.load_calibration(str(path))) as owner:
        program = owner.indexed(gather_recipe())
        source, index = torch.randn(50, 128), torch.randint(50, (10,))
        out = program.execute(source, index)
        assert torch.equal(bits(out), bits(source.index_select(0, index).to(cuda_device, dtype=torch.float16)))
        assert owner.stats()['advice'] == {'advice_missing_keys': 1}
        assert owner.stats()['dispatches'] == {'cpu_reference': 1}


@pytest.mark.gpu
def test_typed_single_calls_never_claim_the_direct_path(advised, cuda_device):
    q = torch.randint(-127, 128, (64, 96), dtype=torch.int8)
    scale = torch.rand(64) * 0.05 + 1e-3
    program = advised.typed(weight_recipe('float32'), q, parameters={'scale': scale})
    assert program.resolved is not None and program.resolved.direct
    out = program.execute(q)
    assert torch.equal(out, (q.to(cuda_device).float() * scale.to(cuda_device)[:, None]))
    reason, = advised.stats()['advice']
    assert reason in ('advice_prefers_pipelined', 'advice_missing_keys')
    assert 'direct' not in advised.stats()['dispatches']


@pytest.mark.gpu
def test_ops_take_the_direct_path_and_stay_traceable(advised, cuda_device):
    from reloc_torch import fast_ops
    program = advised.indexed(gather_recipe())
    source, index = torch.randn(300, 128), torch.randint(300, (64,))
    expected = program.execute(source, index)

    def fn(x, i):
        return program.op(x, i).float().sum(dim=1)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(source, index)
    assert explanation.graph_break_count == 0
    compiled = torch.compile(fn, fullgraph=True, backend='aot_eager')
    torch.testing.assert_close(compiled(source, index), expected.float().sum(dim=1), rtol=0, atol=0)
    torch.library.opcheck(fast_ops.INDEXED_OP, (source, index, program._op_args[1], (64, 128), (128, 1),
                                                  cuda_device, torch.float16))
    assert advised.stats()['dispatches'] == {'direct': advised.stats()['calls']}


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_weights_direct_group_is_bit_exact(advised, cuda_device, dtype):
    from reloc_torch import PreparedWireWeights
    from reloc_torch.fast import FastWeights
    gen = torch.Generator().manual_seed(7)
    pairs = []
    for rows, cols in ((256, 384), (384, 256), (33, 65), (1024, 3072)):
        q = torch.randint(-127, 128, (rows, cols), dtype=torch.int8, generator=gen)
        pairs.append((q, torch.rand(cols, generator=gen) * 0.05 + 1e-3))
    torch_dtype = getattr(torch, dtype)

    def reference(q, s):
        return (q.to(cuda_device).float() * s.to(cuda_device)).t().contiguous().to(torch_dtype)

    with PreparedWireWeights(cuda_device, compiler=advised._compiler, dtype=dtype) as weights:
        assert weights.advice is not None
        assert weights.advice.receive_kernel == {'float32': 'dequant_s8_f32', 'float16': 'dequant_s8_f16'}[dtype]
        for i, (q, s) in enumerate(pairs):
            weights.prepare(str(i), q, s)
        loader = FastWeights(advised, weights)
        outputs = loader.load_many(['0', '1', '2', '3'])
        torch.cuda.synchronize()
        for (q, s), out in zip(pairs, outputs):
            assert out.dtype == torch_dtype and torch.equal(bits(out), bits(reference(q, s)))
        seen = []
        with loader.prefetch([['0', '1'], ['2'], ['3', '0']], slots=2) as window:
            for group in window:
                seen.append([t.clone() for t in group])
        torch.cuda.synchronize()
        assert [len(group) for group in seen] == [2, 1, 2]
        assert torch.equal(bits(seen[2][0]), bits(reference(*pairs[3])))
        assert torch.equal(bits(seen[2][1]), bits(reference(*pairs[0])))
    stats = advised.stats()
    assert stats['advice'] == {'advice_prefers_direct': 4} and stats['dispatches'] == {'direct': 9}


@pytest.mark.gpu
def test_kv_groups_stay_grouped_with_an_advisory_reason(advised, cuda_device, tmp_path):
    import pyreloc
    from reloc_torch.fast import KVCacheSwap
    swap = KVCacheSwap(advised, 16, 64)
    caches = [torch.randn(16, 12, 64, device=cuda_device) for _ in range(8)]
    restored = swap.restore(swap.evict(caches))
    torch.cuda.synchronize()
    for cache, back in zip(caches, restored):
        assert torch.equal(back.view(torch.int32), cache.view(torch.int32))
    swap.drain()
    assert advised.stats()['advice'] == {'advice_prefers_grouped': 2}
    assert advised.stats()['dispatches'] == {'layout_cuda': 16}
    # A calibration pricing the per-member intercept absurdly high: separate
    # calls are cheaper on paper, the group still runs, the reason says so.
    path = tmp_path / 'split.cal'
    path.write_text(SYNTH.replace('overhead.group_member_ms 0.01', 'overhead.group_member_ms 50'))
    advised.calibration = pyreloc.load_calibration(str(path))
    swap.restore(swap.evict(caches))
    swap.drain()
    assert advised.stats()['advice'] == {'advice_prefers_grouped': 2, 'advice_prefers_split': 2}
    assert advised.stats()['dispatches'] == {'layout_cuda': 32}


@pytest.mark.gpu
def test_mixed_group_stays_grouped(advised, cuda_device):
    """A direct-capable weight and a layout member in one submit: the runtime
    never splits the caller's group; the native group runs with an advisory reason."""
    from reloc_torch import PreparedWireWeights
    from reloc_torch.fast import KVCacheSwap, _WireEntry
    q = torch.randint(-127, 128, (64, 96), dtype=torch.int8)
    scale = torch.rand(96) * 0.05 + 1e-3
    cache = torch.randn(12, 16, 64).pin_memory()
    swap = KVCacheSwap(advised, 16, 64)
    layout = swap._program(swap._restore_recipe, cache)
    with PreparedWireWeights(cuda_device, compiler=advised._compiler) as weights:
        weights.prepare('w', q, scale)
        template, source, s = weights.fast_entries(['w'])[0]
        entry = _WireEntry(advised, template, source, s, 'float32', advice=weights.advice)
        assert entry.resolved is not None and entry.resolved.direct
        completion = advised.submit([(entry, source), (layout, cache)])
        w, k = completion.wait()
        assert torch.equal(w, (q.to(cuda_device).float() * scale.to(cuda_device)).t().contiguous())
        assert torch.equal(k, cache.transpose(0, 1).contiguous().to(cuda_device))
        assert completion.report['placement_reason'] in ('advice_prefers_grouped', 'advice_prefers_split')
        assert completion.report['predicted_ms']['direct'] is None
        dispatches = advised.stats()['dispatches']
        assert 'direct' not in dispatches and dispatches.get('cpu_stages_cuda_stages@0') == 1
