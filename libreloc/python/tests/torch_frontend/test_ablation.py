"""Ablation knobs and kernel-only entry points (bench/beat_torch attribution)."""
import pytest
import torch
import pyreloc

from reloc_torch.fast import FastTransfers, _load_native_views
from test_fast import dlrm_recipe, gather_recipe


def native():
    helper = _load_native_views()
    if helper is None:
        pytest.skip('libtorch helper not built (RELOC_ENABLE_TORCH_EXT)')
    return helper


@pytest.fixture
def fast(compiler, cuda_device):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        yield owner


@pytest.fixture
def knobs_off():
    yield
    for name in pyreloc.ablations():
        pyreloc.set_ablation(name, False)


def test_knobs_default_off_toggle_and_reject_unknown_names(knobs_off):
    assert pyreloc.ablations() == {'scalar_dequant': False, 'unfused_dequant_f16': False}
    pyreloc.set_ablation('scalar_dequant', True)
    assert pyreloc.ablations()['scalar_dequant'] is True
    pyreloc.set_ablation('scalar_dequant', False)
    assert pyreloc.ablations()['scalar_dequant'] is False
    with pytest.raises(ValueError, match='unknown ablation knob'):
        pyreloc.set_ablation('no_such_knob', True)


@pytest.mark.gpu
def test_dequant_kernel_matches_torch_bit_for_bit_with_and_without_the_knob(cuda_device, knobs_off):
    helper = native()
    q = torch.randint(-128, 128, (96, 1024), dtype=torch.int8, device=cuda_device)
    scale = torch.rand(96, device=cuda_device) * 0.1 + 1e-3
    expected = q.float() * scale[:, None]
    for knob in (False, True):
        pyreloc.set_ablation('scalar_dequant', knob)
        out = torch.full(q.shape, float('nan'), device=cuda_device)
        assert helper.dequant_s8_f32(q, scale, out, torch.cuda.current_stream().cuda_stream) is out
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int32), expected.view(torch.int32)), knob
    with pytest.raises(ValueError):
        helper.dequant_s8_f32(q, scale[:5], torch.empty_like(expected))
    with pytest.raises(ValueError):
        helper.dequant_s8_f32(q.cpu(), scale, torch.empty_like(expected))


@pytest.mark.gpu
def test_pooled_bags_kernel_matches_embedding_bag_into_a_pinned_wire(cuda_device):
    helper = native()
    gen = torch.Generator().manual_seed(11)
    tables = [torch.randn(700, 64, generator=gen) for _ in range(5)]
    bags = 41
    indices, offsets = [], []
    for _ in tables:
        lengths = torch.randint(1, 5, (bags,), generator=gen)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, 700, (int(lengths.sum()),), generator=gen))
    reference = torch.stack([torch.nn.functional.embedding_bag(i, t, o, mode='sum')
                             for t, i, o in zip(tables, indices, offsets)]).transpose(0, 1)
    for dtype in (torch.float16, torch.float32):
        wire = torch.empty((bags, 5, 64), dtype=dtype, pin_memory=True)
        assert helper.pooled_bags(tables, indices, offsets, wire, 4) is wire
        assert torch.equal(wire.view(torch.int16 if dtype == torch.float16 else torch.int32),
                           reference.to(dtype).view(torch.int16 if dtype == torch.float16 else torch.int32))
    with pytest.raises(ValueError):
        helper.pooled_bags(tables, indices, offsets, torch.empty((bags, 4, 64), dtype=torch.float16), 4)
    bad = [i.clone() for i in indices]
    bad[0][0] = 700
    with pytest.raises(pyreloc.TransferError, match='index_out_of_range'):
        helper.pooled_bags(tables, bad, offsets, torch.empty((bags, 5, 64), dtype=torch.float16), 4)


@pytest.mark.gpu
def test_gather_rows_kernel_matches_index_select_into_a_pinned_wire(compiler, cuda_device):
    helper = native()
    source = torch.randn(5000, 128)
    index = torch.randint(5000, (777,))
    compiled = compiler.compile(gather_recipe())
    bindings = compiled.bind_values(source, index)
    wire = torch.empty((777, 128), dtype=torch.float16, pin_memory=True)
    assert helper.gather_rows(compiled.decoded_plan, bindings, source, index, wire, 4) is wire
    assert torch.equal(wire.view(torch.int16), source.index_select(0, index).half().view(torch.int16))
    with pytest.raises(pyreloc.BindError, match='index_out_of_range'):
        helper.gather_rows(compiled.decoded_plan, bindings, source, torch.tensor([5000] * 777), wire, 4)
    with pytest.raises(ValueError):
        helper.gather_rows(compiled.decoded_plan, bindings, source, index, wire[:10], 4)


@pytest.mark.gpu
def test_producers_without_chunk_overlap_give_the_same_bytes(fast, cuda_device):
    source = torch.randn(3000, 128)
    index = torch.randint(3000, (2000,))
    indexed = fast.indexed(gather_recipe())
    pipelined = indexed.execute(source, index)
    whole = indexed.execute(source, index, pipeline=False)
    assert torch.equal(pipelined.view(torch.int16), whole.view(torch.int16))
    gen = torch.Generator().manual_seed(3)
    tables = [torch.randn(500, 64, generator=gen) for _ in range(26)]
    bags = 512
    indices, offsets = [], []
    for _ in tables:
        lengths = torch.randint(1, 5, (bags,), generator=gen)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, 500, (int(lengths.sum()),), generator=gen))
    pooled = fast.pooled(tables)
    assert torch.equal(pooled.execute(indices, offsets).view(torch.int16),
                       pooled.execute(indices, offsets, pipeline=False).view(torch.int16))


@pytest.mark.gpu
def test_scalar_dequant_knob_keeps_typed_programs_bit_exact(fast, cuda_device, knobs_off):
    from reloc_torch import PreparedWireWeights
    from reloc_torch.fast import FastWeights
    q = torch.randint(-127, 128, (1024, 2048), dtype=torch.int8)
    scale = torch.rand(2048) * 0.05 + 1e-3
    with PreparedWireWeights(cuda_device, compiler=fast._compiler) as weights:
        weights.prepare('w', q, scale)
        loader = FastWeights(fast, weights)
        quad = loader.load_many(['w'])[0].clone()
        pyreloc.set_ablation('scalar_dequant', True)
        scalar = loader.load_many(['w'])[0]
    assert torch.equal(quad.view(torch.int32), scalar.view(torch.int32))
    assert torch.equal(quad, (q.to(cuda_device).float() * scale.to(cuda_device)).t().contiguous())
