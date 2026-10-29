"""Precision additions: f16 weight outputs (fused dequantize pair), bf16 casts, 16-bit KV swaps."""
import pytest
import torch
import pyreloc

from reloc_torch import PreparedWireWeights
from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap
from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides


def spec(shape, dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


def transposed_cast_recipe(dtype, policy, source_dtype='float32', tables=26, dim=64):
    t, b = Const(tables), Symbol('s1')
    return Recipe(spec((t, b, Const(dim)), source_dtype), (Transpose((1, 0, 2)), Cast(dtype, policy)),
                  spec((b, t, Const(dim)), dtype), 'h2d')


@pytest.fixture
def fast(compiler, cuda_device):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        yield owner


@pytest.fixture
def knobs_off():
    yield
    for name in pyreloc.ablations():
        pyreloc.set_ablation(name, False)


def bits(tensor):
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(1024, 4096), (333, 1024)])
def test_f16_wire_weights_match_torch_bit_for_bit_fused_and_unfused(fast, cuda_device, shape, knobs_off):
    q = torch.randint(-127, 128, shape, dtype=torch.int8)
    scale = torch.rand(shape[1]) * 0.05 + 1e-3
    expected = (q.to(cuda_device).float() * scale.to(cuda_device)).t().contiguous().half()
    with PreparedWireWeights(cuda_device, compiler=fast._compiler, dtype='float16') as weights:
        info = weights.prepare('w', q, scale)
        assert info['output_dtype'] == 'float16' and info['output_bytes'] == q.numel() * 2
        assert weights.load('w').dtype == torch.float16
        assert torch.equal(bits(weights.load('w')), bits(expected))
        loader = FastWeights(fast, weights)
        fused = loader.load_many(['w'])[0]
        assert fused.dtype == torch.float16 and torch.equal(bits(fused), bits(expected))
        pyreloc.set_ablation('unfused_dequant_f16', True)
        unfused = loader.load_many(['w'])[0]
        assert torch.equal(bits(unfused), bits(expected))
        pyreloc.set_ablation('scalar_dequant', True)
        pyreloc.set_ablation('unfused_dequant_f16', False)
        assert torch.equal(bits(loader.load_many(['w'])[0]), bits(expected))
    with pytest.raises(ValueError):
        PreparedWireWeights(cuda_device, compiler=fast._compiler, dtype='bfloat16')


@pytest.mark.gpu
@pytest.mark.parametrize('implementation', ['', 'cpu_stages_cuda_stages@0'])
def test_bf16_narrowing_program_matches_torch_rounding(fast, cuda_device, implementation):
    source = torch.randn(26, 40, 64) * 1e3
    source[0, 0, :8] = torch.tensor([1 + 2 ** -8, 1 + 3 * 2 ** -9, 2 ** -133, 2 ** -134, 3.3895313892515355e38,
                                     3.4e38, -0., float('inf')])
    program = fast.typed(transposed_cast_recipe('bfloat16', 'ieee_rne'), source, implementation=implementation)
    expected = source.transpose(0, 1).contiguous().to(cuda_device, dtype=torch.bfloat16)
    out = program.execute(source)
    assert out.dtype == torch.bfloat16 and out.shape == expected.shape
    assert torch.equal(bits(out), bits(expected))


@pytest.mark.gpu
@pytest.mark.parametrize('implementation', ['', 'cpu_stages_cuda_stages@0'])
def test_bf16_widening_program_is_exact(fast, cuda_device, implementation):
    source = (torch.randn(26, 17, 64) * 1e4).to(torch.bfloat16)
    program = fast.typed(transposed_cast_recipe('float32', 'exact', source_dtype='bfloat16'), source,
                         implementation=implementation)
    expected = source.transpose(0, 1).contiguous().to(cuda_device, dtype=torch.float32)
    assert torch.equal(bits(program.execute(source)), bits(expected))


def test_bf16_to_f16_has_no_table(compiler):
    from reloc_torch.artifact import UnsupportedRecipe
    recipe = transposed_cast_recipe('float16', 'ieee_rne', source_dtype='bfloat16')
    with pytest.raises((UnsupportedRecipe, RuntimeError)):
        compiler.compile(recipe)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_kv_swap_round_trips_16bit_caches(fast, cuda_device, dtype):
    torch_dtype = getattr(torch, dtype)
    swap = KVCacheSwap(fast, 16, 64, dtype=dtype)
    caches = [torch.randn(16, 9, 64, device=cuda_device).to(torch_dtype) for _ in range(6)]
    stored = swap.evict(caches)
    for cache, host in zip(caches, stored):
        assert host.dtype == torch_dtype and host.shape == (9, 16, 64) and host.is_pinned()
        assert torch.equal(bits(host), bits(cache.transpose(0, 1).contiguous().cpu()))
    restored = swap.restore(stored)
    torch.cuda.synchronize()
    for cache, back in zip(caches, restored):
        assert back.dtype == torch_dtype and torch.equal(bits(back), bits(cache))
    swap.drain()
    with pytest.raises(ValueError):
        KVCacheSwap(fast, 16, 63, dtype=dtype)
    with pytest.raises(ValueError):
        swap.evict([torch.randn(16, 9, 64, device=cuda_device)])
