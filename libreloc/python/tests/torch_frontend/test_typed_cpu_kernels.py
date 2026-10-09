"""Tiled CPU stages keep global padding coordinates and chunk-local storage."""
import pytest
import torch

from reloc_torch import TransferResources, dispatch
from reloc_torch.recipe import Cast, Fill, Pad, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Add, Const, Symbol, dense_strides


def spec(shape, dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


@pytest.mark.gpu
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
def test_padded_tiled_stage_chain_and_gpu_cuts(compiler, cuda_device, direction):
    n, m = Symbol('s0'), Symbol('s1')
    recipe = Recipe(spec((n, m), 'float32'),
        (Transpose((1, 0)), Cast('float16', 'ieee_rne'),
         Pad(0, Const(3), Const(5), Fill('float16', 0x8000)),
         Pad(1, Const(2), Const(7), Fill('float16', 0x8000)), Cast('float32', 'exact')),
        spec((Add(Const(8), m), Add(Const(9), n)), 'float32'), direction)
    compiled = compiler.compile(recipe)
    device = cuda_device if direction == 'h2d' else 'cpu'
    paths = ['cpu_reference', 'cpu_stages_cuda_stages@1' if direction == 'h2d'
             else 'cuda_stages_then_cpu@1']
    stream = torch.cuda.Stream(device=cuda_device)
    with TransferResources(max_typed_live_bytes=8 << 20) as owner, torch.cuda.stream(stream):
        saved = []
        for rows, columns in ((33, 65), (129, 257)):
            host = torch.randn(rows, columns)
            host[0, :4] = torch.tensor([-0., float('inf'), 2**-24, 65520.])
            expected = torch.full((columns+8, rows+9), -0., dtype=torch.float32)
            expected[3:columns+3, 2:rows+2] = host.t().half().float()
            source = host if direction == 'h2d' else host.to(cuda_device)
            for path in paths:
                for pipeline in (False, True):
                    request = dispatch.prepare_typed_transfer(compiled, source, device, implementation=path)
                    output = dispatch.execute_typed_transfer(request, resources=owner, gather_threads=8,
                        pipeline=pipeline, n_buffers=2, chunk_size=(rows+9)*2*7)
                    assert output.report['host_kernel'] == 'tiled_transpose'
                    assert torch.equal(output.tensor.cpu().view(torch.int32), expected.view(torch.int32))
                    saved.append((output.tensor, expected.clone()))
            source.fill_(3.)
        stream.synchronize()
        for tensor, expected in saved:
            assert torch.equal(tensor.cpu().view(torch.int32), expected.view(torch.int32))
