"""Selection and held-out chains frozen before selection; no manual Sym recipes."""
import torch
import torch.ao.quantization.fx._decomposed  # register audited operators


def selection_cast(x):
    return x.t().contiguous().to('cuda:0', torch.float16)


def selection_layout(x):
    return x.permute(2, 0, 1).contiguous().to('cuda:0')


def selection_dequant(q, s, z):
    return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
        q, s, z, -128, 127, torch.int8).t().contiguous().to('cuda:0', torch.float16)


def selection_index(x, index):
    return x.index_select(0, index).to('cuda:0', torch.float16)


def held_cast_chain(x):
    return x.to(torch.float16).permute(2, 0, 3, 1).contiguous().to('cuda:0', torch.float32)


def held_layout(x):
    return x.permute(3, 1, 0, 2).contiguous().to('cuda:0')


def held_dequant_axis0(q, s, z):
    return torch.ops.quantized_decomposed.dequantize_per_channel(
        q, s, z, 0, -128, 127, torch.int8).permute(2, 1, 0).contiguous().to('cuda:0', torch.float16)


def held_index_rank3(x, index):
    return x.index_select(0, index).to('cuda:0', torch.float16)


def inputs(case, n, step):
    g = torch.Generator().manual_seed(265000 + step)
    strided = case.startswith('held_') and step % 12 == 11
    if case == 'selection_cast': return (torch.randn(n, 192, generator=g),)
    if case == 'selection_layout': return (torch.randn(3, n, 40, generator=g),)
    if case == 'selection_dequant':
        return (torch.randint(-128, 128, (n, 48), dtype=torch.int8, generator=g),
                torch.tensor(.05 + .001 * step), torch.tensor(0 if step % 2 else 3, dtype=torch.int32))
    if case == 'selection_index':
        return torch.randn(1024, 96, generator=g), torch.randint(0, 1024, (n,), generator=g)
    if case == 'held_cast_chain':
        x = torch.randn(5, n * (2 if strided else 1), 11, 7, generator=g)
        return (x[:, ::2] if strided else x,)
    if case == 'held_layout':
        x = torch.randn(n * (2 if strided else 1), 3, 5, 17, generator=g)
        return (x[::2] if strided else x,)
    if case == 'held_dequant_axis0':
        return (torch.randint(-128, 128, (23, n, 7), dtype=torch.int8, generator=g),
                torch.rand(23, generator=g) * .2 + .01,
                torch.zeros(23, dtype=torch.int32) if step % 2 else torch.randint(-2, 3, (23,), dtype=torch.int32, generator=g))
    if case == 'held_index_rank3':
        return torch.randn(733, 5, 19, generator=g), torch.randint(0, 733, (n,), generator=g)
    raise ValueError(case)
