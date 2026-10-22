"""reloc_torch.fast: prepared programs, groups, KV swaps, pooled producers."""
import pytest
import torch
import pyreloc

from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap, _load_native_views
from reloc_torch.recipe import Cast, IndexSelect, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides


def spec(shape, dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


def dlrm_recipe(tables=26, dim=64):
    t, b = Const(tables), Symbol('s1')
    source, destination = (t, b, Const(dim)), (b, t, Const(dim))
    return Recipe(spec(source, 'float32'), (Transpose((1, 0, 2)), Cast('float16', 'ieee_rne')),
                  spec(destination, 'float16'), 'h2d')


def gather_recipe(columns=128):
    n, k = Symbol('s0'), Symbol('k')
    return Recipe(spec((n, Const(columns)), 'float32'),
                  (IndexSelect(TensorSpec((k,), (Const(1),), Const(0), 'int64')), Cast('float16', 'ieee_rne')),
                  spec((k, Const(columns)), 'float16'), 'h2d')


@pytest.fixture
def fast(compiler, cuda_device):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        yield owner


@pytest.mark.gpu
@pytest.mark.parametrize('batch', [1, 7, 333, 2048])
def test_typed_program_matches_eager_and_rejects_other_signatures(fast, cuda_device, batch):
    source = torch.randn(26, batch, 64)
    source[0, 0, :4] = torch.tensor([-0., float('inf'), 2 ** -24, 65520.])
    program = fast.typed(dlrm_recipe(), source)
    assert program.selection['implementation'] == 'cpu_reference'
    expected = source.transpose(0, 1).contiguous().to(cuda_device, dtype=torch.float16)
    for pipeline in (True, False):
        out = program.execute(source, pipeline=pipeline)
        assert out.shape == (batch, 26, 64) and out.is_contiguous()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    source.neg_()
    assert torch.equal(program.execute(source).view(torch.int16),
                       source.transpose(0, 1).contiguous().to(cuda_device, dtype=torch.float16).view(torch.int16))
    with pytest.raises((ValueError, RuntimeError)):
        program.execute(torch.randn(26, batch + 1, 64))
    with pytest.raises((ValueError, RuntimeError)):
        program.execute(torch.randn(26, batch, 64).half())
    assert fast.stats()['dispatches']['cpu_reference'] == 3


@pytest.mark.gpu
def test_strided_runs_kernel_is_reported_for_rank_three_layouts(fast):
    program = fast.typed(dlrm_recipe(), torch.randn(26, 40, 64))
    program.execute(torch.randn(26, 40, 64))
    # The dispatch report of the last call is folded into stats; probe the
    # template's own selection of the host kernel through a direct request.
    request = pyreloc.prepare_dispatch_from_template(
        program.template, program.source_view(torch.randn(26, 40, 64)),
        program.output_view(torch.empty(40, 26, 64, dtype=torch.float16, device='cuda')))
    assert request.report['host_kernel'] == 'strided_runs'


@pytest.mark.gpu
@pytest.mark.parametrize('rows,count', [(1, 1), (9, 1), (1000, 777), (5000, 4096)])
def test_indexed_program_matches_eager_and_checks_indices(fast, cuda_device, rows, count):
    program = fast.indexed(gather_recipe())
    source = torch.randn(rows, 128)
    index = torch.randint(0, rows, (count,))
    out = program.execute(source, index)
    expected = source.index_select(0, index).to(cuda_device, dtype=torch.float16)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    index[0] = rows
    with pytest.raises(Exception, match='index_out_of_range'):
        program.execute(source, index)
    index[0] = -1
    with pytest.raises(Exception, match='index_out_of_range'):
        program.execute(source, index)
    with pytest.raises(ValueError):
        program.execute(source, index[:1].to(torch.int16))


def test_bind_indexed_from_pointer_matches_bytes_binding(compiler):
    compiled = compiler.compile(gather_recipe(8))
    index = torch.tensor([3, 0, 7, 7, 1])
    bindings = compiled.bind_values(torch.randn(8, 8), index)
    by_bytes = pyreloc.bind_indexed(compiled.decoded_plan, bindings, ('int64', [5], index.numpy().tobytes()))
    by_pointer = pyreloc.bind_indexed_from_pointer(compiled.decoded_plan, bindings, 'int64', [5],
                                                   index.data_ptr(), index.numel() * 8)
    assert by_pointer.indices == by_bytes.indices == [3, 0, 7, 7, 1]
    narrow = index.to(torch.int32)
    recipe32 = Recipe(spec((Symbol('s0'), Const(8)), 'float32'),
                      (IndexSelect(TensorSpec((Symbol('k'),), (Const(1),), Const(0), 'int32')),),
                      spec((Symbol('k'), Const(8)), 'float32'), 'h2d')
    compiled32 = compiler.compile(recipe32)
    narrow_bindings = compiled32.bind_values(torch.randn(8, 8), narrow)
    assert pyreloc.bind_indexed_from_pointer(compiled32.decoded_plan, narrow_bindings, 'int32', [5],
                                             narrow.data_ptr(), 20).indices == [3, 0, 7, 7, 1]
    bad = torch.tensor([3, 8])
    with pytest.raises(pyreloc.BindError, match='index_out_of_range'):
        pyreloc.bind_indexed_from_pointer(compiled.decoded_plan, compiled.bind_values(torch.randn(8, 8), bad),
                                          'int64', [2], bad.data_ptr(), 16)


@pytest.mark.gpu
@pytest.mark.parametrize('length', [1, 8, 33])
def test_kv_swap_round_trips_bit_exactly_with_pinned_views(fast, cuda_device, length):
    swap = KVCacheSwap(fast, 16, 64)
    caches = [torch.randn(16, length, 64, device=cuda_device) for _ in range(8)]
    caches[0][0, 0, 0] = -0.
    stored = swap.evict(caches)
    assert all(t.is_pinned() and t.shape == (length, 16, 64) and t.is_contiguous() for t in stored)
    for cache, host in zip(caches, stored):
        assert torch.equal(host.view(torch.int32), cache.transpose(0, 1).contiguous().cpu().view(torch.int32))
    restored = swap.restore(stored)
    torch.cuda.synchronize()
    for cache, back in zip(caches, restored):
        assert back.device == cache.device and back.shape == cache.shape and back.is_contiguous()
        assert torch.equal(back.view(torch.int32), cache.view(torch.int32))
    # Mixed shapes in one call fall back to per-item kernels but stay exact.
    mixed = caches[:3] + [torch.randn(16, length + 2, 64, device=cuda_device)]
    stored = swap.evict(mixed)
    for cache, host in zip(mixed, stored):
        assert torch.equal(host, cache.transpose(0, 1).contiguous().cpu())
    swap.drain()
    assert fast.stats()['dispatches']['layout_cuda'] == 8 + 8 + 4


@pytest.mark.gpu
def test_group_d2h_into_pageable_destination_and_fresh_outputs(fast, cuda_device):
    swap = KVCacheSwap(fast, 4, 8)
    cache = torch.randn(4, 5, 8, device=cuda_device)
    program = swap._program(swap._evict_recipe, cache)
    pageable = torch.empty(5, 4, 8)
    completion = fast.submit([(program, cache)], outputs=[pageable], device_layout=True, gather_threads=1)
    out, = completion.wait()
    assert out is pageable and not out.is_pinned()
    assert torch.equal(out, cache.transpose(0, 1).contiguous().cpu())
    first = fast.submit([(program, cache)], device_layout=False, gather_threads=1).wait()[0]
    second = fast.submit([(program, cache)], device_layout=True, gather_threads=1).wait()[0]
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, second)


@pytest.mark.gpu
@pytest.mark.parametrize('bags,dtype', [(1, 'float16'), (64, 'float16'), (513, 'float32'), (2048, 'float16')])
def test_pooled_producer_matches_embedding_bag_sum_bit_for_bit(fast, cuda_device, bags, dtype):
    generator = torch.Generator().manual_seed(bags)
    tables = [torch.randn(rows, 64, generator=generator) for rows in (10, 1000, 100_000)]
    bag = [torch.nn.EmbeddingBag.from_pretrained(t, freeze=True, mode='sum') for t in tables]
    indices, offsets = [], []
    for table in tables:
        counts = torch.randint(0, 5, (bags,), generator=generator)       # includes empty bags
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, table.shape[0], (int(counts.sum()),), generator=generator))
    program = fast.pooled(tables, dtype=dtype)
    out = program.execute(indices, offsets)
    expected = torch.stack([b(i, o) for b, i, o in zip(bag, indices, offsets)]).transpose(0, 1).contiguous()
    expected = expected.to(cuda_device, dtype=getattr(torch, dtype))
    assert out.shape == (bags, 3, 64)
    width = torch.int16 if dtype == 'float16' else torch.int32
    assert torch.equal(out.view(width), expected.view(width))
    bad = [i.clone() for i in indices]
    if bad[2].numel():
        bad[2][-1] = tables[2].shape[0]
        with pytest.raises(RuntimeError, match='index_out_of_range'):
            program.execute(bad, offsets)
    broken = [o.clone() for o in offsets]
    broken[1][0] = 1
    with pytest.raises(RuntimeError, match='nondecreasing|start at zero'):
        program.execute(indices, broken)


@pytest.mark.gpu
def test_asynchronous_producers_match_blocking_results(fast, cuda_device):
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(4000, 128, generator=generator)
    index = torch.randint(0, 4000, (2500,), generator=generator)
    gather = fast.indexed(gather_recipe())
    completion = gather.submit(source, index)
    assert completion.report is None
    out, = completion.wait_stream()
    assert torch.equal(out, source.index_select(0, index).to(cuda_device, dtype=torch.float16))
    assert completion.done() and completion.report['items'][0]['host_pipeline'] in ('streamed', 'chunked', 'whole_single_chunk')
    tables = [torch.randn(300, 32, generator=generator) for _ in range(3)]
    offsets = [torch.arange(0, 2 * 50, 2) for _ in tables]
    indices = [torch.randint(0, 300, (100,), generator=generator) for _ in tables]
    pooled = fast.pooled(tables, dtype='float32')
    blocking = pooled.execute(indices, offsets)
    asynchronous, = pooled.submit(indices, offsets).wait_stream()
    torch.cuda.synchronize()
    assert torch.equal(blocking, asynchronous)
    expected = torch.stack([torch.nn.functional.embedding_bag(i, t, o, mode='sum') for t, i, o in
                            zip(tables, indices, offsets)]).transpose(0, 1).contiguous().to(cuda_device)
    assert torch.equal(blocking, expected)
    # The index operand is retained by the completion: mutating it after
    # submission must not change what was transferred.
    completion = gather.submit(source, index)
    frozen = source.index_select(0, index).to(cuda_device, dtype=torch.float16)
    index.fill_(0)
    out, = completion.wait()
    assert torch.equal(out, frozen)


@pytest.mark.gpu
def test_fast_weights_groups_and_prefetch_match_reference(fast, cuda_device):
    from reloc_torch import PreparedWireWeights
    generator = torch.Generator().manual_seed(7)
    pairs = []
    for rows, cols in ((64, 96), (96, 64), (33, 65)):
        weight = torch.randn(rows, cols, generator=generator)
        scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-6)
        pairs.append((torch.round(weight / scale).clamp(-127, 127).to(torch.int8), scale))
    with PreparedWireWeights(cuda_device, compiler=fast._compiler) as weights:
        for index, (q, s) in enumerate(pairs):
            weights.prepare(str(index), q, s)
        loader = FastWeights(fast, weights)
        outputs = loader.load_many(['0', '1', '2'])
        for (q, s), out in zip(pairs, outputs):
            assert torch.equal(out, (q.to(cuda_device).float() * s.to(cuda_device)).t().contiguous())
        seen = []
        with loader.prefetch([['0', '1'], ['2'], ['1', '0'], ['2', '2', '0']], slots=2) as window:
            for group in window:
                seen.append([t.clone() for t in group])
        assert [len(group) for group in seen] == [2, 1, 2, 3]
        q, s = pairs[2]
        assert torch.equal(seen[1][0], (q.to(cuda_device).float() * s.to(cuda_device)).t().contiguous())
        assert torch.equal(seen[3][0], seen[3][1])
    stats = fast.stats()
    assert stats['groups'] == 5 and stats['dispatches']['cpu_stages_cuda_stages@0'] == 11


@pytest.mark.gpu
def test_native_view_helper_agrees_with_python_descriptors(cuda_device):
    _native_views = _load_native_views()
    if _native_views is None:
        pytest.skip('built without RELOC_ENABLE_TORCH_EXT')
    from reloc_torch.transport import _storage_view
    base = torch.randn(3, 4, 5, device=cuda_device)
    view = base[1:, :, 2:]
    native = _native_views.view_from_tensor(view)
    python = _storage_view(view, 'cuda', view.device.index)
    for name in ('base', 'capacity_bytes', 'offset_bytes', 'extents', 'strides', 'element_size', 'kind', 'device'):
        assert getattr(native, name) == getattr(python, name), name
    host = torch.randn(2, 6).half()[:, 1:]
    native = _native_views.view_from_tensor(host)
    python = _storage_view(host, 'host', -1)
    assert (native.base, native.offset_bytes, native.extents, native.strides) == (
        python.base, python.offset_bytes, python.extents, python.strides)
    with pytest.raises(TypeError):
        _native_views.view_from_tensor([1, 2])
