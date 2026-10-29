"""reloc_torch.fast_ops: prepared fast-lane programs as torch.compile primitives."""
import importlib

import pytest
import torch

from reloc_torch import fast_ops
from reloc_torch.artifact import UnsupportedRecipe
from reloc_torch.fast import FastTransfers
from test_fast import dlrm_recipe, gather_recipe


@pytest.fixture
def fast(compiler, cuda_device):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        yield owner


def graph_breaks():
    return dict(torch._dynamo.utils.counters.get('graph_break', {}))


def test_schemas_are_explicit_functional_and_registered_once():
    assert str(fast_ops.TYPED_OP._schema) == 'reloc_torch::fast_typed' + fast_ops.TYPED_SCHEMA
    assert str(fast_ops.INDEXED_OP._schema) == 'reloc_torch::fast_indexed' + fast_ops.INDEXED_SCHEMA
    assert str(fast_ops.POOLED_OP._schema) == 'reloc_torch::fast_pooled' + fast_ops.POOLED_SCHEMA
    for overload in (fast_ops.TYPED_OP, fast_ops.INDEXED_OP, fast_ops.POOLED_OP):
        assert not overload._schema.is_mutable and overload._schema.returns[0].alias_info is None
    before = (fast_ops.TYPED_OP, fast_ops.INDEXED_OP, fast_ops.POOLED_OP)
    reloaded = importlib.reload(fast_ops)
    assert (reloaded.TYPED_OP, reloaded.INDEXED_OP, reloaded.POOLED_OP) == before


def test_unknown_handle_fails_without_fabricating_output():
    with pytest.raises(RuntimeError, match='unknown or closed'):
        fast_ops.TYPED_OP(torch.zeros(2), 'reloc-0-deadbeef', [2], [1], torch.device('cpu'), torch.float32)


@pytest.mark.gpu
def test_typed_op_is_captured_whole_and_matches_execute(fast, cuda_device):
    source = torch.randn(26, 48, 64)
    program = fast.typed(dlrm_recipe(), source)
    expected = program.execute(source)

    def fn(x):
        return program.op(x).float().sum(dim=1)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(source)
    assert explanation.graph_break_count == 0 and explanation.graph_count == 1
    targets = [node.target for node in explanation.graphs[0].graph.nodes if node.op == 'call_function']
    assert fast_ops.TYPED_OP in targets
    compiled = torch.compile(fn, fullgraph=True, backend='aot_eager')
    dispatches = fast.stats()['dispatches']['cpu_reference']
    for _ in range(2):
        torch.testing.assert_close(compiled(source), expected.float().sum(dim=1), rtol=0, atol=0)
    assert fast.stats()['dispatches']['cpu_reference'] == dispatches + 2, 'fake kernel must not execute the program'
    assert torch.equal(program.op(source).view(torch.int16), expected.view(torch.int16))
    with pytest.raises((ValueError, RuntimeError)):
        program.op(torch.randn(26, 49, 64))


@pytest.mark.gpu
def test_fake_kernel_gives_the_destination_metadata_without_a_transfer(fast, cuda_device):
    from torch._subclasses.fake_tensor import FakeTensorMode
    source = torch.randn(26, 5, 64)
    program = fast.typed(dlrm_recipe(), source)
    dispatches = dict(fast.stats()['dispatches'])
    with FakeTensorMode() as mode:
        out = program.op(mode.from_tensor(source))
    assert tuple(out.shape) == (5, 26, 64) and out.dtype == torch.float16 and out.device == torch.device(cuda_device)
    assert out.is_contiguous()
    assert fast.stats()['dispatches'] == dispatches


@pytest.mark.gpu
def test_inductor_schedules_compute_around_the_op(fast, cuda_device):
    from torch._inductor import metrics
    source = torch.randn(26, 32, 64)
    program = fast.typed(dlrm_recipe(), source)
    weight = torch.randn(64, 64, device=cuda_device)

    def fn(x):
        return torch.relu(program.op(x).float() @ weight).sum()

    before = metrics.generated_kernel_count
    compiled = torch.compile(fn, fullgraph=True, backend='inductor', options={'triton.cudagraphs': False})
    with torch.no_grad():
        actual = compiled(source)
        source.neg_()
        again = compiled(source)
    assert metrics.generated_kernel_count > before
    torch.testing.assert_close(actual, fn(source.neg()), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(again, fn(source), rtol=1e-4, atol=1e-3)
    assert not graph_breaks()


@pytest.mark.gpu
def test_indexed_op_follows_the_index_length(fast, cuda_device):
    source = torch.randn(3000, 128)
    program = fast.indexed(gather_recipe())
    torch._dynamo.reset()
    compiled = torch.compile(lambda x, i: program.op(x, i).float() * 2, fullgraph=True, backend='aot_eager',
                             dynamic=True)
    for count in (7, 1024):
        index = torch.randint(3000, (count,))
        expected = source.index_select(0, index).to(cuda_device, dtype=torch.float16).float() * 2
        actual = compiled(source, index)
        assert tuple(actual.shape) == (count, 128)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not graph_breaks()
    with pytest.raises(UnsupportedRecipe, match='index_out_of_range'):
        program.op(source, torch.tensor([3000]))


@pytest.mark.gpu
def test_pooled_op_matches_embedding_bag_inside_a_compiled_region(fast, cuda_device):
    gen = torch.Generator().manual_seed(7)
    tables = [torch.randn(500, 64, generator=gen) for _ in range(4)]
    program = fast.pooled(tables, dtype='float16')
    bags = 33
    indices, offsets = [], []
    for _ in tables:
        lengths = torch.randint(1, 5, (bags,), generator=gen)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, 500, (int(lengths.sum()),), generator=gen))
    reference = torch.stack([torch.nn.functional.embedding_bag(i, t, o, mode='sum')
                             for t, i, o in zip(tables, indices, offsets)]).transpose(0, 1).to(cuda_device, torch.float16)
    torch._dynamo.reset()
    compiled = torch.compile(lambda i, o: program.op(i, o).float().sum(dim=2), fullgraph=True, backend='aot_eager')
    torch.testing.assert_close(compiled(indices, offsets), reference.float().sum(dim=2), rtol=0, atol=0)
    assert torch.equal(program.op(indices, offsets).view(torch.int16), reference.view(torch.int16))
    assert not graph_breaks()


@pytest.mark.gpu
def test_opcheck_passes_for_the_typed_op(fast, cuda_device):
    source = torch.randn(26, 9, 64)
    program = fast.typed(dlrm_recipe(), source)
    torch.library.opcheck(fast_ops.TYPED_OP, (source, *program._op_args))


@pytest.mark.gpu
def test_closed_owner_invalidates_the_handle(compiler, cuda_device):
    owner = FastTransfers(cuda_device, compiler=compiler, gather_threads=2)
    source = torch.randn(26, 3, 64)
    program = owner.typed(dlrm_recipe(), source)
    handle = program.handle
    assert fast_ops.lookup_program(handle) is program
    owner.close()
    with pytest.raises(RuntimeError, match='unknown or closed'):
        program.op(source)
    del program
    with pytest.raises(RuntimeError, match='unknown or closed'):
        fast_ops.lookup_program(handle)


@pytest.mark.gpu
@pytest.mark.parametrize('kind', ['typed', 'indexed', 'pooled'])
def test_program_collection_releases_registry_and_owned_storage(fast, kind):
    import gc
    import weakref
    before = len(fast_ops.PROGRAMS)
    table = torch.randn(16, 64)
    table_ref = weakref.ref(table)
    if kind == 'typed':
        program = fast.typed(dlrm_recipe(), torch.randn(26, 3, 64))
    elif kind == 'indexed':
        program = fast.indexed(gather_recipe())
    else:
        program = fast.pooled([table])
    handle = program.handle
    assert fast_ops.lookup_program(handle) is program
    del program, table
    gc.collect()
    assert len(fast_ops.PROGRAMS) == before
    assert table_ref() is None
    with pytest.raises(RuntimeError, match='unknown or closed'):
        fast_ops.lookup_program(handle)


@pytest.mark.gpu
def test_owner_close_removes_live_program_registrations(fast):
    before = len(fast_ops.PROGRAMS)
    programs = [fast.typed(dlrm_recipe(), torch.randn(26, 3, 64)),
                fast.indexed(gather_recipe()), fast.pooled([torch.randn(16, 64)])]
    assert len(fast_ops.PROGRAMS) == before + 3
    fast.close()
    fast.close()
    assert len(fast_ops.PROGRAMS) == before
    for program in programs:
        with pytest.raises(RuntimeError, match='unknown or closed'):
            fast_ops.lookup_program(program.handle)
