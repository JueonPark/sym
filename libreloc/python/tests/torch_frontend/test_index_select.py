"""CPU row indirection and its fused torch.compile H2D region."""
import pytest
import torch
import pyreloc

from reloc_torch import RelocBackend, compat, import_graph
from reloc_torch.index_select import bind_index_select, prepare_index_select_transfer
from reloc_torch.transport import _storage_view


SPELLINGS = [
    lambda x, i: torch.index_select(x, 0, i),
    lambda x, i: torch.index_select(input=x, dim=0, index=i),
    lambda x, i: x.index_select(0, i),
    lambda x, i: x.index_select(dim=-2, index=i),
    lambda x, i: torch.ops.aten.index_select.default(x, 0, i),
    lambda x, i: torch.ops.aten.index_select.default(self=x, dim=0, index=i),
]


def transfer(x, dtype=torch.float16):
    return x.to(device='cuda:0', dtype=dtype)


@pytest.mark.parametrize('select', SPELLINGS)
def test_import_spellings(select, compiler):
    x, index = torch.randn(8, 5), torch.tensor([7, 2, 7])
    gm = torch.fx.symbolic_trace(lambda x, i: transfer(select(x, i)))
    before = str(gm.graph)
    report = import_graph(gm, (x, index))
    assert not report.exclusions
    candidate, = report.candidates
    assert candidate.index == 'i'
    assert len(candidate.members) == 2
    compiled = compiler.compile(candidate.recipe)
    assert bind_index_select(compiled, x, index) == {}
    backend = RelocBackend(compiler=compiler)
    try:
        rewritten = backend(gm, (x, index))
        assert backend.stats()['replaced_regions'] == 1
        assert 'index_select_transfer' in str(rewritten.rewritten.graph)
        assert str(gm.graph) == before
    finally:
        backend.close()


@pytest.mark.parametrize('source_dtype,dest_dtype', [
    (torch.float32, torch.float16), (torch.float16, torch.float32),
    (torch.float32, torch.float32), (torch.float16, torch.float16),
    (torch.int8, torch.int8),
])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('shape', [(7,), (7, 129), (7, 3, 5)])
def test_native_row_order_repeats_casts_and_more_output_rows(source_dtype, dest_dtype, index_dtype, shape):
    x = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(source_dtype)
    index = torch.tensor([6, 0, 3, 6, 1, 5, 0, 4, 2], dtype=index_dtype)
    expected = x.index_select(0, index).to(dest_dtype)
    output = torch.empty_like(expected)
    source_view = _storage_view(x, 'host', -1)
    program = pyreloc.prepare_index_select_program(
        source_view, index.numpy().tobytes(), index.element_size(), str(dest_dtype).removeprefix('torch.'))
    capability = pyreloc.query_capability(program, 'h2d', 'cuda')
    assert [row['implementation'] for row in capability['eligible']] == ['cpu_reference']
    request = pyreloc.prepare_dispatch(program, source_view, _storage_view(output, 'host', -1), 'h2d')
    report = pyreloc.execute_dispatch(request, gather_threads=3)
    assert torch.equal(output, expected)
    assert report['source_bytes'] == x.numel() * x.element_size()
    assert report['payload_bytes_transferred'] == output.numel() * output.element_size()
    with pytest.raises(pyreloc.TransferError, match='already_executed'):
        pyreloc.execute_dispatch(request)


def test_native_cast_rounding_edges():
    x = torch.tensor([0., -0., 1., 1.00048828125, 65504., 65520.,
                      2**-24, 2**-25, float('inf'), -float('inf'), float('nan')]).reshape(1, -1)
    index = torch.tensor([0, 0])
    out = torch.empty(2, x.numel(), dtype=torch.float16)
    view = _storage_view(x, 'host', -1)
    program = pyreloc.prepare_index_select_program(view, index.numpy().tobytes(), 8, 'float16')
    request = pyreloc.prepare_dispatch(program, view, _storage_view(out, 'host', -1), 'h2d')
    pyreloc.execute_dispatch(request)
    expected = x.index_select(0, index).half()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize('indices', [[-1], [8], []])
def test_native_bounds_checked_before_launch(indices):
    source = torch.zeros(8, 4)
    index = torch.tensor(indices, dtype=torch.int64)
    with pytest.raises(pyreloc.TransferError, match='index_out_of_range|empty_tensor'):
        pyreloc.prepare_index_select_program(_storage_view(source, 'host', -1), index.numpy().tobytes(), 8, 'float32')


def test_native_source_shape_and_capacity_rechecked():
    source = torch.zeros(8, 4)
    index = torch.tensor([7, 0])
    program = pyreloc.prepare_index_select_program(_storage_view(source, 'host', -1), index.numpy().tobytes(), 8, 'float32')
    output = torch.empty(2, 4)
    with pytest.raises(pyreloc.TransferError, match='plan_mismatch'):
        pyreloc.prepare_dispatch(program, _storage_view(source.reshape(4, 8), 'host', -1),
                                 _storage_view(output, 'host', -1), 'h2d')
    bad = pyreloc.BufferView(base=source.data_ptr(), capacity_bytes=4, offset_bytes=0, extents=[8, 4],
                            strides=[4, 1], element_size=4, kind='host')
    with pytest.raises(pyreloc.TransferError, match='insufficient_capacity'):
        pyreloc.prepare_index_select_program(bad, index.numpy().tobytes(), 8, 'float32')


def test_native_output_size_overflow_is_checked_without_reading_source():
    source = pyreloc.BufferView(base=1, capacity_bytes=2**62, offset_bytes=0,
        extents=[1, 2**60], strides=[2**60, 1], element_size=4, kind='host')
    with pytest.raises(pyreloc.TransferError, match='integer_overflow'):
        pyreloc.prepare_index_select_program(source, torch.zeros(3, dtype=torch.int64).numpy().tobytes(), 8, 'float32')


def test_empty_indices_and_standalone_selection_keep_torch():
    x, index = torch.ones(8, 4), torch.empty(0, dtype=torch.int64)
    report = import_graph(torch.fx.symbolic_trace(lambda x, i: transfer(x.index_select(0, i))), (x, index))
    assert not report.candidates
    assert 'empty_tensor' in {e.reason for e in report.exclusions}
    assert not import_graph(torch.fx.symbolic_trace(lambda x, i: x.index_select(0, i)), (x, index)).candidates


def test_direct_negative_view_source_guard(compiler):
    from reloc_torch.artifact import UnsupportedRecipe

    x, index = torch.ones(8, 4), torch.tensor([0, 1])
    candidate, = import_graph(torch.fx.symbolic_trace(lambda x, i: transfer(x.index_select(0, i))), (x, index)).candidates
    artifact = compiler.compile(candidate.recipe)
    with pytest.raises(UnsupportedRecipe, match='dense CPU source'):
        bind_index_select(artifact, torch._neg_view(x), index)


@pytest.mark.parametrize('function,reason', [
    (lambda x, i: transfer(x.index_select(1, i)), 'unsupported_index_select_dim'),
    (lambda x, i: transfer(x.t().index_select(0, i)), 'unsupported_index_select'),
    (lambda x, i: transfer(x.index_select(0, i).flatten()), 'unsupported_index_select'),
    (lambda x, i: transfer(torch.index_select(x, 0, i, out=x)), 'unrecognized_fx_target'),
    (lambda x, i: (transfer(x.index_select(0, i)), x.index_select(0, i)), None),
])
def test_unsupported_scopes(function, reason):
    # Separate selection calls are independent, not an escaping intermediate.
    report = import_graph(torch.fx.symbolic_trace(function), (torch.ones(8, 4), torch.tensor([0, 1])))
    if reason is None:
        assert report.candidates
    else:
        assert not report.candidates
        assert reason in {e.reason for e in report.exclusions}


def test_escaping_selection_and_index_mutation_rejected():
    def escaping(x, index):
        selected = x.index_select(0, index)
        return selected, transfer(selected)
    def mutated(x, index):
        selected = x.index_select(0, index)
        index.add_(1)
        return transfer(selected)
    for fn, reason in [(escaping, 'escaping_intermediate'), (mutated, 'unknown_side_effect')]:
        report = import_graph(torch.fx.symbolic_trace(fn), (torch.ones(8, 4), torch.tensor([0, 1])))
        assert not report.candidates
        assert reason in {e.reason for e in report.exclusions}


def test_symbolic_rows_and_width_bind_from_correct_operands(compiler):
    fn = lambda x, i: transfer(x.index_select(0, i))
    gm = compat.symbolic_capture(fn, torch.ones(8, 5), torch.tensor([0, 1, 0]))
    candidate, = import_graph(gm).candidates
    compiled = compiler.compile(candidate.recipe)
    for rows, width, selected in [(8, 5, 3), (12, 7, 19), (3, 11, 2)]:
        x, index = torch.ones(rows, width), torch.arange(selected) % rows
        bindings = bind_index_select(compiled, x, index)
        assert tuple(d.evaluate(bindings) for d in compiled.logical_destination.shape) == (selected, width)


@pytest.fixture
def cuda():
    if not pyreloc.cuda_enabled or not torch.cuda.is_available():
        pytest.skip('CUDA runtime unavailable')


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_compile_dynamic_and_changing_indices(cuda, compiler, dtype):
    fn = lambda x, i: transfer(torch.index_select(x, 0, i), dtype)
    backend = RelocBackend(compiler=compiler)
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    try:
        for rows, width, count in [(8, 5, 3), (12, 7, 19), (8, 5, 3)]:
            x = torch.randn(rows, width)
            # A strided vector and repeated rows are both legal indices.
            index = (torch.arange(count * 2) % rows)[::2]
            actual = compiled(x, index)
            expected = fn(x, index)
            assert torch.equal(actual, expected)
            assert actual.data_ptr() != expected.data_ptr()
        stats = backend.stats()
        assert stats['runtime_executions'] == 3
        assert not stats['fallbacks']
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize('numpy_write', [False, True])
def test_prepared_index_mutation_rejected(cuda, compiler, numpy_write):
    x, index = torch.randn(8, 5), torch.tensor([0, 1, 0])
    candidate, = import_graph(torch.fx.symbolic_trace(lambda x, i: transfer(x.index_select(0, i))), (x, index)).candidates
    artifact = compiler.compile(candidate.recipe)
    request = prepare_index_select_transfer(artifact, x, index, 'cuda:0')
    if numpy_write:
        index.numpy()[0] = 3  # No Tensor version bump: recheck actual values.
    else:
        index[0] = 3
    with pytest.raises(RuntimeError, match='indices changed'):
        request.recheck()


@pytest.mark.gpu
def test_invalid_index_preserves_torch_error_and_does_not_launch(cuda, compiler):
    x, index = torch.randn(8, 5), torch.tensor([0, 1, 0])
    fn = lambda x, i: transfer(x.index_select(0, i))
    backend = RelocBackend(compiler=compiler)
    try:
        compiled = backend(torch.fx.symbolic_trace(fn), (x, index))
        with pytest.raises((IndexError, RuntimeError), match='out of|bounds'):
            compiled(x, torch.tensor([0, -1, 0]))
        assert backend.stats().get('runtime_executions', 0) == 0
    finally:
        backend.close()


@pytest.mark.gpu
def test_index_select_gradients_use_original_graph(cuda, compiler):
    x = torch.randn(8, 5, requires_grad=True)
    index = torch.tensor([0, 1, 0])
    fn = lambda x, i: transfer(x.index_select(0, i), torch.float32)
    backend = RelocBackend(compiler=compiler)
    try:
        compiled = backend(torch.fx.symbolic_trace(fn), (x, index))
        compiled(x, index).sum().backward()
        expected = torch.zeros_like(x)
        expected[0] = 2
        expected[1] = 1
        assert torch.equal(x.grad, expected)
        assert backend.stats()['replaced_regions'] == 0
    finally:
        backend.close()


@pytest.mark.gpu
def test_negative_views_preserve_values(cuda, compiler):
    x, index = torch.randn(8, 5), torch.tensor([0, 1, 0])
    fn = lambda x, i: transfer(x.index_select(0, i))
    backend = RelocBackend(compiler=compiler)
    try:
        compiled = backend(torch.fx.symbolic_trace(fn), (x, index))
        negative = torch._neg_view(x)
        assert torch.equal(compiled(negative, index), fn(negative, index))
        indices = torch._neg_view(-index)
        assert torch.equal(compiled(x, indices), fn(x, indices))
        # Torch's Negative dispatch key resolves these views before entering
        # the custom op. Direct bridge calls are guarded separately above.
        assert backend.stats()['runtime_executions'] == 2
        assert not backend.stats()['fallbacks']
    finally:
        backend.close()
