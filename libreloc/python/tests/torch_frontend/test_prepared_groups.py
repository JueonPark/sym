"""Reusable group bindings keep storage validation and report lifetimes."""
import pytest
import torch
import pyreloc

from reloc_torch.fast import FastTransfers, _load_native_views
from reloc_torch.runtime import bind_plan, bind_symbols
from test_transport import transpose
from test_fast import dlrm_recipe


def member(plan, source, output, direction='h2d'):
    def device(t):
        return t.device.index if t.is_cuda else -1
    return (plan, direction, str(source.dtype).removeprefix('torch.'),
            str(output.dtype).removeprefix('torch.'), list(source.shape), list(source.stride()), device(source),
            list(output.shape), list(output.stride()), device(output))


@pytest.fixture
def cpu_binder(compiler):
    native = _load_native_views()
    if native is None:
        pytest.skip('native Torch metadata bridge not built')
    source, output = torch.ones(8, 12), torch.empty(12, 8)
    compiled = compiler.compile(transpose())
    bound = bind_plan(compiled, bind_symbols(compiled, source))
    binder = native.PreparedTensorGroup([member(bound, source, output)])
    return binder, source, output


def test_native_binder_reuses_structure_with_fresh_pointer_and_offset(cpu_binder):
    binder, source, output = cpu_binder
    group = binder.prepare([source], [output])
    shifted = torch.arange(97, dtype=torch.float32)[1:].view(8, 12)
    replacement = torch.empty(100, dtype=torch.float32)[4:].view(12, 8)
    binder.rebind(group, [shifted], [replacement])
    assert group.rebind_count == 1
    assert binder.stats() == dict(group_constructions=1, rebinds=1, view_descriptors=2)
    assert group.report['logical_transfers'] == 1


@pytest.mark.parametrize('change', ['shape', 'stride', 'source_dtype', 'output_dtype', 'source_grad', 'output_grad', 'negative_view', 'subclass'])
def test_native_metadata_guard_is_atomic(cpu_binder, change):
    binder, source, output = cpu_binder
    group = binder.prepare([source], [output])
    bad_source, bad_output = source, output
    if change == 'shape':bad_source = source.reshape(12, 8)
    if change == 'stride':bad_source = torch.empty_strided((8, 12), (1, 8))
    if change == 'source_dtype':bad_source = source.to(torch.int32)
    if change == 'output_dtype':bad_output = output.to(torch.int32)
    if change == 'source_grad':bad_source = source.clone().requires_grad_()
    if change == 'output_grad':bad_output = output.clone().requires_grad_()
    if change == 'negative_view':bad_source = torch._neg_view(source)
    if change == 'subclass':
        class Derived(torch.Tensor):pass
        bad_source = source.as_subclass(Derived)
    before = group.report
    with pytest.raises((ValueError, TypeError, pyreloc.TransferError)):
        binder.rebind(group, [bad_source], [bad_output])
    assert group.rebind_count == 0 and group.report == before
    binder.rebind(group, [source + 1], [torch.empty_like(output)])
    assert group.rebind_count == 1


def test_native_binder_rejects_wrong_handle_and_aliases(cpu_binder, compiler):
    binder, source, output = cpu_binder
    group = binder.prepare([source], [output])
    compiled = compiler.compile(transpose())
    bound = bind_plan(compiled, bind_symbols(compiled, source))
    other = _load_native_views().PreparedTensorGroup([member(bound, source, output)])
    with pytest.raises((ValueError, pyreloc.TransferError), match='different|identity|binder'):
        other.rebind(group, [source], [output])
    with pytest.raises(pyreloc.TransferError, match='group_alias'):
        binder.rebind(group, [source], [source.view(12, 8)])
    assert group.rebind_count == 0
    two = _load_native_views().PreparedTensorGroup([member(bound, source, output)] * 2)
    group2 = two.prepare([source, source], [output, torch.empty_like(output)])
    with pytest.raises(pyreloc.TransferError, match='group_alias'):
        two.rebind(group2, [source, source], [output, output])
    assert group2.rebind_count == 0


def test_compact_rebind_checks_capacity_overflow_and_count(cpu_binder):
    binder, source, output = cpu_binder
    group = binder.prepare([source], [output])
    good = (source.data_ptr(), source.nbytes, 0, output.data_ptr(), output.nbytes, 0)
    for bad in [[], [good, good], [(good[0], 4, 0, *good[3:])],
                [(good[0], good[1], (1 << 64)-1, *good[3:])],
                [(0, good[1], 0, *good[3:])]]:
        with pytest.raises((ValueError, pyreloc.TransferError)):
            pyreloc.rebind_dispatch_group(group, bad)
        assert group.rebind_count == 0
    pyreloc.rebind_dispatch_group(group, [good])
    assert group.rebind_count == 1


@pytest.mark.gpu
def test_native_report_snapshot_survives_rebind_and_counts_once(compiler, cuda_device):
    source = torch.randn(26, 7, 64)
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        program = owner.typed(dlrm_recipe(), source)
        output = program.allocate()
        binder = _load_native_views().PreparedTensorGroup([member(program.template, source, output)])
        group = binder.prepare([source], [output])
        counters = pyreloc.GroupCounters()
        completion = pyreloc.submit_dispatch_group(group, resources=owner.native,
            owners=((source,), (output,)), caller_stream=torch.cuda.current_stream().cuda_stream,
            gather_threads=4)
        completion.wait()
        expected = group.report
        snapshot = counters.capture(group)
        assert counters.stats()['groups'] == 1
        assert counters.stats()['logical_transfers'] == 1
        assert counters.stats()['payload_bytes'] == expected['items'][0]['payload_bytes_transferred']
        assert counters.stats()['report_materializations'] == 0
        output2 = program.allocate()
        binder.rebind(group, [source.neg()], [output2])
        assert snapshot.report == expected
        assert counters.stats()['report_materializations'] == 1
        assert not group.report['items'][0]['executed']
        assert torch.equal(output, source.transpose(0, 1).contiguous().to(cuda_device, dtype=torch.float16))


@pytest.mark.gpu
def test_native_pending_group_cannot_be_rebound_or_accounted(compiler, cuda_device):
    source = torch.randn(26, 7, 64)
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        program = owner.typed(dlrm_recipe(), source)
        output = program.allocate()
        binder = _load_native_views().PreparedTensorGroup([member(program.template, source, output)])
        group = binder.prepare([source], [output])
        completion = pyreloc.submit_dispatch_group(group, resources=owner.native,
            owners=((source,), (output,)), gather_threads=4)
        with pytest.raises(pyreloc.TransferError, match='group_pending'):
            binder.rebind(group, [source], [output])
        with pytest.raises(ValueError, match='completed'):
            pyreloc.GroupCounters().capture(group)
        completion.wait()
        binder.rebind(group, [source], [output])
        assert completion.query()  # The old completed handle must not touch a reused slot.
        completion.wait()
        assert group.rebind_count == 1 and not group.report['items'][0]['executed']


@pytest.mark.gpu
def test_native_failed_group_is_never_recycled(compiler, cuda_device):
    source = torch.randn(26, 7, 64)
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        program = owner.typed(dlrm_recipe(), source)
        output = program.allocate()
        binder = _load_native_views().PreparedTensorGroup([member(program.template, source, output)])
        group = binder.prepare([source], [output])
        with pytest.raises(pyreloc.TransferError):
            pyreloc.submit_dispatch_group(group, resources=owner.native,
                owners=((source,), (output,)), gather_threads=4, max_scratch_bytes=4)
        assert group.failed
        with pytest.raises(pyreloc.TransferError, match='group_failed'):
            binder.rebind(group, [source], [output])
        with pytest.raises(ValueError, match='completed'):
            pyreloc.GroupCounters().capture(group)


def test_rebind_preserves_existing_unaligned_host_storage_contract(cpu_binder):
    binder, source, output = cpu_binder
    # torch.frombuffer can expose valid storage at a byte-shifted address;
    # the existing full view validator admits it. Rebinding must agree.
    shifted = torch.frombuffer(bytearray(source.nbytes + 1), dtype=source.dtype,
                               count=source.numel(), offset=1).view_as(source)
    group = binder.prepare([shifted], [output])
    binder.rebind(group, [shifted], [torch.empty_like(output)])
    assert group.rebind_count == 1
