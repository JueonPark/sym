"""Grouped transfers: ownership, completion, changing values and bounded scratch."""
import dataclasses
import gc
from concurrent.futures import ThreadPoolExecutor
import weakref

import pytest
import torch
import pyreloc

from reloc_torch import TransferResources, prepare_transfer_group, execute_transfer_group
from reloc_torch import dispatch, transport
from test_execution_templates import matrix
from test_transport import transpose, pad_recipe, reference


def test_group_rejects_invalid_members_before_cuda():
    for members in ([], [None] * 257):
        with pytest.raises(ValueError):
            prepare_transfer_group(members)
    with pytest.raises(TypeError):
        prepare_transfer_group([object()])
    with pytest.raises(pyreloc.TransferError, match="invalid_group"):
        pyreloc.prepare_dispatch_group([])


def test_native_group_checks_all_views_and_aliases_on_cpu(compiler):
    from reloc_torch.runtime import bind_symbols, bind_plan
    from reloc_torch.transport import _storage_view
    compiled = compiler.compile(transpose())
    src = torch.ones(8, 12)
    dst, other = torch.empty(12, 8), torch.empty(12, 8)
    bound = bind_plan(compiled, bind_symbols(compiled, src))
    def entry(out):
        return (bound, _storage_view(src, 'host', -1), _storage_view(out, 'host', -1), 'h2d')
    group = pyreloc.prepare_dispatch_group([entry(dst), entry(other)])
    assert group.report['logical_transfers'] == 2
    assert not group.consumed
    with pytest.raises(pyreloc.TransferError, match="group_alias"):
        pyreloc.prepare_dispatch_group([entry(dst), entry(dst)])
    with pytest.raises(pyreloc.TransferError, match="group_alias"):
        pyreloc.prepare_dispatch_group([entry(src.view(12, 8))])
    with pytest.raises(pyreloc.TransferError):
        pyreloc.prepare_dispatch_group([entry(dst), entry(torch.empty(2))])
    assert not group.consumed


@pytest.mark.gpu
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
@pytest.mark.parametrize('pinning', ['auto', 'pinned', 'pageable'])
def test_layout_groups_match_individual_with_live_outputs(compiler, cuda_device, direction, dtype, pinning):
    compiled = compiler.compile(transpose(dtype, direction))
    retained = []
    with TransferResources() as owner:
        for step in range(3):
            sources = [(torch.arange(r*c) % 100 + step).to(getattr(torch, dtype)).reshape(r, c)
                       for r, c in ((16, 33), (7, 19), (32, 16))]
            if direction == 'd2h':
                sources = [source.to(cuda_device) for source in sources]
            target = cuda_device if direction == 'h2d' else 'cpu'
            members = [transport.prepare_transfer(compiled, source, target) for source in sources]
            group = prepare_transfer_group(members)
            result = execute_transfer_group(group, resources=owner, pinning=pinning, gather_threads=1)
            assert result.descriptors == group.destinations
            assert result.report['event_waits'] == result.report['event_records'] == 1
            assert result.report['caller_waits'] == 1
            assert result.report['copy_calls'] == result.report['payload_copy_calls'] == 3
            assert result.report['packing_bytes'] == 0
            assert result.report['scratch_peak_bytes'] <= 64 << 20
            for source, out in zip(sources, result.tensors):
                expected = source.t().contiguous().cpu()
                individual = transport.execute_transfer(transport.prepare_transfer(compiled, source, target))
                assert torch.equal(out.cpu(), expected) and torch.equal(individual.cpu(), expected)
                retained.append((out, expected))
            with pytest.raises(RuntimeError, match='already executed'):
                execute_transfer_group(group, resources=owner)
            with pytest.raises(RuntimeError, match='already executed'):
                transport.execute_transfer(members[0])
            refs = [weakref.ref(source) for source in sources]
            del source, sources, members, group, result
            gc.collect()
            assert all(ref() is None for ref in refs)
        assert len({out.data_ptr() for out, _ in retained}) == len(retained)
        assert all(torch.equal(out.cpu(), expected) for out, expected in retained)


@pytest.mark.gpu
@pytest.mark.parametrize('implementation', ['cuda_dequant_relocate', 'cpu_reference', 'cpu_stages_cuda_stages@0'])
def test_typed_groups_share_only_identical_parameter_uploads(matrix, cuda_device, implementation):
    outputs = []
    with TransferResources() as owner:
        for step in range(3):
            sources = [torch.full((16 + i, 32), i + step + 1, dtype=torch.int8) for i in range(4)]
            scales = [torch.full((32,), value + step) for value in (.5, .5, .25, .5)]
            members = [dispatch.prepare_typed_transfer(matrix, source, cuda_device,
                parameters={'scale': scale}, implementation=implementation, threads=1)
                for source, scale in zip(sources, scales)]
            result = execute_transfer_group(prepare_transfer_group(members), resources=owner, gather_threads=1)
            report = result.report
            assert report['event_waits'] == report['caller_waits'] == 1
            if implementation != 'cpu_reference':
                assert report['parameter_uploads'] == report['parameter_reuses'] == 2
                assert report['parameter_upload_bytes'] == 2 * 32 * 4
                assert report['copy_calls'] == 6
            for source, scale, out in zip(sources, scales, result.tensors):
                expected = (source.float() * scale).t().contiguous()
                assert torch.equal(out.cpu(), expected)
                outputs.append((out, expected))
        assert all(torch.equal(out.cpu(), expected) for out, expected in outputs)


@pytest.mark.gpu
def test_mixed_typed_layout_and_directions_with_padding(matrix, compiler, cuda_device):
    q, scale = torch.full((16, 32), -7, dtype=torch.int8), torch.full((32,), .5)
    src = torch.arange(24, dtype=torch.float32).reshape(4, 6).to(cuda_device)
    layout = compiler.compile(pad_recipe(direction='d2h'))
    result = execute_transfer_group(prepare_transfer_group([
        dispatch.prepare_typed_transfer(matrix, q, cuda_device, parameters={'scale': scale}),
        transport.prepare_transfer(layout, src, 'cpu')]))
    assert torch.equal(result.tensors[0].cpu(), (q.float() * scale).t().contiguous())
    assert torch.equal(result.tensors[1], reference('pad', src).cpu())
    assert result.report['event_waits'] == 1


@pytest.mark.gpu
def test_group_preflight_staleness_and_device_checks_launch_nothing(matrix, compiler, cuda_device):
    src, scale = torch.ones(16, 32, dtype=torch.int8), torch.ones(32)
    def member(device=cuda_device):
        return dispatch.prepare_typed_transfer(matrix, src, device, parameters={'scale': scale})
    request = member()
    with pytest.raises(ValueError, match='twice'):
        prepare_transfer_group([request, request])
    group = prepare_transfer_group([request, member()])
    with TransferResources() as owner:
        scale.fill_(2.)
        with pytest.raises(RuntimeError, match='stale'):
            execute_transfer_group(group, resources=owner)
        assert not group.consumed and owner.stats()['typed']['requests'] == 0
        group = prepare_transfer_group([member()])
        src.resize_(8, 64)
        with pytest.raises(RuntimeError, match='stale'):
            execute_transfer_group(group, resources=owner)
        assert owner.stats()['typed']['requests'] == 0
    if torch.cuda.device_count() > 1:
        src.resize_(16, 32)
        with pytest.raises(ValueError, match='same CUDA device'):
            prepare_transfer_group([member(), member(torch.device('cuda', (cuda_device.index+1)%torch.cuda.device_count()))])


@pytest.mark.gpu
def test_partial_group_limit_failure_drains_and_releases_every_owner(matrix, cuda_device):
    owner = TransferResources(max_typed_retained_bytes=4096)
    sources = [torch.ones(16, 32, dtype=torch.int8) for _ in range(3)]
    scale = torch.ones(32)
    members = [dispatch.prepare_typed_transfer(matrix, source, cuda_device,
        parameters={'scale': scale}, implementation='cuda_dequant_relocate', threads=1) for source in sources]
    group = prepare_transfer_group(members)
    # First member needs 512 + 128 host + 128 device bytes. The next source
    # allocation fails after the first payload/kernel has been submitted.
    refs = [weakref.ref(source) for source in sources]
    with pytest.raises(RuntimeError, match='live-byte limit'):
        execute_transfer_group(group, resources=owner, max_scratch_bytes=1000, gather_threads=1)
    assert group.consumed and all(request.consumed for request in members)
    assert group.native.report['copy_calls'] >= 2
    assert owner.stats()['typed']['retained_bytes'] == 0
    assert not owner.stats()['typed']['quarantined']
    del sources, members, group
    gc.collect()
    assert all(ref() is None for ref in refs)
    owner.close()


@pytest.mark.gpu
def test_group_stream_ordering_concurrency_and_smaller_budget(compiler, cuda_device):
    compiled = compiler.compile(transpose(direction='d2h'))
    def run(value):
        with torch.cuda.device(cuda_device), TransferResources() as owner:
            stream = torch.cuda.Stream(device=cuda_device)
            with torch.cuda.stream(stream):
                sources = [torch.empty(64, 96, device=cuda_device) for _ in range(3)]
                torch.cuda._sleep(1_000_000)
                for src in sources:
                    src.fill_(value)
                result = execute_transfer_group(prepare_transfer_group([
                    transport.prepare_transfer(compiled, src, 'cpu') for src in sources]),
                    resources=owner, max_scratch_bytes=100000, gather_threads=1)
            assert all(torch.equal(t, torch.full((96, 64), value)) for t in result.tensors)
            # Enforce a smaller cap even when a previous call retained more.
            small = torch.ones(8, 8, device=cuda_device)
            result2 = execute_transfer_group(prepare_transfer_group([
                transport.prepare_transfer(compiled, small, 'cpu')]), resources=owner,
                max_scratch_bytes=1024, gather_threads=1)
            assert result2.report['scratch_peak_bytes'] <= 1024
            return result.tensors
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(run, [1., 2., 3.]))
    assert len({out.data_ptr() for group in results for out in group}) == 9


@pytest.mark.gpu
@pytest.mark.parametrize('mode', [1, 2, 3, 4])
def test_group_partial_submission_and_completion_faults(mode, cuda_device):
    import os
    from pathlib import Path
    import subprocess
    import sys

    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    assert shim.is_file(), f'build typed_dispatch_faults: {shim}'
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('group_fault_scenario.py')), str(mode)],
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout


@pytest.mark.gpu
def test_concurrent_groups_serialize_safely_on_one_owner(matrix, cuda_device):
    with TransferResources(max_typed_live_bytes=2 << 20) as owner:
        def run(i):
            with torch.cuda.device(cuda_device):
                src, scale = torch.full((64, 96), i + 1, dtype=torch.int8), torch.full((96,), .25)
                members = [dispatch.prepare_typed_transfer(matrix, src, cuda_device,
                    parameters={'scale': scale}, implementation='cuda_dequant_relocate', threads=1) for _ in range(3)]
                result = execute_transfer_group(prepare_transfer_group(members), resources=owner, gather_threads=1)
                assert all(torch.equal(out.cpu(), src.t().float() * .25) for out in result.tensors)
                return result.tensors
        with ThreadPoolExecutor(max_workers=3) as workers:
            outputs = list(workers.map(run, range(6)))
        assert len({out.data_ptr() for group in outputs for out in group}) == 18
        assert owner.stats()['typed']['requests'] == 6
        assert owner.stats()['typed']['peak_live_bytes'] <= 2 << 20


@pytest.mark.gpu
def test_grouped_typed_partitions_and_special_values(compiler, cuda_device):
    import numpy as np
    from test_dispatch import _spec, oracle_quantize
    from reloc_torch.recipe import BindingParam, Quantize, Dequantize, Recipe, Transpose
    from reloc_torch.symbolic import Const, Symbol

    n = Symbol('s0')
    quant = compiler.compile(Recipe(_spec((n, 3), 'float32'),
        (Quantize('int8', BindingParam('s','float32',(Const(3),)), None, 1, 'symmetric_rne'),
         Transpose((1,0))), _spec((3,n), 'int8'), 'h2d'))
    source = torch.randn(32, 3) * 40
    source[0] = torch.tensor([float('nan'), float('inf'), -float('inf')])
    scale = torch.tensor([1., .5, .25])
    probe = dispatch.prepare_typed_transfer(quant, source, cuda_device, parameters={'s':scale})
    expected = np.ascontiguousarray(oracle_quantize(source.numpy(),scale.numpy()).T)
    with TransferResources() as owner:
        for row in probe.capability['eligible']:
            members = [dispatch.prepare_typed_transfer(quant, source.clone(), cuda_device,
                parameters={'s':scale.clone()}, implementation=row['implementation']) for _ in range(2)]
            result = execute_transfer_group(prepare_transfer_group(members), resources=owner)
            for out in result.tensors:
                np.testing.assert_array_equal(out.cpu().numpy(), expected)
            assert result.report['event_waits'] == 1
    dequant = compiler.compile(Recipe(_spec((n,), 'int8'),
        (Dequantize('float32', BindingParam('s','float32',()), None, None, 'affine'),),
        _spec((n,), 'float32'), 'd2h'))
    source = torch.arange(-128, 128, dtype=torch.int8).to(cuda_device)
    probe = dispatch.prepare_typed_transfer(dequant, source, 'cpu', parameters={'s':torch.tensor(.25)})
    with TransferResources() as owner:
        for row in probe.capability['eligible']:
            members = [dispatch.prepare_typed_transfer(dequant, source, 'cpu',
                parameters={'s':torch.tensor(.25)}, implementation=row['implementation']) for _ in range(2)]
            result = execute_transfer_group(prepare_transfer_group(members), resources=owner)
            assert all(torch.equal(out, source.cpu().float()*.25) for out in result.tensors)
            assert result.report['event_waits'] == 1
