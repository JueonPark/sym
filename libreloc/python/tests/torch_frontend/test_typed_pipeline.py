"""Typed transform/H2D ring qualification on real CUDA, including partial failure."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import torch
import pyreloc

from reloc_torch import TransferResources, dispatch
from reloc_torch.recipe import BindingParam, Cast, Dequantize, Fill, Pad, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Add, Const, Symbol, dense_strides


def spec(shape, dtype):
    shape = tuple(Const(n) if isinstance(n, int) else n for n in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


@pytest.mark.gpu
@pytest.mark.parametrize('buffers', [1, 2, 4])
def test_chunked_corpus_matches_exact_oracle_bytes(buffers, cuda_device):
    from reloc_torch import CompiledRecipe
    from test_typed_conformance_gpu import CORPUS, FIXTURES, array

    chunked = 0
    with TransferResources() as owner:
        for name in FIXTURES:
            meta = json.loads((CORPUS / f'{name}.json').read_text())
            compiled = CompiledRecipe.load(CORPUS / f'{name}.reloc')
            params = {n: torch.from_numpy(array(p).copy()) for n, p in meta['parameters'].items()}
            for binding in meta['bindings']:
                source = torch.from_numpy(array(binding['source']).copy())
                expected = array(binding['expected']).tobytes()
                probe = dispatch.prepare_typed_transfer(compiled, source, cuda_device, parameters=params)
                for row in probe.capability['eligible']:
                    label = row['implementation']
                    if label != 'cpu_reference' and not label.startswith('cpu_stages_cuda_stages@'):
                        continue
                    request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                        parameters=params, implementation=label)
                    out = dispatch.execute_typed_transfer(request, resources=owner,
                        n_buffers=buffers, chunk_size=1, pinning='pinned')
                    assert out.tensor.cpu().numpy().tobytes() == expected, (name, label, buffers)
                    chunked += out.report['host_pipeline'] == 'chunked'
                    assert out.report['host_buffers'] <= buffers
    assert chunked > 0


@pytest.mark.gpu
@pytest.mark.parametrize('pinning', ['pinned', 'pageable', 'auto'])
@pytest.mark.parametrize('implementation', ['cpu_reference', 'cpu_stages_cuda_stages@1'])
def test_changing_shapes_padding_and_cpu_gpu_split(compiler, cuda_device, pinning, implementation):
    a, b = Symbol('s0'), Symbol('s1')
    compiled = compiler.compile(Recipe(spec((a, b, 33), 'float32'),
        (Transpose((1, 0, 2)), Cast('float16', 'ieee_rne'),
         Pad(0, Const(3), Const(4), Fill('float16', 0x3c00)), Cast('float32', 'exact')),
        spec((Add(Const(7), b), a, 33), 'float32'), 'h2d'))
    saved = []
    with TransferResources(max_typed_live_bytes=2 << 20) as owner:
        for step in range(3):
            source = torch.randn(23 + step, 17 + step, 33)
            expected = torch.cat((torch.ones(3, 23 + step, 33),
                source.transpose(0, 1).half().float(), torch.ones(4, 23 + step, 33)))
            for buffers, pipeline in ((1, True), (2, True), (4, True), (2, False)):
                request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                    implementation=implementation)
                out = dispatch.execute_typed_transfer(request, resources=owner,
                    n_buffers=buffers, chunk_size=2 * (23 + step) * 33 * 2,
                    pipeline=pipeline, pinning=pinning, gather_threads=3)
                assert torch.equal(out.tensor.cpu(), expected)
                assert out.report['host_pipeline'] == ('chunked' if pipeline else 'whole_disabled')
                saved.append((out.tensor, expected.clone()))
            source.fill_(17.)
        assert all(torch.equal(out.cpu(), expected) for out, expected in saved)
        assert len({out.data_ptr() for out, _ in saved}) == len(saved)


@pytest.mark.gpu
@pytest.mark.parametrize('implementation', ['cpu_reference', 'cpu_stages_cuda_stages@2'])
def test_chunked_channel_parameters_refresh_with_shapes_and_values(compiler, cuda_device, implementation):
    a, b = Symbol('s0'), Symbol('s1')
    compiled = compiler.compile(Recipe(spec((a, b), 'int8'),
        (Dequantize('float32', BindingParam('scale', 'float32', (b,)), None, 1, 'affine'),
         Transpose((1, 0)), Cast('float16', 'ieee_rne'), Cast('float32', 'exact')),
        spec((b, a), 'float32'), 'h2d'))
    saved = []
    with TransferResources() as owner:
        for step in range(3):
            source = torch.randint(-127, 128, (23 + step, 13 + step), dtype=torch.int8)
            scale = torch.arange(1, 14 + step, dtype=torch.float32) * .25
            for _ in range(2):
                expected = (source.float() * scale).t().half().float().contiguous()
                request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                    parameters={'scale': scale}, implementation=implementation)
                out = dispatch.execute_typed_transfer(request, resources=owner,
                    n_buffers=2, chunk_size=1, pinning='pinned', gather_threads=3)
                assert out.report['host_pipeline'] == 'chunked'
                assert torch.equal(out.tensor.cpu(), expected)
                saved.append((out.tensor, expected))
                source.neg_()
                scale.mul_(.5)
        assert all(torch.equal(out.cpu(), expected) for out, expected in saved)


@pytest.mark.gpu
def test_ring_is_retained_bounded_and_orders_after_current_stream(compiler, cuda_device):
    n = Symbol('s0')
    compiled = compiler.compile(Recipe(spec((n,), 'float32'),
        (Cast('float16', 'ieee_rne'),), spec((n,), 'float16'), 'h2d'))
    stream = torch.cuda.Stream(device=cuda_device)
    with TransferResources(max_typed_retained_bytes=32768, max_typed_live_bytes=32768) as owner:
        for step in range(3):
            source = torch.arange(65539 + step, dtype=torch.float32).remainder_(211)
            request = dispatch.prepare_typed_transfer(compiled, source, cuda_device, policy='original_cpu')
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000)
                producer = torch.cuda.Event()
                producer.record()
                out = dispatch.execute_typed_transfer(request, resources=owner,
                    n_buffers=2, chunk_size=16384, pinning='pinned')
            assert producer.query(), 'private copies did not wait for the caller stream'
            assert torch.equal(out.tensor.cpu(), source.half())
            assert out.report['host_chunks'] > 2 and out.report['host_buffers'] == 2
            assert owner.stats()['typed']['host_allocations'] == 2
            assert owner.stats()['typed']['peak_live_bytes'] == 32768
            ref = weakref.ref(source)
            del source, request
            gc.collect()
            assert ref() is None
        source = torch.ones(65539)
        request = dispatch.prepare_typed_transfer(compiled, source, cuda_device, policy='original_cpu')
        with pytest.raises(RuntimeError, match='live-byte limit'):
            dispatch.execute_typed_transfer(request, resources=owner, pipeline=False)
        assert not owner.stats()['typed']['quarantined']
        assert owner.stats()['typed']['retained_bytes'] == 0


@pytest.mark.gpu
@pytest.mark.parametrize('mode', [1, 2, 3, 4])
@pytest.mark.parametrize('indexed', [False, True])
def test_pipeline_failure_drains_or_quarantines_every_owner(mode, indexed, cuda_device):
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    assert shim.is_file()
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable,
        str(Path(__file__).with_name('typed_pipeline_fault_scenario.py')), str(mode), str(int(indexed))],
        env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout


@pytest.mark.gpu
def test_pipeline_options_fail_before_consumption(compiler, cuda_device):
    compiled = compiler.compile(Recipe(spec((64,), 'float32'),
        (Cast('float16', 'ieee_rne'),), spec((64,), 'float16'), 'h2d'))
    request = dispatch.prepare_typed_transfer(compiled, torch.ones(64), cuda_device)
    for options in ({'chunk_size': 0}, {'chunk_size': -1}, {'chunk_size': 1 << 64},
                    {'chunk_size': True}, {'pipeline': 'yes'}):
        with pytest.raises((ValueError, TypeError)):
            dispatch.execute_typed_transfer(request, **options)
        assert not request.consumed
