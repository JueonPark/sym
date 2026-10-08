"""Actual pending CUDA work, ownership, stream ordering, and bounded prefetch."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import weakref

import pytest
import torch

from reloc_torch import TransferQueue, dispatch, prepare_transfer_group
from reloc_torch.recipe import Cast, Recipe, TensorSpec
from reloc_torch.symbolic import Const, Symbol, dense_strides

pytestmark = pytest.mark.gpu


def artifact(compiler, direction='h2d'):
    n = Symbol('n')
    def spec(dtype):
        return TensorSpec((n,), dense_strides((n,)), Const(0), dtype)
    return compiler.compile(Recipe(spec('float32'), (Cast('float16', 'ieee_rne'),),
                                   spec('float16'), direction))


def group(compiled, source, device='cuda:0'):
    return prepare_transfer_group([dispatch.prepare_typed_transfer(compiled, source, device,
                                   implementation='cpu_reference', threads=2)])


def delayed(stream):
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        event = torch.cuda.Event()
        event.record()
    return event


def test_pending_h2d_consumer_event_and_owned_inputs(compiler, cuda_device):
    compiled = artifact(compiler)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        queue.submit(group(compiled, torch.ones(65536))).close()  # warm allocators
        x = torch.arange(65536, dtype=torch.float32).remainder_(127)
        expected = x.half()
        source_ref = weakref.ref(x)
        before = delayed(producer)
        prepared = group(compiled, x)
        handle = queue.submit(prepared, producer_stream=producer)
        assert not before.query() and not handle.done(), 'submission waited for GPU completion'
        del prepared, x
        gc.collect()
        assert source_ref() is not None
        with torch.cuda.stream(consumer):
            output, = handle.wait_stream()
            actual = output + 1
            handle.release()
        assert torch.equal(actual.cpu(), expected + 1)
        assert handle.done()
        handle.close()
        gc.collect()
        assert source_ref() is None
        assert queue.stats()['held'] == 0
        with pytest.raises(RuntimeError, match='closed'):
            handle.wait()


def test_d2h_waits_for_producer_and_host_transform(compiler, cuda_device):
    compiled = artifact(compiler, 'd2h')
    producer = torch.cuda.Stream(device=cuda_device)
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        x = torch.zeros(65536, device=cuda_device)
        queue.submit(group(compiled, x, 'cpu')).close()
        with torch.cuda.stream(producer):
            before = delayed(producer)
            x.fill_(17.5)
            handle = queue.submit(group(compiled, x, 'cpu'))
        assert not before.query() and not handle.done()
        with pytest.raises(Exception, match='host_completion_required'):
            handle.wait_stream()
        result = handle.wait()
        assert torch.equal(result.tensors[0], torch.full((65536,), 17.5, dtype=torch.float16))
        assert result.report['items'][0]['executed']
        handle.close()


def test_output_budget_cancel_and_close_drain(compiler, cuda_device):
    compiled = artifact(compiler)
    queue = TransferQueue(cuda_device, gather_threads=2, max_output_bytes=512)
    queue.submit(group(compiled, torch.ones(128))).close()
    producer = torch.cuda.Stream()
    event = delayed(producer)
    first = queue.submit(group(compiled, torch.ones(128)), producer_stream=producer)
    # The next submission drains the prior producer before scratch reuse.
    second = queue.submit(group(compiled, torch.full((128,), 2.)))
    assert event.query()
    extra = group(compiled, torch.ones(128))
    with pytest.raises(BufferError):
        queue.submit(extra)
    assert not extra.consumed
    first.cancel()
    queue.submit(extra)
    assert queue.stats()['peak_output_bytes'] == 512
    queue.close()
    assert queue.stats()['held'] == 0
    assert queue.stats()['resources']['retained_bytes'] == 0
    queue.close()
    with pytest.raises(RuntimeError, match='closed'):
        second.wait()


def test_prefetch_retires_previous_consumer_not_later_kernels(compiler, cuda_device):
    compiled = artifact(compiler)
    with TransferQueue(cuda_device, gather_threads=2, max_output_bytes=512) as queue:
        checks = []
        groups = (group(compiled, torch.full((128,), float(i))) for i in range(5))
        with queue.prefetch(groups) as window:
            for i, (weights,) in enumerate(window):
                torch.cuda._sleep(1_000_000)
                checks.append((weights + 2, i + 2))
        assert all(torch.equal(value.cpu(), torch.full((128,), expected, dtype=torch.float16))
                   for value, expected in checks)
        assert queue.stats()['peak_in_flight'] == 2
        assert queue.stats()['peak_output_bytes'] == 512
        assert queue.stats()['held'] == 0
        assert queue.stats()['resources']['context_creations'] == 1


def test_prefetch_early_exit_and_model_failure_drain(compiler, cuda_device):
    compiled = artifact(compiler)
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        with pytest.raises(ValueError, match='model failed'):
            with queue.prefetch(group(compiled, torch.ones(128)) for _ in range(4)) as window:
                output, = next(window)
                output.mul_(2)
                raise ValueError('model failed')
        assert queue.stats()['held'] == 0
        assert queue.stats()['submitted'] == 1


def test_concurrent_submissions_and_same_request_consumption(compiler, cuda_device):
    compiled = artifact(compiler)
    with TransferQueue(cuda_device, gather_threads=2, max_in_flight=4) as queue:
        requests = [group(compiled, torch.full((256,), float(i))) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as threads:
            handles = list(threads.map(queue.submit, requests))
        for i, handle in enumerate(handles):
            assert torch.equal(handle.wait().tensors[0].cpu(), torch.full((256,), float(i), dtype=torch.float16))
            handle.close()
        assert queue.stats()['peak_in_flight'] == 4
        with pytest.raises(RuntimeError, match='already executed'):
            queue.submit(requests[0])


def test_stale_source_metadata_and_fresh_values(compiler, cuda_device):
    compiled = artifact(compiler)
    x = torch.ones(128)
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        for value in (1., 7., -3.):
            x.fill_(value)  # supported only after prior host completion
            handle = queue.submit(group(compiled, x))
            assert torch.equal(handle.wait().tensors[0].cpu(), x.half())
            handle.close()
        prepared = group(compiled, x)
        x.resize_(256)
        with pytest.raises(RuntimeError, match='stale'):
            queue.submit(prepared)
        assert not prepared.consumed


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork guard requires Unix')
def test_inherited_queue_rejects_before_any_lock(compiler, cuda_device):
    compiled = artifact(compiler)
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        handle = queue.submit(group(compiled, torch.ones(128)))
        pid = os.fork()
        if pid == 0:
            try:
                handle.wait()
            except RuntimeError as error:
                os._exit(0 if 'another process' in str(error) else 2)
            os._exit(3)
        assert os.waitpid(pid, 0)[1] == 0
        handle.close()


@pytest.mark.parametrize('mode', [1, 2, 3, 4, 5, 6])
def test_submission_and_completion_failure_ownership(mode, cuda_device):
    from pathlib import Path
    import subprocess
    import sys
    import pyreloc
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('async_fault_scenario.py')), str(mode)],
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout


def test_two_consumer_streams_and_output_retirement(compiler, cuda_device):
    compiled = artifact(compiler)
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        handle = queue.submit(group(compiled, torch.full((65536,), 3.)))
        streams = [torch.cuda.Stream() for _ in range(2)]
        actual = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                output, = handle.wait_stream()
                delayed(stream)
                actual.append(output + i)
        ref = weakref.ref(output)
        del output
        assert ref() is not None
        handle.release()
        with pytest.raises(RuntimeError, match='released'):
            handle.wait_stream()
        handle.close()
        assert all(s.query() for s in streams)
        gc.collect()
        assert ref() is None
        for i, result in enumerate(actual):
            assert torch.equal(result.cpu(), torch.full((65536,), 3. + i, dtype=torch.float16))


@pytest.mark.parametrize('cleanup', ['handle', 'resources'])
def test_native_raii_and_owner_close_drain_pending_work(compiler, cuda_device, cleanup):
    from reloc_torch import TransferResources
    from reloc_torch.group import _execute_transfer_group
    compiled = artifact(compiler)
    stream = torch.cuda.Stream()
    with TransferResources(max_typed_background_workers=1) as resources:
        # Warm allocation/worker setup before the deliberately pending event.
        _execute_transfer_group(group(compiled, torch.ones(65536)), resources=resources,
                                gather_threads=2)
        source = torch.full((65536,), 11.)
        ref = weakref.ref(source)
        with torch.cuda.stream(stream):
            before = delayed(stream)
            native, outputs, _ = _execute_transfer_group(group(compiled, source),
                resources=resources, gather_threads=2, submit=True)
        assert not before.query()
        del source
        gc.collect()
        assert ref() is not None
        if cleanup == 'resources':
            resources.close()
            assert native.query()
        del native  # destructor also drains when public resources remain open
        assert before.query()
        gc.collect()
        assert ref() is None
        assert torch.equal(outputs[0].cpu(), torch.full((65536,), 11., dtype=torch.float16))


def test_host_wait_registers_current_gpu_consumer(compiler, cuda_device):
    compiled = artifact(compiler)
    consumer = torch.cuda.Stream()
    with TransferQueue(cuda_device, gather_threads=2) as queue:
        handle = queue.submit(group(compiled, torch.ones(65536)))
        with torch.cuda.stream(consumer):
            output, = handle.wait().tensors
            event = delayed(consumer)
            actual = output + 2
        del output
        assert not event.query()
        handle.close()
        assert consumer.query()
        assert torch.equal(actual.cpu(), torch.full((65536,), 3., dtype=torch.float16))
