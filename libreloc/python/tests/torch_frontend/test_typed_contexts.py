"""Bounded typed leases across devices, streams, workers and async queues."""
import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import torch
import pyreloc
from reloc_torch import TransferQueue, TransferResources, dispatch, prepare_transfer_group
from reloc_torch.recipe import Cast, Recipe, TensorSpec
from reloc_torch.symbolic import Const, Symbol

pytestmark = pytest.mark.gpu


def artifact(compiler):
    shape = (Symbol('s0'),)
    return compiler.compile(Recipe(TensorSpec(shape, (Const(1),), Const(0), 'float32'),
        (Cast('float16', 'ieee_rne'),),
        TensorSpec(shape, (Const(1),), Const(0), 'float16'), 'h2d'))


def request(compiled, source, device=0, threads=1):
    return dispatch.prepare_typed_transfer(compiled, source, f'cuda:{device}',
        threads=threads, implementation='cpu_reference')


def group(compiled, source, device=0):
    return prepare_transfer_group((request(compiled, source, device),))


def owner(contexts=2, per_device=1, **extra):
    return TransferResources(max_typed_contexts=contexts,
        max_typed_contexts_per_device=per_device, max_typed_background_workers=2,
        max_typed_streams=4, max_typed_retained_bytes=2 << 20,
        max_typed_live_bytes=2 << 20, **extra)


def test_alternating_devices_retain_contexts_and_fresh_values(compiler):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    compiled = artifact(compiler)
    saved = []
    with torch.cuda.device(0), owner() as resources:
        for step, device in enumerate((0, 1, 0, 1, 0, 1)):
            stream = torch.cuda.Stream(device=device)
            source = torch.full((65539 + step,), step + .25)
            with torch.cuda.stream(stream):
                result = dispatch.execute_typed_transfer(request(compiled, source, device, 2),
                    resources=resources, gather_threads=2, n_streams=1, pinning='pinned')
                saved.append((result.tensor, source.half()))
            source.fill_(99)
            assert torch.cuda.current_device() == 0
        stats = resources.stats()['typed']
        assert stats['context_creations'] == 2 and stats['hits'] == 4
        assert stats['background_workers'] == 2 and stats['streams'] == 2
        assert stats['retained_bytes'] == stats['live_bytes'] <= 2 << 20
        assert stats['peak_live_bytes'] <= 2 << 20
        assert {c['device'] for c in stats['contexts']} == {0, 1}
        assert all(c['live_limit'] == 1 << 20 for c in stats['contexts'])
        resources.clear()
        assert resources.stats()['typed']['live_bytes'] == 0
        for output, expected in saved:
            assert torch.equal(output.cpu(), expected)


def test_configuration_reuse_and_lru_eviction(compiler, cuda_device):
    compiled = artifact(compiler)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip('requires two allowed CPUs')
    source = torch.ones(65539)
    original = set(cpus)
    def transfer(resources, cpuset, streams):
        os.sched_setaffinity(0, cpuset)
        return dispatch.execute_typed_transfer(request(compiled, source, cuda_device.index),
            resources=resources, gather_threads=1, n_streams=streams, pinning='pinned').tensor
    try:
        with owner(per_device=2) as resources:
            for cpuset in ({cpus[0]}, {cpus[1]}, {cpus[0]}, {cpus[1]}):
                assert torch.equal(transfer(resources, cpuset, 1).cpu(), source.half())
            stats = resources.stats()['typed']
            assert stats['context_creations'] == 2 and stats['hits'] == 2
            assert {tuple(c['affinity']) for c in stats['contexts']} == {(cpus[0],), (cpus[1],)}
            transfer(resources, {cpus[1]}, 2)
            assert resources.stats()['typed']['evictions'] == 1
            assert resources.stats()['typed']['context_creations'] == 3
    finally:
        os.sched_setaffinity(0, original)


@pytest.mark.parametrize('same_device', [False, True])
def test_async_leases_preserve_independent_order_and_shared_queue_lifetime(compiler, same_device):
    if not same_device and torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    compiled = artifact(compiler)
    devices = (0, 0 if same_device else 1)
    with owner(per_device=2 if same_device else 1) as resources:
        queues = [TransferQueue(f'cuda:{d}', resources=resources, gather_threads=1,
                  max_scratch_bytes=1 << 20) for d in devices]
        # Same-device slots need two simultaneous leases to warm both contexts.
        from reloc_torch.group import _execute_transfer_group
        warm = [_execute_transfer_group(group(compiled, torch.ones(65539), d),
                    resources=resources, gather_threads=1, submit=True,
                    max_scratch_bytes=1 << 20)[0] for d in devices]
        for handle in warm: handle.wait()
        del warm, handle
        producer = torch.cuda.Stream(device=devices[0])
        with torch.cuda.stream(producer):
            torch.cuda._sleep(500_000_000)
            delayed = torch.cuda.Event()
            delayed.record()
        first = queues[0].submit(group(compiled, torch.full((65539,), 3.), devices[0]),
                                 producer_stream=producer)
        second = queues[1].submit(group(compiled, torch.full((65539,), 7.), devices[1]))
        snapshot = resources.stats()['typed']
        assert snapshot['active_contexts'] == snapshot['peak_active_contexts'] == 2
        assert not delayed.query()
        second_output, = second.wait().tensors
        assert not delayed.query(), 'independent context waited for the other producer'
        second.close()
        queues[1].close()
        assert not resources.closed and not queues[0].stats()['owns_resources']
        assert torch.equal(second_output.cpu(), torch.full((65539,), 7., dtype=torch.float16))
        first_output, = first.wait().tensors
        first.close()
        queues[0].close()
        assert torch.equal(first_output.cpu(), torch.full((65539,), 3., dtype=torch.float16))
        assert resources.stats()['typed']['active_contexts'] == 0


def test_pending_admission_timeout_releases_admission_and_allows_retry(compiler, cuda_device):
    compiled = artifact(compiler)
    from reloc_torch.group import _execute_transfer_group
    with owner(contexts=1, typed_acquire_timeout_ms=0) as resources:
        # Warm before delaying a producer, so lazy CUDA allocations do not synchronize it.
        _execute_transfer_group(group(compiled, torch.ones(65539), cuda_device.index),
                                resources=resources, gather_threads=1)
        producer = torch.cuda.Stream(device=cuda_device)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(200_000_000)
            native, _, _ = _execute_transfer_group(group(compiled, torch.ones(65539), cuda_device.index),
                resources=resources, gather_threads=1, submit=True)
        pending = request(compiled, torch.ones(65539), cuda_device.index)
        with pytest.raises(RuntimeError, match='acquire_timeout'):
            dispatch.execute_typed_transfer(pending, resources=resources, gather_threads=1, n_streams=1)
        # Frontend requests remain one-shot, including failed execution attempts.
        assert pending.consumed and resources.stats()['typed']['queued'] == 0
        native.wait()
        pending = request(compiled, torch.ones(65539), cuda_device.index)
        output = dispatch.execute_typed_transfer(pending, resources=resources, gather_threads=1, n_streams=1)
        assert torch.equal(output.tensor.cpu(), torch.ones(65539, dtype=torch.float16))


@pytest.mark.parametrize('cleanup', ['clear', 'close'])
def test_maintenance_drains_every_pending_context_and_releases_sources(compiler, cleanup):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    from reloc_torch.group import _execute_transfer_group
    compiled = artifact(compiler)
    resources = owner()
    pending, outputs, refs = [], [], []
    for d in (0, 1):
        source = torch.full((65539,), d + 1.)
        refs.append(weakref.ref(source))
        handle, tensors, _ = _execute_transfer_group(group(compiled, source, d),
            resources=resources, gather_threads=1, submit=True)
        pending.append(handle)
        outputs.extend(tensors)
    del source, handle
    getattr(resources, cleanup)()
    assert all(handle.query() for handle in pending)
    pending.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
    stats = resources.stats()['typed']
    assert stats['active_contexts'] == stats['live_bytes'] == stats['streams'] == 0
    for i, output in enumerate(outputs):
        assert torch.equal(output.cpu(), torch.full((65539,), i + 1., dtype=torch.float16))
    resources.close()


def test_quarantine_keeps_failed_owners_and_drains_healthy_sibling(cuda_device):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('typed_context_fault_scenario.py'))],
        env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout


def test_blocking_calls_overlap_and_refresh_parameter_bindings(compiler):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from reloc_torch.recipe import BindingParam, Dequantize
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    shape = (Symbol('s0'),)
    spec = lambda dtype: TensorSpec(shape, (Const(1),), Const(0), dtype)
    compiled = compiler.compile(Recipe(spec('int8'),
        (Dequantize('float32', BindingParam('s', 'float32', ()), None, None, 'affine'),
         Cast('float16', 'ieee_rne')), spec('float16'), 'h2d'))
    gate = Barrier(2)
    with owner() as resources, ThreadPoolExecutor(2) as executor:
        def worker(device):
            saved = []
            for step in range(4):
                source = torch.full((262147,), step + device, dtype=torch.int8)
                scale = torch.tensor(.125 * (step + 1))
                prepared = dispatch.prepare_typed_transfer(compiled, source, f'cuda:{device}',
                    parameters={'s': scale}, threads=2, implementation='cpu_reference')
                gate.wait(timeout=10)
                result = dispatch.execute_typed_transfer(prepared, resources=resources,
                    gather_threads=2, n_streams=1, pinning='pinned')
                saved.append((result.tensor, (source.float() * scale).half()))
                source.fill_(99)
                scale.fill_(99)
            return saved
        futures = [executor.submit(worker, device) for device in (0, 1)]
        for future in futures:
            for output, expected in future.result(timeout=30):
                assert torch.equal(output.cpu(), expected)
        stats = resources.stats()['typed']
        assert stats['peak_active_contexts'] == 2
        assert stats['context_creations'] == 2 and stats['hits'] == 6
        assert stats['peak_live_bytes'] <= 2 << 20
