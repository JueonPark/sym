"""Integrated CUDA resource qualification; longer runs are opt-in via the environment."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pyreloc
import pytest
import torch

from reloc_torch import AUTO, RelocBackend, TransferResources
from reloc_torch.runtime import TransportAdapter
from test_transport import transpose


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not pyreloc.cuda_enabled or not torch.cuda.is_available(),
    reason='requires a CUDA runtime and GPU')]
ROUNDS = int(os.environ.get('SYM_RESOURCE_STRESS_ROUNDS', '12'))
assert ROUNDS > 0


@pytest.fixture(autouse=True)
def bound_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def values(shape, seed):
    return (torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape).remainder_(251).add_(seed))


def assert_bounds(stats, contexts, staging, workers, buffers, retained=None):
    assert stats['contexts'] <= contexts
    assert stats['allocated_staging_bytes'] + stats['reserved_staging_bytes'] <= staging
    retained = staging if retained is None else retained
    assert stats['retained_allocated_bytes'] + stats['retained_reserved_bytes'] <= retained
    assert stats['background_workers'] + stats['reserved_workers'] <= workers
    assert stats['streams'] <= contexts * 2
    assert stats['outstanding_events'] <= contexts * buffers
    assert stats['quarantined'] == stats['quarantine_bytes'] == stats['failures'] == 0
    assert all(device['contexts'] <= 2 for device in stats['devices'])


def assert_drained(stats):
    for name in ['contexts', 'allocated_staging_bytes', 'reserved_staging_bytes',
                 'background_workers', 'reserved_workers', 'streams', 'outstanding_events']:
        assert stats[name] == 0, (name, stats)
    for created, retired in [('staging_allocations', 'staging_frees'),
                             ('stream_creations', 'stream_destructions'),
                             ('worker_creations', 'worker_joins'),
                             ('event_creations', 'event_retirements')]:
        assert stats[created] == stats[retired]


@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('policy', ['default', 'auto', 'borrowed'])
def test_compiled_streams_immediate_consumers_and_retained_outputs(compiler, cuda_device,
                                                                 direction, policy):
    owner = TransferResources(
        max_contexts=2, max_retained_bytes=32 << 20, max_live_staging_bytes=48 << 20,
        max_background_workers=4) if policy == 'borrowed' else AUTO
    options = {} if policy == 'default' else {'transfer_resources': owner}
    backend = RelocBackend(compiler=compiler, **options, cache_capacity=1,
                           transfer_options=dict(n_buffers=4, n_streams=2, gather_threads=3))
    destination = cuda_device if direction == 'h2d' else 'cpu'
    streams = [torch.cuda.default_stream(cuda_device), torch.cuda.Stream(device=cuda_device),
               torch.cuda.Stream(device=cuda_device)]

    def transfer(x):
        return x.t().contiguous().to(destination)

    def padded_transfer(x):
        return torch.nn.functional.pad(x.t().contiguous(), (1, 2, 0, 1)).to(destination)

    functions = [torch.compile(fn, backend=backend, dynamic=True)
                 for fn in (transfer, padded_transfer)]
    held = []
    calls = 0
    try:
        for iteration in range(ROUNDS):
            for index, shape in enumerate([(65, 67), (1024, 4096), (1009, 4093)]):
                expected_source = values(shape, iteration)
                stream = streams[(iteration + index) % len(streams)]
                with torch.cuda.device(cuda_device), torch.cuda.stream(stream):
                    source = expected_source.clone() if direction == 'h2d' else expected_source.to(cuda_device)
                    # For D2H this producer is queued on the current caller stream.
                    source.add_(1)
                    outputs = [fn(source) for fn in functions]
                    calls += len(outputs)
                    del source
                expected = expected_source.add(1).t().contiguous()
                oracles = [expected, torch.nn.functional.pad(expected, (1, 2, 0, 1))]
                # The blocking contract permits consumption on another stream
                # immediately, without an event or device-wide synchronization.
                with torch.cuda.stream(streams[(iteration + index + 1) % len(streams)]):
                    consumed = [(out + 3).cpu() for out in outputs]
                for out, used, oracle in zip(outputs, consumed, oracles):
                    assert torch.equal(used.cpu(), oracle + 3)
                    assert torch.equal(out.cpu().view(torch.uint8), oracle.view(torch.uint8))
                    held.append((out, oracle))
                assert len({out.data_ptr() for out, _ in held}) == len(held)
                # Allocate/fill independent storage while old outputs stay live.
                with torch.cuda.stream(stream):
                    pressure = [torch.empty_like(outputs[0]).fill_(-17) for _ in range(3)]
                    del pressure
                for old, oracle in held:
                    assert torch.equal(old.cpu(), oracle)
                held = held[-6:]
                stats = backend.stats()
                assert stats['runtime_executions'] == calls  # no Torch fallback
                assert_bounds(stats['transfer_resources'], 2, 32 << 20, 4, 4)
                assert stats['transfer_resources']['outstanding_events'] == 0
            if iteration == 0:
                compiles = backend.stats()['dynamo_compiles']
            else:
                assert backend.stats()['dynamo_compiles'] == compiles
    finally:
        backend.close()
        if policy == 'borrowed':
            assert not owner.closed
            # Closing the borrower must leave its owner's cache usable.
            adapter = TransportAdapter(transfer_resources=owner)
            try:
                src = values((65, 67), 99)
                call = adapter.preflight(compiler.compile(transpose()), src, cuda_device)
                assert torch.equal(adapter.execute(call).cpu(), src.t().contiguous())
            finally:
                adapter.close()
                owner.close()
    assert_drained(backend.stats()['transfer_resources'])


@pytest.mark.parametrize('multiple_devices', [False, True])
def test_concurrent_devices_shapes_clear_and_bounded_events(compiler, cuda_device, multiple_devices):
    if multiple_devices and torch.cuda.device_count() < 2:
        pytest.skip('multiple-device qualification requires two CUDA GPUs')
    devices = [cuda_device.index] if not multiple_devices else [0, 1]
    recipes = {direction: compiler.compile(transpose(direction=direction))
               for direction in ['h2d', 'd2h']}
    owner = TransferResources(max_contexts=4, max_contexts_per_device=2,
                              max_background_workers=8, max_retained_bytes=64 << 20,
                              max_live_staging_bytes=96 << 20, acquire_timeout_ms=30000)
    start = threading.Barrier(4)

    def worker(index):
        device = torch.device('cuda', devices[(index // 2) % len(devices)])
        direction = 'h2d' if index % 2 == 0 else 'd2h'
        destination = device if direction == 'h2d' else torch.device('cpu')
        adapter = TransportAdapter(transfer_resources=owner,
                                   transfer_options=dict(n_buffers=2, gather_threads=3))
        held = []
        with torch.cuda.device(device):
            streams = [torch.cuda.default_stream(device), torch.cuda.Stream(device=device)]
            try:
                start.wait(timeout=30)
                for iteration in range(ROUNDS * 2):
                    shape = [(65, 67), (513, 517), (1025, 1031)][iteration % 3]
                    expected = values(shape, index * 1000 + iteration)
                    with torch.cuda.stream(streams[iteration % 2]):
                        source = expected if direction == 'h2d' else expected.to(device)
                        call = adapter.preflight(recipes[direction], source, destination)
                        out = adapter.execute(call)
                        assert out.device == destination
                        assert torch.cuda.current_device() == device.index
                        used = (out + 1).cpu()
                        del source, call
                    assert torch.equal(used.cpu(), expected.t().contiguous() + 1)
                    if iteration in (0, ROUNDS * 2 - 1):
                        held.append((out, expected.t().contiguous()))
                    assert_bounds(owner.stats(), 4, 96 << 20, 8, 2, retained=64 << 20)
                    if index == 0 and iteration % 7 == 0:
                        owner.clear()  # may overlap other callers' admitted work
                for out, expected in held:
                    assert torch.equal(out.cpu(), expected)
            finally:
                adapter.close()
        assert not owner.closed
        return device.index

    try:
        with ThreadPoolExecutor(4) as workers:
            futures = [workers.submit(worker, i) for i in range(4)]
            observed = {future.result(timeout=180) for future in futures}
        assert observed == set(devices)
        stats = owner.stats()
        assert stats['requests'] == ROUNDS * 8
        assert stats['outstanding_events'] == stats['waiters'] == stats['leased'] == 0
        assert stats['event_creations'] == stats['event_retirements']
        # clear() can remove all idle entries of a device whose callers finished
        # earlier. Per-device snapshots describe live entries, not historical use.
        assert {d['device'] for d in stats['devices']} <= set(devices)
        assert all(d['contexts'] <= 2 for d in stats['devices'])
        assert stats['peak_live_staging_bytes'] <= 96 << 20
    finally:
        owner.close()
    assert_drained(owner.stats())
