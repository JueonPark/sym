"""Shared frontend owners, lifecycle races, and opt-in compiled/eager reuse."""
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyreloc
import pytest
import torch

from reloc_torch import AUTO, RelocBackend, TransferResources, eager_transfers
from reloc_torch.runtime import ExecutionError, TransportAdapter, execute_or_fallback
from test_resources import host_request
from test_transport import _values, transpose


@pytest.fixture
def host_bridge(monkeypatch):
    # Only CPU control-flow tests substitute preflight/stream discovery. The
    # execution bridge, native cache, copies and gathers remain real HostBackend.
    from reloc_torch import transport

    monkeypatch.setattr(transport, 'prepare_transfer',
                        lambda compiled, src, device, **kw: host_request(compiled, src))
    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)


@pytest.mark.parametrize('factory', [RelocBackend, TransportAdapter])
@pytest.mark.parametrize('options,error', [
    ([], TypeError), ({'resources': None}, ValueError),
    ({'n_buffers': 0}, ValueError), ({'n_streams': -1}, ValueError),
    ({'gather_threads': -1}, ValueError), ({'n_buffers': 1.5}, TypeError),
    ({'n_streams': True}, TypeError), ({'gather_threads': 1 << 31}, ValueError),
    ({'gather_pool': object()}, TypeError),
])
def test_settings_are_validated_before_any_transfer(factory, options, error):
    with pytest.raises(error):
        factory(transfer_options=options)
    with pytest.raises(TypeError, match='transfer_resources'):
        factory(transfer_resources=object())


def test_custom_runtime_owns_its_policy(counting_runtime):
    for options in [{'transfer_resources': AUTO}, {'transfer_options': {}},
                    {'transfer_resources': TransferResources()}]:
        with pytest.raises(ValueError, match='custom runtime owns its policy'):
            RelocBackend(runtime=counting_runtime, **options)
    backend = RelocBackend(runtime=counting_runtime)
    backend.close()
    assert backend.stats()['transfer_resources'] is None


def test_options_are_copied_and_external_pool_is_borrowed(compiler, host_bridge):
    pool = pyreloc.GatherPool(3)
    options = dict(n_buffers=1, n_streams=3, gather_threads=999, gather_pool=pool)
    with TransferResources(max_background_workers=0) as resources:
        backend = RelocBackend(compiler=compiler, transfer_resources=resources,
                               transfer_options=options)
        options.update(n_streams=0, gather_pool=None)
        source = torch.arange(64, dtype=torch.float32)
        entry = backend.eager_entry(source, 'cpu', lambda x: x.clone())
        assert torch.equal(execute_or_fallback(entry, source, None, 'cpu'), source)
        stats = backend.stats()['transfer_resources']
        assert (stats['stream_creations'], stats['worker_creations']) == (3, 0)
        backend.close()
        assert not pool.closed and not resources.closed
        assert entry.closed
        with pytest.raises(RuntimeError, match='closed'):
            execute_or_fallback(entry, source, None, 'cpu')
    assert not pool.closed
    pool.close()


def test_defaults_and_unused_auto_do_not_create_resources(compiler, host_bridge):
    source = torch.arange(64, dtype=torch.float32)
    for policy in [None, AUTO]:
        backend = RelocBackend(compiler=compiler, transfer_resources=policy)
        assert backend.stats()['transfer_resources'] is None
        adapter = backend.runtime
        call = adapter.preflight(compiler.compile(transpose()), source.reshape(8, 8), 'cpu')
        assert backend.stats()['transfer_resources'] is None  # preflight allocates none
        if policy is None:
            adapter.execute(call)
            assert backend.stats()['transfer_resources'] is None
        backend.close()
        assert backend.stats()['transfer_resources'] is None
        with pytest.raises(RuntimeError, match='closed'):
            adapter.execute(call)
        with pytest.raises(RuntimeError, match='closed'):
            backend.runtime


def test_fake_graph_execution_never_materializes_auto_resources(compiler):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from test_backend import capture, transfer

    backend = RelocBackend(compiler=compiler, transfer_resources=AUTO)
    graph = backend(capture(lambda x: transfer(x.t().contiguous())), None)
    before = backend.stats()
    assert before['replaced_regions'] == 1
    with FakeTensorMode() as mode:
        output = graph(mode.from_tensor(torch.ones(3, 4)))
    output = output[0] if isinstance(output, (tuple, list)) else output
    assert output.shape == (4, 3) and output.device.type == 'cuda'
    assert backend.stats() == before
    assert before['transfer_resources'] is None
    graph.release()
    backend.close()


def test_concurrent_first_execution_materializes_one_owner(compiler, host_bridge, monkeypatch):
    from reloc_torch import runtime

    constructing, release = threading.Event(), threading.Event()
    created = []

    def create():
        constructing.set()
        assert release.wait(5)
        owner = TransferResources(max_contexts=1)
        created.append(owner)
        return owner

    monkeypatch.setattr(runtime, 'TransferResources', create)
    backend = RelocBackend(compiler=compiler, transfer_resources=AUTO,
                           transfer_options={'n_buffers': 1, 'gather_threads': 2})
    compiled = compiler.compile(transpose())
    start = threading.Barrier(6)

    def run(seed):
        start.wait(timeout=5)
        adapter = backend.runtime
        source = _values((33, 67), torch.float32) + seed
        output = adapter.execute(adapter.preflight(compiled, source, 'cpu'))
        assert torch.equal(output, source.t().contiguous())
        return adapter, output

    with ThreadPoolExecutor(6) as threads:
        futures = [threads.submit(run, seed) for seed in range(6)]
        try:
            assert constructing.wait(5)
        finally:
            release.set()
        results = [future.result(timeout=10) for future in futures]
    assert len(created) == 1
    assert len({id(adapter) for adapter, _ in results}) == 1
    assert len({out.data_ptr() for _, out in results}) == 6
    stats = backend.stats()['transfer_resources']
    assert (stats['misses'], stats['hits'], stats['contexts']) == (1, 5, 1)
    backend.close()
    assert created[0].closed and created[0].stats()['contexts'] == 0


@pytest.mark.parametrize('inject_adapter', [False, True])
def test_borrowers_and_entries_do_not_close_shared_owner(compiler, host_bridge, inject_adapter):
    with TransferResources(max_contexts=1) as resources:
        adapter = TransportAdapter(transfer_resources=resources)
        options = {'runtime': adapter} if inject_adapter else {'transfer_resources': resources}
        first = RelocBackend(compiler=compiler, **options)
        second = RelocBackend(compiler=compiler, **options)
        source = torch.arange(64, dtype=torch.float32)
        entry = first.eager_entry(source, 'cpu', lambda x: x.clone())
        execute_or_fallback(entry, source, None, 'cpu')
        entry.close()
        assert not resources.closed
        first.close()
        assert not resources.closed
        next_entry = second.eager_entry(source + 1, 'cpu', lambda x: x.clone())
        assert torch.equal(execute_or_fallback(next_entry, source + 1, None, 'cpu'), source + 1)
        assert resources.stats()['hits'] == 1
        second.close()
        assert not resources.closed
        adapter.close()
        assert not resources.closed


def test_admission_failure_propagates_without_torch_replay(compiler, host_bridge):
    with TransferResources(max_contexts=0) as resources:
        backend = RelocBackend(compiler=compiler, transfer_resources=resources)
        source = torch.arange(64, dtype=torch.float32)
        entry = backend.eager_entry(source, 'cpu', lambda x: pytest.fail('unexpected replay'))
        with pytest.raises(ExecutionError, match='resource_limit'):
            execute_or_fallback(entry, source, None, 'cpu')
        assert entry.fallback_calls == 0
        assert backend.stats()['runtime_executions'] == 1
        backend.close()


@pytest.mark.parametrize('mode', ['allocation', 'copy', 'throw_copy', 'complete_failure', 'wait'])
def test_native_failure_propagates_without_original_region_replay(compiler, host_bridge, mode):
    harness = pytest.importorskip('_reloc_transfer_test', reason='build-only native fault harness')
    control = harness.Control(mode=mode, gated=False)
    with TransferResources() as resources:
        resources._native = control.cache()
        backend = RelocBackend(compiler=compiler, transfer_resources=resources)
        source = torch.arange(64, dtype=torch.float32)
        entry = backend.eager_entry(source, 'cpu', lambda x: pytest.fail('unexpected replay'))
        try:
            with pytest.raises(ExecutionError, match='backend_failure'):
                execute_or_fallback(entry, source, None, 'cpu')
            assert entry.fallback_calls == 0
            assert backend.stats()['runtime_executions'] == 1
            assert control.stats()['copies'] == (0 if mode == 'allocation' else 1)
            assert resources.stats()['contexts'] == 0
        finally:
            backend.close()


def test_close_wins_before_native_admission(compiler, host_bridge, monkeypatch):
    from reloc_torch import transport

    ready, release = threading.Event(), threading.Event()
    execute = transport.execute_transfer

    def gated(request, **options):
        ready.set()
        assert release.wait(5)
        return execute(request, **options)

    monkeypatch.setattr(transport, 'execute_transfer', gated)
    backend = RelocBackend(compiler=compiler, transfer_resources=AUTO)
    source = torch.arange(64, dtype=torch.float32)
    entry = backend.eager_entry(source, 'cpu', lambda x: pytest.fail('unexpected replay'))
    with ThreadPoolExecutor(1) as threads:
        running = threads.submit(execute_or_fallback, entry, source, None, 'cpu')
        try:
            assert ready.wait(5)
            backend.close()
            assert backend.stats()['transfer_resources']['contexts'] == 0
        finally:
            release.set()
        with pytest.raises(ExecutionError, match='resources_closed'):
            running.result(timeout=5)
    assert entry.fallback_calls == 0


def test_typed_dispatch_does_not_materialize_layout_cache(monkeypatch):
    adapter = TransportAdapter(transfer_resources=AUTO)
    tensor = torch.arange(6, dtype=torch.float32)
    report = object()
    request = object()
    calls = []

    def execute(prepared):
        calls.append(prepared)
        return SimpleNamespace(tensor=tensor, report=report)

    monkeypatch.setattr(adapter, '_dispatch', lambda: SimpleNamespace(execute_typed_transfer=execute))
    call = SimpleNamespace(compiled=SimpleNamespace(typed=True), request=request)
    assert adapter.execute(call) is tensor and call.report is report
    assert calls == [request] and adapter.resource_stats() is None
    adapter.close()


def scenario_close():
    from _reloc_transfer_test import Control
    from reloc_torch import CompilerClient, runtime, transport
    from reloc_torch.cache import REGISTRY

    control = Control()
    owner = TransferResources()
    owner._native = control.cache()
    runtime.TransferResources = lambda: owner
    transport.prepare_transfer = lambda compiled, src, device, **kw: host_request(compiled, src)
    transport.compat.cuda_stream_handle = lambda _: None
    backend = RelocBackend(compiler=CompilerClient.from_environment(), transfer_resources=AUTO,
                           cache_capacity=1)
    source = torch.arange(128, dtype=torch.float32)
    entry = backend.eager_entry(source, 'cpu', lambda x: x.clone())
    registration = REGISTRY.register(entry)
    with ThreadPoolExecutor(3) as threads:
        running = threads.submit(execute_or_fallback, entry, source, None, 'cpu')
        try:
            assert control.wait_for_copy()
            # Artifact/registration eviction and entry closure do not
            # invalidate an admitted lease.
            backend.compile_recipe(transpose())
            registration.release()
            entry.close()
            assert owner.stats()['leased'] == 1
            close_started = queue.Queue()
            original_close = backend.runtime.close

            def close_adapter():
                close_started.put(True)
                original_close()

            backend.runtime.close = close_adapter
            closing = threads.submit(backend.close)
            assert close_started.get(timeout=5)
            also_closing = threads.submit(backend.close)
            assert close_started.get(timeout=5)
            # Would deadlock if close held either Python owner lock. The outer
            # subprocess timeout bounds regressions in native close as well.
            assert backend.stats()['closed']
            assert not closing.done() and not also_closing.done()
            with pytest.raises(RuntimeError, match='closed'):
                execute_or_fallback(entry, source, None, 'cpu')
        finally:
            control.release()
        assert torch.equal(running.result(timeout=5), source)
        closing.result(timeout=5)
        also_closing.result(timeout=5)
    assert owner.closed and owner.stats()['contexts'] == 0
    assert control.stats()['allocations'] == control.stats()['frees']


def scenario_fork():
    backend = RelocBackend(transfer_resources=AUTO)
    adapter = backend.runtime
    # Fork while the parent holds both locks: child rejection must precede them.
    with backend._lock, adapter._lock:
        pid = os.fork()
        if pid == 0:
            for action in [backend.stats, backend.close, lambda: backend.runtime,
                           adapter.resource_stats, adapter.close, lambda: adapter.execute(None)]:
                try:
                    action()
                except RuntimeError as error:
                    assert 'process_mismatch' in str(error)
                else:
                    os._exit(1)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    backend.close()


@pytest.mark.parametrize('scenario', ['close', 'fork'])
def test_lifecycle_subprocess(scenario):
    if scenario == 'close':
        pytest.importorskip('_reloc_transfer_test', reason='build-only native fault harness')
    if scenario == 'fork' and not hasattr(os, 'fork'):
        pytest.skip('requires fork')
    result = subprocess.run([sys.executable, __file__, scenario], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('policy', ['auto', 'borrowed'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.int8])
def test_compiled_and_eager_share_across_recipes_and_shapes(compiler, cuda_device, direction, policy, dtype):
    resources = AUTO if policy == 'auto' else TransferResources(max_contexts=1)
    backend = RelocBackend(compiler=compiler, transfer_resources=resources, cache_capacity=1,
                           transfer_options={'n_buffers': 1, 'gather_threads': 3})
    source_device = 'cpu' if direction == 'h2d' else cuda_device
    destination = cuda_device if direction == 'h2d' else 'cpu'

    def transpose_transfer(x):
        return x.t().contiguous().to(destination)

    def pad_transfer(x):
        return torch.nn.functional.pad(x, (1, 2, 0, 1)).to(destination)

    functions = [(torch.compile(fn, backend=backend, dynamic=True), fn)
                 for fn in [transpose_transfer, pad_transfer]]
    outputs = []
    stream = torch.cuda.Stream(device=cuda_device)
    try:
        for iteration in range(2):
            with torch.no_grad(), torch.cuda.stream(stream):
                for shape in [(65, 67), (33, 39)]:
                    source = _values(shape, dtype, source_device) + iteration
                    for compiled, original in functions:
                        outputs.append((compiled(source), original(source).cpu()))
                    with eager_transfers(backend=backend):
                        output = source.to(destination)
                    outputs.append((output, source.cpu()))
            stats = backend.stats()['transfer_resources']
            counters = tuple(stats[name] for name in ['contexts', 'staging_allocations',
                                                      'stream_creations', 'worker_creations'])
            if iteration == 0:
                warmed = counters
            else:
                assert counters == warmed
        assert stats['misses'] == 1 and stats['hits'] == 11
        assert backend.stats()['runtime_executions'] == 12
        assert backend.stats()['cache_entries'] == 1
        assert len({out.data_ptr() for out, _ in outputs}) == 12
        for output, expected in outputs:
            assert torch.equal(output.cpu(), expected)
    finally:
        backend.close()
        if policy == 'borrowed':
            assert not resources.closed
            resources.close()
    assert backend.stats()['transfer_resources']['contexts'] == 0


if __name__ == '__main__':
    globals()['scenario_' + sys.argv[1]]()
