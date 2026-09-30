"""Explicit direct-call resource sharing, fresh outputs and buffer ownership."""
import gc
import os
import pickle
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import pyreloc

from reloc_torch import TransferResources
from reloc_torch.transport import PreparedTransfer, execute_transfer, prepare_transfer
from test_transport import TORCH_DTYPES, _values, pad_recipe, reference, transpose


def host_request(compiled, source):
    """CPU test double for the bridge; real native HostBackend, no simulated CUDA."""
    from reloc_torch.runtime import bind_plan, bind_symbols, destination_descriptor
    from reloc_torch.transport import _storage_view

    bindings = bind_symbols(compiled, source)
    bound = bind_plan(compiled, bindings)
    view = _storage_view(source, 'host', -1)
    span = pyreloc.validate_transfer_source(bound, view, 'd2h')
    destination = destination_descriptor(compiled, bindings, torch.device('cpu'))
    return PreparedTransfer(compiled, source, bindings, bound, destination,
                            'd2h', torch.device('cpu'), view, span)


def test_facade_is_lazy_without_runtime_or_cuda_initialization():
    code = """
import sys, importlib.abc
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyreloc': raise ModuleNotFoundError('native runtime unavailable')
finder = NoRuntime()
sys.meta_path.insert(0, finder)
from reloc_torch import TransferResources
assert 'torch' not in sys.modules and 'pyreloc' not in sys.modules
try: TransferResources()
except ModuleNotFoundError: pass
else: raise AssertionError('construction requires the native runtime')
assert 'torch' not in sys.modules
sys.meta_path.remove(finder)
import torch
assert not torch.cuda.is_initialized()
with TransferResources() as resources:
    assert resources.stats()['contexts'] == 0
    assert resources.stats()['devices'] == []
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('cached', [False, True])
def test_direct_bridge_owns_current_buffers_with_both_resource_policies(compiler, monkeypatch, cached):
    from reloc_torch import transport

    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)
    compiled = compiler.compile(transpose())
    resources = TransferResources() if cached else None
    calls, references, retained = [], [], []
    original = pyreloc.execute_transfer

    def observed(native, **kwargs):
        assert kwargs['resources'] is (resources.native if cached else None)
        assert len(kwargs['owners']) == 2
        references.append(tuple(weakref.ref(owner) for owner in kwargs['owners']))
        calls.append(native)
        return original(native, **kwargs)

    monkeypatch.setattr(pyreloc, 'execute_transfer', observed)
    for n in [67, 33, 65]:
        src = _values((n, 67), torch.float32) + n
        prepared = host_request(compiled, src)
        out = execute_transfer(prepared, resources=resources, gather_threads=3)
        assert prepared.consumed and prepared.request.consumed
        assert torch.equal(out, src.t().contiguous())
        retained.append((out, out.clone()))
        del src, prepared, out
        assert all(source() is None for source, _ in references)
    for out, expected in retained:
        assert torch.equal(out, expected)
    assert len({id(request) for request in calls}) == 3
    del out, expected
    retained.clear()
    gc.collect()
    assert all(output() is None for _, output in references)
    if cached:
        s = resources.stats()
        assert (s['misses'], s['hits'], s['staging_allocations']) == (1, 2, 1)
        assert s['worker_creations'] == 2
        resources.clear()
        assert resources.stats()['contexts'] == 0
        resources.close()


def test_direct_rejections_preserve_preflight_and_consumption(compiler, monkeypatch):
    from reloc_torch import transport

    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)
    compiled = compiler.compile(transpose())
    src = _values((33, 67), torch.float32)
    prepared = host_request(compiled, src)
    with pytest.raises(TypeError, match='TransferResources'):
        execute_transfer(prepared, resources=object())
    assert not prepared.consumed
    src.resize_(67, 33)
    with pytest.raises(RuntimeError, match='stale'):
        execute_transfer(prepared, resources=TransferResources())
    assert not prepared.consumed
    with TransferResources(max_contexts=0) as resources:
        prepared = host_request(compiled, src)
        with pytest.raises(RuntimeError, match='resource_limit'):
            execute_transfer(prepared, resources=resources)
        assert prepared.consumed and prepared.request.consumed
        assert resources.stats()['requests'] == 1
        with pytest.raises(RuntimeError, match='already executed'):
            execute_transfer(prepared, resources=resources)
    with pytest.raises(RuntimeError, match='resources_closed'):
        execute_transfer(host_request(compiled, src), resources=resources)
    with pytest.raises(TypeError, match='serialized'):
        pickle.dumps(resources)


def test_direct_borrowed_workers_win_and_are_not_retained(compiler, monkeypatch):
    from reloc_torch import transport

    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)
    compiled = compiler.compile(transpose())
    source = _values((1025, 1031), torch.float32)
    pool = pyreloc.GatherPool(3)
    weak_pool = weakref.ref(pool)
    with TransferResources(max_background_workers=0) as resources:
        out = execute_transfer(host_request(compiled, source), resources=resources,
                               gather_pool=pool, gather_threads=999)
        assert torch.equal(out, source.t().contiguous())
        assert resources.stats()['worker_creations'] == 0
        assert not pool.closed
        del pool
        gc.collect()
        assert weak_pool() is None


def test_one_prepared_request_cannot_race_during_output_allocation(compiler, monkeypatch):
    from reloc_torch import transport

    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)
    source = _values((33, 67), torch.float32)
    request = host_request(compiler.compile(transpose()), source)
    allocating, release = threading.Event(), threading.Event()
    original = torch.empty_strided

    def allocate(*args, **kwargs):
        allocating.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'empty_strided', allocate)
    with ThreadPoolExecutor(1) as workers:
        first = workers.submit(execute_transfer, request)
        try:
            assert allocating.wait(5)
            with pytest.raises(RuntimeError, match='already executed'):
                execute_transfer(request)
        finally:
            release.set()
        assert torch.equal(first.result(timeout=5), source.t().contiguous())


def test_output_allocation_failure_releases_prepared_request_guard(compiler, monkeypatch):
    from reloc_torch import transport

    monkeypatch.setattr(transport.compat, 'cuda_stream_handle', lambda _: None)
    source = _values((33, 67), torch.float32)
    request = host_request(compiler.compile(transpose()), source)
    original = torch.empty_strided

    def fail(*args, **kwargs):
        raise RuntimeError('injected allocation failure')

    monkeypatch.setattr(torch, 'empty_strided', fail)
    with pytest.raises(RuntimeError, match='injected allocation failure'):
        execute_transfer(request)
    assert not request.consumed
    monkeypatch.setattr(torch, 'empty_strided', original)
    assert torch.equal(execute_transfer(request), source.t().contiguous())


def scenario_quarantine():
    from _reloc_transfer_test import Control
    from reloc_torch import CompilerClient, transport

    transport.compat.cuda_stream_handle = lambda _: None
    compiled = CompilerClient.from_environment().compile(transpose())
    control = Control(mode='unknown')
    resources = TransferResources()
    resources._native = control.cache()  # inject only through the build-only harness
    references = []
    original = pyreloc.execute_transfer

    def observed(request, **kwargs):
        references.extend(weakref.ref(owner) for owner in kwargs['owners'])
        return original(request, **kwargs)

    pyreloc.execute_transfer = observed
    source = _values((33, 67), torch.float32)
    prepared = host_request(compiled, source)
    try:
        execute_transfer(prepared, resources=resources)
    except RuntimeError as error:
        assert 'completion_unknown' in str(error)
    else:
        raise AssertionError('the injected failure must propagate')
    assert prepared.consumed
    del source, prepared
    gc.collect()
    assert all(ref() is not None for ref in references)
    with pytest.raises(pyreloc.TransferError, match='completion_unknown'):
        resources.close()
    control.release()
    control.drain_unknown()
    del resources
    gc.collect()
    assert all(ref() is not None for ref in references)


def scenario_fork():
    from reloc_torch import transport

    resources = TransferResources()
    child = os.fork()
    if child == 0:
        # The foreign-process check must run before looking at a request,
        # importing Torch in the call, or allocating its output.
        for action in [resources.stats, resources.close, resources.clear,
                       lambda: transport.execute_transfer(None, resources=resources)]:
            try:
                action()
            except RuntimeError as error:
                if 'process_mismatch' not in str(error): os._exit(2)
            else:
                os._exit(3)
        os._exit(0)
    assert os.waitpid(child, 0)[1] == 0
    resources.close()


@pytest.mark.parametrize('scenario', ['quarantine', 'fork'])
def test_frontend_lifetime_and_process_boundaries(scenario):
    pytest.importorskip('_reloc_transfer_test', reason='requires the build-only fault harness')
    if scenario == 'fork' and not hasattr(os, 'fork'):
        pytest.skip('requires fork')
    result = subprocess.run([sys.executable, __file__, scenario], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


gpu = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not pyreloc.cuda_enabled or not torch.cuda.is_available(),
                                reason='requires CUDA runtime and GPU')


@gpu
@needs_cuda
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int8'])
@pytest.mark.parametrize('layout', ['transpose', 'pad'])
def test_cached_cuda_current_data_outputs_and_streams(compiler, cuda_device, monkeypatch,
                                                     direction, dtype, layout):
    compiled = compiler.compile((transpose if layout == 'transpose' else pad_recipe)(dtype, direction))
    streams = [torch.cuda.Stream(device=cuda_device), torch.cuda.Stream(device=cuda_device)]
    observed, refs, outputs = [], [], []
    original = pyreloc.execute_transfer

    def record(request, **kwargs):
        observed.append(kwargs['caller_stream'])
        refs.append(tuple(weakref.ref(owner) for owner in kwargs['owners']))
        return original(request, **kwargs)

    monkeypatch.setattr(pyreloc, 'execute_transfer', record)
    with TransferResources() as resources:
        for round in range(3):
            stream = streams[round % 2]
            with torch.cuda.stream(stream):
                source = _values((513, 517), TORCH_DTYPES[dtype],
                                 'cpu' if direction == 'h2d' else cuda_device)
                source.add_(round)
                prepared = prepare_transfer(compiled, source,
                                            cuda_device if direction == 'h2d' else 'cpu')
                out = execute_transfer(prepared, resources=resources, n_buffers=2, gather_threads=3)
                expected = reference(layout, source).cpu()
            assert observed[-1] == stream.cuda_stream
            assert torch.equal(out.cpu().view(torch.uint8), expected.view(torch.uint8))
            outputs.append((out, expected))
            del source, prepared, out, expected
            assert all(src() is None for src, _ in refs)
            current = resources.stats()
            if round == 0:
                warm = current
            else:
                for counter in ['staging_allocations', 'stream_creations', 'worker_creations']:
                    assert current[counter] == warm[counter]
            for old, oracle in outputs:
                assert torch.equal(old.cpu().view(torch.uint8), oracle.view(torch.uint8))
        assert resources.stats()['hits'] == 2
        assert resources.stats()['outstanding_events'] == 0
        del old, oracle
        outputs.clear()
        gc.collect()
        assert all(dst() is None for _, dst in refs)
    assert resources.stats()['contexts'] == 0


@gpu
@needs_cuda
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
def test_cached_cuda_growth_and_independent_outputs(compiler, cuda_device, direction):
    compiled = compiler.compile(transpose(direction=direction))
    outputs = []
    with TransferResources(max_contexts=1) as resources:
        for shape in [(33, 67), (513, 517), (1025, 1031), (65, 67)]:
            source = _values(shape, torch.float32, 'cpu' if direction == 'h2d' else cuda_device)
            request = prepare_transfer(compiled, source, cuda_device if direction == 'h2d' else 'cpu')
            out = execute_transfer(request, resources=resources, n_buffers=1, gather_threads=3)
            outputs.append((out, source.t().contiguous().cpu()))
        for out, expected in outputs:
            assert torch.equal(out.cpu(), expected)
        s = resources.stats()
        assert (s['misses'], s['growths'], s['hits']) == (1, 2, 1)
        assert (s['stream_creations'], s['worker_creations']) == (2, 2)


@gpu
@needs_cuda
def test_cached_cuda_python_threads_share_one_context(compiler, cuda_device):
    compiled = compiler.compile(transpose())
    with TransferResources(max_contexts=1) as resources:
        def transfer(seed):
            source = _values((513, 517), torch.float32) + seed
            request = prepare_transfer(compiled, source, cuda_device)
            output = execute_transfer(request, resources=resources, n_buffers=1, gather_threads=2)
            assert torch.equal(output.cpu(), source.t().contiguous())
            return output
        with ThreadPoolExecutor(2) as workers:
            outputs = list(workers.map(transfer, range(6)))
        assert len({out.data_ptr() for out in outputs}) == 6
        assert resources.stats()['contexts'] == 1 and resources.stats()['hits'] == 5


if __name__ == '__main__':
    globals()['scenario_' + sys.argv[1]]()
