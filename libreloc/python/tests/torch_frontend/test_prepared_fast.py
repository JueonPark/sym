"""Prepared fast groups: values, bounded ownership and independent completions."""
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import pytest
import torch

from reloc_torch import PreparedWireWeights
from reloc_torch.fast import FastTransfers, FastWeights
from test_fast import dlrm_recipe

pytestmark = pytest.mark.gpu


def owner(compiler, cuda_device, **kwargs):
    return FastTransfers(cuda_device, compiler=compiler, gather_threads=4, **kwargs)


def expected(source, device):
    return source.transpose(0, 1).contiguous().to(device, dtype=torch.float16)


@pytest.mark.parametrize('report_mode', ['native', 'eager'])
def test_reuse_preserves_outputs_and_lazy_reports_after_close(compiler, cuda_device, report_mode):
    runtime = owner(compiler, cuda_device, report_mode=report_mode, contexts=1)
    source = torch.arange(26 * 7 * 64, dtype=torch.float32).reshape(26, 7, 64)
    program = runtime.typed(dlrm_recipe(), source)
    handle = runtime.prepare_group([program])
    completions, references, outputs = [], [], []
    for i in range(6):
        # Same address, different values; later a different base and offset.
        source.fill_(i + 1)
        if i == 3:
            source = torch.full((source.numel() + 3,), 4.)[3:].reshape(source.shape)
        references.append(expected(source, cuda_device))
        completion = handle.submit([source])
        assert completion.report is None
        outputs.append(completion.wait()[0])
        completions.append(completion)
    stats = runtime.stats()
    assert stats['groups'] == stats['logical_transfers'] == 6
    assert stats['prepared_groups']['group_constructions'] == 1
    assert stats['prepared_groups']['rebinds'] == 5
    assert stats['prepared_groups']['view_descriptors'] == 2
    assert stats['report_materializations'] == (0 if report_mode == 'native' else 6)
    handle.close()
    runtime.close()
    assert len({out.data_ptr() for out in outputs}) == 6
    for completion, output, reference in zip(completions, outputs, references):
        assert completion.done() and torch.equal(output, reference)
        report = completion.report
        assert report['logical_transfers'] == 1 and len(report['items']) == 1
        report['items'][0]['implementation'] = 'caller mutation'
        assert completion.report['items'][0]['implementation'] != 'caller mutation'
    handle.close()
    with pytest.raises(RuntimeError, match='closed'):
        handle.submit([source])


@pytest.mark.parametrize('change', ['shape', 'stride', 'dtype', 'device', 'output', 'output_count', 'source_count', 'grad'])
def test_invalid_rebinding_allows_retry(compiler, cuda_device, change):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device, contexts=1) as runtime:
        handle = runtime.prepare_group([runtime.typed(dlrm_recipe(), source)])
        handle.submit([source]).wait()
        sources, outputs = [source], None
        if change == 'shape': sources = [source.reshape(26, 14, 32)]
        if change == 'stride': sources = [source.transpose(1, 2).contiguous().transpose(1, 2)]
        if change == 'dtype': sources = [source.double()]
        if change == 'device': sources = [source.to(cuda_device)]
        if change == 'grad': sources = [source.clone().requires_grad_()]
        if change == 'output': outputs = [torch.empty(26, 7, 64, device=cuda_device, dtype=torch.float16)]
        if change == 'output_count': outputs = []
        if change == 'source_count': sources = []
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            handle.submit(sources, outputs=outputs)
        result, = handle.submit([source + 2]).wait()
        assert torch.equal(result, expected(source + 2, cuda_device))
        stats = runtime.stats()
        assert stats['groups'] == 2
        assert stats['prepared_groups']['group_constructions'] == 1
        assert stats['prepared_groups']['rebinds'] == 1


def test_bounded_cache_eviction_preserves_held_completions(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device, max_prepared_groups=1, contexts=1) as runtime:
        first = runtime.typed(dlrm_recipe(), source)
        second = runtime.typed(dlrm_recipe(), source)
        a = runtime.submit([(first, source)])
        b = runtime.submit([(second, source * 2)])
        assert torch.equal(a.wait()[0], expected(source, cuda_device))
        assert torch.equal(b.wait()[0], expected(source * 2, cuda_device))
        stats = runtime.stats()['prepared_groups']
        assert stats['cached_handles'] == 1 and stats['evictions'] == 1
        assert stats['group_constructions'] == 2 and stats['live_slots'] <= 1
        assert a.report['items'] == b.report['items']


def test_shared_handle_concurrent_calls_count_each_completion_once(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device, contexts=2) as runtime:
        handle = runtime.prepare_group([runtime.typed(dlrm_recipe(), source)])
        def submit(i):
            completion = handle.submit([source * i])
            out, = completion.wait()
            assert torch.equal(out, expected(source * i, cuda_device))
            return completion
        with ThreadPoolExecutor(max_workers=4) as pool:
            completions = list(pool.map(submit, range(1, 17)))
            list(pool.map(lambda _: completions[0].wait(), range(16)))
        stats = runtime.stats()
        assert stats['groups'] == 16
        assert stats['prepared_groups']['group_constructions'] <= 2
        assert stats['prepared_groups']['live_slots'] <= 2
        assert stats['prepared_groups']['rebinds'] >= 14


def test_weight_revision_replaces_implicit_handle_and_retains_old_inflight_values(compiler, cuda_device):
    q = torch.arange(64 * 96).remainder(127).to(torch.int8).reshape(64, 96)
    scale = torch.ones(96)
    with owner(compiler, cuda_device) as runtime, PreparedWireWeights(cuda_device, compiler=compiler) as weights:
        weights.prepare('w', q, scale)
        loader = FastWeights(runtime, weights)
        old = loader.submit(['w'])
        old_source = weakref.ref(loader._entries['w'][1])
        weights.prepare('w', q, scale * 2)
        new = loader.submit(['w'])
        assert torch.equal(old.wait()[0], q.float().t().contiguous().to(cuda_device))
        assert torch.equal(new.wait()[0], (q.float() * 2).t().contiguous().to(cuda_device))
        gc.collect()
        assert old_source() is None
        assert runtime.stats()['prepared_groups']['cached_handles'] == 1
        assert old.report['logical_transfers'] == new.report['logical_transfers'] == 1


@pytest.mark.parametrize('prepared,bridge', [(False, True), (True, False)])
def test_legacy_and_absent_bridge_remain_functional(compiler, cuda_device, monkeypatch, prepared, bridge):
    import reloc_torch.fast as module
    with owner(compiler, cuda_device, prepared_groups=prepared) as runtime:
        if not bridge:
            monkeypatch.setattr(module, '_native_views', None)
        source = torch.ones(26, 7, 64)
        program = runtime.typed(dlrm_recipe(), source)
        for i in (1, 2):
            output, = runtime.submit([(program, source * i)]).wait()
            assert torch.equal(output, expected(source * i, cuda_device))
        assert runtime.stats()['groups'] == 2
        assert runtime.stats()['prepared_groups']['group_constructions'] == 0


def test_process_guard_precedes_locks(compiler, cuda_device, monkeypatch):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device) as runtime:
        program = runtime.typed(dlrm_recipe(), source)
        handle = runtime.prepare_group([program])
        completion = handle.submit([source])
        completion.wait()
        class InheritedLock:
            def __enter__(self):
                pytest.fail('inherited process lock was touched')
            def __exit__(self, *exc):
                pass
        with monkeypatch.context() as m:
            m.setattr(runtime, '_pid', -1)
            m.setattr(runtime, '_lock', InheritedLock())
            for action in (runtime.close, runtime.stats, handle.close, completion.wait, completion.done,
                           lambda: completion.report, lambda: handle.submit([source]),
                           lambda: runtime.submit([(program, source)]), lambda: runtime.prepare_group([program])):
                with pytest.raises(RuntimeError, match='another process'):
                    action()


def test_submit_close_race_drains_all_successful_submissions(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    runtime = owner(compiler, cuda_device)
    handle = runtime.prepare_group([runtime.typed(dlrm_recipe(), source)])
    first = handle.submit([source])
    def submit(i):
        try:
            return handle.submit([source * i])
        except RuntimeError as error:
            assert 'closed' in str(error)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(submit, i) for i in range(1, 9)]
        runtime.close()
        results = [task.result() for task in pending]
    assert first.done()
    for i, completion in enumerate(results, 1):
        if completion is not None:
            assert completion.done()
            assert torch.equal(completion.wait()[0], expected(source * i, cuda_device))
    assert runtime._resources.stats()['typed']['retained_bytes'] == 0


def test_supplied_offset_outputs_alias_rejection_and_resource_reuse(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    runtime = owner(compiler, cuda_device, contexts=1)
    program = runtime.typed(dlrm_recipe(), source)
    handle = runtime.prepare_group([program, program])
    shape = (7, 26, 64)
    block = torch.empty(2 * source.numel() + 5, device=cuda_device, dtype=torch.float16)
    outputs = [block[5:5 + source.numel()].view(shape), block[5 + source.numel():].view(shape)]
    handle.submit([source, source * 2], outputs=outputs).wait()
    before = runtime.stats()['resources']
    with pytest.raises(Exception, match='group_alias'):
        handle.submit([source, source], outputs=[outputs[0], outputs[0]])
    for i in range(3, 9):
        result = handle.submit([source * i, source * (i + 1)], outputs=outputs).wait()
        assert result[0] is outputs[0] and result[1] is outputs[1]
        assert torch.equal(result[0], expected(source * i, cuda_device))
        assert torch.equal(result[1], expected(source * (i + 1), cuda_device))
    after = runtime.stats()['resources']
    for key in ('context_creations', 'host_allocations', 'device_allocations', 'retained_bytes'):
        assert after[key] == before[key]
    runtime.close()
    assert runtime._resources.stats()['typed']['retained_bytes'] == 0


def test_failed_execution_never_recycles_failed_slot(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device, contexts=1, max_scratch_bytes=4) as runtime:
        handle = runtime.prepare_group([runtime.typed(dlrm_recipe(), source)])
        with pytest.raises(RuntimeError, match='scratch|budget'):
            handle.submit([source])
        assert runtime.stats()['groups'] == 0
        assert runtime.stats()['prepared_groups']['live_slots'] == 0
        # A second rejected attempt must construct a new slot, never recycle the failed one.
        with pytest.raises(RuntimeError, match='scratch|budget'):
            handle.submit([source])
        assert runtime.stats()['prepared_groups']['group_constructions'] == 2
        handle.close()


@pytest.mark.parametrize('mode', [1, 2, 3, 4])
def test_prepared_partial_failure_keeps_quarantine_ownership(cuda_device, mode):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import pyreloc
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('prepared_fast_fault_scenario.py')),
                             str(mode)], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout


def test_collected_handle_releases_metadata_without_losing_construction_counts(compiler, cuda_device):
    source = torch.ones(26, 7, 64)
    with owner(compiler, cuda_device) as runtime:
        handle = runtime.prepare_group([runtime.typed(dlrm_recipe(), source)])
        handle.submit([source]).wait()
        handle_ref = weakref.ref(handle)
        del handle
        gc.collect()
        assert handle_ref() is None
        stats = runtime.stats()['prepared_groups']
        assert stats['live_slots'] == 0 and stats['group_constructions'] == 1


def test_explicit_handle_keeps_parameter_revision_after_replacement(compiler, cuda_device):
    q = torch.ones(64, 96, dtype=torch.int8)
    scale = torch.ones(96)
    with owner(compiler, cuda_device) as runtime, PreparedWireWeights(cuda_device, compiler=compiler) as weights:
        weights.prepare('w', q, scale)
        loader = FastWeights(runtime, weights)
        program, source = loader._items(['w'])[0]
        handle = runtime.prepare_group([program])
        first = handle.submit([source])
        weights.prepare('w', q, scale * 3)
        replacement = loader.submit(['w'])
        old_again = handle.submit([source])
        assert torch.equal(first.wait()[0], torch.ones(96, 64, device=cuda_device))
        assert torch.equal(old_again.wait()[0], torch.ones(96, 64, device=cuda_device))
        assert torch.equal(replacement.wait()[0], torch.full((96, 64), 3., device=cuda_device))
