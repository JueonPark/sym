"""Explicit wire representation and asynchronous source ownership."""
import gc
import weakref

import pytest
import torch


def owner(**kwargs):
    from reloc_torch.consumer_layout import PackedWireRestore
    return PackedWireRestore('cuda:0', **kwargs)


@pytest.fixture
def packed():
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    return torch.randn(4,7,4,8,dtype=torch.float16,pin_memory=True)


def test_wire_restore_grouped_offsets_fresh_values_and_held_outputs(packed):
    restore = owner()
    values = list(packed[1:].unbind(0))
    try:
        first = restore.restore(values); restore.drain()
        assert restore.stats()['copy_calls'] == 1
        assert all(x.is_contiguous() for x in first)
        assert torch.equal(torch.stack(first).cpu(), packed[1:])
        snapshot = [x.clone() for x in first]
        packed.add_(2)
        later = restore.restore(values); restore.close()
        assert restore.stats()['pending_host_bytes'] == 0
        assert torch.equal(torch.stack(later).cpu(), packed[1:])
        for x,y in zip(first,snapshot): assert torch.equal(x,y)
        with pytest.raises(RuntimeError,match='closed'): restore.restore(values)
    finally: restore.close()


def test_wire_restore_separate_storage_uses_safe_individual_copies(packed):
    restore = owner()
    try:
        values = [packed[0].clone(), packed[1].clone()]
        out = restore.restore(values); restore.drain()
        assert restore.stats()['copy_calls'] == 2
        assert all(torch.equal(a.cpu(),b) for a,b in zip(out,values))
    finally: restore.close()


def test_wire_preflight_rejects_every_bad_input_before_submission(packed):
    restore = owner()
    try:
        for bad in (packed[0][...,::2], packed[0].cuda(), packed[0].float(), packed[0][0]):
            with pytest.raises((TypeError,ValueError)):
                restore.restore([packed[0],bad])
            assert restore.stats()['copy_calls'] == 0
        assert restore.restore([]) == []
    finally: restore.close()


def test_wire_budget_and_pending_limit(packed):
    from reloc_torch.memory import MemoryBudgetError
    with pytest.raises((ValueError,TypeError)): owner(max_inflight=0)
    restore = owner(max_bytes=16, max_inflight=1)
    try:
        with pytest.raises(MemoryBudgetError): restore.restore([packed[0]])
        assert restore.stats()['copy_calls'] == 0
    finally: restore.close()
    restore = owner(max_inflight=1)
    try:
        for _ in range(4): restore.restore([packed[0]])
        assert restore.stats()['pending_groups'] <= 1
    finally: restore.close()


def test_wire_temporary_host_storage_lives_until_copy_and_gc_fences(packed):
    restore = owner()
    source = packed.clone().pin_memory(); weak = weakref.ref(source)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        out = restore.restore([source[0]])
    del source; gc.collect()
    # The copied host view keeps the underlying allocation alive even when its
    # original tensor object can disappear; numerical output is authoritative.
    del restore; gc.collect()
    assert stream.query()
    assert torch.equal(out[0].cpu(), packed[0])
    assert weak() is None


def test_wire_cross_stream_handoff_and_input_revision(packed):
    restore = owner()
    producer = torch.cuda.Stream(); consumer = torch.cuda.Stream()
    try:
        with torch.cuda.stream(producer):
            values = restore.restore(list(packed.unbind(0)))
        with torch.cuda.stream(consumer):
            consumer.wait_stream(producer)
            result = torch.cat([x.transpose(0,1) for x in values], dim=1)
            for x in values: x.record_stream(consumer)
        del values; restore.close(); consumer.synchronize()
        assert torch.equal(result.cpu(), torch.cat([x.transpose(0,1) for x in packed],dim=1))
    finally: restore.close()


def test_kv_owner_exposes_wire_without_changing_default_restore(compiler, packed):
    from reloc_torch.fast import FastTransfers, KVCacheSwap
    with FastTransfers('cuda:0',compiler=compiler,gather_threads=1,pinning='pinned') as fast:
        swap = KVCacheSwap(fast,4,8,dtype='float16')
        cache = [torch.randn(4,7,8,device='cuda',dtype=torch.float16) for _ in range(2)]
        stored = swap.evict(cache)
        wire = swap.restore_wire(stored)
        restored = swap.restore(stored)
        swap.drain()
        assert all(torch.equal(a.transpose(0,1),b) for a,b in zip(wire,cache))
        assert all(torch.equal(a,b) for a,b in zip(restored,cache))
    assert all(torch.equal(a.transpose(0,1),b) for a,b in zip(wire,cache))
    with pytest.raises(RuntimeError,match='closed'): swap.restore_wire(stored)


def test_wire_partial_submission_failure_fences_and_poison(packed, monkeypatch):
    restore = owner()
    original = torch.Tensor.copy_
    calls = 0
    def copy(target, source, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('injected second copy failure')
        return original(target, source, **kwargs)
    stream = torch.cuda.Stream()
    monkeypatch.setattr(torch.Tensor, 'copy_', copy)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            with pytest.raises(RuntimeError,match='injected'):
                restore.restore([packed[0].clone(), packed[1].clone()])
        assert stream.query()
        assert restore.stats()['poisoned'] and restore.stats()['failed_groups'] == 0
        with pytest.raises(RuntimeError,match='poisoned'): restore.restore([packed[0]])
    finally: restore.close()


def test_kv_wire_admission_uses_owner_declared_envelope(compiler, packed):
    from reloc_torch.fast import FastTransfers, KVCacheSwap
    from reloc_torch.memory import MemoryBudgetError
    with FastTransfers('cuda:0',compiler=compiler,gather_threads=1,contexts=1,max_scratch_bytes=128) as fast:
        swap = KVCacheSwap(fast,4,8,dtype='float16')
        with pytest.raises(MemoryBudgetError): swap.restore_wire([packed[0]])
        assert swap._wire_restore.stats()['copy_calls'] == 0


def test_wire_device_guard_restores_callers_current_device(packed):
    if torch.cuda.device_count() < 2: pytest.skip('second GPU hidden or unavailable')
    restore = owner()
    try:
        with torch.cuda.device(1):
            out = restore.restore([packed[0]])
            assert torch.cuda.current_device() == 1
        restore.close()
        assert out[0].device.index == 0 and torch.equal(out[0].cpu(),packed[0])
    finally: restore.close()


def test_kv_wire_rejects_incompatible_declared_cache_metadata(compiler, packed):
    from reloc_torch.fast import FastTransfers, KVCacheSwap
    with FastTransfers('cuda:0',compiler=compiler,gather_threads=1) as fast:
        swap = KVCacheSwap(fast,4,8,dtype='float16')
        for bad in (packed[0].float(), torch.zeros(7,5,8,dtype=torch.float16),
                    torch.zeros(7,4,6,dtype=torch.float16)):
            with pytest.raises(ValueError): swap.restore_wire([bad])
        assert swap._wire_restore is None


def test_fast_owner_close_fences_wire_copy_without_explicit_drain(compiler, packed):
    from reloc_torch.fast import FastTransfers, KVCacheSwap
    fast = FastTransfers('cuda:0',compiler=compiler,gather_threads=1)
    swap = KVCacheSwap(fast,4,8,dtype='float16')
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            out = swap.restore_wire([packed[0]])
        fast.close()
        assert stream.query() and swap._wire_restore.stats()['closed']
        assert torch.equal(out[0].cpu(),packed[0])
    finally: fast.close()
