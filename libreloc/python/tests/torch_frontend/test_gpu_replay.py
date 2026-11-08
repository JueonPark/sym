"""CUDA replay's observable storage, freshness, failure and stream contract."""
import os

import pytest
import torch


def replay(fn, **policy):
    from reloc_torch import GPUReplay, GPUReplayPolicy
    return GPUReplay(fn, device='cuda:0', policy=GPUReplayPolicy(**policy))


@pytest.mark.parametrize('options', [dict(slots=0), dict(slots=True), dict(max_buckets=0),
                                    dict(max_bytes=-1), dict(enabled=1)])
def test_invalid_policy(options):
    from reloc_torch import GPUReplayPolicy
    with pytest.raises((ValueError, TypeError)):
        GPUReplayPolicy(**options)


@pytest.mark.gpu
@pytest.mark.parametrize('enabled', [False, True])
def test_fresh_parameters_and_held_aliases_after_close(cuda_device, enabled):
    with torch.no_grad(), replay(lambda x, p, *, bias: {'y': x * p + bias}, enabled=enabled) as owner:
        x, p = torch.arange(16., device=cuda_device), torch.full((16,), 2., device=cuda_device)
        owner.prepare(x, p, bias=3)
        owner.freeze()
        old = owner(x, p, bias=3)['y']
        held = old[::2]
        for value in (4., 7., -2.):
            y = owner(x + 1, torch.full_like(p, value), bias=3)['y']
            torch.testing.assert_close(y, (x + 1) * value + 3)
        torch.testing.assert_close(held, torch.arange(0., 16., 2., device=cuda_device) * 2 + 3)
        stats = owner.stats()
        assert stats['slots'] == 2
        assert stats['replays'] == (4 if enabled else 0)
        assert stats['retained_bytes'] <= stats['max_bytes']
        assert stats['pool_reserved_bytes'] > 0 if enabled else stats['pool_reserved_bytes'] == 0
    torch.testing.assert_close(old, x * 2 + 3)
    assert owner.stats()['retained_bytes'] == 0
    with torch.no_grad(), pytest.raises(RuntimeError, match='closed'):
        owner(x, p, bias=3)


@pytest.mark.gpu
def test_exact_shapes_bounded_growth_and_revision_invalidation(cuda_device):
    with torch.no_grad(), replay(lambda x: x + 2, max_buckets=1) as owner:
        x = torch.ones(7, device=cuda_device)
        owner.prepare(x)
        torch.testing.assert_close(owner(torch.ones(9, device=cuda_device)), torch.full((9,), 3., device=cuda_device))
        assert owner.stats()['fallbacks']['bucket_limit'] == 1
        assert owner.stats()['buckets'] == 1
        owner.invalidate()
        owner.prepare(torch.ones(9, device=cuda_device))
        owner.freeze()
        torch.testing.assert_close(owner(x), x + 2)
        assert owner.stats()['fallbacks']['frozen_shape'] == 1
        assert owner.stats()['invalidations'] == 1


@pytest.mark.gpu
@pytest.mark.parametrize('limit', [1, 4096])
def test_pool_reservation_admission_and_cleanup(cuda_device, limit):
    with torch.no_grad(), replay(lambda x: x.sin(), max_bytes=limit) as owner:
        x = torch.ones(16, device=cuda_device)
        torch.testing.assert_close(owner(x), x.sin())
        assert owner.stats()['fallbacks']['memory_budget'] == 1
        assert owner.stats()['retained_bytes'] == 0
        assert owner.stats()['slots'] == 0


@pytest.mark.gpu
def test_stream_order_and_occupied_slots(cuda_device):
    with torch.no_grad(), replay(lambda x: x * 3, slots=2) as owner:
        x = torch.ones(16, device=cuda_device)
        owner.prepare(x)
        owner.freeze()
        streams = [torch.cuda.Stream() for _ in range(3)]
        torch.cuda.synchronize()
        outputs = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(20_000_000)
                value = torch.full_like(x, i + 1)
                outputs.append(owner(value))
        torch.cuda.synchronize()
        for i, value in enumerate(outputs):
            torch.testing.assert_close(value, torch.full_like(x, 3 * (i + 1)))
        assert owner.stats()['fallbacks']['busy_slots'] == 1
        assert owner.stats()['replays'] == 2


@pytest.mark.gpu
def test_producer_dependency_before_input_refresh(cuda_device):
    with torch.no_grad(), replay(lambda x: x + 1) as owner:
        x = torch.zeros(32, device=cuda_device)
        owner.prepare(x)
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(5_000_000)
            x.fill_(9)
        with torch.cuda.stream(consumer):
            consumer.wait_stream(producer)
            result = owner(x)
        consumer.synchronize()
        torch.testing.assert_close(result, torch.full_like(x, 10))


@pytest.mark.gpu
def test_clean_capture_failure_falls_back_once_and_remembers(cuda_device):
    # .item() is legal eagerly but forbidden during actual CUDA capture.
    def needs_host(x):
        return x + x.sum().item()
    with torch.no_grad(), replay(needs_host) as owner:
        x = torch.ones(8, device=cuda_device)
        torch.testing.assert_close(owner(x), torch.full_like(x, 9))
        torch.testing.assert_close(owner(x * 2), torch.full_like(x, 18))
        assert owner.stats()['capture_failures'] == 1
        assert owner.stats()['slots'] == 0
        assert owner.stats()['fallbacks']['capture_failed'] == 2


@pytest.mark.gpu
def test_mutating_region_uses_original_once(cuda_device):
    with torch.no_grad(), replay(lambda x: x.add_(1)) as owner:
        x = torch.ones(8, device=cuda_device)
        result = owner(x)
        torch.testing.assert_close(x, torch.full_like(x, 2))
        assert result is x
        assert owner.stats()['fallbacks']['input_mutation'] == 1


@pytest.mark.gpu
def test_warmup_error_is_not_retried(cuda_device):
    def invalid(x):
        raise ValueError('bad region')
    with torch.no_grad(), replay(invalid) as owner:
        with pytest.raises(ValueError, match='bad region'):
            owner(torch.ones(8, device=cuda_device))
        assert owner.stats()['slots'] == 0


@pytest.mark.gpu
def test_inference_metadata_device_and_settings_guards(cuda_device):
    owner = replay(lambda x: x + 1)
    x = torch.ones(8, device=cuda_device)
    try:
        with pytest.raises(RuntimeError, match='no_grad'):
            owner(x)
        with torch.no_grad():
            owner.prepare(x)
            owner.freeze()
            for value in (torch.ones(8), x[::2], x[:0]):
                torch.testing.assert_close(owner(value), value + 1)
            with torch.autocast('cuda'):
                torch.testing.assert_close(owner(x), x + 1)
            assert owner.stats()['replays'] == 0
            assert sum(owner.stats()['fallbacks'].values()) == 4
    finally:
        owner.close()


@pytest.mark.gpu
def test_process_guard_precedes_locks_and_cuda(cuda_device, monkeypatch):
    owner = replay(lambda x: x + 1)
    monkeypatch.setattr(os, 'getpid', lambda: -1)
    for action in (owner.stats, owner.close, owner.freeze, owner.invalidate):
        with pytest.raises(RuntimeError, match='process'):
            action()
