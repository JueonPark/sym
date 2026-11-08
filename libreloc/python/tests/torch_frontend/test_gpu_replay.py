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


@pytest.mark.gpu
def test_backend_owned_region_after_automatic_transfer(compiler, cuda_device):
    from reloc_torch import RelocBackend, GPUReplayPolicy, PreparedExecutionPolicy
    backend = RelocBackend(compiler=compiler, prepared_execution=PreparedExecutionPolicy(min_bytes=0, min_reuses=0))
    transfer = torch.compile(lambda x: x.t().contiguous().to(cuda_device, torch.float16), backend=backend, fullgraph=True)
    try:
        with torch.no_grad():
            x = torch.arange(12.).reshape(3, 4)
            device = transfer(x)
            region = backend.prepare_gpu_region(lambda y, p: y * p, device,
                         torch.full_like(device, 2), policy=GPUReplayPolicy(slots=1))
            region.freeze()
            old = region(device, torch.full_like(device, 3))
            result = region(transfer(x + 10), torch.full_like(device, 4))
            torch.testing.assert_close(result.cpu(), ((x + 10).t() * 4).half())
            torch.testing.assert_close(old.cpu(), (x.t() * 3).half())
            stats = backend.stats()
            assert stats['prepared_execution']['executions'] == 2
            assert stats['gpu_replay'][0]['replays'] > 0
        backend.close()
        torch.testing.assert_close(old.cpu(), (x.t() * 3).half())
        with torch.no_grad(), pytest.raises(RuntimeError, match='closed'):
            region(device, device)
    finally:
        backend.close()


@pytest.mark.gpu
def test_backend_failed_region_creation_has_no_retained_owner(cuda_device):
    from reloc_torch import RelocBackend
    backend = RelocBackend()
    try:
        def invalid(x):
            raise ValueError('invalid region')
        with torch.no_grad(), pytest.raises(ValueError, match='invalid region'):
            backend.prepare_gpu_region(invalid, torch.ones(4, device=cuda_device))
        assert backend.stats()['gpu_replay'] == []
    finally:
        backend.close()


@pytest.mark.gpu
def test_replay_execution_failure_poisoned_without_fallback(cuda_device, monkeypatch):
    def fail(graph):
        raise RuntimeError('injected driver submission failure')
    with torch.no_grad(), replay(lambda x: x + 1) as owner:
        x = torch.ones(4, device=cuda_device)
        owner.prepare(x)
        monkeypatch.setattr(torch.cuda.CUDAGraph, 'replay', fail)
        with pytest.raises(RuntimeError, match='driver submission'):
            owner(x)
        assert owner.stats()['poisoned']
        assert owner.stats()['fallbacks'] == {}
        with pytest.raises(RuntimeError, match='poisoned'):
            owner(x)


@pytest.mark.gpu
def test_inference_mode_and_changed_math_settings(cuda_device):
    with torch.inference_mode(), replay(lambda x: x * 2) as owner:
        x = torch.ones(8, device=cuda_device)
        owner.prepare(x).freeze()
        torch.testing.assert_close(owner(x + 3), torch.full_like(x, 8))
        prior = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = not prior
            torch.testing.assert_close(owner(x), x * 2)
            assert owner.stats()['fallbacks']['frozen_shape'] == 1
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior


@pytest.mark.gpu
def test_device_change_never_reuses_graph(cuda_device):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires second GPU')
    with torch.no_grad(), replay(lambda x: x + 1) as owner:
        owner.prepare(torch.ones(4, device=cuda_device))
        x = torch.ones(4, device='cuda:1')
        torch.testing.assert_close(owner(x), x + 1)
        assert owner.stats()['replays'] == 0
        assert owner.stats()['fallbacks']['metadata'] == 1


@pytest.mark.gpu
def test_changed_default_dtype_does_not_replay_stale_output(cuda_device):
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.no_grad(), replay(lambda x: torch.ones_like(x, dtype=torch.get_default_dtype())) as owner:
            x = torch.ones(4, device=cuda_device)
            owner.prepare(x).freeze()
            torch.set_default_dtype(torch.float64)
            actual = owner(x)
            assert actual.dtype == torch.float64
            assert owner.stats()['fallbacks']['frozen_shape'] == 1
    finally:
        torch.set_default_dtype(prior)


@pytest.mark.gpu
def test_changed_cudnn_policy_uses_guarded_fallback(cuda_device):
    prior = torch.backends.cudnn.benchmark
    try:
        with torch.no_grad(), replay(lambda x: x + 2) as owner:
            x = torch.ones(4, device=cuda_device)
            owner.prepare(x).freeze()
            torch.backends.cudnn.benchmark = not prior
            torch.testing.assert_close(owner(x), x + 2)
            assert owner.stats()['fallbacks']['frozen_shape'] == 1
    finally:
        torch.backends.cudnn.benchmark = prior


@pytest.mark.gpu
def test_fallback_preserves_temporary_input_on_another_stream(cuda_device):
    with torch.no_grad(), replay(lambda x: x + 1, max_bytes=1) as owner:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        # Load the add/fill kernels before the delayed-stream lifetime check.
        warm = torch.full((16384,), 9., device=cuda_device) + 1
        torch.cuda.synchronize()
        del warm
        with torch.cuda.stream(producer):
            value = torch.full((16384,), 9., device=cuda_device)
        with torch.cuda.stream(consumer):
            consumer.wait_stream(producer)
            torch.cuda._sleep(100_000_000)
            output = owner(value)
        del value
        # Reuse the producer's allocation while the consumer is delayed. The
        # wrapper must register its use even when graph admission falls back.
        with torch.cuda.stream(producer):
            replacement = torch.full((16384,), -100., device=cuda_device)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 10.))
        assert owner.stats()['fallbacks']['memory_budget'] == 1


@pytest.mark.gpu
def test_close_waits_for_inflight_output_copy(cuda_device):
    with torch.no_grad():
        owner = replay(lambda x: x * 2)
        x = torch.ones(16, device=cuda_device)
        owner.prepare(x).freeze()
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(30_000_000)
            output = owner(x + 1)
        owner.close()
        assert consumer.query()
        torch.testing.assert_close(output, torch.full_like(output, 4.))
        assert owner.stats()['retained_bytes'] == 0


@pytest.mark.gpu
def test_explicit_code_revision_invalidation_keeps_old_outputs(cuda_device):
    with torch.no_grad(), replay(lambda x: x + 2) as owner:
        x = torch.ones(16, device=cuda_device)
        owner.prepare(x).freeze()
        old = owner(x)
        owner.invalidate()
        owner.function = lambda x: x - 7
        owner.prepare(x).freeze()
        new = owner(x)
        torch.testing.assert_close(new, torch.full_like(x, -6))
        torch.testing.assert_close(old, torch.full_like(x, 3))


@pytest.mark.gpu
@pytest.mark.parametrize('setting', ['matmul_precision', 'fp16_accumulation', 'fp16_split_k',
                                    'bf16_split_k', 'math_sdp_reduction', 'sdp_priority',
                                    'deterministic_warn_only', 'sdp_overrideable'])
def test_changed_precision_and_selection_policy_falls_back(cuda_device, setting):
    from contextlib import ExitStack
    with ExitStack() as restore:
        if setting == 'matmul_precision':
            prior = torch.get_float32_matmul_precision()
            restore.callback(torch.set_float32_matmul_precision, prior)
            torch.set_float32_matmul_precision('high')
            change = lambda: torch.set_float32_matmul_precision('medium')
        elif setting == 'fp16_accumulation':
            flags = torch.backends.cuda.matmul
            prior = flags.allow_fp16_accumulation
            restore.callback(setattr, flags, 'allow_fp16_accumulation', prior)
            change = lambda: setattr(flags, 'allow_fp16_accumulation', not prior)
        elif setting in ('fp16_split_k', 'bf16_split_k'):
            flags = torch.backends.cuda.matmul
            name = 'allow_' + setting[:4] + '_reduced_precision_reduction'
            prior = (getattr(flags, name), getattr(flags, name + '_split_k'))
            restore.callback(setattr, flags, name, prior)
            setattr(flags, name, (False, True))
            change = lambda: setattr(flags, name, (False, False))
        elif setting == 'math_sdp_reduction':
            flags = torch.backends.cuda
            prior = flags.fp16_bf16_reduction_math_sdp_allowed()
            restore.callback(flags.allow_fp16_bf16_reduction_math_sdp, prior)
            change = lambda: flags.allow_fp16_bf16_reduction_math_sdp(not prior)
        elif setting == 'sdp_priority':
            prior = torch._C._get_sdp_priority_order()
            restore.callback(torch._C._set_sdp_priority_order, prior)
            change = lambda: torch._C._set_sdp_priority_order(list(reversed(prior)))
        elif setting == 'sdp_overrideable':
            prior = torch._C._get_overrideable_sdp_enabled()
            restore.callback(torch._C._set_sdp_use_overrideable, prior)
            change = lambda: torch._C._set_sdp_use_overrideable(not prior)
        else:
            prior = (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled())
            restore.callback(torch.use_deterministic_algorithms, prior[0], warn_only=prior[1])
            torch.use_deterministic_algorithms(True, warn_only=False)
            change = lambda: torch.use_deterministic_algorithms(True, warn_only=True)
        with torch.no_grad(), replay(lambda x: x @ x) as owner:
            x = torch.eye(4, device=cuda_device)
            owner.prepare(x).freeze()
            change()
            torch.testing.assert_close(owner(x), x @ x)
            assert owner.stats()['replays'] == 0
            assert owner.stats()['fallbacks']['frozen_shape'] == 1


@pytest.mark.gpu
@pytest.mark.skipif(not hasattr(torch.Tensor, 'refine_names'), reason='Torch build has no named tensors')
def test_named_axes_use_original_callable(cuda_device):
    with torch.no_grad(), replay(lambda x: x.sum('N')) as owner:
        x = torch.tensor([[1., 2.], [3., 4.]], device=cuda_device).refine_names('N', 'M')
        owner.prepare(x).freeze()
        renamed = x.rename('M', 'N')
        torch.testing.assert_close(owner(renamed).rename(None), renamed.sum('N').rename(None))
        assert owner.stats()['captures'] == 0
        assert owner.stats()['fallbacks']['metadata'] == 1


@pytest.mark.gpu
@pytest.mark.parametrize('enabled', [False, True])
def test_garbage_collection_retires_inflight_replay(cuda_device, enabled):
    import weakref
    with torch.no_grad():
        x = torch.ones(8192, device=cuda_device)
        owner = replay(lambda x: x + 3, enabled=enabled)
        owner.prepare(x).freeze()
        consumer = torch.cuda.Stream()
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(100_000_000)
            output = owner(x)
        reference = weakref.ref(owner)
        del owner
        assert reference() is None
        assert consumer.query(), 'owner destruction must retire in-flight private storage'
        torch.testing.assert_close(output, torch.full_like(x, 4))
