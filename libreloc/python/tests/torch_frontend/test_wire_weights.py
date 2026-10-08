"""Owned compact snapshots: exact arithmetic, replacement, budgets and retirement."""
import gc
import weakref

import pytest
import torch

from reloc_torch import PreparedWireWeights


def inputs(rows=71, cols=29):
    q = torch.arange(rows * cols).remainder(256).sub(128).to(torch.int8).reshape(rows, cols)
    s = torch.linspace(.001, .25, cols)
    return q, s


def expected(q, s):
    return (q.float() * s).t().contiguous()


def owner(compiler, **kwargs):
    return PreparedWireWeights(compiler=compiler, pin_memory=False, **kwargs)


def test_preparation_owns_only_snapshot_and_no_public_tensor_alias(compiler):
    q, s = inputs()
    qr, sr = weakref.ref(q), weakref.ref(s)
    with owner(compiler) as weights:
        info = weights.prepare('w', q, s)
        assert info['shape'] == (29, 71)
        assert info['wire_bytes'] == q.numel()
        assert info['payload_bytes'] == q.numel() + s.numel() * 4
        assert info['prepared_bytes'] == q.numel() + s.numel() * 8 + 4
        assert info['pinned_bytes'] == 0
        assert not any(isinstance(v, torch.Tensor) for v in info.values())
        with pytest.raises(TypeError):
            info['shape'] = (1,)
        del q, s
        gc.collect()
        assert qr() is None and sr() is None
        assert weights.describe('w') is info
        weights.invalidate('w')
        assert weights.stats()['prepared_bytes'] == 0
        with pytest.raises(KeyError):
            weights.describe('w')
    assert weights.stats()['closed']


def test_atomic_replacement_capacity_and_failed_binding(compiler):
    q, s = inputs()
    size = q.numel() + s.numel() * 8 + 4
    with owner(compiler, max_prepared_bytes=size) as weights:
        first = weights.prepare('w', q, s)
        with pytest.raises(BufferError):
            weights.prepare('w', q, s)
        assert weights.describe('w') is first
        weights.invalidate()
        assert weights.prepare('w', q, s)['revision'] == 2
    with owner(compiler, max_prepared_bytes=2*size) as weights:
        first = weights.prepare('w', q, s)
        for invalid in (s * 0, s * float('nan'), s * float('inf'), -s):
            with pytest.raises(Exception, match='scale'):
                weights.prepare('w', q, invalid)
            assert weights.describe('w') is first
            assert weights.stats()['prepared_bytes'] == size
        second = weights.prepare('w', q, s * 2)
        assert second['revision'] == 2
        assert weights.stats()['prepared_bytes'] == size
        assert weights.stats()['peak_prepared_bytes'] == size * 2


@pytest.mark.parametrize('change', [lambda q,s:(q.float(),s), lambda q,s:(q.t(),s),
    lambda q,s:(q,s.double()), lambda q,s:(q,s[:-1]), lambda q,s:(q,s.requires_grad_()),
    lambda q,s:(q[:0],s)])
def test_rejects_unsupported_representation_without_publication(compiler, change):
    with owner(compiler) as weights:
        with pytest.raises(ValueError):
            weights.prepare('bad', *change(*inputs()))
        assert weights.stats()['entries'] == weights.stats()['prepared_bytes'] == 0


def test_entry_budget_close_and_process_guard(compiler, monkeypatch):
    weights = owner(compiler, max_entries=1)
    weights.prepare('w', *inputs())
    with pytest.raises(BufferError):
        weights.prepare('next', *inputs())
    from reloc_torch import wire_weights
    with monkeypatch.context() as m:
        m.setattr(wire_weights.os, 'getpid', lambda: -1)
        for call in (weights.stats, weights.close, lambda: weights.describe('w')):
            with pytest.raises(RuntimeError, match='another process'):
                call()
    weights.close()
    weights.close()
    assert weights.stats()['prepared_bytes'] == 0
    for call in (lambda: weights.prepare('w', *inputs()), weights.invalidate,
                 lambda: weights.describe('w'), lambda: weights.submit(['w'])):
        with pytest.raises(RuntimeError, match='closed'):
            call()


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(1, 1), (1, 129), (127, 1), (71, 29), (1024, 513)])
def test_exact_layout_channel_extremes_fresh_outputs_and_direct_wire(compiler, cuda_device, shape):
    q, s = inputs(*shape)
    with PreparedWireWeights(cuda_device, compiler=compiler) as weights:
        weights.prepare('w', q, s)
        first = weights.load_many(['w'])
        second = weights.load('w')
        assert torch.equal(first.tensors[0].cpu(), expected(q, s))
        assert torch.equal(second.cpu(), expected(q, s))
        assert second.shape == (shape[1], shape[0]) and second.is_contiguous()
        assert second.dtype == torch.float32
        assert first.tensors[0].data_ptr() != second.data_ptr()
        first.tensors[0].zero_()
        assert torch.equal(weights.load('w').cpu(), expected(q, s))
        report = first.report
        assert report['host_transform_bytes'] == report['packing_bytes'] == 0
        assert report['items'][0]['host_pipeline'] == 'direct_dense'
        assert report['items'][0]['wire_bytes'] == q.numel()
        assert report['items'][0]['payload_bytes_transferred'] == q.numel() + s.numel()*4
        assert weights.stats()['pinned_bytes'] == q.numel()
    assert weights.stats()['prepared_bytes'] == weights.stats()['pinned_bytes'] == 0


@pytest.mark.gpu
def test_mutation_aliases_replacement_state_dict_scale_require_explicit_reprepare(compiler, cuda_device, monkeypatch):
    q, s = inputs()
    module = torch.nn.Module()
    module.register_buffer('q', q)
    module.register_buffer('s', s)
    old = expected(q, s)
    with PreparedWireWeights(cuda_device, compiler=compiler) as weights:
        weights.prepare('w', module.q, module.s)
        module.q.data.fill_(7)
        module.s.numpy()[:] = .5
        module.q = torch.full_like(module.q, 17)
        module.load_state_dict({'q': torch.full_like(q, -29), 's': torch.full_like(s, .25)})
        # No tensor-value snapshot/equality helper is permitted on reuse.
        from reloc_torch import dispatch
        with monkeypatch.context() as m:
            m.setattr(dispatch, '_parameter_snapshot', lambda *a: pytest.fail('scanned values on reuse'))
            assert torch.equal(weights.load('w').cpu(), old)
        weights.prepare('w', module.q, module.s)
        assert torch.equal(weights.load('w').cpu(), expected(module.q, module.s))


@pytest.mark.gpu
def test_invalidated_inflight_revision_remains_charged_until_completion(compiler, cuda_device):
    q, s = inputs(1024, 1024)
    size = q.numel() + s.numel()*8 + 4
    with PreparedWireWeights(cuda_device, compiler=compiler, max_prepared_bytes=size,
                             max_pinned_bytes=q.numel()) as weights:
        weights.prepare('w', q, s)
        handle = weights.submit(['w'])
        weights.invalidate('w')
        assert weights.stats()['entries'] == 0
        assert weights.stats()['prepared_bytes'] == size
        with pytest.raises(BufferError):
            weights.prepare('w', q, s)
        assert torch.equal(handle.wait().tensors[0].cpu(), expected(q, s))
        handle.close()
        assert weights.stats()['prepared_bytes'] == 0
        weights.prepare('w', q, s)
        old = weights.submit(['w'])
        weights.close()
        assert weights.stats()['prepared_bytes'] == 0
        with pytest.raises(RuntimeError, match='closed'):
            old.wait()


@pytest.mark.gpu
def test_replacement_with_outstanding_revision_and_bounded_prefetch(compiler, cuda_device):
    q, s = inputs()
    with PreparedWireWeights(cuda_device, compiler=compiler, max_output_bytes=q.numel()*8) as weights:
        weights.prepare('w', q, s)
        old = weights.submit(['w'])
        weights.prepare('w', q, s*2)
        assert torch.equal(old.wait().tensors[0].cpu(), expected(q, s))
        old.close()
        with weights.prefetch([['w']] * 4) as window:
            outputs = [(tensor + 1) for (tensor,) in window]
        assert all(torch.equal(value.cpu(), expected(q, s*2) + 1) for value in outputs)
        stats = weights.stats()['queue']
        assert stats['held'] == 0 and stats['peak_in_flight'] == 2
        assert stats['peak_output_bytes'] == q.numel()*8
        with weights.prefetch([['w']] * 3) as window:
            next(window)
        assert weights.stats()['queue']['held'] == 0


@pytest.mark.gpu
@pytest.mark.parametrize('mode', [1, 2, 3, 4, 5, 6])
def test_failed_wire_upload_releases_or_quarantines_charged_snapshot(mode, cuda_device):
    from pathlib import Path
    import os
    import subprocess
    import sys
    import pyreloc
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    env = dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('async_fault_scenario.py')),
                             str(mode), 'wire'], env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout
