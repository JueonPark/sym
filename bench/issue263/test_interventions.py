"""Ablations must preserve live data, revisions, stream dependencies and outputs."""
import pytest
import torch

from bench.issue263.instrument import Instrumentation
from bench.issue263.interventions import Intervention
from reloc_torch import PreparedWireWeights
from reloc_torch import fast


def test_signature_bypass_requires_explicit_qualified_metadata_contract():
    with pytest.raises(ValueError, match='qualified metadata'):
        with Intervention('no_check'):
            pass


def test_interventions_restore_callables_after_failure():
    finish, check = fast.FastCompletion._finish, fast._Program.check
    with pytest.raises(RuntimeError, match='intentional'):
        with Intervention('both', qualified_metadata=True):
            raise RuntimeError('intentional')
    assert fast.FastCompletion._finish is finish
    assert fast._Program.check is check


@pytest.mark.parametrize('variant', ['baseline', 'no_report', 'no_check', 'both'])
@pytest.mark.parametrize('mode', ['off', 'counters'])
def test_live_kv_revisions_and_held_outputs_survive_diagnostic_bypass(variant, mode):
    import controls
    torch.manual_seed(263)
    held = []
    with Intervention(variant, qualified_metadata=True), Instrumentation(mode) as meter:
        meter.install(controls)
        with fast.FastTransfers(gather_threads=1, contexts=2) as owner, PreparedWireWeights(dtype='float16') as weights:
            q = torch.randint(-127, 128, (13, 7), dtype=torch.int8)
            scales = torch.rand(7)
            producer = fast.FastWeights(owner, weights)
            swap = fast.KVCacheSwap(owner, 2, 6, dtype='float16')
            for revision in range(2):
                if revision:
                    q.neg_()
                    scales.mul_(1.5)
                weights.prepare('weight', q, scales)
                expected = (q.float() * scales).t().contiguous().half()
                with meter.request(f'weights/{revision}'):
                    with producer.prefetch([['weight']] * 3) as loaded:
                        outputs = [next(loaded)[0] for _ in range(3)]
                    torch.cuda.synchronize()
                for output in outputs:
                    assert torch.equal(output.cpu(), expected)
                    held.append((output, expected.clone()))
                caches = [torch.randn(2, 5 + revision, 6, device='cuda').half() for _ in range(2)]
                with meter.request(f'kv/{revision}'):
                    restored = swap.restore(swap.evict(caches))
                    torch.cuda.synchronize()
                for actual, expected_cache in zip(restored, caches):
                    assert torch.equal(actual, expected_cache)
                    held.append((actual, expected_cache.cpu()))
            swap.drain()
        torch.cuda.synchronize()
        for output, expected in held:
            assert torch.equal(output.cpu(), expected)
        resources = owner._resources.stats()['typed']
        assert resources['retained_bytes'] == resources['live_bytes'] == 0
        assert resources['streams'] == resources['background_workers'] == 0
    if mode == 'counters':
        assert len(meter.records) == 4
        assert all(sum(r['exclusive_ns'].values()) == r['elapsed_ns'] for r in meter.records)
        assert sum(len(r['groups']) for r in meter.records) == 10
