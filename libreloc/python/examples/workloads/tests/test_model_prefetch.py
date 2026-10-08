"""Real model consumers, bounded weight lifetime, routing and fresh parameters."""
import pytest
import torch
import common
from llm_offload import OffloadedGPT, generate
from moe_experts import OffloadedMoE

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')]


@pytest.mark.parametrize('row', ['cpu_reference', 'cuda_dequant_relocate'])
def test_decode_prefetch_preserves_tokens_logits_and_kv(row):
    device = torch.device('cuda:0')
    sizes = dict(vocab=128, d_model=64, heads=4, layers=3, d_ff=128, prompt=8, new_tokens=3)
    model = OffloadedGPT(sizes, device, torch.Generator().manual_seed(223))
    def evict(t):
        return t.transpose(0, 1).contiguous().cpu()
    def restore(t):
        return t.transpose(0, 1).contiguous().to(device)
    reference = lambda q, s: common.WeightFetcher.reference(q, s, device)
    report = common.Report('prefetch_llm')
    with common.WeightFetcher(report, implementation=row) as fetcher, torch.no_grad():
        for revision in range(2):
            # Same allocations, changed quantized values and parameter values.
            for layer in model.layers:
                for q, scale in layer.values():
                    q.neg_()
                    scale.mul_(.5)
            prompt = torch.arange(revision, revision + 8, device=device)
            expected = generate(model, prompt, 3, reference, evict, restore)
            actual = generate(model, prompt, 3, reference, evict, restore, prefetch=fetcher.prefetch)
            assert actual[0] == expected[0]
            assert common.same_tensor(actual[1], expected[1])
        stats = report.data['weight_prefetch']['cuda:0']
        assert stats['held'] == 0 and stats['peak_in_flight'] == 2
        assert stats['totals']['logical_transfers'] == 2 * 3 * 3 * 4
        assert report.data['bytes']['weights']['transfers'] == 72
    assert all(q.stats()['closed'] and q.stats()['resources']['retained_bytes'] == 0
               for q in fetcher._queues.values())


def test_moe_prefetch_waits_for_routing_and_never_loads_inactive_experts():
    device = torch.device('cuda:0')
    model = OffloadedMoE(dict(d_model=64, d_ff=128, experts=8, blocks=2), [device],
                        torch.Generator().manual_seed(223))
    report = common.Report('prefetch_moe')
    with common.WeightFetcher(report, implementation='cuda_dequant_relocate') as fetcher, torch.no_grad():
        for tokens in (1, 32):
            x = torch.randn(tokens, 64, device=device)
            expected_routes, actual_routes = [], []
            expected = model.forward(x, common.WeightFetcher.reference, expected_routes.append)
            before = sum(q.stats()['submitted'] for q in fetcher._queues.values())
            actual = model.forward(x, common.WeightFetcher.reference, actual_routes.append,
                                   prefetch=fetcher.prefetch)
            after = sum(q.stats()['submitted'] for q in fetcher._queues.values())
            assert expected_routes == actual_routes
            assert after - before == sum(map(len, actual_routes))
            if tokens == 1:
                assert all(len(active) == 2 for active in actual_routes)
            assert common.same_tensor(actual, expected)
        model.devices = [device, torch.device('cuda:1')]
        with pytest.raises(ValueError, match='one CUDA device'):
            model.forward(x, None, lambda _: None, prefetch=fetcher.prefetch)
