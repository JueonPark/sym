"""Caller-owned typed resources: lifetime failures and real changing transfers."""
from types import SimpleNamespace

import pytest
import torch

import common


@pytest.fixture
def fake_dispatch(monkeypatch):
    """Exercise caller cleanup even when native execution/close raises."""
    import reloc_torch
    from reloc_torch import dispatch
    from reloc_torch.compiler import CompilerClient

    state = SimpleNamespace(owners=[], calls=[], fail_execute=False, fail_close=False)
    monkeypatch.setattr(CompilerClient, "from_environment",
                        lambda: SimpleNamespace(compile=lambda recipe: object()))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dispatch, "prepare_typed_transfer", lambda *args, **kwargs: object())

    class Owner:
        def __init__(self, **kwargs):
            self.closed = False
            self.ordinal = len(state.owners)
            state.owners.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True
            if state.fail_close and self.ordinal == 1:
                raise RuntimeError("close failed")

        def stats(self):
            return {"typed": {"closed": self.closed}}

    def execute(request, *, resources):
        state.calls.append(resources)
        if state.fail_execute:
            raise RuntimeError("execution failed")
        return SimpleNamespace(tensor=torch.zeros(1), report={
            "implementation": "cpu_reference", "source_bytes": 1, "wire_bytes": 4,
            "destination_bytes": 4, "payload_bytes_transferred": 4})

    monkeypatch.setattr(reloc_torch, "TransferResources", Owner)
    monkeypatch.setattr(dispatch, "execute_typed_transfer", execute)
    return state


@pytest.mark.parametrize("policy", ["retained", "per-call"])
def test_canonical_device_ownership_and_per_call_control(fake_dispatch, policy):
    state = fake_dispatch
    report = common.Report("lifecycle")
    fetcher = common.WeightFetcher(report, resource_policy=policy)
    assert not state.owners
    with fetcher:
        for device in ("cuda", "cuda:0", "cuda:1", "cuda:0"):
            fetcher.fetch(torch.ones(1), None, device)
    if policy == "retained":
        assert len(state.owners) == 2
        assert state.calls == [state.owners[i] for i in (0, 0, 1, 0)]
        assert all(owner.closed for owner in state.owners)
    else:
        assert not state.owners and state.calls == [None] * 4
    assert report.data["weight_resources"]["policy"] == policy
    fetcher.close()  # idempotent; a closed fetcher can never recreate resources
    with pytest.raises(RuntimeError, match="closed"):
        fetcher.fetch(torch.ones(1), None, "cuda")


@pytest.mark.parametrize("failure", ["execution", "model", "close"])
def test_every_owner_closes_when_execution_model_or_close_fails(fake_dispatch, failure):
    state = fake_dispatch
    fetcher = common.WeightFetcher(common.Report("failure"))
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with fetcher:
            for device in ("cuda:0", "cuda:1"):
                fetcher.fetch(torch.ones(1), None, device)
            if failure == "execution":
                state.fail_execute = True
                fetcher.fetch(torch.ones(1), None, "cuda:0")
            elif failure == "model":
                raise RuntimeError("model failed")
            else:
                state.fail_close = True
    assert len(state.owners) == 2 and all(owner.closed for owner in state.owners)
    with pytest.raises(RuntimeError, match="closed"):
        fetcher.fetch(torch.ones(1), None, "cuda:0")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["retained", "per-call"])
@pytest.mark.parametrize("row", ["cpu_reference", "cuda_dequant_relocate"])
def test_warm_fetches_refresh_payloads_scales_shapes_and_streams(policy, row):
    report = common.Report("changing")
    fetcher = common.WeightFetcher(report, implementation=row, resource_policy=policy)
    streams = [torch.cuda.Stream(device=0) for _ in range(2)]
    outputs = []
    generator = torch.Generator().manual_seed(218)
    shapes = [(256, 512), (512, 256), (128, 96), (512, 256), (256, 512), (128, 96)]
    inputs = {shape: (torch.randint(-127, 128, shape, dtype=torch.int8, generator=generator),
                      torch.ones(shape[1])) for shape in shapes}
    with fetcher:
        for i, shape in enumerate(shapes):
            q, scale = inputs[shape]
            q.fill_(i - 3)  # same storages get new payloads and parameter values
            scale.fill_(2.0 ** (i - 2))
            expected = (q.float() * scale).t().contiguous()
            with torch.cuda.stream(streams[i % 2]):
                output = fetcher.fetch(q, scale, "cuda:0")
                consumer = output + 1
                actual = consumer.cpu()
            assert torch.equal(actual, expected + 1)
            outputs.append((output, expected))
            if policy == "retained":
                stats = fetcher.resource_stats()["cuda:0"]
                assert stats["retained_bytes"] <= fetcher.retained_bytes
                assert stats["context_creations"] == 1 and stats["requests"] == i + 1
                assert stats["hits"] == i
                if i == 1:
                    warm = stats
                elif i > 1:
                    for key in ("device_allocations", "host_allocations", "context_creations",
                                "streams", "background_workers"):
                        assert stats[key] == warm[key], (key, stats, warm)
        assert len({output.data_ptr() for output, _ in outputs}) == len(outputs)
    for output, expected in outputs:
        assert common.same_tensor(output.cpu(), expected)
    for stats in fetcher.resource_stats().values():
        assert stats["closed"] and stats["retained_bytes"] == stats["streams"] == stats["background_workers"] == 0
    assert report.data["bytes"]["weights"]["transfers"] == len(shapes)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_live_budget_failure_closes_the_native_owner():
    report = common.Report("limited")
    fetcher = common.WeightFetcher(report, implementation="cuda_dequant_relocate", live_bytes=1)
    with pytest.raises(RuntimeError, match="limit exceeded"):
        with fetcher:
            fetcher.fetch(torch.ones(64, 64, dtype=torch.int8), torch.ones(64), "cuda:0")
    stats = fetcher.resource_stats()["cuda:0"]
    assert stats["closed"] and not stats["quarantined"]
    assert stats["retained_bytes"] == stats["streams"] == stats["background_workers"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('policy', ['retained', 'per-call'])
def test_fetch_many_heterogeneous_consumer_groups_and_report(policy):
    report = common.Report('grouped')
    fetcher = common.WeightFetcher(report, implementation='cuda_dequant_relocate', resource_policy=policy)
    saved = []
    with fetcher:
        for iteration in range(3):
            scale = torch.full((32,), .5 * (iteration + 1))
            weights = [(torch.full((rows, 32), i + 1, dtype=torch.int8), scale)
                       for i, rows in enumerate([16, 32, 64])]
            outputs = fetcher.fetch_many(weights, 'cuda:0')
            for (q, s), out in zip(weights, outputs):
                expected = common.WeightFetcher.reference(q, s, 'cuda:0')
                assert common.same_tensor(out, expected)
                saved.append((out, expected))
        assert all(common.same_tensor(out, expected) for out, expected in saved)
        totals = report.data['transfer_groups']
        assert totals['groups'] == 3 and totals['logical_transfers'] == 9
        assert totals['parameter_uploads'] == 3 and totals['parameter_reuses'] == 6
        assert totals['event_waits'] == totals['caller_waits'] == 3
        assert report.data['bytes']['weights']['transfers'] == 9
        if policy == 'retained':
            assert fetcher.resource_stats()['cuda:0']['requests'] == 3
    with pytest.raises(RuntimeError, match='closed'):
        fetcher.fetch_many(weights, 'cuda:0')
