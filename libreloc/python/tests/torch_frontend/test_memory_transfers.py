"""Real transfers and iterator backpressure over bounded output ownership."""
import gc

import pytest
import torch

from reloc_torch import PreparedWireWeights
from reloc_torch.fast import FastPrefetch, FastTransfers, FastWeights, KVCacheSwap
from reloc_torch.memory import MemoryBudgetError, PackedOutputArena
from test_fast import dlrm_recipe
from test_memory import descriptor


@pytest.fixture(autouse=True)
def packed_bridge():
    from reloc_torch.fast import _load_native_views
    bridge = _load_native_views()
    if bridge is None or not hasattr(bridge, 'PackedStorage'):
        pytest.skip('packed storage requires the optional native Torch bridge')


@pytest.mark.gpu
@pytest.mark.parametrize("prepared", [False, True])
def test_fast_groups_pack_live_outputs_without_overwriting_aliases(compiler, cuda_device, prepared):
    arena = PackedOutputArena(cuda_device, max_bytes=2048)
    owner = FastTransfers(cuda_device, compiler=compiler, gather_threads=1,
                          prepared_groups=prepared, output_arena=arena)
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    program = owner.typed(dlrm_recipe(2, 4), source)
    first = owner.submit([(program, source), (program, source + 1)]).wait()
    assert first[0].untyped_storage()._cdata == first[1].untyped_storage()._cdata
    alias = first[0].detach()[1:]
    reference = alias.clone()
    second = owner.submit([(program, source + 2), (program, source + 3)]).wait()
    assert first[0].data_ptr() != second[0].data_ptr()
    assert torch.equal(second[0], (source + 2).transpose(0, 1).contiguous().half().to(cuda_device))
    assert torch.equal(alias, reference)
    assert owner.stats()["output_arena"]["live_bytes"] == 1024
    owner.close()
    assert not arena.stats()["closed"]  # The owner borrows the arena.
    arena.close()
    assert torch.equal(alias, reference)


@pytest.mark.gpu
def test_explicit_prepared_handle_uses_the_same_output_budget(compiler, cuda_device):
    source = torch.ones(2, 1, 4)
    with PackedOutputArena(cuda_device, max_bytes=256) as arena:
        with FastTransfers(cuda_device, compiler=compiler, gather_threads=1, output_arena=arena) as owner:
            handle = owner.prepare_group([owner.typed(dlrm_recipe(2, 4), source)])
            output = handle.submit([source]).wait()
            pointer = output[0].data_ptr()
            with pytest.raises(MemoryBudgetError):
                handle.submit([source + 1])
            assert owner.stats()["groups"] == 1
            del output
            gc.collect()
            new = handle.submit([source + 2]).wait()
            assert new[0].data_ptr() == pointer and torch.all(new[0] == 3)


@pytest.mark.gpu
def test_output_alias_validation_precedes_any_group_launch(compiler, cuda_device):
    source = torch.ones(2, 1, 4)
    with PackedOutputArena(cuda_device, max_bytes=512) as arena:
        with FastTransfers(cuda_device, compiler=compiler, output_arena=arena) as owner:
            program = owner.typed(dlrm_recipe(2, 4), source)
            shared = torch.full((1, 2, 4), 43., dtype=torch.float16, device=cuda_device)
            with pytest.raises(RuntimeError, match="group_alias"):
                owner.submit([(program, source)] * 2, outputs=[shared, shared])
            assert owner.stats()["groups"] == 0
            assert torch.all(shared == 43)
            with pytest.raises((ValueError, RuntimeError)):
                owner.submit([(program, source.double())])
            assert owner.stats()["groups"] == 0


@pytest.mark.gpu
def test_arena_rejects_implicit_d2h_but_explicit_cpu_outputs_remain_valid(compiler, cuda_device):
    with PackedOutputArena(cuda_device, max_bytes=512) as arena:
        with FastTransfers(cuda_device, compiler=compiler, output_arena=arena) as owner:
            swap = KVCacheSwap(owner, 4, 8)
            source = torch.randn(4, 5, 8, device=cuda_device)
            program = swap._program(swap._evict_recipe, source)
            with pytest.raises(ValueError, match="device|CUDA"):
                owner.submit([(program, source)], device_layout=True)
            assert owner.stats()["groups"] == 0
            cpu = torch.empty(5, 4, 8)
            actual = owner.submit([(program, source)], outputs=[cpu], device_layout=True).wait()[0]
            assert actual is cpu
            assert torch.equal(actual, source.transpose(0, 1).contiguous().cpu())


class ImmediateCompletion:
    def __init__(self, outputs):
        self.outputs = outputs

    def done(self):
        return True

    def wait(self):
        return self.outputs

    def wait_stream(self):
        return self.outputs


class CpuProducer:
    """A real bounded arena producer for the iterator's generic owner interface."""
    def __init__(self, arena):
        self.arena = arena

    def submit(self, number, **options):
        outputs = self.arena.allocate([descriptor((1,))])
        outputs[0].fill_(number)
        return ImmediateCompletion(outputs)


def test_prefetch_defers_speculation_without_dropping_or_reordering_groups():
    with PackedOutputArena("cpu", max_bytes=512) as arena:
        with FastPrefetch(CpuProducer(arena), range(20), slots=2) as window:
            observed = []
            for values in window:
                observed.append(values[0].item())
            assert observed == list(range(20))
        assert arena.stats()["refusals"] > 0
        assert arena.stats()["allocated_bytes"] <= 512


def test_prefetch_reports_held_output_exhaustion_without_waiting_forever():
    with PackedOutputArena("cpu", max_bytes=512) as arena:
        with FastPrefetch(CpuProducer(arena), range(20), slots=2) as window:
            first, second = next(window), next(window)
            with pytest.raises(MemoryBudgetError):
                next(window)
            assert (first[0].item(), second[0].item()) == (0, 1)


def prepare_weights(weights):
    q = torch.arange(128).remainder(97).to(torch.int8).reshape(8, 16)
    for i in range(3):
        weights.prepare(str(i), q + i, torch.ones(16))
    return q


@pytest.mark.gpu
def test_weight_subgroups_use_packed_outputs_and_current_snapshot_revisions(compiler, cuda_device):
    with PackedOutputArena(cuda_device, max_bytes=4096) as arena:
        with FastTransfers(cuda_device, compiler=compiler, gather_threads=1) as owner:
            with PreparedWireWeights(cuda_device, compiler=compiler, dtype="float16") as weights:
                q = prepare_weights(weights)
                loader = FastWeights(owner, weights, output_arena=arena, max_group_members=1,
                                     max_prefetch_slots=1)
                first = loader.submit(["0", "1", "2"])
                values = first.wait()
                assert len(first.report["subgroups"]) == 3
                assert owner.stats()["groups"] == owner.stats()["logical_transfers"] == 3
                assert len({x.untyped_storage()._cdata for x in values}) == 1
                for i, value in enumerate(values):
                    assert torch.equal(value, (q.float() + i).t().contiguous().half().to(cuda_device))
                weights.prepare("0", q, torch.full((16,), 2.))
                updated = loader.submit(["0"]).wait()[0]
                assert torch.equal(updated, (q.float() * 2).t().contiguous().half().to(cuda_device))
                assert torch.equal(values[0], q.float().t().contiguous().half().to(cuda_device))
                assert loader.stats()["output_arena"]["live_bytes"] > 0
                assert owner.stats()["output_arena"] is None


@pytest.mark.gpu
def test_weight_subgroup_validates_all_destinations_before_first_launch(compiler, cuda_device):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=1) as owner:
        with PreparedWireWeights(cuda_device, compiler=compiler, dtype="float16") as weights:
            q = prepare_weights(weights)
            loader = FastWeights(owner, weights, max_group_members=1)
            outputs = [torch.full((16, 8), 37., dtype=torch.float16, device=cuda_device) for _ in range(3)]
            with pytest.raises(RuntimeError, match="group_alias"):
                loader.submit(["0", "1", "2"], outputs=[outputs[0], outputs[1], outputs[0]])
            assert owner.stats()["groups"] == 0 and torch.all(outputs[0] == 37)
            result = loader.submit(["0", "1", "2"], outputs=outputs).wait()
            assert result[0] is outputs[0]
            assert torch.equal(result[0], q.float().t().contiguous().half().to(cuda_device))


@pytest.mark.gpu
def test_partial_subgroup_failure_drains_prior_work_without_replay(compiler, cuda_device, monkeypatch):
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=1) as owner:
        with PreparedWireWeights(cuda_device, compiler=compiler, dtype="float16") as weights:
            prepare_weights(weights)
            loader = FastWeights(owner, weights, max_group_members=1)
            submit = owner.submit
            calls = []
            def fail_second(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected second-group failure")
                return submit(*args, **kwargs)
            monkeypatch.setattr(owner, "submit", fail_second)
            with pytest.raises(RuntimeError, match="second-group failure"):
                loader.submit(["0", "1", "2"])
            assert len(calls) == 2
            assert owner.stats()["groups"] == 1
