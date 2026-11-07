"""Physical-budget estimates and output storage must not shorten consumer lifetimes."""
import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from reloc_torch.runtime import ConcreteDescriptor


def descriptor(shape, dtype="float32", device="cpu"):
    strides, stride = [], 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return ConcreteDescriptor(tuple(shape), tuple(reversed(strides)), dtype, torch.device(device))


def test_plan_accounts_for_consumer_group_and_reduces_depth():
    from reloc_torch.memory import plan_transfer_memory
    # Two-slot live envelope: 3*100 output + 2*20 scratch + 30 workspace = 370.
    deep = plan_transfer_memory([100] * 4, [20] * 4, available_bytes=370,
                                workspace_bytes=30, safety_bytes=0)
    assert (deep.slots, deep.contexts, deep.output_bytes, deep.predicted_device_bytes) == (2, 2, 300, 370)
    shallow = plan_transfer_memory([100] * 4, [20] * 4, available_bytes=369,
                                   workspace_bytes=30, safety_bytes=0)
    assert (shallow.slots, shallow.output_bytes, shallow.predicted_device_bytes) == (1, 200, 250)


def test_plan_resident_prefix_uses_actual_group_sizes_and_external_holdings():
    from reloc_torch.memory import plan_transfer_memory
    plan = plan_transfer_memory([40, 60, 100, 100], [5, 10, 20, 20], available_bytes=380,
                                workspace_bytes=30, retained_bytes=10, safety_bytes=20,
                                max_resident_groups=2)
    assert (plan.resident_groups, plan.slots, plan.resident_bytes) == (2, 1, 100)
    assert plan.predicted_device_bytes == 380
    assert plan.headroom_bytes == 0


def test_plan_can_reduce_group_scratch_without_claiming_smaller_outputs():
    from reloc_torch.memory import plan_transfer_memory
    plan = plan_transfer_memory([100, 100], [40, 40], subgroup_scratch_bytes=[10, 10],
                                available_bytes=320, workspace_bytes=0, safety_bytes=0)
    assert (plan.slots, plan.subgroup_members, plan.output_bytes) == (2, 1, 300)
    assert plan.predicted_device_bytes == 320


def test_plan_has_no_transfer_resources_when_fully_resident_or_empty():
    from reloc_torch.memory import plan_transfer_memory
    plan = plan_transfer_memory([40, 60], [10, 20], available_bytes=120,
                                workspace_bytes=20, safety_bytes=0, max_resident_groups=2)
    assert (plan.resident_groups, plan.slots, plan.contexts, plan.output_bytes) == (2, 0, 0, 0)
    assert plan.predicted_device_bytes == 120
    empty = plan_transfer_memory([], [], available_bytes=20, workspace_bytes=20, safety_bytes=0)
    assert (empty.resident_groups, empty.slots, empty.predicted_device_bytes) == (0, 0, 20)


def test_plan_refuses_infeasible_storage_before_execution():
    from reloc_torch.memory import MemoryBudgetError, plan_transfer_memory
    with pytest.raises(MemoryBudgetError):
        plan_transfer_memory([100], [20], available_bytes=219, workspace_bytes=0, safety_bytes=0)


def test_plan_without_device_scratch_still_supplies_valid_native_quota():
    from reloc_torch.memory import plan_transfer_memory
    plan = plan_transfer_memory([100], [0], available_bytes=200, workspace_bytes=0, safety_bytes=0)
    assert plan.device_scratch_per_context == 0
    assert plan.native_scratch_bytes == 1  # Native resources require a positive limit.


@pytest.mark.parametrize("changed", [
    {"available_bytes": True}, {"available_bytes": -1}, {"workspace_bytes": -1},
    {"available_bytes": 1 << 64}, {"max_slots": 0}, {"max_slots": 100},
    {"max_resident_groups": 2}, {"retained_bytes": -1}, {"safety_bytes": -1},
    {"group_output_bytes": [0]}, {"group_scratch_bytes": []},
    {"subgroup_scratch_bytes": [21]},
])
def test_plan_rejects_invalid_or_unrepresentable_requirements(changed):
    from reloc_torch.memory import plan_transfer_memory
    kwargs = dict(group_output_bytes=[100], group_scratch_bytes=[20], available_bytes=500,
                  workspace_bytes=0, safety_bytes=0)
    kwargs.update(changed)
    with pytest.raises((TypeError, ValueError)):
        plan_transfer_memory(**kwargs)


def test_packed_outputs_are_aligned_disjoint_and_preserve_dtype():
    from reloc_torch.memory import PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=768) as arena:
        outputs = arena.allocate([descriptor((3, 5)), descriptor((7,), "float16"), descriptor((2,), "int64")])
        first, second, third = outputs
        first.fill_(1.5); second.fill_(2.5); third.fill_(7)
        assert [t.dtype for t in outputs] == [torch.float32, torch.float16, torch.int64]
        assert [t.data_ptr() - first.data_ptr() for t in outputs] == [0, 256, 512]
        assert all(t.untyped_storage()._cdata == first.untyped_storage()._cdata for t in outputs)
        assert torch.equal(first, torch.full((3, 5), 1.5))
        assert torch.equal(second, torch.full((7,), 2.5, dtype=torch.float16))
        assert torch.equal(third, torch.full((2,), 7, dtype=torch.int64))
        assert arena.stats()["allocated_bytes"] == 768


def test_arena_does_not_overwrite_detached_alias_after_tensor_release():
    from reloc_torch.memory import MemoryBudgetError, PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=256) as arena:
        outputs = arena.allocate([descriptor((16,))])
        outputs[0].fill_(19)
        alias = outputs[0].detach()[3:]
        pointer = outputs[0].data_ptr()
        del outputs
        gc.collect()
        with pytest.raises(MemoryBudgetError):
            arena.allocate([descriptor((16,))])
        assert torch.equal(alias, torch.full((13,), 19.0))
        del alias
        gc.collect()
        replacement = arena.allocate([descriptor((16,))])
        assert replacement[0].data_ptr() == pointer
        assert arena.stats()["reuses"] == 1


def test_arena_evicts_only_expired_blocks_for_different_size():
    from reloc_torch.memory import MemoryBudgetError, PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=768) as arena:
        held = arena.allocate([descriptor((1,))])
        old = arena.allocate([descriptor((1,))])
        del old
        large = arena.allocate([descriptor((100,))])
        assert arena.stats()["allocated_bytes"] == 768
        held[0].fill_(17)
        with pytest.raises(MemoryBudgetError):
            arena.allocate([descriptor((1,))])
        assert held[0].item() == 17
        assert large[0].numel() == 100


def test_packing_without_recycling_releases_expired_capacity():
    from reloc_torch.memory import PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=256, reuse=False) as arena:
        first = arena.allocate([descriptor((16,))])
        del first
        second = arena.allocate([descriptor((16,))])
        assert second[0].numel() == 16
        assert arena.stats()["allocations"] == 2
        assert arena.stats()["reuses"] == 0


def test_packing_only_does_not_report_expired_storage_as_retained():
    from reloc_torch.memory import PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=256, reuse=False) as arena:
        values = arena.allocate([descriptor((16,))])
        del values
        gc.collect()
        stats = arena.stats()
        assert (stats["allocated_bytes"], stats["live_bytes"], stats["reusable_bytes"]) == (0, 0, 0)


@pytest.mark.gpu
def test_packing_only_uses_normal_allocator_lifetime_without_recycling_barrier(cuda_device):
    from reloc_torch.memory import PackedOutputArena
    with PackedOutputArena(cuda_device, max_bytes=1024, reuse=False) as arena:
        first = arena.allocate([descriptor((256,), device=cuda_device)])
        first[0].fill_(31)
        del first
        second = arena.allocate([descriptor((256,), device=cuda_device)])
        assert second[0].numel() == 256
        assert arena.stats()["completion_barriers"] == 0


@pytest.mark.parametrize("bad", [
    ConcreteDescriptor((3, 4), (1, 3), "float32", torch.device("cpu")),
    descriptor((0,)), descriptor((-1,)), descriptor((1 << 62,)),
    ConcreteDescriptor((3,), (1,), "unknown", torch.device("cpu")),
])
def test_arena_rejects_unsupported_metadata_without_consuming_capacity(bad):
    from reloc_torch.memory import PackedOutputArena
    with PackedOutputArena("cpu", max_bytes=256) as arena:
        with pytest.raises((TypeError, ValueError)):
            arena.allocate([bad])
        assert arena.stats()["allocated_bytes"] == 0
        assert arena.allocate([descriptor((1,))])[0].shape == (1,)


def test_close_preserves_held_outputs_and_rejects_new_submissions():
    from reloc_torch.memory import PackedOutputArena
    arena = PackedOutputArena("cpu", max_bytes=256)
    values = arena.allocate([descriptor((16,))]); values[0].fill_(23)
    alias = values[0][4:]
    arena.close(); arena.close()
    del arena, values
    gc.collect()
    assert torch.equal(alias, torch.full((12,), 23.0))
    retired = PackedOutputArena("cpu", max_bytes=256); retired.close()
    with pytest.raises(RuntimeError, match="closed"):
        retired.allocate([descriptor((1,))])


@pytest.mark.gpu
def test_recycling_waits_for_released_cross_stream_consumer(cuda_device):
    from reloc_torch.memory import PackedOutputArena
    spec = descriptor((256,), device=cuda_device)
    with PackedOutputArena(cuda_device, max_bytes=1024) as arena:
        outputs = arena.allocate([spec]); outputs[0].fill_(13)
        torch.cuda.synchronize()
        consumer = torch.cuda.Stream(device=cuda_device)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(5_000_000)
            observed = outputs[0].clone()
        pointer = outputs[0].data_ptr()
        del outputs
        gc.collect()
        replacement = arena.allocate([spec])
        replacement[0].fill_(99)
        assert replacement[0].data_ptr() == pointer
        consumer.synchronize()
        assert torch.equal(observed.cpu(), torch.full((256,), 13.0))


@pytest.mark.gpu
@pytest.mark.parametrize("explicit_close", [True, False])
def test_after_retirement_record_stream_preserves_future_consumer(cuda_device, explicit_close):
    from reloc_torch.memory import PackedOutputArena
    arena = PackedOutputArena(cuda_device, max_bytes=1 << 20)
    outputs = arena.allocate([descriptor((1 << 18,), device=cuda_device)])
    outputs[0].fill_(29)
    alias = outputs[0].detach()
    if explicit_close:
        arena.close()
    del arena, outputs
    gc.collect()
    consumer = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(10_000_000)
        observed = alias.clone()
    alias.record_stream(consumer)
    del alias
    gc.collect()
    # Allocations on the original/default stream may otherwise recycle the address.
    pressure = [torch.full((1 << 18,), float(i), device=cuda_device) for i in range(16)]
    consumer.synchronize()
    assert torch.equal(observed.cpu(), torch.full((1 << 18,), 29.0))
    assert len(pressure) == 16


@pytest.mark.gpu
def test_uncertain_cuda_completion_reports_and_quarantines_backing(cuda_device):
    scenario = Path(__file__).with_name("memory_fault_scenario.py")
    process = subprocess.run([sys.executable, str(scenario)], text=True, capture_output=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
    assert '"quarantined_bytes": 1048576' in process.stdout


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
def test_arena_rejects_inherited_use_before_lock_or_cuda():
    from reloc_torch.memory import PackedOutputArena
    arena = PackedOutputArena("cpu", max_bytes=256)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            arena.allocate([descriptor((1,))])
        except RuntimeError:
            os.write(write_fd, b"rejected")
        finally:
            os._exit(0)
    os.close(write_fd)
    result = os.read(read_fd, 32); os.close(read_fd)
    os.waitpid(pid, 0); arena.close()
    assert result == b"rejected"
