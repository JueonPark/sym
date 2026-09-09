import dataclasses
import gc
import weakref

import pytest
import torch

from reloc_torch import observe_transfers
from reloc_torch.eligibility import classify


def _records(inventory, operator):
    return [record for record in inventory.records if record.operator == operator]


def test_observer_preserves_copy_noop_cast_alias_and_values():
    x = torch.arange(6, dtype=torch.float32)
    with observe_transfers() as inventory:
        same = x.to("cpu")
        copied = x.to("cpu", copy=True)
        casted = x.to(torch.float16)
        viewed = x.view(2, 3)
    assert same is x
    assert copied.data_ptr() != x.data_ptr()
    assert torch.equal(copied, x)
    assert casted.dtype == torch.float16
    assert viewed.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    copies = _records(inventory, "aten._to_copy.default")
    assert len(copies) == 2
    assert copies[0].aliases_source is False
    assert _records(inventory, "aten.view.default")[0].aliases_source is True


def test_phase_nested_modes_cleanup_and_frozen_records():
    x = torch.arange(4)
    outer = observe_transfers()
    inner = observe_transfers()
    with outer:
        with outer.phase("weights"):
            x.to("cpu", copy=True)
        with inner:
            x.to(torch.float32)
        x.to(torch.float64)
    assert outer.records[0].phase == "weights"
    assert len(outer.records) == 3
    assert len(inner.records) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        outer.records[0].phase = "changed"
    with pytest.raises(RuntimeError, match="boom"):
        with observe_transfers():
            raise RuntimeError("boom")
    assert torch.equal(x + 1, torch.arange(1, 5))


def test_mutation_preserves_destination_identity_and_clears_history():
    source = torch.arange(6).view(2, 3).transpose(0, 1)
    destination = torch.empty_like(source)
    with observe_transfers() as inventory:
        result = destination.copy_(source)
    assert result is destination
    assert torch.equal(destination, source)
    record = _records(inventory, "aten.copy_.default")[0]
    assert record.mutates
    assert record.source.shape == tuple(source.shape)
    assert record.destination.shape == tuple(destination.shape)
    assert record.layout_history == ()


def test_schema_binding_handles_out_and_positional_copy_options():
    left = torch.arange(4, dtype=torch.float32)
    right = torch.ones(4)
    out = torch.empty(4)
    copied = torch.empty(4)
    with observe_transfers() as inventory:
        result = torch.add(left, right, out=out)
        copy_result = copied.copy_(left, True)
    assert result is out
    assert copy_result is copied
    assert torch.equal(out, left + right)
    add_record = _records(inventory, "aten.add.out")[0]
    copy_record = _records(inventory, "aten.copy_.default")[0]
    assert add_record.mutates and copy_record.mutates
    assert copy_record.non_blocking is True
    assert copy_record.source.shape == tuple(left.shape)
    assert copy_record.destination.shape == tuple(copied.shape)


def test_records_do_not_keep_tensors_alive_and_history_is_bounded():
    with observe_transfers() as inventory:
        tensor = torch.arange(16)
        for _ in range(40):
            tensor = tensor.view(4, 4).transpose(0, 1)
        tensor = tensor.to("cpu", copy=True)
        reference = weakref.ref(tensor)
    assert len(inventory.records[-1].layout_history) <= 16
    del tensor
    gc.collect()
    assert reference() is None


def test_unrelated_arithmetic_does_not_inherit_layout_provenance():
    source = torch.arange(8)
    with observe_transfers() as inventory:
        view = source.view(2, 4)
        computed = view + 1
        computed.to("cpu", copy=True)
    assert inventory.records[-1].operator == "aten._to_copy.default"
    assert inventory.records[-1].layout_history == ()


def test_parameter_grad_sparse_and_subclass_metadata_are_safe():
    parameter = torch.nn.Parameter(torch.ones(3))
    sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (3,))

    class ChildTensor(torch.Tensor):
        pass

    child = torch.Tensor._make_subclass(ChildTensor, torch.ones(3), False)
    with observe_transfers() as inventory:
        converted = parameter.to(torch.float16)
        sparse_copy = sparse.to("cpu", copy=True)
        child_copy = child.to("cpu", copy=True)
        with torch.no_grad():
            frozen_copy = parameter.to(torch.float64)
    assert sparse_copy.layout == sparse.layout
    assert sparse_copy.dtype == sparse.dtype
    assert type(child_copy) is ChildTensor
    assert frozen_copy.requires_grad is False
    copies = _records(inventory, "aten._to_copy.default")
    assert converted.requires_grad is True
    assert copies[0].source.requires_grad is True
    assert copies[0].destination.requires_grad is False
    assert copies[0].source.is_subclass is False
    assert copies[1].source.layout == "sparse_coo"
    assert copies[1].source.storage_capacity_bytes is None
    assert copies[2].source.is_subclass is True


def test_operator_exception_is_unchanged_and_recorded():
    x = torch.ones(2, 3)
    y = torch.ones(4, 2)
    with pytest.raises(Exception) as baseline:
        torch.mm(x, y)
    inventory = observe_transfers()
    with pytest.raises(type(baseline.value)) as observed:
        with inventory:
            torch.mm(x, y)
    assert str(observed.value) == str(baseline.value)
    assert inventory.records[-1].failure_type == type(observed.value).__name__
    assert inventory.records[-1].failure_message == str(observed.value)
    assert classify(inventory.records[-1]).reason == "operator_failed"


@pytest.mark.gpu
def test_gpu_transfer_apis_weights_and_pinning(cuda_device):
    pageable = torch.arange(8, dtype=torch.float32)
    pinned = pageable.pin_memory()
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(8))
    module.register_buffer("scale", torch.full((8,), 2.0))
    state = {"weight": torch.full((8,), 3.0), "scale": torch.full((8,), 4.0)}
    with observe_transfers() as inventory:
        pageable_gpu = pageable.to(cuda_device)
        pinned_gpu = pinned.cuda(non_blocking=True)
        back = pageable_gpu.cpu()
        copied = pageable_gpu.to(cuda_device, copy=True)
        module.to(cuda_device)
        with inventory.phase("weight_loading"):
            module.load_state_dict(state)
    assert torch.equal(back, pageable)
    assert copied.data_ptr() != pageable_gpu.data_ptr()
    assert torch.equal(module.weight.cpu(), state["weight"])
    assert torch.equal(module.scale.cpu(), state["scale"])
    transfers = _records(inventory, "aten._to_copy.default")
    assert any(r.source.pinned and r.non_blocking for r in transfers)
    assert any(not r.source.pinned and r.source.device_type == "cpu" for r in transfers)
    assert any(r.source.device_type == "cuda" and r.destination.device_type == "cpu" for r in transfers)
    assert any(r.source.requires_grad and r.destination.device_type == "cuda" for r in transfers)
    assert any(not r.source.requires_grad and r.destination.device_type == "cuda" for r in transfers)
    weights = [r for r in _records(inventory, "aten.copy_.default") if r.phase == "weight_loading"]
    assert len(weights) == 2
    assert all(r.mutates for r in weights)


@pytest.mark.gpu
def test_cross_device_copy_records_source_and_destination(cuda_device):
    source = torch.arange(4, device=cuda_device)
    destination = torch.empty(4)
    with observe_transfers() as inventory:
        result = destination.copy_(source)
    assert result is destination
    assert torch.equal(destination, source.cpu())
    record = _records(inventory, "aten.copy_.default")[0]
    assert record.source.device_type == "cuda"
    assert record.destination.device_type == "cpu"
