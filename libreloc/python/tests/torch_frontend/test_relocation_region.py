"""Observable values, ownership and physical bindings for the issue 270 model."""
import json

import pyreloc
import pytest
import torch

from bench.issue270.ablate import CORPUS, bindings, compare, fixture
from bench.issue270.region import Invocation, capture, lower, materialized, transfer


def exposed(call, source, prefix):
    producer = source.clone(memory_format=torch.contiguous_format)
    value = transfer(call, producer)
    return torch.cat((prefix, value), dim=0), producer, value


def mixed(call, source, prefix):
    producer = source.clone(memory_format=torch.contiguous_format)
    value = transfer(call, producer)
    return torch.cat((prefix, value), dim=0), value.flatten()


def mutating(call, source, prefix):
    producer = source.clone(memory_format=torch.contiguous_format)
    source.view(-1).zero_()
    value = transfer(call, producer)
    return torch.cat((prefix, value), dim=0)


@pytest.mark.parametrize("name,index", bindings())
def test_region_matches_committed_bits_and_forwards_both_buffers(name, index):
    for row in compare(name, index, "cpu"):
        assert row["temporary_bytes"] == [row["source_bytes"] + row["destination_bytes"], 0]


@pytest.mark.parametrize("program", [exposed, mixed])
def test_visible_intermediates_keep_independent_storage(program):
    call, source, prefix, expected = fixture("quantize_edges", 0, "cpu")
    source_bits = source.view(torch.uint8).clone()
    out, *held = lower(capture(program))(call, source, prefix)
    assert torch.equal(out[prefix.size(0):], expected)
    held[-1].zero_()
    assert torch.equal(out[prefix.size(0):], expected)
    if program is exposed:
        held[0].zero_()
        assert torch.equal(source.view(torch.uint8), source_bits)


def test_alias_write_between_snapshot_and_transfer_preserves_old_value():
    call, source, prefix, expected = fixture("quantize_edges", 0, "cpu")
    out = lower(capture(mutating))(call, source, prefix)
    assert torch.count_nonzero(source).item() == 0
    assert torch.equal(out[prefix.size(0):], expected)


def test_cat_promotion_falls_back_and_later_calls_preserve_held_outputs():
    call, source, prefix, expected = fixture("quantize_edges", 0, "cpu")
    run = lower(capture(materialized))
    promoted = run(call, source, prefix.float())
    assert promoted.dtype == torch.float32
    assert torch.equal(promoted[prefix.size(0):], expected.float())
    held = run(call, source, prefix)
    source.zero_()
    assert torch.count_nonzero(run(call, source, prefix)[prefix.size(0):]) == 0
    assert torch.equal(held[prefix.size(0):], expected)


@pytest.mark.gpu
def test_every_eligible_h2d_cut_forwards_buffers_on_device(cuda_device):
    for name, index in bindings():
        for row in compare(name, index, str(cuda_device)):
            assert row["temporary_bytes"][1] == 0, row


def test_only_private_edge_is_forwarded():
    def keep_source(call, source, prefix):
        p = source.clone(memory_format=torch.contiguous_format)
        t = transfer(call, p)
        return torch.cat((prefix, t), dim=0), p

    def keep_transfer(call, source, prefix):
        p = source.clone(memory_format=torch.contiguous_format)
        t = transfer(call, p)
        return torch.cat((prefix, t), dim=0), t

    call, source, prefix, expected = fixture("quantize_edges", 0, "cpu")
    for program, source_private in ((keep_source, False), (keep_transfer, True)):
        out, held = lower(capture(program))(call, source, prefix)
        assert (source.data_ptr() == call.source_address) == source_private
        assert (out[prefix.size(0):].data_ptr() == call.destination_address) != source_private
        held.zero_()
        assert torch.equal(out[prefix.size(0):], expected)


def test_rebinding_channel_values_keeps_old_revision_and_held_output():
    name = "quantize_channel_transpose_sym"
    meta = json.loads((CORPUS / f"{name}.json").read_text())
    plan = pyreloc.load_typed_plan((CORPUS / f"{name}.bin").read_bytes())
    old, source, prefix, expected = fixture(name, 0, "cpu")
    run = lower(capture(materialized))
    held = run(old, source, prefix)
    # All channels now share scale 1. The old binding has scales [1, 1/2, 1/4].
    scales = torch.ones(3, dtype=torch.float32)
    params = {"s": ("float32", [3], scales.numpy().tobytes())}
    fresh = Invocation(pyreloc.bind_typed(plan, meta["bindings"][0]["symbols"], params), old.device)
    revised = run(fresh, source, prefix)[1:]
    oracle = source.nan_to_num(nan=-128).clamp(-128, 127).round().to(torch.int8).T
    assert torch.equal(revised, oracle) and not torch.equal(revised, expected)
    assert torch.equal(held[1:], expected) and torch.equal(run(old, source, prefix)[1:], expected)


def test_cut_before_late_padding_is_rejected_before_launch():
    call, _, _, _ = fixture("quant_dequant_padded", 0, "cpu")
    with pytest.raises(pyreloc.TransferError, match="implementation_unavailable"):
        pyreloc.select_dispatch(call.bound, "h2d", "cuda", implementation="cpu_stages_cuda_stages@1")
    assert call.report is None


def test_fresh_cat_version_and_storage_free_source():
    call, source, prefix, _ = fixture("quantize_edges", 0, "cpu")
    original = capture(materialized)
    run = lower(original)
    assert run(call, source, prefix)._version == original(call, source, prefix)._version
    zeros = torch._efficientzerotensor(source.shape, dtype=source.dtype)
    assert torch.equal(run(call, zeros, prefix), original(call, zeros, prefix))
    wrapped = torch._to_functional_tensor(prefix)
    assert torch.equal(run(call, source, wrapped), original(call, source, wrapped))


@pytest.mark.gpu
def test_held_region_output_survives_cross_stream_use_and_next_call(cuda_device):
    call, source, prefix, expected = fixture("quantize_edges", 0, cuda_device)
    run = lower(capture(materialized))
    first = run(call, source, prefix)
    consumer = torch.cuda.Stream(device=cuda_device)
    consumer.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(5_000_000)
        observed = first[1:].clone()
        first.record_stream(consumer)
    # A completed transfer allows another invocation. The caller still owns first.
    source.zero_()
    second = run(call, source, prefix)
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    consumer.synchronize()
    assert torch.equal(observed.cpu(), expected)
    assert torch.equal(first[1:].cpu(), expected)
    assert torch.count_nonzero(second[1:]).item() == 0
