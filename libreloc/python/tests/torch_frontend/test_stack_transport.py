"""Stacked preflight (torch.stack support): every input is guarded and validated as one
slice of the logical source before anything is allocated or launched."""
import types

import pytest
import torch

import pyreloc
from conftest import stacked_recipe


@pytest.fixture
def compiled(compiler):
    return compiler.compile(stacked_recipe(count=3, dim=1))


def inputs(rows=4, columns=5):
    return [torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) + 100 * i
            for i in range(3)]


def test_host_preflight_binds_once_and_rechecks_every_input(compiled):
    from reloc_torch.runtime import prepare_stacked_host_call

    xs = inputs()
    call = prepare_stacked_host_call(compiled, xs, torch.device("cpu"))
    assert call.sources == tuple(xs) and call.src is xs[0]
    assert call.bindings == {"s0": 4, "s1": 5}
    assert call.destination.shape == (4, 3, 5)
    call.recheck()
    xs[2].unsqueeze_(0)
    with pytest.raises(RuntimeError, match="stale prepared call"):
        call.recheck()


@pytest.mark.parametrize(("make", "reason"), [
    (lambda xs: [xs[0], xs[1]], "stack_count"),
    (lambda xs: [xs[0], xs[1].half(), xs[2]], "stack_dtype_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 6), xs[2]], "stack_shape_mismatch"),
    (lambda xs: [xs[0], torch.ones(4, 10)[:, ::2], xs[2]], "unsupported_layout"),
    (lambda xs: [xs[0], torch.ones(22)[2:].view(4, 5), xs[2]], "storage_offset"),
])
def test_host_preflight_rejections_have_stable_reasons(compiled, make, reason):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.runtime import prepare_stacked_host_call

    with pytest.raises(UnsupportedRecipe) as error:
        prepare_stacked_host_call(compiled, make(inputs()), torch.device("cpu"))
    assert error.value.reason == reason


def test_transport_rejects_before_allocation(compiled):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.transport import prepare_stacked_transfer

    xs = inputs()
    for kwargs, device, reason in (
        ({}, "cpu", "direction_mismatch"),
        ({"non_blocking": True}, "cuda", "nonblocking_unavailable"),
    ):
        with pytest.raises(UnsupportedRecipe) as error:
            prepare_stacked_transfer(compiled, xs, device, **kwargs)
        assert error.value.reason == reason
    with pytest.raises(UnsupportedRecipe) as error:
        prepare_stacked_transfer(compiled, xs[:2], "cuda")
    assert error.value.reason == "plan_mismatch"
    if not (pyreloc.cuda_enabled and torch.cuda.is_available()):
        with pytest.raises(UnsupportedRecipe) as error:
            prepare_stacked_transfer(compiled, xs, "cuda")
        assert error.value.reason == "cuda_unavailable"


def test_transport_adapter_routes_stacked_preflight_and_refuses_typed(compiled):
    from reloc_torch.artifact import UnsupportedRecipe
    from reloc_torch.runtime import TransportAdapter

    adapter = TransportAdapter()
    try:
        with pytest.raises(UnsupportedRecipe) as error:
            adapter.preflight_stacked(types.SimpleNamespace(typed=True), inputs(), torch.device("cuda"))
        assert error.value.reason == "typed_transform_unavailable"
        with pytest.raises(UnsupportedRecipe) as error:
            adapter.preflight_stacked(compiled, inputs(), torch.device("cpu"))
        assert error.value.reason == "direction_mismatch"
    finally:
        adapter.close()
