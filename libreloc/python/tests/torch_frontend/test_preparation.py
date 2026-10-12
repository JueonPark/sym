"""Cold preparation must preserve live-value semantics and explicit budgets."""
import gc
import os
import weakref

import pytest
import torch

from reloc_torch import CompilerClient, PreparedTransfers
from reloc_torch.recipe import BindingParam, Cast, Dequantize, IndexSelect, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides


def recipe(kind="typed", direction="h2d"):
    shape = (Symbol("N"), Const(16))
    spec = lambda s, d: TensorSpec(s, dense_strides(s), Const(0), d)
    if kind == "indexed":
        index = TensorSpec((Symbol("M"),), (Const(1),), Const(0), "int64")
        return Recipe(spec(shape, "float32"), (IndexSelect(index), Cast("float16", "ieee_rne")),
                      spec((Symbol("M"), Const(16)), "float16"), direction)
    if kind == "weight":
        return Recipe(spec(shape, "int8"),
                      (Dequantize("float32", BindingParam("scale", "float32", (Const(16),)),
                                  None, 1, "affine"), Transpose((1, 0))),
                      spec(tuple(reversed(shape)), "float32"), direction)
    return Recipe(spec(shape, "float32"),
                  (Cast("float16", "ieee_rne"),) if kind == "typed" else (Transpose((1, 0)),),
                  spec(shape if kind == "typed" else tuple(reversed(shape)),
                       "float16" if kind == "typed" else "float32"), direction)


@pytest.mark.parametrize("kind", ["typed", "layout", "indexed", "weight"])
def test_preparation_live_inputs_outputs_and_no_example_ownership(compiler, cuda_device, kind):
    source = torch.arange(1024, dtype=torch.float32).reshape(64, 16)
    if kind == "weight":
        source = source.to(torch.int8)
    index = torch.tensor([5, 1, 9], dtype=torch.int64)
    scale = torch.linspace(.01, .1, 16)
    args = dict(indices=index) if kind == "indexed" else dict(parameters={"scale": scale}) if kind == "weight" else {}
    if kind == "weight":
        args["implementation"] = "cuda_dequant_relocate"
    reference = lambda x: (x[index].half() if kind == "indexed" else
                           (x.float() * scale).t().contiguous() if kind == "weight" else
                           x.half() if kind == "typed" else x.t().contiguous())
    session = PreparedTransfers({"test": recipe(kind)}, compiler=compiler,
                                transfer_options={"pinning": "pageable", "gather_threads": 2})
    session.warm("test", source, cuda_device, **args)
    before = session.stats()
    source.add_(1)
    index[0] = 2
    scale.mul_(2)
    expected = reference(source).clone()
    first = session.transfer("test", source, cuda_device, **args)
    source.add_(1)
    index[1] = 3
    scale.mul_(2)
    second = session.transfer("test", source, cuda_device, **args)
    assert torch.equal(first.cpu(), expected)
    assert torch.equal(second.cpu(), reference(source))
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)
    after = session.stats()
    assert after["resources"]["typed"]["context_creations"] == before["resources"]["typed"]["context_creations"]
    assert after["serving_calls"] == 2 and after["warm_calls"] == 1
    assert after["first_execution_ms"] > 0 and after["subsequent_execution_ms"] > 0
    references = [weakref.ref(v) for v in (source, index, scale)]
    del source, index, scale, args
    gc.collect()
    assert all(ref() is None for ref in references)
    session.close()
    assert session.stats()["retirement_ms"] > 0
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.transfer("test", torch.ones(4, 16), cuda_device)


def test_preparation_layout_d2h_and_shape_growth(compiler, cuda_device):
    with PreparedTransfers({"test": recipe("layout", "d2h")}, compiler=compiler,
                           max_output_bytes=4096) as session:
        session.warm("test", torch.arange(64., device=cuda_device).reshape(4, 16), "cpu")
        for size in (8, 2, 32):
            source = torch.randn(size, 16, device=cuda_device)
            assert torch.equal(session.transfer("test", source, "cpu"), source.cpu().t())
        with pytest.raises(ValueError, match="max_output_bytes"):
            session.transfer("test", torch.ones(128, 16, device=cuda_device), "cpu")


def test_preparation_live_budget_is_enforced(compiler, cuda_device):
    with PreparedTransfers({"test": recipe()}, compiler=compiler,
                           resource_options={"max_typed_live_bytes": 1}) as session:
        with pytest.raises(RuntimeError, match="limit|budget"):
            session.warm("test", torch.ones(128, 16), cuda_device)
        assert session.stats()["resources"]["typed"]["live_bytes"] <= 1


@pytest.mark.parametrize("limits", [{"max_typed_live_bytes": 0}, {"max_live_staging_bytes": None}])
def test_preparation_requires_finite_live_limits(compiler, limits):
    with pytest.raises(ValueError, match="positive integer"):
        PreparedTransfers({"test": recipe()}, compiler=compiler, resource_options=limits)


def test_preparation_fork_rejected_before_lock(monkeypatch, compiler):
    session = PreparedTransfers({"test": recipe()}, compiler=compiler)
    with monkeypatch.context() as patch:
        patch.setattr(os, "getpid", lambda: -1)
        for call in (session.stats, session.close, session.initialize_cuda):
            with pytest.raises(RuntimeError, match="process_mismatch"):
                call()
    session.close()
