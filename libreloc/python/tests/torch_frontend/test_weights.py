"""Prepared inference weights: live slots, freshness, lifecycle (T4 Task 2)."""
import gc
import weakref

import pytest
import torch


gpu = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def weights_api():
    from reloc_torch import weights

    return weights


@pytest.fixture
def cpu_backend(compiler, counting_runtime):
    from reloc_torch import RelocBackend

    backend = RelocBackend(compiler=compiler, runtime=counting_runtime)
    yield backend
    backend.close()


def _module():
    module = torch.nn.Module()
    module.register_buffer("weight", torch.arange(24.0).reshape(4, 6))
    module.layer = torch.nn.Module()
    module.layer.weight = torch.nn.Parameter(torch.arange(24.0).reshape(4, 6) + 100)
    return module


def expected(tensor):
    return tensor.detach().t().contiguous()


@gpu
@needs_cuda
def test_prepared_weight_follows_live_slot(backend, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = torch.nn.Module()
    module.register_buffer("weight", torch.arange(24.).reshape(4, 6))
    with torch.no_grad(), prepare_weights(
            module, {"weight": transpose_weight_recipe}, backend=backend,
            stable=True) as prepared:
        first = prepared.get("weight", device="cuda")
        module.weight.add_(1)
        second = prepared.get("weight", device="cuda")
        torch.testing.assert_close(second, module.weight.t().contiguous().cuda(),
                                   rtol=0, atol=0)
        assert not torch.equal(first, second)
        module.weight = torch.full_like(module.weight, 9)
        third = prepared.get("weight", device="cuda")
        torch.testing.assert_close(third, torch.full_like(third, 9), rtol=0, atol=0)
    stats = backend.stats()
    assert stats["weight_preparations"] == 3
    assert stats["weight_invalidations"] == 2
    assert stats["runtime_executions"] == 3


def test_buffers_parameters_and_nested_names_relocate_without_touching_the_module(cpu_backend, counting_runtime, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    parameter = module.layer.weight
    recipes = {"weight": transpose_weight_recipe, "layer.weight": transpose_weight_recipe}
    with torch.no_grad(), prepare_weights(module, recipes, backend=cpu_backend) as prepared:
        for name in recipes:
            source = module.get_buffer(name) if name == "weight" else module.get_parameter(name)
            out = prepared.get(name, device="cpu")
            assert torch.equal(out, expected(source))
            assert out.stride() == (4, 1) and out.storage_offset() == 0
            assert out.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert module.layer.weight is parameter
    assert parameter.requires_grad is True
    assert isinstance(module.layer.weight, torch.nn.Parameter)
    assert counting_runtime.executions == 2
    stats = cpu_backend.stats()
    assert stats["plan_compiles"] == 1 and stats["weight_preparations"] == 0


def test_stable_preparation_reuses_until_bytes_change(cpu_backend, counting_runtime, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    with torch.no_grad(), prepare_weights(module, {"weight": transpose_weight_recipe}, backend=cpu_backend, stable=True) as prepared:
        first = prepared.get("weight", device="cpu")
        second = prepared.get("weight", device="cpu")
        assert torch.equal(first, second) and first.data_ptr() != second.data_ptr()
        stats = cpu_backend.stats()
        assert stats["weight_preparations"] == 1 and stats["weight_invalidations"] == 0
        first.fill_(-1)  # a caller's output mutation must not poison the cache
        assert torch.equal(prepared.get("weight", device="cpu"), expected(module.weight))
        module.weight.add_(1)  # version counter bump
        assert torch.equal(prepared.get("weight", device="cpu"), expected(module.weight))
        module.weight.data[0, 0] = 777  # .data write: only the byte snapshot sees it
        assert torch.equal(prepared.get("weight", device="cpu"), expected(module.weight))
        module.weight.numpy()[1, 1] = 555  # NumPy write: no version bump either
        assert torch.equal(prepared.get("weight", device="cpu"), expected(module.weight))
        module.weight.copy_(torch.full((4, 6), float("nan")))
        prepared.get("weight", device="cpu")
        preparations = cpu_backend.stats()["weight_preparations"]
        prepared.get("weight", device="cpu")  # unchanged NaNs are not a mutation
        assert cpu_backend.stats()["weight_preparations"] == preparations
        stats = cpu_backend.stats()
        assert stats["weight_preparations"] == 5
        assert stats["weight_invalidations"] == 4
        assert stats["plan_compiles"] == 1  # the weight recipe only: a CPU target copies the prepared layout
        assert counting_runtime.executions == cpu_backend.stats()["runtime_executions"]
        assert not stats["fallbacks"]  # the CPU target never routes through a device transfer


def test_replacement_load_state_dict_and_tied_parameters(cpu_backend, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    module.layer.tied = module.layer.weight  # tie: one Parameter, two names
    recipes = {"weight": transpose_weight_recipe, "layer.weight": transpose_weight_recipe, "layer.tied": transpose_weight_recipe}
    with torch.no_grad(), prepare_weights(module, recipes, backend=cpu_backend, stable=True) as prepared:
        for name in recipes:
            prepared.get(name, device="cpu")
        assert cpu_backend.stats()["weight_preparations"] == 3
        module.weight = torch.full((4, 6), 9.0)  # slot replacement
        assert torch.equal(prepared.get("weight", device="cpu"), torch.full((6, 4), 9.0))
        module.load_state_dict({"weight": torch.ones(4, 6), "layer.weight": torch.zeros(4, 6), "layer.tied": torch.zeros(4, 6)})
        assert torch.equal(prepared.get("weight", device="cpu"), torch.ones(6, 4))
        assert torch.equal(prepared.get("layer.weight", device="cpu"), torch.zeros(6, 4))
        module.layer.weight.add_(3)  # mutation through the tie invalidates both names
        assert torch.equal(prepared.get("layer.tied", device="cpu"), torch.full((6, 4), 3.0))
        assert module.layer.tied is module.layer.weight
        stats = cpu_backend.stats()
        assert stats["weight_invalidations"] >= 4


def test_changed_shape_or_dtype_falls_back_to_pytorch_with_a_reason(cpu_backend, counting_runtime, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    with torch.no_grad(), prepare_weights(module, {"weight": transpose_weight_recipe}, backend=cpu_backend, stable=True) as prepared:
        prepared.get("weight", device="cpu")
        module.weight = torch.arange(30.0).reshape(5, 6)
        out = prepared.get("weight", device="cpu")
        assert torch.equal(out, module.weight.t().contiguous())
        module.weight = torch.arange(24.0, dtype=torch.float16).reshape(4, 6)
        out = prepared.get("weight", device="cpu")
        assert out.dtype == torch.float16 and torch.equal(out, module.weight.t().contiguous())
    stats = cpu_backend.stats()
    # Both the shape change and the dtype change violate the recipe's source
    # descriptor guard; each fell back to PyTorch with that reason.
    assert stats["fallbacks"].get("source_descriptor", 0) >= 2
    # The first get prepared the layout through the CPU relocation executor and
    # copied it for the CPU target; no adapter transfer ran at any point.
    assert counting_runtime.executions == 0
    assert stats["weight_preparations"] == 1


def test_deleted_slot_dead_module_and_close_raise_clearly(cpu_backend, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    prepared = prepare_weights(module, {"weight": transpose_weight_recipe}, backend=cpu_backend, stable=True)
    with torch.no_grad():
        held = prepared.get("weight", device="cpu")
        del module.weight
        with pytest.raises(RuntimeError, match="no longer exists"):
            prepared.get("weight", device="cpu")
        with pytest.raises(KeyError):
            prepared.get("layer.weight", device="cpu")
    weak = weakref.ref(module)
    del module
    gc.collect()
    assert weak() is None
    with pytest.raises(RuntimeError, match="garbage collected"):
        prepared.get("weight", device="cpu")
    prepared.invalidate()
    prepared.close()
    prepared.close()
    assert prepared.closed
    with pytest.raises(RuntimeError, match="closed"):
        prepared.get("weight", device="cpu")
    with pytest.raises(RuntimeError, match="closed"):
        prepared.invalidate()
    assert held.shape == (6, 4)


def test_explicit_invalidation_exception_cleanup_and_release(cpu_backend, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    with pytest.raises(ValueError):
        with torch.no_grad(), prepare_weights(module, {"weight": transpose_weight_recipe}, backend=cpu_backend, stable=True) as prepared:
            prepared.get("weight", device="cpu")
            retained = weakref.ref(prepared._states["weight"].prepared)
            prepared.invalidate("weight")
            assert cpu_backend.stats()["weight_invalidations"] == 1
            prepared.get("weight", device="cpu")
            retained = weakref.ref(prepared._states["weight"].prepared)
            raise ValueError("boom")
    assert prepared.closed
    gc.collect()
    assert retained() is None


def test_grad_enabled_parameter_replays_through_pytorch(cpu_backend, counting_runtime, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    with prepare_weights(module, {"layer.weight": transpose_weight_recipe}, backend=cpu_backend) as prepared:
        out = prepared.get("layer.weight", device="cpu")
    assert out.requires_grad is True
    assert torch.equal(out.detach(), expected(module.layer.weight))
    assert module.layer.weight.requires_grad is True
    assert counting_runtime.executions == 0
    assert cpu_backend.stats()["fallbacks"]["requires_grad"] == 1


def test_unstable_preparation_always_reads_current_values(cpu_backend, counting_runtime, transpose_weight_recipe):
    from reloc_torch import prepare_weights

    module = _module()
    with torch.no_grad(), prepare_weights(module, {"weight": transpose_weight_recipe}, backend=cpu_backend) as prepared:
        prepared.get("weight", device="cpu")
        module.weight.data[2, 3] = 42.0
        assert torch.equal(prepared.get("weight", device="cpu"), expected(module.weight))
    stats = cpu_backend.stats()
    assert stats["weight_preparations"] == 0 and stats["weight_invalidations"] == 0
    assert counting_runtime.executions == 2


def test_replay_matches_pytorch_for_pad_and_reshape(compiler):
    from reloc_torch.recipe import Fill, Pad, Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, FloorDiv, Symbol, dense_strides, operation_shape
    from reloc_torch.weights import replay

    n = Symbol("s0")
    shape = (n,)
    operations = (Reshape((Const(2), FloorDiv(n, 2))), Transpose((1, 0)), Pad(0, Const(1), Const(2), Fill("float32", 0x3F800000)))
    current = shape
    for operation in operations:
        current = operation_shape(current, operation)
    recipe = Recipe(
        TensorSpec(shape, dense_strides(shape), Const(0), "float32"),
        operations,
        TensorSpec(current, dense_strides(current), Const(0), "float32"),
        "h2d",
    )
    x = torch.arange(6.0)
    out = replay(recipe, x)
    reference = torch.nn.functional.pad(x.reshape(2, 3).t().contiguous(), (0, 0, 1, 2), value=1.0)
    assert torch.equal(out, reference) and out.stride() == reference.stride()


def test_public_entry_point_is_lazy_and_torch_free():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys; from reloc_torch import prepare_weights, PreparedWeights; assert callable(prepare_weights); assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
