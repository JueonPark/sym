"""Owned prefold bridge conformance and the typed capability gate (T4 Task 3).

The bridge wraps the existing prefolder; its output must be byte-identical to
the declared int8 semantics q = rne(clamp(x * invScale, -128, 127)) with NaN
mapping to -128. The typed recipe contract (C3/C4/R3) does not exist yet, so
the frontend gate reports that and float weight loading never quantizes.

Handoff fact for C4/R3: the v0 binder coalesces an identity relocation into a
single axis, and the prefolder needs a distinct channel axis, so
``s8_quant_pack`` is unreachable through a compiled identity artifact; the
fused ``s8_gather_quant`` path is reachable for every layout that keeps the
channel axis (transpose here).
"""
import gc
import subprocess
import sys
import weakref

import pytest
import torch


def reference_q(x, inv_scales):
    """Declared semantics: clamp (max then min, NaN -> -128), then RNE."""
    scaled = x * inv_scales.reshape(-1, *([1] * (x.dim() - 1)))
    clamped = torch.clamp(scaled, -128.0, 127.0)
    clamped = torch.where(torch.isnan(scaled), torch.full_like(scaled, -128.0), clamped)
    return torch.round(clamped).to(torch.int8)


def _bound(compiler, recipe, source):
    import pyreloc

    compiled = compiler.compile(recipe)
    return pyreloc.bind(pyreloc.load_plan(compiled.plan_bytes), compiled.bind_values(source))


def _identity(rank=2, dtype="float32"):
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import Const, Symbol, dense_strides

    shape = tuple(Symbol(f"s{i}") for i in range(rank))
    descriptor = TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    return Recipe(descriptor, (), descriptor, "h2d")


def _transpose(dtype="float32"):
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol

    rows, columns = Symbol("s0"), Symbol("s1")
    return Recipe(
        TensorSpec((rows, columns), (columns, Const(1)), Const(0), dtype),
        (Transpose((1, 0)),),
        TensorSpec((columns, rows), (rows, Const(1)), Const(0), dtype),
        "h2d",
    )


def test_prefold_bindings_import_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, pyreloc; pyreloc.prefold_s8; pyreloc.PrefoldHandle; pyreloc.PrefoldError; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gather_quant_matches_declared_semantics_including_ties_clipping_and_nan(compiler):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    # Source (6, 4); the transposed destination (4, 6) has four channels.
    x = torch.tensor([
        [0.5, 1.5, 2.5, -0.5, -1.5, -2.5],          # ties: RNE -> 0, 2, 2, 0, -2, -2
        [200.0, -200.0, 127.4, -128.6, 0.0, -0.0],  # clipping and signed extremes
        [float("nan"), float("inf"), -float("inf"), 3.49999, 3.5, -3.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ], dtype=torch.float32).t().contiguous()
    inv = torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float32)
    bound = _bound(compiler, _transpose(), x)
    assert bound.extents == [4, 6]
    with pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_gather_quant") as handle:
        assert handle.nbytes == x.numel() and not handle.closed
        image = torch.empty(4, 6, dtype=torch.int8)
        handle.copy_to(*as_ptr(image))
    assert handle.closed
    expected = reference_q(x.t().contiguous(), inv)
    assert torch.equal(image, expected), (image, expected)
    assert image[2, 0] == -128  # NaN maps to -128 by the declared max-then-min clamp
    assert image[0, 2] == 2 and image[0, 4] == -2  # ties to even
    assert image[1, 0] == 127 and image[1, 1] == -128  # clipping


@pytest.mark.parametrize("gather_threads", [1, 2, 0])
def test_gather_quant_transposes_and_quantizes_per_output_channel(compiler, gather_threads):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    x = (torch.arange(24, dtype=torch.float32).reshape(4, 6) - 11.5) * 3.0
    inv = torch.tensor([1.0, 0.5, 0.25, 2.0, 1.0, 0.125], dtype=torch.float32)  # one per dst row
    bound = _bound(compiler, _transpose(), x)
    assert bound.extents == [6, 4]
    with pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_gather_quant", gather_threads=gather_threads) as handle:
        image = torch.empty(6, 4, dtype=torch.int8)
        handle.copy_to(*as_ptr(image))
    assert torch.equal(image, reference_q(x.t().contiguous(), inv))


def test_compiled_identity_plans_coalesce_away_the_channel_axis(compiler):
    """Recorded handoff fact: s8_quant_pack needs an uncoalesced channel axis."""
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    bound = _bound(compiler, _identity(), x)
    assert bound.extents == [24]
    with pytest.raises(pyreloc.PrefoldError, match="rank >= 2"):
        pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(torch.ones(24, dtype=torch.float32)), output_spec="s8_quant_pack")


def test_torch_facing_bridge_returns_an_owned_int8_image(compiler):
    from reloc_torch.prefold import prefold_s8_image

    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    inv = torch.full((6,), 0.5, dtype=torch.float32)
    image = prefold_s8_image(_bound(compiler, _transpose(), x), x, inv, output_spec="s8_gather_quant")
    assert image.dtype == torch.int8 and image.shape == (6, 4)
    assert torch.equal(image, reference_q(x.t().contiguous(), inv))
    with pytest.raises(TypeError):
        prefold_s8_image(_bound(compiler, _transpose(), x), x.to(torch.float16), inv, output_spec="s8_gather_quant")
    with pytest.raises(ValueError):
        prefold_s8_image(_bound(compiler, _transpose(), x), x, inv, output_spec="s8_something")


def test_validation_rejects_bad_inputs_before_any_kernel(compiler):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    inv = torch.ones(6, dtype=torch.float32)
    bound = _bound(compiler, _transpose(), x)
    with pytest.raises(ValueError, match="output_spec"):
        pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_other")
    with pytest.raises(pyreloc.PrefoldError, match="exactly 6 float32"):
        pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(torch.ones(4, dtype=torch.float32)), output_spec="s8_gather_quant")
    with pytest.raises(pyreloc.PrefoldError, match="too small"):
        pyreloc.prefold_s8(bound, x.data_ptr(), x.numel() * 4 - 1, *as_ptr(inv), output_spec="s8_gather_quant")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        scales = inv.clone()
        scales[1] = bad
        with pytest.raises(pyreloc.PrefoldError, match="positive and finite"):
            pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(scales), output_spec="s8_gather_quant")
    with pytest.raises(pyreloc.PrefoldError, match="non-null"):
        pyreloc.prefold_s8(bound, 0, 96, *as_ptr(inv), output_spec="s8_gather_quant")
    with pytest.raises(pyreloc.PrefoldError, match="identity"):
        pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_quant_pack")
    with pytest.raises(pyreloc.PrefoldError, match="rank >= 2"):
        rank1 = _bound(compiler, _identity(rank=1), torch.arange(6, dtype=torch.float32))
        pyreloc.prefold_s8(rank1, *as_ptr(torch.arange(6, dtype=torch.float32)), *as_ptr(torch.ones(6)), output_spec="s8_gather_quant")
    half_source = x.to(torch.float16)
    half = _bound(compiler, _transpose("float16"), half_source)
    with pytest.raises(pyreloc.PrefoldError, match="float32"):
        pyreloc.prefold_s8(half, *as_ptr(half_source), *as_ptr(inv), output_spec="s8_gather_quant")
    with pytest.raises(ValueError):
        pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_gather_quant", gather_threads=-1)


def test_handle_lifecycle_copy_after_close_and_small_destination(compiler):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    inv = torch.ones(6, dtype=torch.float32)
    bound = _bound(compiler, _transpose(), x)
    handle = pyreloc.prefold_s8(bound, *as_ptr(x), *as_ptr(inv), output_spec="s8_gather_quant")
    small = torch.empty(10, dtype=torch.int8)
    with pytest.raises(ValueError, match="too small"):
        handle.copy_to(*as_ptr(small))
    image = torch.empty(24, dtype=torch.int8)
    handle.copy_to(*as_ptr(image))
    assert torch.equal(image.reshape(6, 4), reference_q(x.t().contiguous(), inv))
    handle.close()
    handle.close()
    assert handle.closed and handle.nbytes == 0
    with pytest.raises(pyreloc.PrefoldError, match="closed"):
        handle.copy_to(*as_ptr(image))
    reference = weakref.ref(handle)
    del handle
    gc.collect()
    assert reference() is None


def test_typed_capability_gate_blocks_quantized_preparation_of_float_weights(compiler, transpose_weight_recipe, counting_runtime):
    from reloc_torch import RelocBackend, prepare_weights
    from reloc_torch.prefold import CAPABILITY_REASON, prefold_eligibility, typed_prefold_capability

    assert typed_prefold_capability() == CAPABILITY_REASON
    assert prefold_eligibility(transpose_weight_recipe) == "not_typed_recipe"

    class Typed:
        value_transforms = ("quantize",)

    assert prefold_eligibility(Typed()) == CAPABILITY_REASON
    backend = RelocBackend(compiler=compiler, runtime=counting_runtime)
    module = torch.nn.Module()
    module.register_buffer("weight", torch.arange(24.0).reshape(4, 6))
    with torch.no_grad(), prepare_weights(module, {"weight": transpose_weight_recipe}, backend=backend, stable=True) as prepared:
        out = prepared.get("weight", device="cpu")
    assert out.dtype == torch.float32  # ordinary float loading keeps dtype and values
    assert torch.equal(out, module.weight.t().contiguous())
    backend.close()
