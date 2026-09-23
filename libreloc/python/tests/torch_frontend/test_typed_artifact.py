"""Typed artifacts (C3, issue #143) through the frontend bridge.

Typed recipes compile with ``--typed`` into schema-2 / wire-v1 artifacts;
admission checks the stages, fills and parameter declarations against the
recipe and against the plan the runtime decoded; the portable format carries
them as ``format_version`` 2 and loads in a fresh process; stale or foreign
artifacts are rejected; layout-only artifacts are untouched.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

import pyreloc


def _spec(shape, dtype):
    from reloc_torch.recipe import TensorSpec
    from reloc_torch.symbolic import Const, dense_strides

    shape = tuple(Const(d) if isinstance(d, int) else d for d in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


@pytest.fixture
def quantize_transpose_recipe():
    """f32 [B, 3] -> per-channel int8 with a runtime scale over the three
    channels -> transposed [3, B]: the channel moves to result axis 0."""
    from reloc_torch.recipe import BindingParam, Quantize, Recipe, Transpose
    from reloc_torch.symbolic import Const, Symbol

    b = Symbol("s0")
    quantize = Quantize("int8", BindingParam("s", "float32", (Const(3),)), None, 1, "symmetric_rne")
    return Recipe(_spec((b, 3), "float32"), (quantize, Transpose((1, 0))), _spec((3, b), "int8"), "h2d")


@pytest.fixture
def cast_pad_recipe():
    """f32 [6] -> f16 -> padded [8] with an f16 fill of 1.0 (bits 0x3C00)."""
    from reloc_torch.recipe import Cast, Fill, Pad, Recipe
    from reloc_torch.symbolic import Const

    operations = (Cast("float16", "ieee_rne"), Pad(0, Const(1), Const(1), Fill("float16", 0x3C00)))
    return Recipe(_spec((6,), "float32"), operations, _spec((8,), "float16"), "h2d")


@pytest.fixture
def dequantize_recipe():
    """int8 [N] -> f32 with a runtime per-tensor scale and an inline zero point of -3."""
    from reloc_torch.recipe import BindingParam, Dequantize, InlineParam, Recipe
    from reloc_torch.symbolic import Symbol

    n = Symbol("s0")
    dequantize = Dequantize(
        "float32", BindingParam("s", "float32", ()), InlineParam("int32", (), (0xFFFFFFFD,)), None, "affine"
    )
    return Recipe(_spec((n,), "int8"), (dequantize,), _spec((n,), "float32"), "h2d")


def test_typed_recipe_compiles_to_a_schema_2_wire_v1_artifact(compiler, quantize_transpose_recipe):
    from reloc_torch.symbolic import GuardError

    compiled = compiler.compile(quantize_transpose_recipe)
    assert compiled.typed and compiled.wire_version == 1
    assert compiled.manifest["schema_version"] == 2
    assert compiled.symbols == ("s0",)
    assert [(p.name, p.dtype, len(p.extents)) for p in compiled.parameters] == [("s", "float32", 1)]
    assert (compiled.logical_source.dtype, compiled.logical_destination.dtype) == ("float32", "int8")
    assert pyreloc.wire_version(compiled.plan_bytes) == 1
    symbols = compiled.bind_values(torch.zeros(4, 3))
    assert symbols == {"s0": 4}
    assert compiled.parameter_extents(symbols) == {"s": ("float32", (3,))}
    plan = pyreloc.load_typed_plan(compiled.plan_bytes)
    scales = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)
    bound = pyreloc.bind_typed(plan, symbols, {"s": ("float32", [3], scales.numpy().tobytes())})
    assert (bound.source_bytes, bound.destination_bytes, bound.parameter_bytes) == (48, 12, 12)
    assert [c["bytes"] for c in bound.cuts] == [48, 12]
    assert bound.layout.typed and bound.requirements == ["typed_execution_dispatch"]
    # The frontend guards run before the binder: the source dtype is the recipe's.
    with pytest.raises(GuardError, match="source_descriptor"):
        compiled.bind_values(torch.zeros(4, 3, dtype=torch.float16))
    with pytest.raises(GuardError, match="positive_extent"):
        compiled.bind_values(torch.zeros(0, 3))
    # Same artifact, another batch: the parameter extent is the constant channel count.
    assert compiled.parameter_extents(compiled.bind_values(torch.zeros(7, 3))) == {"s": ("float32", (3,))}


def test_typed_compilation_is_deterministic_and_pins_both_digests(compiler, quantize_transpose_recipe):
    left = compiler.compile(quantize_transpose_recipe)
    right = compiler.compile(quantize_transpose_recipe)
    assert left.plan_bytes == right.plan_bytes and left.manifest == right.manifest
    assert left.manifest["plan_sha256"] == hashlib.sha256(left.plan_bytes).hexdigest()


def test_emitted_typed_mlir_spells_every_stage(quantize_transpose_recipe, cast_pad_recipe, dequantize_recipe):
    from reloc_torch.mlir_emit import emit_mlir

    text = emit_mlir(quantize_transpose_recipe)
    assert (
        'reloc.quantize %x axis 1 scale(#reloc.binding<"s" : [3], f32>) policy symmetric_rne'
        ' : !sym.tensor<["s0", 3], f32> -> !sym.tensor<["s0", 3], i8>'
    ) in text
    assert 'reloc.transpose %v0 perm [1, 0] : !sym.tensor<["s0", 3], i8> -> !sym.tensor<[3, "s0"], i8>' in text
    text = emit_mlir(cast_pad_recipe)
    assert "reloc.cast %x policy ieee_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], f16>" in text
    assert "value (0x3C00 : f16) : !sym.tensor<[6], f16> -> !sym.tensor<[8], f16>" in text
    text = emit_mlir(dequantize_recipe)
    assert (
        'reloc.dequantize %x scale(#reloc.binding<"s" : [], f32>) zero_point(dense<-3> : tensor<i32>)'
        " policy affine : !sym.tensor<[\"s0\"], i8> -> !sym.tensor<[\"s0\"], f32>"
    ) in text


def test_cast_pad_fills_and_dequantize_parameters_are_admitted(compiler, cast_pad_recipe, dequantize_recipe):
    compiled = compiler.compile(cast_pad_recipe)
    assert compiled.parameters == ()
    assert compiled.manifest["fills"] == [{"dst_axis": 0, "stage": 1, "dtype": "float16", "bits": "3c00"}]
    plan = pyreloc.load_typed_plan(compiled.plan_bytes)
    bound = pyreloc.bind_typed(plan, compiled.bind_values(torch.zeros(6)))
    # 6 f32 in; the pad enters after the cast, so the f16 boundary holds 8.
    assert (bound.source_bytes, bound.destination_bytes) == (24, 16)
    assert [c["bytes"] for c in bound.cuts] == [24, 16]

    compiled = compiler.compile(dequantize_recipe)
    assert [(p.name, p.dtype, p.extents) for p in compiled.parameters] == [("s", "float32", ())]
    stage, = compiled.manifest["stages"]
    assert stage["zero_point"] == {"kind": "inline", "dtype": "int32", "shape": [], "bits": ["fffffffd"]}
    symbols = compiled.bind_values(torch.zeros(5, dtype=torch.int8))
    assert compiled.parameter_extents(symbols) == {"s": ("float32", ())}
    plan = pyreloc.load_typed_plan(compiled.plan_bytes)
    bound = pyreloc.bind_typed(plan, symbols, {"s": ("float32", [], torch.tensor(0.5).numpy().tobytes())})
    assert (bound.source_bytes, bound.destination_bytes, bound.parameter_bytes) == (5, 20, 8)


def test_layout_only_artifacts_are_unchanged(compiler, identity_recipe):
    from reloc_torch.cache import artifact_key

    compiled = compiler.compile(identity_recipe)
    assert not compiled.typed and compiled.wire_version == 0 and compiled.parameters == ()
    assert compiled.manifest["schema_version"] == 1 and "stages" not in compiled.manifest
    assert json.loads(compiled.to_bytes())["format_version"] == 1
    key = artifact_key(identity_recipe, compiler_identity="c", runtime_capability="r")
    assert (key.wire_version, key.value_transform) == (0, "layout_only")


def test_cache_keys_separate_typed_artifacts(identity_recipe, quantize_transpose_recipe):
    from reloc_torch.cache import artifact_key

    typed = artifact_key(quantize_transpose_recipe, compiler_identity="c", runtime_capability="r")
    assert (typed.wire_version, typed.value_transform) == (1, "typed")
    assert typed != artifact_key(identity_recipe, compiler_identity="c", runtime_capability="r")
    hash(typed)


def test_portable_typed_artifact_round_trips_in_a_fresh_process(compiler, quantize_transpose_recipe, tmp_path):
    from reloc_torch.artifact import CompiledRecipe

    compiled = compiler.compile(quantize_transpose_recipe)
    path = tmp_path / "typed.reloc"
    compiled.save(path)
    assert json.loads(path.read_bytes())["format_version"] == 2
    reloaded = CompiledRecipe.load(path)
    assert reloaded == compiled
    script = (
        "import hashlib, json, sys\n"
        "from reloc_torch.artifact import CompiledRecipe\n"
        "c = CompiledRecipe.load(sys.argv[1])\n"
        "print(json.dumps({'wire': c.wire_version, 'typed': c.typed, 'symbols': list(c.symbols),"
        " 'parameters': [(p.name, p.dtype, len(p.extents)) for p in c.parameters],"
        " 'sha': hashlib.sha256(c.plan_bytes).hexdigest(), 'schema': c.manifest['schema_version']}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True, env=dict(os.environ)
    )
    assert json.loads(result.stdout) == {
        "wire": 1,
        "typed": True,
        "symbols": ["s0"],
        "parameters": [["s", "float32", 1]],
        "sha": hashlib.sha256(compiled.plan_bytes).hexdigest(),
        "schema": 2,
    }


def _mutated(compiled, mutate):
    payload = json.loads(compiled.to_bytes())
    mutate(payload)
    return json.dumps(payload).encode("utf-8")


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    [
        ("format_1_for_typed", lambda p: p.update(format_version=1), "format version does not match"),
        ("schema_1", lambda p: p["manifest"].update(schema_version=1), "schema version mismatch"),
        ("wire_0", lambda p: p["manifest"].update(wire_version=0), "wire version mismatch"),
        ("missing_section", lambda p: p["manifest"].pop("parameters"), "invalid success fields"),
        ("stage_policy", lambda p: p["manifest"]["stages"][0].update(policy="affine"), "recipe transform"),
        ("stage_axis", lambda p: p["manifest"]["stages"][0].update(axis=0), "channel axis mismatch"),
        ("stage_shape", lambda p: p["manifest"]["stages"][0].update(shape=[["const", 4], ["const", 3]]), "operand shape"),
        ("scale_name", lambda p: p["manifest"]["stages"][0]["scale"].update(name="t"), "parameter mismatch"),
        ("channel_dim", lambda p: p["manifest"]["stages"][0]["channel"].update(expr=["dim", 5]), "out of range"),
        ("channel_symbol", lambda p: p["manifest"]["stages"][0]["channel"].update(expr=["symbol", "zz"]), "not a plan symbol"),
        ("parameter_dtype", lambda p: p["manifest"]["parameters"][0].update(dtype="float16"), "recipe bindings"),
        ("extra_fill", lambda p: p["manifest"]["fills"].append({"dst_axis": 0, "stage": 0, "dtype": "int8", "bits": "0"}), "fill count"),
        ("decimal_bits", lambda p: p["manifest"]["stages"][0].update(scale={"kind": "inline", "dtype": "float32", "shape": [3], "bits": ["0.5", "0.25", "0.125"]}), "lowercase hex"),
        ("recipe_policy", lambda p: p["recipe"]["operations"][0].update(policy="affine"), "recipe transform"),
        ("recipe_layout_only", lambda p: p["recipe"]["operations"].pop(0), "format version does not match"),
        ("plan_digest", lambda p: p["manifest"].update(plan_sha256="0" * 64), "plan digest mismatch"),
    ],
)
def test_stale_or_foreign_typed_artifacts_are_rejected(compiler, quantize_transpose_recipe, name, mutate, match):
    from reloc_torch.artifact import CompiledRecipe

    compiled = compiler.compile(quantize_transpose_recipe)
    with pytest.raises(RuntimeError, match=match):
        CompiledRecipe.from_bytes(_mutated(compiled, mutate))


def test_a_v0_plan_never_passes_as_a_typed_artifact(compiler, quantize_transpose_recipe, identity_recipe):
    import base64

    from reloc_torch.artifact import CompiledRecipe

    typed = compiler.compile(quantize_transpose_recipe)
    layout = compiler.compile(identity_recipe)

    def swap_plan(payload):
        payload["plan_base64"] = base64.b64encode(layout.plan_bytes).decode("ascii")
        payload["manifest"]["plan_sha256"] = hashlib.sha256(layout.plan_bytes).hexdigest()

    with pytest.raises(RuntimeError, match="invalid plan|wire version"):
        CompiledRecipe.from_bytes(_mutated(typed, swap_plan))
    # And a layout-only artifact stays loadable as format 1.
    assert CompiledRecipe.from_bytes(layout.to_bytes()) == layout


def test_typed_recipes_are_rejected_by_the_verifier_when_ill_typed(compiler):
    """A verifier-invalid typed chain is a compiler error (exit 1), not an
    admitted artifact and not an UnsupportedRecipe."""
    from reloc_torch.recipe import Cast, Recipe

    recipe = Recipe(_spec((4,), "float32"), (Cast("float16", "exact"),), _spec((4,), "float16"), "h2d")
    with pytest.raises(RuntimeError, match="compiler exited with status 1"):
        compiler.compile(recipe)


def test_typed_fold_bails_are_declared_unsupported(compiler):
    from reloc_torch import UnsupportedRecipe
    from reloc_torch.recipe import Fill, InlineParam, Pad, Quantize, Recipe
    from reloc_torch.symbolic import Const

    # Pad before a per-channel quantize: the fill has no single channel.
    scale = InlineParam("float32", (3,), (0x3F000000, 0x3E800000, 0x3E000000))
    operations = (
        Pad(0, Const(1), Const(1), Fill("float32", 0x3F800000)),
        Quantize("int8", scale, None, 1, "symmetric_rne"),
    )
    recipe = Recipe(_spec((6, 3), "float32"), operations, _spec((8, 3), "int8"), "h2d")
    with pytest.raises(UnsupportedRecipe) as failure:
        compiler.compile(recipe)
    assert failure.value.reason == "fold_unsupported"
