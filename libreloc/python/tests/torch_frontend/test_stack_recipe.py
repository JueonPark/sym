"""Stacked recipes (torch.stack support): Recipe.stack_inputs, the
compiled artifact, its serialization and the guarded logical binding."""
import json

import pytest
import torch

from conftest import stacked_recipe
from reloc_torch.recipe import Cast, Recipe, TensorSpec
from reloc_torch.symbolic import Const, GuardError, Symbol


def test_stack_inputs_must_describe_a_layout_only_h2d_logical_source():
    good = stacked_recipe()
    assert good.stack_inputs == 3 and not good.typed
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError, match="stack_inputs"):
            Recipe(good.source, good.operations, good.destination, "h2d", stack_inputs=bad)
    with pytest.raises(ValueError, match="leading"):
        Recipe(good.source, good.operations, good.destination, "h2d", stack_inputs=4)
    with pytest.raises(ValueError, match="host-to-device"):
        Recipe(good.source, good.operations, good.destination, "d2h", stack_inputs=3)
    half = TensorSpec(good.destination.shape, good.destination.strides, Const(0), "float16")
    with pytest.raises(ValueError, match="layout-only"):
        Recipe(good.source, good.operations + (Cast("float16", "ieee_rne"),), half, "h2d", stack_inputs=3)
    assert Recipe(good.source, good.operations, good.destination, "h2d").stack_inputs == 0


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stacked_recipe_compiles_through_the_unchanged_exporter(compiler, dim):
    compiled = compiler.compile(stacked_recipe(dim=dim))
    assert compiled.recipe.stack_inputs == 3
    assert compiled.logical_source.shape == (Const(3), Symbol("s0"), Symbol("s1"))
    assert {s.name: s.axis for s in compiled.symbol_sources} == {"s0": 1, "s1": 2}
    assert not compiled.typed
    assert compiled.decoded_plan is not None


def test_serialization_round_trips_stack_inputs_and_keeps_plain_artifacts_unchanged(compiler, tmp_path):
    from reloc_torch.artifact import CompiledRecipe

    stacked = compiler.compile(stacked_recipe())
    path = tmp_path / "stacked.reloc.json"
    stacked.save(path)
    payload = json.loads(path.read_text())
    assert payload["format_version"] == 1
    assert payload["recipe"]["stack_inputs"] == 3
    loaded = CompiledRecipe.load(path)
    assert loaded.recipe == stacked.recipe and loaded.plan_bytes == stacked.plan_bytes
    plain = compiler.compile(Recipe(stacked.recipe.source, stacked.recipe.operations,
                                    stacked.recipe.destination, "h2d"))
    assert "stack_inputs" not in json.loads(plain.to_bytes())["recipe"]
    bad = dict(payload, recipe=dict(payload["recipe"], stack_inputs=0))
    with pytest.raises(RuntimeError, match="stack_inputs"):
        CompiledRecipe.from_bytes(json.dumps(bad).encode())


def test_bind_stacked_values_guards_every_input(compiler):
    compiled = compiler.compile(stacked_recipe())
    xs = [torch.ones(4, 5) for _ in range(3)]
    assert compiled.bind_stacked_values(xs) == {"s0": 4, "s1": 5}
    cases = [
        (xs[:2], "stack_count"),
        ([xs[0], torch.ones(4, 5, dtype=torch.float16), xs[2]], "stack_dtype_mismatch"),
        ([xs[0], torch.ones(4, 6), xs[2]], "stack_shape_mismatch"),
        ([torch.ones(5, 4).t(), xs[1], xs[2]], "source_descriptor"),
    ]
    for sources, reason in cases:
        with pytest.raises(GuardError) as error:
            compiled.bind_stacked_values(sources)
        assert error.value.reason == reason


def test_stacked_and_plain_recipes_never_share_a_cache_key():
    from reloc_torch.cache import artifact_key

    stacked = stacked_recipe()
    plain = Recipe(stacked.source, stacked.operations, stacked.destination, "h2d")
    keys = {artifact_key(r, compiler_identity="c", runtime_capability="r") for r in (stacked, plain)}
    assert len(keys) == 2
