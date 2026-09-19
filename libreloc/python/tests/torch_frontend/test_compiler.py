"""Real R1 exporter to frontend/runtime artifact agreement."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch


def _static_recipe(shape, operations, destination_shape, dtype="float32"):
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import Const, dense_strides

    source_shape = tuple(Const(value) for value in shape)
    destination_shape = tuple(Const(value) for value in destination_shape)
    return Recipe(
        TensorSpec(source_shape, dense_strides(source_shape), Const(0), dtype),
        tuple(operations),
        TensorSpec(destination_shape, dense_strides(destination_shape), Const(0), dtype),
        "h2d",
    )


def _transpose_recipe(shape=(2, 3), dtype="float32"):
    from reloc_torch.recipe import Transpose

    return _static_recipe(shape, (Transpose((1, 0)),), tuple(reversed(shape)), dtype)


def _reshape_recipe(shape, destination_shape, dtype="float32"):
    from reloc_torch.recipe import Reshape
    from reloc_torch.symbolic import Const

    return _static_recipe(
        shape,
        (Reshape(tuple(Const(value) for value in destination_shape)),),
        destination_shape,
        dtype,
    )


def _relocate(compiled, source, destination_shape):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    plan = pyreloc.load_plan(compiled.plan_bytes)
    bound = pyreloc.bind(plan, compiled.bind_values(source))
    actual = np.empty(int(np.prod(destination_shape)), dtype=source.numpy().dtype)
    pyreloc.relocate(bound, *as_ptr(source), *as_ptr(actual))
    return actual.reshape(destination_shape), bound


def _mutating_exporter(tmp_path, compiler, mode):
    script = tmp_path / f"export-{mode}.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, os, pathlib, subprocess, sys\n"
        f"mode = {mode!r}\n"
        f"real = {str(compiler.executable)!r}\n"
        "if mode == 'crash':\n"
        "  print('deliberate compiler crash', file=sys.stderr)\n"
        "  raise SystemExit(17)\n"
        "result = subprocess.run([real, *sys.argv[1:]])\n"
        "if result.returncode != 0:\n"
        "  raise SystemExit(result.returncode)\n"
        "manifest_path = pathlib.Path(sys.argv[sys.argv.index('--manifest') + 1])\n"
        "plan_path = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "if mode == 'malformed':\n"
        "  manifest_path.write_text('{broken')\n"
        "  raise SystemExit(0)\n"
        "if mode == 'missing_manifest':\n"
        "  manifest_path.unlink()\n"
        "  raise SystemExit(0)\n"
        "if mode == 'missing_plan':\n"
        "  plan_path.unlink()\n"
        "  raise SystemExit(0)\n"
        "manifest = json.loads(manifest_path.read_text())\n"
        "if mode == 'missing_symbols': manifest['symbols'] = []\n"
        "elif mode == 'dtype': manifest['logical_source']['dtype'] = 'float16'\n"
        "elif mode == 'constraint': manifest['constraints']['divisibility'] = []\n"
        "elif mode == 'plan_digest': manifest['plan_sha256'] = '0' * 64\n"
        "elif mode == 'stale': manifest['input_sha256'] = os.environ['T3_STALE_DIGEST']\n"
        "elif mode == 'bool_schema': manifest['schema_version'] = True\n"
        "elif mode == 'bool_extent': manifest['logical_source']['shape'][0] = ['const', True]\n"
        "elif mode == 'symbol_order': manifest['symbols'].reverse()\n"
        "elif mode == 'malformed_plan':\n"
        "  plan_path.write_bytes(plan_path.read_bytes()[:-1])\n"
        "  manifest['plan_sha256'] = hashlib.sha256(plan_path.read_bytes()).hexdigest()\n"
        "manifest_path.write_text(json.dumps(manifest))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_real_export_is_deterministic_across_processes(compiler, split_transpose_recipe):
    left = compiler.compile(split_transpose_recipe)
    right = compiler.compile(split_transpose_recipe)
    assert left.plan_bytes == right.plan_bytes
    assert left.manifest == right.manifest
    assert left.symbols == ("s0",)
    assert left.wire_version == 0
    assert left.compiler.name == "sym-reloc-export"


def test_one_artifact_multiple_bindings(compiler, split_transpose_recipe):
    import pyreloc
    from pyreloc.torch_interop import as_ptr

    compiled = compiler.compile(split_transpose_recipe)
    plan = pyreloc.load_plan(compiled.plan_bytes)
    for n in (64, 128, 192):
        x = torch.arange(n, dtype=torch.float32)
        symbols = compiled.bind_values(x)
        bound = pyreloc.bind(plan, symbols)
        actual = np.empty(n, dtype=np.float32)
        pyreloc.relocate(bound, *as_ptr(x), *as_ptr(actual))
        expected = x.reshape(n // 64, 64).t().contiguous().numpy()
        np.testing.assert_array_equal(actual.reshape(expected.shape), expected)


def test_actual_captured_fx_candidate_compiles_and_rebinds(compiler):
    from reloc_torch import import_graph
    from reloc_torch.compat import symbolic_capture

    def transfer(x):
        return torch.ops.aten._to_copy.default(x, device=torch.device("cuda:0"))

    gm = symbolic_capture(
        lambda x: transfer(x.reshape(x.shape[0] // 64, 64).transpose(0, 1).contiguous()),
        torch.ones(128),
    )
    candidate, = import_graph(gm).candidates
    compiled = compiler.compile(candidate.recipe)
    for n in (128, 192):
        source = torch.arange(n, dtype=torch.float32)
        actual, _ = _relocate(compiled, source, (64, n // 64))
        expected = source.reshape(n // 64, 64).t().contiguous().numpy()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("name", "recipe", "source", "destination_shape", "expected"),
    [
        (
            "identity",
            lambda: _static_recipe((2, 3), (), (2, 3)),
            lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3),
            (2, 3),
            lambda x: x.numpy(),
        ),
        (
            "transpose",
            _transpose_recipe,
            lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3),
            (3, 2),
            lambda x: x.t().contiguous().numpy(),
        ),
        (
            "static_merge",
            lambda: _reshape_recipe((2, 3), (6,)),
            lambda: torch.arange(6, dtype=torch.float32).reshape(2, 3),
            (6,),
            lambda x: x.reshape(6).numpy(),
        ),
        (
            "static_split",
            lambda: _reshape_recipe((6,), (2, 3)),
            lambda: torch.arange(6, dtype=torch.float32),
            (2, 3),
            lambda x: x.reshape(2, 3).numpy(),
        ),
    ],
)
def test_static_layout_recipe_matrix(compiler, name, recipe, source, destination_shape, expected):
    source = source()
    actual, _ = _relocate(compiler.compile(recipe()), source, destination_shape)
    np.testing.assert_array_equal(actual, expected(source))


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "numpy_dtype"),
    [
        ("float32", torch.float32, np.float32),
        ("float16", torch.float16, np.float16),
        ("int8", torch.int8, np.int8),
    ],
)
def test_layout_only_dtypes_preserve_exact_bytes(compiler, dtype, torch_dtype, numpy_dtype):
    recipe = _transpose_recipe(dtype=dtype)
    source = torch.arange(6, dtype=torch_dtype).reshape(2, 3)
    actual, _ = _relocate(compiler.compile(recipe), source, (3, 2))
    expected = source.t().contiguous().numpy().astype(numpy_dtype, copy=False)
    assert actual.tobytes() == expected.tobytes()


def test_constant_pad_and_transpose_pad_values(compiler):
    from reloc_torch.recipe import Fill, Pad, Transpose
    from reloc_torch.symbolic import Const

    pad = _static_recipe(
        (4,),
        (Pad(0, Const(1), Const(1), Fill("float32", 0x3FC00000)),),
        (6,),
    )
    source = torch.arange(4, dtype=torch.float32)
    actual, _ = _relocate(compiler.compile(pad), source, (6,))
    np.testing.assert_array_equal(actual, np.pad(source.numpy(), (1, 1), constant_values=np.float32(1.5)))

    transpose_pad = _static_recipe(
        (2, 3),
        (
            Transpose((1, 0)),
            Pad(0, Const(1), Const(2), Fill("float32", 0x80000000)),
        ),
        (6, 2),
    )
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    actual, _ = _relocate(compiler.compile(transpose_pad), source, (6, 2))
    expected = np.pad(source.t().contiguous().numpy(), ((1, 2), (0, 0)), constant_values=np.float32(-0.0))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "numpy_dtype", "bits", "fill"),
    [
        ("float32", torch.float32, np.float32, 0x3EAAAAAB, np.array([0x3EAAAAAB], dtype=np.uint32).view(np.float32)[0]),
        ("float16", torch.float16, np.float16, 0x3555, np.array([0x3555], dtype=np.uint16).view(np.float16)[0]),
        ("int8", torch.int8, np.int8, 0xFE, np.int8(-2)),
    ],
)
def test_pad_fill_dtypes_preserve_representable_bits(
    compiler, dtype, torch_dtype, numpy_dtype, bits, fill
):
    from reloc_torch.recipe import Fill, Pad
    from reloc_torch.symbolic import Const

    recipe = _static_recipe(
        (3,),
        (Pad(0, Const(1), Const(2), Fill(dtype, bits)),),
        (6,),
        dtype,
    )
    source = torch.arange(3, dtype=torch_dtype)
    actual, _ = _relocate(compiler.compile(recipe), source, (6,))
    expected = np.pad(source.numpy(), (1, 2), constant_values=fill).astype(numpy_dtype, copy=False)
    assert actual.tobytes() == expected.tobytes()


def test_guards_precede_standalone_binding_and_invalid_binds_fail(compiler, split_transpose_recipe):
    import pyreloc
    from reloc_torch.symbolic import GuardError

    compiled = compiler.compile(split_transpose_recipe)
    plan = pyreloc.load_plan(compiled.plan_bytes)
    launched = False
    for source, reason in (
        (torch.empty(0, dtype=torch.float32), "positive_extent"),
        (torch.empty(130, dtype=torch.float32), "divisibility"),
    ):
        with pytest.raises(GuardError, match=reason):
            symbols = compiled.bind_values(source)
            launched = True
            pyreloc.bind(plan, symbols)
    assert launched is False
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind(plan, {})
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind(plan, {"s0": 128, "extra": 1})
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind(plan, {"s0": 130})

    class HugeTensor:
        shape = (2**63,)
        dtype = "float32"

        @staticmethod
        def stride():
            return (1,)

        @staticmethod
        def storage_offset():
            return 0

    with pytest.raises(GuardError, match="integer_overflow"):
        compiled.bind_values(HugeTensor())


def test_repeated_symbol_guard_and_logical_rank_survive_plan_coalescing(compiler):
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import Const, GuardError, Symbol

    n = Symbol("s0")
    descriptor = TensorSpec((n, n), (n, Const(1)), Const(0), "float32")
    compiled = compiler.compile(Recipe(descriptor, (), descriptor, "h2d"))
    with pytest.raises(GuardError, match="repeated_symbol"):
        compiled.bind_values(torch.empty(4, 5, dtype=torch.float32))
    assert len(compiled.logical_source.shape) == 2
    assert len(compiled.logical_destination.shape) == 2
    assert tuple(expr.evaluate({"s0": 4}) for expr in compiled.logical_destination.shape) == (4, 4)
    import pyreloc
    bound = pyreloc.bind(pyreloc.load_plan(compiled.plan_bytes), compiled.bind_values(torch.empty(4, 4)))
    assert bound.extents == [16]


def test_real_fold_bail_preserves_candidate_original(compiler):
    from reloc_torch import UnsupportedRecipe
    from reloc_torch.fx_import import Candidate
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const

    source = TensorSpec((Const(2), Const(3)), (Const(3), Const(1)), Const(0), "float32")
    destination = TensorSpec((Const(6),), (Const(1),), Const(0), "float32")
    recipe = Recipe(source, (Transpose((1, 0)), Reshape((Const(6),))), destination, "h2d")
    original = lambda x: x.t().reshape(6)
    candidate = Candidate("x", "reshape", ("transpose", "reshape"), recipe, (), (), original)
    with pytest.raises(UnsupportedRecipe) as failure:
        compiler.compile(candidate.recipe)
    assert failure.value.reason == "fold_unsupported"
    assert torch.equal(candidate.original(torch.arange(6).reshape(2, 3)), torch.tensor([0, 3, 1, 4, 2, 5]))


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("malformed", "malformed manifest"),
        ("missing_manifest", "complete plan/manifest pair"),
        ("missing_plan", "complete plan/manifest pair"),
        ("missing_symbols", "symbol"),
        ("dtype", "source descriptor mismatch"),
        ("constraint", "constraints mismatch"),
        ("plan_digest", "plan digest mismatch"),
        ("malformed_plan", "invalid plan"),
        ("bool_schema", "must be an integer"),
        ("bool_extent", "must be an integer"),
        ("crash", "status 17.*deliberate compiler crash"),
    ],
)
def test_malformed_or_mismatched_exports_are_rejected(
    compiler, split_transpose_recipe, tmp_path, mode, match
):
    from reloc_torch import CompilerClient

    client = CompilerClient(_mutating_exporter(tmp_path, compiler, mode))
    with pytest.raises(RuntimeError, match=match):
        client.compile(split_transpose_recipe)


def test_stale_manifest_for_same_shape_identity_is_rejected(
    compiler, tmp_path, monkeypatch
):
    from reloc_torch import CompilerClient
    from reloc_torch.mlir_emit import emit_mlir
    from reloc_torch.recipe import Transpose

    identity = _static_recipe((4, 4), (), (4, 4))
    transpose = _static_recipe((4, 4), (Transpose((1, 0)),), (4, 4))
    monkeypatch.setenv("T3_STALE_DIGEST", hashlib.sha256(emit_mlir(transpose).encode()).hexdigest())
    client = CompilerClient(_mutating_exporter(tmp_path, compiler, "stale"))
    with pytest.raises(RuntimeError, match="input digest mismatch"):
        client.compile(identity)


def test_manifest_symbols_must_match_exact_wire_order(compiler, tmp_path):
    from reloc_torch import CompilerClient
    from reloc_torch.recipe import Recipe, TensorSpec
    from reloc_torch.symbolic import Const, Symbol

    rows, columns = Symbol("s0"), Symbol("s1")
    descriptor = TensorSpec(
        (rows, columns),
        (columns, Const(1)),
        Const(0),
        "float32",
    )
    recipe = Recipe(descriptor, (), descriptor, "h2d")
    client = CompilerClient(_mutating_exporter(tmp_path, compiler, "symbol_order"))
    with pytest.raises(RuntimeError, match="symbol order"):
        client.compile(recipe)


def test_portable_artifact_reloads_and_executes_in_fresh_process(
    compiler, split_transpose_recipe, tmp_path
):
    compiled = compiler.compile(split_transpose_recipe)
    assert compiled.to_bytes() == compiler.compile(split_transpose_recipe).to_bytes()
    artifact = tmp_path / "split-transpose.reloc.json"
    compiled.save(artifact)
    program = """
import numpy as np
import pyreloc
import torch
from pyreloc.torch_interop import as_ptr
from reloc_torch import CompiledRecipe
c = CompiledRecipe.load(sys.argv[1])
x = torch.arange(192, dtype=torch.float32)
b = pyreloc.bind(pyreloc.load_plan(c.plan_bytes), c.bind_values(x))
y = np.empty(192, dtype=np.float32)
pyreloc.relocate(b, *as_ptr(x), *as_ptr(y))
expected = x.reshape(3, 64).t().contiguous().numpy()
assert y.reshape(expected.shape).tobytes() == expected.tobytes()
assert c.recipe.destination == c.logical_destination
print('portable artifact passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", "import sys;" + program, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable artifact passed"


def test_portable_artifact_revalidates_recipe_association(compiler, split_transpose_recipe):
    from reloc_torch import CompiledRecipe

    payload = json.loads(compiler.compile(split_transpose_recipe).to_bytes())
    payload["recipe"]["operations"] = []
    tampered = json.dumps(payload).encode()
    with pytest.raises(RuntimeError, match="constraints mismatch|destination descriptor mismatch|input digest mismatch"):
        CompiledRecipe.from_bytes(tampered)


def test_compiler_artifact_api_remains_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from reloc_torch import CompilerClient, CompiledRecipe, UnsupportedRecipe; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
