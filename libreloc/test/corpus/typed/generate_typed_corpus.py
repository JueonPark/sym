#!/usr/bin/env python3
"""Regenerate the typed conformance corpus (C4, issue #144).

Every fixture is a frontend ``Recipe`` with value transforms that is

  1. emitted as MLIR (``reloc_torch.mlir_emit``),
  2. compiled through the PUBLIC exporter (``sym-reloc-export --typed``,
     ``reloc_torch.compiler.CompilerClient``) into a wire v1 plan and a
     schema-2 manifest, admitted by the frontend bridge,
  3. replayed by the independent NumPy oracle (``typed_reference.py``) on a
     deterministic source for one or more symbol bindings,
  4. cross-checked once against the runtime's forced CPU baseline
     (``pyreloc.prepare_dispatch`` host to host) before anything is written,
  5. committed as ``<name>.bin`` (plan bytes), ``<name>.reloc`` (the portable
     ``CompiledRecipe`` payload, format 2) and ``<name>.json`` (recipe JSON,
     manifest, MLIR text, bindings with source/expected bit patterns, exact
     parameter bits, footprints and capability rows).

Deterministic: fixed vectors plus seeded random fill; the exporter is
deterministic per build. Run from a configured build environment::

    export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
    export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
    python libreloc/test/corpus/typed/generate_typed_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "libreloc" / "python" / "tests"))
sys.path.append(str(REPO / "libreloc" / "python"))  # after the build tree (pyreloc lives there)

import typed_reference as ref  # noqa: E402

from reloc_torch.artifact import CompiledRecipe  # noqa: E402
from reloc_torch.compiler import CompilerClient  # noqa: E402
from reloc_torch.mlir_emit import emit_mlir  # noqa: E402
from reloc_torch.recipe import (  # noqa: E402
    BindingParam,
    Cast,
    Dequantize,
    Fill,
    InlineParam,
    Pad,
    Quantize,
    Recipe,
    Reshape,
    TensorSpec,
    Transpose,
)
from reloc_torch.symbolic import Const, Symbol, add, dense_strides  # noqa: E402


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def spec(shape, dtype):
    shape = tuple(Const(d) if isinstance(d, int) else d for d in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


def inline_f32(*values):
    return InlineParam("float32", () if len(values) == 1 else (len(values),), tuple(f32bits(v) for v in values))


def inline_i32(value):
    return InlineParam("int32", (), (value & 0xFFFFFFFF,))


S0 = Symbol("s0")

# C1 §6 vectors.
HALF_BOUNDARIES = [2.0 ** -24, 2.0 ** -25, 1.5 * 2.0 ** -25, 2.0 ** -14, 65504.0, 65519.99, 65520.0,
                   1e5, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -12, -0.0, float("inf"), -float("inf")]
HALF_WITNESS_BITS = [0x0001, 0x0400, 0x7BFF, 0x7C00, 0xFC00, 0x8000, 0x3C00, 0x3C01]
QUANT_TIES = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
QUANT_LIMITS = [-129.0, -128.0, -127.0, 126.0, 127.0, 128.0]
SPECIALS = [float("nan"), float("inf"), -float("inf"), -0.0]


def random_f32(rng, count, low=-70.0, high=70.0):
    return rng.uniform(low, high, size=count).astype(np.float32)


def fill_f32(rng, shape, head, low=-70.0, high=70.0):
    count = int(np.prod(shape))
    values = list(np.asarray(head, dtype=np.float32))[:count]
    values += list(random_f32(rng, count - len(values), low, high))
    return np.asarray(values, dtype=np.float32).reshape(shape)


def fill_i8(rng, shape, head=()):
    count = int(np.prod(shape))
    values = list(np.asarray(head, dtype=np.int8))[:count]
    values += list(rng.integers(-128, 128, size=count - len(values), dtype=np.int64).astype(np.int8))
    return np.asarray(values, dtype=np.int8).reshape(shape)


def fill_f16_bits(rng, shape, head=()):
    count = int(np.prod(shape))
    bits = list(head)[:count]
    while len(bits) < count:
        b = int(rng.integers(0, 0x10000))
        if (b & 0x7C00) != 0x7C00:  # finite only: NaN payloads are outside conformance
            bits.append(b)
    return ref.from_bits("float16", bits, tuple(shape))


# A fixture: recipe, description, parameter arrays, bindings (symbol values)
# and a deterministic source builder (symbols, rng) -> array.
FIXTURES = []


def fixture(name, description, recipe, source, bindings=({},), parameters=None):
    FIXTURES.append(dict(name=name, description=description, recipe=recipe, source=source,
                         bindings=tuple(bindings), parameters=parameters or {}))


fixture(
    "cast_transpose_f16",
    "layout + cast: f32 [4, 6] transposed then narrowed to f16 with C1's binary16 boundary vectors",
    Recipe(spec((4, 6), "float32"), (Transpose((1, 0)), Cast("float16", "ieee_rne")), spec((6, 4), "float16"), "h2d"),
    lambda symbols, rng: fill_f32(rng, (4, 6), HALF_BOUNDARIES, -70000.0, 70000.0),
)
fixture(
    "widen_reshape",
    "cast + layout: f16 [2, 12] widened exactly then reshaped to [6, 4], C1's binary16 -> binary32 witness bits first",
    Recipe(spec((2, 12), "float16"), (Cast("float32", "exact"), Reshape((Const(6), Const(4)))), spec((6, 4), "float32"), "h2d"),
    lambda symbols, rng: fill_f16_bits(rng, (2, 12), HALF_WITNESS_BITS),
)
fixture(
    "quantize_channel_transpose_sym",
    "layout + quantize: f32 [s0, 3] per-channel (runtime scale over 3 channels) then transposed to [3, s0]; two bindings",
    Recipe(spec((S0, 3), "float32"),
           (Quantize("int8", BindingParam("s", "float32", (Const(3),)), None, 1, "symmetric_rne"), Transpose((1, 0))),
           spec((3, S0), "int8"), "h2d"),
    lambda symbols, rng: fill_f32(rng, (symbols["s0"], 3), SPECIALS),
    bindings=({"s0": 5}, {"s0": 8}),
    parameters={"s": np.array([1.0, 0.5, 0.25], dtype=np.float32)},
)
fixture(
    "quantize_then_pad",
    "pad order witness: quantize at scale 0.5, then pad the s8 result with code 1 (stays 1)",
    Recipe(spec((6,), "float32"),
           (Quantize("int8", inline_f32(0.5), None, None, "symmetric_rne"), Pad(0, Const(1), Const(1), Fill("int8", 1))),
           spec((8,), "int8"), "h2d"),
    lambda symbols, rng: np.array([0.3, 0.75, -0.75, 2.0, -128.0, 100.0], dtype=np.float32),
)
fixture(
    "pad_then_quantize",
    "pad order witness: pad f32 1.0 first, then quantize at scale 0.5 (the fill becomes code 2)",
    Recipe(spec((6,), "float32"),
           (Pad(0, Const(1), Const(1), Fill("float32", f32bits(1.0))), Quantize("int8", inline_f32(0.5), None, None, "symmetric_rne")),
           spec((8,), "int8"), "h2d"),
    lambda symbols, rng: np.array([0.3, 0.75, -0.75, 2.0, -128.0, 100.0], dtype=np.float32),
)
fixture(
    "quant_dequant_padded",
    "f32 -> s8 -> f32 with a padded destination: distinct source, wire and destination footprints; two bindings",
    Recipe(spec((S0,), "float32"),
           (Quantize("int8", inline_f32(0.5), None, None, "symmetric_rne"),
            Dequantize("float32", inline_f32(0.5), None, None, "affine"),
            Pad(0, Const(1), Const(2), Fill("float32", f32bits(1.5)))),
           spec((add(S0, Const(3)),), "float32"), "h2d"),
    lambda symbols, rng: fill_f32(rng, (symbols["s0"],), QUANT_TIES + SPECIALS + [0.3, 64.0, -64.5]),
    bindings=({"s0": 13}, {"s0": 6}),
)
fixture(
    "dequant_channel_runtime_transpose",
    "dequantize + layout: s8 [2, 3, 4] per-channel runtime scales and zero points on axis 1, then transposed to [3, 2, 4]",
    Recipe(spec((2, 3, 4), "int8"),
           (Dequantize("float32", BindingParam("scales", "float32", (Const(3),)), BindingParam("zero_points", "int32", (Const(3),)), 1, "affine"),
            Transpose((1, 0, 2))),
           spec((3, 2, 4), "float32"), "h2d"),
    lambda symbols, rng: (np.arange(24, dtype=np.int64) * 37 % 256 - 128).astype(np.int8).reshape(2, 3, 4),
    parameters={"scales": np.array([0.5, 1.0, 2.0], dtype=np.float32), "zero_points": np.array([-128, 0, 127], dtype=np.int32)},
)
fixture(
    "dequant_inline_zp_cast",
    "dequantize -> cast: s8 [8] affine with scale 0.25 and zero point -3, then narrowed to f16 (both roundings kept)",
    Recipe(spec((8,), "int8"),
           (Dequantize("float32", inline_f32(0.25), inline_i32(-3), None, "affine"), Cast("float16", "ieee_rne")),
           spec((8,), "float16"), "h2d"),
    lambda symbols, rng: np.array([-128, -1, 0, 1, 127, 5, -3, 100], dtype=np.int8),
)
fixture(
    "witness_channel",
    "C1's channel witness: [2, 3, 4], axis 1, scales [0.5, 1, 2], x[i] = i - 11.5, channel axis moved to result axis 0",
    Recipe(spec((2, 3, 4), "float32"),
           (Quantize("int8", inline_f32(0.5, 1.0, 2.0), None, 1, "symmetric_rne"), Transpose((1, 0, 2))),
           spec((3, 2, 4), "int8"), "h2d"),
    lambda symbols, rng: (np.arange(24, dtype=np.float32) - np.float32(11.5)).reshape(2, 3, 4),
)
fixture(
    "quantize_edges",
    "symmetric_rne edge cases at scale 0.5: ties, signed-i8 limits, NaN/inf, signed zero, a subnormal and a non-multiple",
    Recipe(spec((16,), "float32"), (Quantize("int8", inline_f32(0.5), None, None, "symmetric_rne"),), spec((16,), "int8"), "h2d"),
    lambda symbols, rng: np.array([v / 2 for v in QUANT_TIES] + [v / 2 for v in QUANT_LIMITS] + SPECIALS[:3] + [1e-40], dtype=np.float32),
)
fixture(
    "reciprocal_formation",
    "the reciprocal is formed once in binary32 from scale 0.3 (x * fl32(1/0.3), not x / 0.3)",
    Recipe(spec((8,), "float32"), (Quantize("int8", inline_f32(0.3), None, None, "symmetric_rne"),), spec((8,), "int8"), "h2d"),
    lambda symbols, rng: np.array([0.1, 0.3, 0.7, 1.1, 2.9, 10.1, 100.3, 1000.7], dtype=np.float32),
)
fixture(
    "dequant_scale_witness",
    "C1's dequantize bits: q in [-128, -1, 0, 1, 127], scale 0.3, zero point 0",
    Recipe(spec((5,), "int8"), (Dequantize("float32", inline_f32(0.3), None, None, "affine"),), spec((5,), "float32"), "h2d"),
    lambda symbols, rng: np.array([-128, -1, 0, 1, 127], dtype=np.int8),
)
fixture(
    "dequant_zp127",
    "C1's dequantize bits with zero point 127: q = -128 gives (-255) * 0.3 = 0xC2990000",
    Recipe(spec((5,), "int8"), (Dequantize("float32", inline_f32(0.3), inline_i32(127), None, "affine"),), spec((5,), "float32"), "h2d"),
    lambda symbols, rng: np.array([-128, -1, 0, 1, 127], dtype=np.int8),
)


def hex_bits(array):
    return [format(b, "x") for b in ref.to_bits(array)]


def view(array, kind="host"):
    import pyreloc

    itemsize = array.dtype.itemsize
    return pyreloc.BufferView(array.ctypes.data, array.nbytes, 0, list(array.shape),
                              [s // itemsize for s in array.strides], itemsize, kind)


def parameter_map(parameters):
    return {name: (value.dtype.name, list(value.shape), value.tobytes()) for name, value in parameters.items()}


def rows(bound, direction, device):
    import pyreloc

    return [r["implementation"] for r in pyreloc.query_capability(bound, direction, device)["eligible"]]


def build(entry, compiler, seed):
    import pyreloc

    recipe = entry["recipe"]
    compiled = compiler.compile(recipe)
    assert compiled.typed and compiled.wire_version == 1, entry["name"]
    payload = json.loads(compiled.to_bytes())
    recipe_json = payload["recipe"]
    plan = pyreloc.load_typed_plan(compiled.plan_bytes)
    parameters = entry["parameters"]
    bindings = []
    for symbols in entry["bindings"]:
        rng = np.random.default_rng(seed)
        source = np.ascontiguousarray(entry["source"](symbols, rng))
        expected = ref.replay(recipe_json, source, symbols, parameters)
        wire_symbols = {name: symbols[name] for name in compiled.symbols}
        bound = pyreloc.bind_typed(plan, wire_symbols, parameter_map(parameters))
        footprints = {}
        for direction in ("h2d", "d2h"):
            actual = np.zeros(expected.shape, dtype=expected.dtype)
            request = pyreloc.prepare_dispatch(bound, view(source), view(actual), direction, policy="original_cpu")
            report = pyreloc.execute_dispatch(request)
            if actual.tobytes() != expected.tobytes():
                raise SystemExit(f"{entry['name']} ({symbols}, {direction}): runtime disagrees with the oracle")
            footprints[direction] = {k: report[k] for k in ("source_bytes", "wire_bytes", "destination_bytes", "parameter_bytes", "wire_boundary")}
        bindings.append({
            "symbols": symbols,
            "source": {"dtype": source.dtype.name, "shape": list(source.shape), "bits": hex_bits(source)},
            "expected": {"dtype": expected.dtype.name, "shape": list(expected.shape), "bits": hex_bits(expected),
                         "sha256": hashlib.sha256(expected.tobytes()).hexdigest()},
            "footprints": footprints,
            "capability": {device: {direction: rows(bound, direction, device) for direction in ("h2d", "d2h")}
                           for device in ("host", "cuda")},
        })
    meta = {
        "name": entry["name"],
        "description": entry["description"],
        "generator_seed": seed,
        "mlir": emit_mlir(recipe),
        "recipe": recipe_json,
        # The compiler block (build identity) changes with every build and is
        # not part of what the fixture pins; the .reloc payload keeps it.
        "manifest": {k: v for k, v in compiled.manifest.items() if k != "compiler"},
        "plan_sha256": hashlib.sha256(compiled.plan_bytes).hexdigest(),
        "symbols": list(compiled.symbols),
        "parameters": {name: {"dtype": value.dtype.name, "shape": list(value.shape), "bits": hex_bits(value)}
                       for name, value in parameters.items()},
        "stages": [{"transform": s["transform"], "policy": s["policy"], "input_dtype": s["input_dtype"],
                    "output_dtype": s["output_dtype"]} for s in compiled.manifest["stages"]],
        "bindings": bindings,
    }
    return compiled, meta


def normalized(path, data):
    """The portable payload carries the compiler's build identity, which a
    fresh build changes legitimately; everything else must be identical."""
    if path.suffix != ".reloc":
        return data
    payload = json.loads(data)
    payload["manifest"].pop("compiler", None)
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--check", action="store_true", help="compare against the committed files instead of writing")
    args = parser.parse_args()
    compiler = CompilerClient.from_environment()
    mismatches = []
    for entry in FIXTURES:
        compiled, meta = build(entry, compiler, args.seed)
        # The manifest's build identity changes with every compiler build;
        # keep it in the file but never compare it.
        text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
        outputs = {
            HERE / f"{entry['name']}.bin": compiled.plan_bytes,
            HERE / f"{entry['name']}.reloc": compiled.to_bytes(),
            HERE / f"{entry['name']}.json": text.encode("utf-8"),
        }
        for path, data in outputs.items():
            if args.check:
                if not path.exists() or normalized(path, path.read_bytes()) != normalized(path, data):
                    mismatches.append(path.name)
            else:
                path.write_bytes(data)
        print(f"{entry['name']}: {len(compiled.plan_bytes)} plan bytes, {len(meta['bindings'])} binding(s)")
    if mismatches:
        raise SystemExit("committed corpus differs from a fresh generation: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
