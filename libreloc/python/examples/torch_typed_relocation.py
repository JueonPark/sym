#!/usr/bin/env python3
"""Explicit typed relocation through the public path (C4, issue #144).

Two typed programs are compiled with the real exporter (``--typed``),
bound with runtime parameters supplied by name, executed, validated against
the independent NumPy reference and reported with their path and byte
counters:

* ``layout + cast``: f32 [B, 6] transposed then narrowed to f16;
* ``layout + quantize / dequantize``: f32 [B, 3] quantized per channel with
  a runtime scale, transposed, and the reverse program that dequantizes
  s8 [3, B] with runtime scales and zero points.

``--host`` (default) runs Torch-free, host to host, through ``pyreloc``:
the forced CPU baseline exactly as CI exercises it. ``--cuda`` runs the same
artifacts through ``reloc_torch.dispatch`` on a real GPU with both policies
and prints the row each one selected. Symbolic rebinding is shown by running
each artifact for two batch sizes.

Environment (see docs/typed-relocation-support.md):
    export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
    export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "libreloc" / "python" / "tests"))  # typed_reference (the oracle)
sys.path.append(str(REPO / "libreloc" / "python"))

import typed_reference as ref  # noqa: E402

from reloc_torch.compiler import CompilerClient  # noqa: E402
from reloc_torch.recipe import BindingParam, Cast, Dequantize, Quantize, Recipe, TensorSpec, Transpose  # noqa: E402
from reloc_torch.symbolic import Const, Symbol, dense_strides  # noqa: E402

B = Symbol("s0")


def spec(shape, dtype):
    shape = tuple(Const(d) if isinstance(d, int) else d for d in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


RECIPES = {
    "layout_cast": Recipe(spec((B, 6), "float32"), (Transpose((1, 0)), Cast("float16", "ieee_rne")),
                          spec((6, B), "float16"), "h2d"),
    "layout_quantize": Recipe(spec((B, 3), "float32"),
                              (Quantize("int8", BindingParam("scale", "float32", (Const(3),)), None, 1, "symmetric_rne"),
                               Transpose((1, 0))),
                              spec((3, B), "int8"), "h2d"),
    "dequantize_layout": Recipe(spec((3, B), "int8"),
                                (Dequantize("float32", BindingParam("scale", "float32", (Const(3),)),
                                            BindingParam("zero_point", "int32", (Const(3),)), 0, "affine"),
                                 Transpose((1, 0))),
                                spec((B, 3), "float32"), "h2d"),
}
PARAMETERS = {
    "scale": np.array([1.0, 0.5, 0.25], dtype=np.float32),
    "zero_point": np.array([-128, 0, 127], dtype=np.int32),
}


def source_for(recipe, batch, rng):
    shape = tuple(ref.evaluate(d, {"s0": batch}) for d in json_recipe(recipe)["source"]["shape"])
    if recipe.source.dtype == "int8":
        return rng.integers(-128, 128, size=shape, dtype=np.int64).astype(np.int8)
    x = rng.uniform(-70, 70, size=shape).astype(np.float32)
    x.reshape(-1)[:3] = [np.nan, np.inf, -np.inf]  # C1 specials
    return x


_recipe_json = {}


def json_recipe(recipe):
    return _recipe_json[id(recipe)]


def view(array, kind="host"):
    import pyreloc

    itemsize = array.dtype.itemsize
    return pyreloc.BufferView(array.ctypes.data, array.nbytes, 0, list(array.shape),
                              [s // itemsize for s in array.strides], itemsize, kind)


def run_host(compiled, name, batch, rng):
    """Torch-free: bind, dispatch host to host with the forced baseline, validate."""
    import pyreloc

    recipe = compiled.recipe
    source = source_for(recipe, batch, rng)
    symbols = {"s0": batch}
    params = {p.name: PARAMETERS[p.name] for p in compiled.parameters}
    expected = ref.replay(json_recipe(recipe), source, symbols, params)
    plan = pyreloc.load_typed_plan(compiled.plan_bytes)
    bound = pyreloc.bind_typed(plan, {n: symbols[n] for n in compiled.symbols},
                               {n: (v.dtype.name, list(v.shape), v.tobytes()) for n, v in params.items()})
    actual = np.zeros(expected.shape, dtype=expected.dtype)
    request = pyreloc.prepare_dispatch(bound, view(source), view(actual), "h2d", policy="original_cpu")
    report = pyreloc.execute_dispatch(request)
    ok = actual.tobytes() == expected.tobytes()
    rows = [r["implementation"] for r in pyreloc.query_capability(bound, "h2d", "cuda")["eligible"]]
    print(f"{name} B={batch}: {'OK' if ok else 'MISMATCH'} via {report['implementation']} "
          f"(source {report['source_bytes']} B, wire {report['wire_bytes']} B, destination "
          f"{report['destination_bytes']} B, parameters {report['parameter_bytes']} B); "
          f"rows a CUDA device end would offer: {rows}")
    return ok


def run_cuda(compiled, name, batch, rng):
    """Through reloc_torch.dispatch on the GPU, both policies."""
    import torch
    from reloc_torch import dispatch

    recipe = compiled.recipe
    source = source_for(recipe, batch, rng)
    symbols = {"s0": batch}
    params = {p.name: PARAMETERS[p.name] for p in compiled.parameters}
    expected = ref.replay(json_recipe(recipe), source, symbols, params)
    tensors = {n: torch.from_numpy(v.copy()) for n, v in params.items()}
    ok = True
    for policy in ("original_cpu", "auto"):
        request = dispatch.prepare_typed_transfer(compiled, torch.from_numpy(source.copy()), "cuda",
                                                  parameters=tensors, policy=policy)
        result = dispatch.execute_typed_transfer(request)
        same = result.tensor.cpu().numpy().tobytes() == expected.tobytes()
        ok = ok and same
        r = result.report
        print(f"{name} B={batch} policy={policy}: {'OK' if same else 'MISMATCH'} via {r['implementation']} "
              f"({r['placement_reason']}; wire {r['wire_bytes']} B, payload {r['payload_bytes_transferred']} B)")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda", action="store_true", help="execute on a CUDA device through reloc_torch.dispatch")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    compiler = CompilerClient.from_environment()
    rng = np.random.default_rng(args.seed)
    ok = True
    for name, recipe in RECIPES.items():
        compiled = compiler.compile(recipe)
        _recipe_json[id(compiled.recipe)] = json.loads(compiled.to_bytes())["recipe"]
        assert compiled.typed and compiled.manifest["schema_version"] == 2
        print(f"== {name}: wire v{compiled.wire_version}, stages "
              f"{[s['transform'] for s in compiled.manifest['stages']]}, parameters "
              f"{[p.name for p in compiled.parameters]}")
        for batch in (4, 9):  # one artifact, two bindings
            ok = (run_cuda if args.cuda else run_host)(compiled, name, batch, rng) and ok
    print("all outputs match the independent reference" if ok else "MISMATCH against the reference")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
