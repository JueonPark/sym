"""Independent scalar reference for typed relocation programs (C4, issue #144).

C1's published policy (docs/reloc-typed-semantics.md §3) written out in
NumPy, with binary32 arithmetic throughout, plus an ordered replay of a
frontend recipe (the JSON form ``reloc_torch.artifact`` emits) on a concrete
source array. Nothing here calls a production kernel, the compiler, or the
runtime: the oracle is the definition, and every conformance test compares
the real path against it.

Comparison rules (stated before any kernel comparison, C4 Task 1):

* casts and quantization are compared bit for bit, NaN payloads excepted
  (C1 promises only "is a NaN" for NaN inputs);
* dequantization is compared bit for bit (0 ulp): C1's formula has exactly
  one rounding, so any other result is a different formula, not noise.
"""
from __future__ import annotations

import numpy as np

DTYPES = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int8": np.dtype(np.int8),
    "int32": np.dtype(np.int32),
}
_BITS = {"float32": np.uint32, "float16": np.uint16, "int8": np.uint8, "int32": np.uint32}


#===----------------------------------------------------------------------===#
# Stage arithmetic (C1 §3).
#===----------------------------------------------------------------------===#

def cast_f32_f16(x):
    """§3.1 ``ieee_rne``: round to nearest even, overflow to inf, subnormals kept."""
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return x.astype(np.float16)


def cast_f16_f32(h):
    """§3.2 ``exact``: every binary16 value is a binary32 value."""
    return np.asarray(h, dtype=np.float16).astype(np.float32)


def _along(param, rank, axis):
    """Broadcast a per-tensor scalar or a per-channel vector along ``axis``."""
    param = np.asarray(param)
    if axis is None:
        if param.ndim != 0 and param.size != 1:
            raise ValueError("per-tensor parameter must be a scalar")
        return param.reshape(())
    shape = [1] * rank
    shape[axis] = param.shape[0]
    return param.reshape(shape)


def quantize_f32_s8(x, scale, axis=None):
    """§3.3 ``symmetric_rne``: inv = fl32(1/scale) once per parameter element,
    t = fl32(x*inv), clamp max-then-min (NaN -> -128), round to nearest even."""
    x = np.asarray(x, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale must be finite and strictly positive")
    inv = (np.float32(1) / scale).astype(np.float32)
    inv = _along(inv, x.ndim, axis)
    with np.errstate(over="ignore", invalid="ignore"):
        t = (x * inv).astype(np.float32)
    clamped = np.minimum(np.maximum(t, np.float32(-128)), np.float32(127))
    clamped = np.where(np.isnan(t), np.float32(-128), clamped)
    return np.rint(clamped).astype(np.int8)


def dequantize_s8_f32(q, scale, zero_point=0, axis=None):
    """§3.4 ``affine``: d = q - zp exactly, y = fl32(d * scale), one rounding."""
    q = np.asarray(q, dtype=np.int8)
    scale = np.asarray(scale, dtype=np.float32)
    zero_point = np.asarray(zero_point, dtype=np.int32)
    if np.any(zero_point < -128) or np.any(zero_point > 127):
        raise ValueError("zero point outside [-128, 127]")
    if zero_point.ndim == 0 or zero_point.size == 1:
        zp = zero_point.reshape(())
    else:
        zp = _along(zero_point, q.ndim, axis)
    d = (q.astype(np.int32) - zp).astype(np.float32)
    return (d * _along(scale, q.ndim, axis)).astype(np.float32)


#===----------------------------------------------------------------------===#
# Bits <-> values.
#===----------------------------------------------------------------------===#

def from_bits(dtype, bits, shape=()):
    """Array of ``dtype`` from integer bit patterns (zero-extended)."""
    raw = np.asarray(list(bits) if not isinstance(bits, int) else [bits], dtype=np.uint64)
    return raw.astype(_BITS[dtype]).view(DTYPES[dtype]).reshape(shape)


def to_bits(array):
    """Integer bit patterns of a contiguous array, row-major."""
    array = np.ascontiguousarray(array)
    return [int(v) for v in array.view(_BITS[array.dtype.name]).reshape(-1)]


#===----------------------------------------------------------------------===#
# Recipe replay (frontend recipe JSON as data).
#===----------------------------------------------------------------------===#

def evaluate(expr, symbols):
    """The manifest/recipe expression vocabulary, evaluated on Python ints."""
    tag = expr[0]
    if tag == "const":
        return int(expr[1])
    if tag == "symbol":
        return int(symbols[expr[1]])
    if tag == "add":
        return evaluate(expr[1], symbols) + evaluate(expr[2], symbols)
    if tag == "mul":
        return evaluate(expr[1], symbols) * evaluate(expr[2], symbols)
    if tag == "floordiv":
        return evaluate(expr[1], symbols) // int(expr[2])
    if tag == "mod":
        return evaluate(expr[1], symbols) % int(expr[2])
    raise ValueError(f"unsupported expression {expr!r}")


def _parameter(param, parameters):
    if param is None:
        return None
    if param["kind"] == "inline":
        return from_bits(param["dtype"], param["bits"], tuple(param["shape"]))
    value = np.asarray(parameters[param["name"]])
    if value.dtype != DTYPES[param["dtype"]]:
        raise ValueError(f"parameter {param['name']!r} is {value.dtype}, declared {param['dtype']}")
    return value


def replay(recipe, source, symbols, parameters=None):
    """Replay ``recipe`` (frontend JSON) on ``source`` and return the dense
    result array. Layout operations are pure element moves; value transforms
    use the stage arithmetic above with the operand's channel axis."""
    parameters = parameters or {}
    a = np.ascontiguousarray(np.asarray(source))
    if a.dtype != DTYPES[recipe["source"]["dtype"]]:
        raise ValueError("source dtype does not match the recipe")
    expected_shape = tuple(evaluate(d, symbols) for d in recipe["source"]["shape"])
    if a.shape != expected_shape:
        raise ValueError(f"source shape {a.shape} does not match the recipe {expected_shape}")
    for op in recipe["operations"]:
        kind = op["kind"]
        if kind == "transpose":
            a = np.transpose(a, op["perm"])
        elif kind == "reshape":
            a = np.ascontiguousarray(a).reshape([evaluate(d, symbols) for d in op["shape"]])
        elif kind == "pad":
            lo, hi = evaluate(op["lo"], symbols), evaluate(op["hi"], symbols)
            widths = [(0, 0)] * a.ndim
            widths[op["axis"]] = (lo, hi)
            fill = from_bits(op["fill"]["dtype"], op["fill"]["bits"])[()]
            if a.dtype != DTYPES[op["fill"]["dtype"]]:
                raise ValueError("pad fill dtype differs from the stage dtype")
            a = np.pad(a, widths, mode="constant", constant_values=fill)
        elif kind == "cast":
            if a.dtype == np.float32 and op["dtype"] == "float16" and op["policy"] == "ieee_rne":
                a = cast_f32_f16(a)
            elif a.dtype == np.float16 and op["dtype"] == "float32" and op["policy"] == "exact":
                a = cast_f16_f32(a)
            else:
                raise ValueError(f"cast {a.dtype} -> {op['dtype']} {op['policy']} has no C1 table")
        elif kind == "quantize":
            if a.dtype != np.float32 or op["dtype"] != "int8" or op["policy"] != "symmetric_rne":
                raise ValueError("quantize outside C1's table")
            zp = _parameter(op["zero_point"], parameters)
            if zp is not None and np.any(np.asarray(zp) != 0):
                raise ValueError("symmetric_rne admits only zero point 0")
            a = quantize_f32_s8(a, _parameter(op["scale"], parameters), op["axis"])
        elif kind == "dequantize":
            if a.dtype != np.int8 or op["dtype"] != "float32" or op["policy"] != "affine":
                raise ValueError("dequantize outside C1's table")
            zp = _parameter(op["zero_point"], parameters)
            a = dequantize_s8_f32(a, _parameter(op["scale"], parameters), 0 if zp is None else zp, op["axis"])
        else:
            raise ValueError(f"unknown operation {kind!r}")
    result = np.ascontiguousarray(a)
    destination = recipe["destination"]
    if result.dtype != DTYPES[destination["dtype"]]:
        raise ValueError("replayed dtype does not reach the destination dtype")
    shape = tuple(evaluate(d, symbols) for d in destination["shape"])
    if result.shape != shape:
        raise ValueError(f"replayed shape {result.shape} does not match the destination {shape}")
    return result


__all__ = (
    "DTYPES",
    "cast_f16_f32",
    "cast_f32_f16",
    "dequantize_s8_f32",
    "evaluate",
    "from_bits",
    "quantize_f32_s8",
    "replay",
    "to_bits",
)
