"""numpy reference executor for corpus recipes (issue #46).

The corpus JSON records the generating op chain symbolically; this module
replays it with numpy.transpose/reshape/pad for a concrete binding. All
three ops are pure element moves (no arithmetic), so replaying on arrays
whose elements are random BIT PATTERNS is bit-exact -- the byte compare is
the oracle.

Dim expression vocabulary (kept in sync with generate_corpus.py):
int | "N" | "N floordiv <k>".
"""
import re

import numpy as np

NP_DTYPES = {"f32": np.float32, "f16": np.float16, "i8": np.int8}


def eval_dim(expr, binding):
    if isinstance(expr, int):
        return expr
    s = expr.strip()
    m = re.fullmatch(r"(\w+) floordiv (\d+)", s)
    if m:
        return binding[m.group(1)] // int(m.group(2))
    if re.fullmatch(r"\w+", s):
        return binding[s]
    raise ValueError(f"unsupported dim expression: {expr!r}")


def eval_shape(shape, binding):
    return [eval_dim(d, binding) for d in shape]


def sample_binding(symbols, rng):
    """Draw a constraint-satisfying binding: multiple_of * factor."""
    return {name: c["multiple_of"] * int(rng.integers(c["min_factor"],
                                                      c["max_factor"] + 1))
            for name, c in symbols.items()}


def random_src(meta, binding, rng):
    """C-contiguous source array of the recipe's dtype whose elements are
    random bit patterns (max byte-level discrimination)."""
    shape = eval_shape(meta["src_shape"], binding)
    dt = NP_DTYPES[meta["dtype"]]
    nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
    raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    return raw.view(dt).reshape(shape)


def reference(meta, binding, src):
    """Replay the recipe: the returned C-contiguous array's bytes are the
    expected dst buffer contents."""
    a = src
    for op in meta["ops"]:
        if op["kind"] == "transpose":
            a = np.transpose(a, op["perm"])
        elif op["kind"] == "reshape":
            a = a.reshape(eval_shape(op["to"], binding))
        elif op["kind"] == "pad":
            widths = [(0, 0)] * a.ndim
            widths[op["axis"]] = (op["lo"], op["hi"])
            a = np.pad(a, widths,
                       constant_values=a.dtype.type(op["value"]))
        else:
            raise ValueError(f"unknown op kind {op['kind']!r}")
    return np.ascontiguousarray(a)


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def mismatch_report(meta, binding, strategy, got, want):
    off = first_diff(got, want)
    lo, hi = max(0, off - 8), off + 8
    return (
        f"pyreloc output differs from numpy reference\n"
        f"  plan: {meta['name']}\n"
        f"  recipe (generating MLIR):\n{meta['mlir']}\n"
        f"  binding: {binding}\n"
        f"  strategy: {strategy}\n"
        f"  first differing byte offset: {off} "
        f"(got {len(got)} bytes, want {len(want)})\n"
        f"  got[{lo}:{hi}]:  {got[lo:hi].hex()}\n"
        f"  want[{lo}:{hi}]: {want[lo:hi].hex()}")
