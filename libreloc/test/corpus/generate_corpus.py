#!/usr/bin/env python3
"""Regenerate the committed plan corpus (issue #46).

Plans come FROM THE COMPILER SIDE, per the issue: each case is a
reloc.{transpose,reshape,pad} chain that is
  1. emitted as textual MLIR,
  2. folded to a single reloc.plan_result by `sym-opt --reloc-fold`
     (chains that bail -- reloc.fallback -- are dropped with a warning),
  3. serialized to wire hex by wrapping the extracted #reloc.plan attribute
     in a "test.plan"() {serialize} op and running
     `sym-opt --test-reloc-utils`,
  4. committed as <name>.bin (wire bytes) + <name>.json (the recipe the
     pytest oracle replays with numpy).

Deterministic for a fixed --seed (default 0); stdlib only. The dim
expression vocabulary shared with libreloc/python/tests/oracle.py is:
int | "N" | "N floordiv <k>".

Usage: generate_corpus.py [--sym-opt PATH] [--seed 0] [--random-count 6]
"""
import argparse
import json
import pathlib
import random
import re
import subprocess
import sys

CORPUS_DIR = pathlib.Path(__file__).resolve().parent
DTYPES = {"f32": 4, "f16": 2, "i8": 1}

# --- dim expressions -------------------------------------------------------

def dim_mlir(d, in_type):
    """Render a dim expression. Symbols are quoted inside !sym.tensor types
    but unquoted in reshape to-lists (see test/dialect/reloc/ops.mlir)."""
    if isinstance(d, int):
        return str(d)
    m = re.fullmatch(r"(\w+) floordiv (\d+)", d)
    if m:
        sym = f'"{m.group(1)}"' if in_type else m.group(1)
        return f"{sym} floordiv {m.group(2)}"
    if re.fullmatch(r"\w+", d):
        return f'"{d}"' if in_type else d
    raise ValueError(f"unsupported dim expression: {d!r}")


def shape_type(shape, dtype):
    dims = ", ".join(dim_mlir(d, in_type=True) for d in shape)
    return f"!sym.tensor<[{dims}], {dtype}>"


# --- op semantics on shapes (mirrors what the oracle does with numpy) ------

def apply_op(shape, op):
    if op["kind"] == "transpose":
        return [shape[p] for p in op["perm"]]
    if op["kind"] == "reshape":
        return list(op["to"])
    if op["kind"] == "pad":
        out = list(shape)
        d = out[op["axis"]]
        if not isinstance(d, int):
            raise ValueError("pads are static-only in this generator")
        out[op["axis"]] = d + op["lo"] + op["hi"]
        return out
    raise ValueError(f"unknown op kind {op['kind']!r}")


def op_mlir(op, in_shape, out_shape, dtype):
    tin, tout = shape_type(in_shape, dtype), shape_type(out_shape, dtype)
    if op["kind"] == "transpose":
        perm = ", ".join(str(p) for p in op["perm"])
        return f"reloc.transpose %V perm [{perm}] : {tin} -> {tout}"
    if op["kind"] == "reshape":
        to = ", ".join(dim_mlir(d, in_type=False) for d in op["to"])
        return f"reloc.reshape %V to [{to}] : {tin} -> {tout}"
    if op["kind"] == "pad":
        return (f"reloc.pad %V axis {op['axis']} lo {op['lo']} hi "
                f"{op['hi']} value ({op['value']:.6e} : {op['dtype']}) : "
                f"{tin} -> {tout}")
    raise ValueError(op["kind"])


def emit_func(case):
    shape = list(case["src_shape"])
    lines = []
    val = "%arg0"
    for i, op in enumerate(case["ops"]):
        out_shape = apply_op(shape, op)
        text = op_mlir({**op, "dtype": case["dtype"]}, shape, out_shape,
                       case["dtype"])
        lines.append(f"  %{i} = " + text.replace("%V", val, 1))
        val, shape = f"%{i}", out_shape
    tin = shape_type(case["src_shape"], case["dtype"])
    tout = shape_type(shape, case["dtype"])
    body = "\n".join(lines)
    return (f"func.func @{case['name']}(%arg0: {tin}) -> {tout} {{\n"
            f"{body}\n  return {val} : {tout}\n}}\n")


# --- case definitions -------------------------------------------------------

def curated_cases():
    t = lambda perm: {"kind": "transpose", "perm": perm}
    r = lambda to: {"kind": "reshape", "to": to}
    p = lambda axis, lo, hi, value: {"kind": "pad", "axis": axis, "lo": lo,
                                     "hi": hi, "value": value}
    div64 = {"N": {"multiple_of": 64, "min_factor": 1, "max_factor": 3}}
    return [
        dict(name="identity_1d", dtype="f32", src_shape=[8],
             ops=[t([0])], symbols={}),
        dict(name="transpose_2d_f32", dtype="f32", src_shape=[32, 64],
             ops=[t([1, 0])], symbols={}),
        dict(name="transpose_2d_i8", dtype="i8", src_shape=[32, 64],
             ops=[t([1, 0])], symbols={}),
        dict(name="transpose_3d_f16", dtype="f16", src_shape=[4, 5, 6],
             ops=[t([2, 0, 1])], symbols={}),
        dict(name="reshape_merge", dtype="f32", src_shape=[4, 6],
             ops=[r([24])], symbols={}),
        dict(name="reshape_split_sym", dtype="f32", src_shape=["N"],
             ops=[r(["N floordiv 64", 64])], symbols=div64),
        dict(name="blocked_reference", dtype="f32", src_shape=[128, 128],
             ops=[r([2, 64, 2, 64]), t([2, 0, 1, 3])], symbols={}),
        dict(name="pad_1d", dtype="f32", src_shape=[30],
             ops=[p(0, 0, 2, 0.0)], symbols={}),
        dict(name="pad_2d_outer", dtype="f32", src_shape=[6, 4],
             ops=[p(0, 1, 1, -2.5)], symbols={}),
        dict(name="transpose_then_pad", dtype="f32", src_shape=[32, 64],
             ops=[t([1, 0]), p(0, 1, 1, 0.0)], symbols={}),
    ]


def random_cases(rng, count):
    """Seeded random transpose/reshape chains over small static shapes.
    Cases that fail to fold are dropped later, so over-generate 3x."""
    cases = []
    for i in range(count * 3):
        rank = rng.choice([2, 3])
        shape = [rng.choice([2, 3, 4, 6, 8]) for _ in range(rank)]
        ops, cur = [], list(shape)
        for _ in range(rng.choice([1, 2, 3])):
            # A rank-1 tensor has only the identity permutation, so
            # "transpose" would spin forever below trying to find a
            # non-identity perm -- force reshape in that case.
            kind = rng.choice(["transpose", "reshape"]) if len(cur) > 1 \
                else "reshape"
            if kind == "transpose":
                perm = list(range(len(cur)))
                while perm == sorted(perm):
                    rng.shuffle(perm)
                ops.append({"kind": "transpose", "perm": perm})
                cur = [cur[p] for p in perm]
            else:
                total = 1
                for d in cur:
                    total *= d
                # random factorization of `total` into 1..3 dims
                dims = []
                rest = total
                for _ in range(rng.choice([0, 1, 2])):
                    divs = [d for d in range(2, min(rest, 9))
                            if rest % d == 0]
                    if not divs:
                        break
                    d = rng.choice(divs)
                    dims.append(d)
                    rest //= d
                dims.append(rest)
                ops.append({"kind": "reshape", "to": dims})
                cur = dims
        dtype = rng.choice(["f32", "f16", "i8"])
        cases.append(dict(name=f"rand_{i:02d}", dtype=dtype,
                          src_shape=shape, ops=ops, symbols={}))
    return cases


# --- sym-opt plumbing -------------------------------------------------------

PLAN_ATTR_RE = re.compile(r"plan\((#reloc\.plan<.+>)\) : ")
PLAN_HEX_RE = re.compile(r"plan_hex\((\w+)\): ([0-9a-f]+)")


def run_sym_opt(sym_opt, args, text):
    proc = subprocess.run([sym_opt, *args, "-"], input=text,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sym-opt {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


def fold_to_plan_attr(sym_opt, func_text):
    """Fold one chain; return the #reloc.plan<...> attr text, or None if
    the fold bailed (reloc.fallback) or produced no single plan_result."""
    out = run_sym_opt(
        sym_opt,
        ["--allow-unregistered-dialect", "--reloc-fold",
         "--mlir-print-local-scope"],
        func_text)
    if "reloc.fallback" in out or out.count("reloc.plan_result") != 1:
        return None
    m = PLAN_ATTR_RE.search(out)
    return m.group(1) if m else None


def serialize_plans(sym_opt, named_attrs):
    ops = "\n".join(
        f'"test.plan"() {{serialize, name = "{name}", plan = {attr}}}'
        f" : () -> ()" for name, attr in named_attrs)
    out = run_sym_opt(sym_opt, ["--allow-unregistered-dialect",
                                "--test-reloc-utils"], ops)
    return {m.group(1): bytes.fromhex(m.group(2))
            for m in PLAN_HEX_RE.finditer(out)}


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym-opt",
                    default=str(CORPUS_DIR.parents[2] /
                                "build/sym/sym/tools/sym-opt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-count", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept, dropped = [], []
    randoms_kept = 0
    for case in curated_cases() + random_cases(rng, args.random_count):
        is_random = case["name"].startswith("rand_")
        if is_random and randoms_kept >= args.random_count:
            continue
        try:
            attr = fold_to_plan_attr(args.sym_opt, emit_func(case))
        except RuntimeError as e:
            # A verifier/parser rejection (e.g. an op form the dialect
            # refuses) drops the case rather than aborting the run.
            print(f"note: {case['name']} rejected by sym-opt:\n{e}",
                  file=sys.stderr)
            attr = None
        if attr is None:
            dropped.append(case["name"])
            continue
        kept.append((case, attr))
        randoms_kept += is_random

    blobs = serialize_plans(args.sym_opt,
                            [(c["name"], a) for c, a in kept])
    for case, _ in kept:
        blob = blobs[case["name"]]
        (CORPUS_DIR / f"{case['name']}.bin").write_bytes(blob)
        meta = dict(case, element_size=DTYPES[case["dtype"]],
                    mlir=emit_func(case), generator_seed=args.seed)
        (CORPUS_DIR / f"{case['name']}.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"kept {len(kept)} cases, dropped (fold bail): {dropped or 'none'}")
    if len(kept) < 10:
        print("ERROR: fewer than 10 corpus entries", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
