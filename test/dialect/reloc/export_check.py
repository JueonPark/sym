"""Compiler-only subprocess contract; no Torch or extension dependency."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

TOOL = shutil.which(sys.argv[1])
assert TOOL, "supported sym-reloc-export executable is missing"
SOURCE = Path(sys.argv[2]).read_text()


def run(root, source, expected=0, reason=None):
    root.mkdir()
    inp, plan, manifest = (root / n for n in ("input.mlir", "plan.bin", "manifest.json"))
    inp.write_text(source)
    proc = subprocess.run([TOOL, str(inp), "--output", str(plan), "--manifest", str(manifest)], capture_output=True, text=True)
    assert proc.returncode == expected, (proc.returncode, proc.stderr, source)
    if expected == 1:
        assert proc.stderr.strip()
        assert not plan.exists() and not manifest.exists()
        return
    meta = json.loads(manifest.read_text())
    assert meta["schema_version"] == 1 and meta["wire_version"] == 0
    if expected == 2:
        assert meta["status"] == "unsupported" and meta["reason"] == reason and meta["detail"]
        assert not plan.exists()
        return
    blob = plan.read_bytes()
    assert meta["status"] == "ok" and meta["plan_count"] == 1
    assert meta["plan_sha256"] == hashlib.sha256(blob).hexdigest()
    assert meta["input_sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
    assert blob[:8] == b"RPLN\0\0\0\0"
    count, = struct.unpack_from("<I", blob, 8)
    pos, symbols = 12, []
    for _ in range(count):
        size, = struct.unpack_from("<I", blob, pos)
        pos += 4
        symbols.append(blob[pos:pos + size].decode())
        pos += size
    assert meta["symbols"] == symbols
    assert meta["compiler"]["name"] == "sym-reloc-export"
    assert meta["compiler"]["interface_version"] == 1
    assert meta["compiler"]["llvm_version"] and meta["compiler"]["build_identity"]
    return blob, manifest.read_bytes(), meta


def function(src, dst, body):
    return f"func.func @test(%t: !sym.tensor<[{src}], f32>) -> !sym.tensor<[{dst}], f32> {{\n{body}\n}}\n"


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    first = run(root / "first", SOURCE)
    assert first[:2] == run(root / "second", SOURCE)[:2]
    meta = first[2]
    assert meta["symbols"] == ["s0"]
    assert meta["logical_source"] == {"shape": [["symbol", "s0"]], "strides": [["const", 1]], "offset": ["const", 0], "dtype": "float32"}
    assert meta["logical_destination"]["shape"] == [["const", 64], ["floordiv", ["symbol", "s0"], 64]]
    assert meta["logical_destination"]["strides"] == [["floordiv", ["symbol", "s0"], 64], ["const", 1]]
    assert meta["constraints"] == {"divisibility": [{"expr": ["symbol", "s0"], "divisor": 64}]}
    identity = function("4, 4", "4, 4", '%0 = reloc.transpose %t perm [0, 1] : !sym.tensor<[4, 4], f32> -> !sym.tensor<[4, 4], f32>\nreturn %0 : !sym.tensor<[4, 4], f32>')
    ident = run(root / "identity", identity)[2]
    assert ident["logical_source"]["shape"] == ident["logical_destination"]["shape"] == [["const", 4], ["const", 4]]
    assert ident["logical_source"]["strides"] == [["const", 4], ["const", 1]]
    trans = run(root / "square_transpose", identity.replace("[0, 1]", "[1, 0]"))[2]
    assert ident["logical_source"] == trans["logical_source"] and ident["input_sha256"] != trans["input_sha256"]
    ordered = function('"s1", "s0"', '"s0", "s1"', '%0 = reloc.transpose %t perm [1, 0] : !sym.tensor<["s1", "s0"], f32> -> !sym.tensor<["s0", "s1"], f32>\nreturn %0 : !sym.tensor<["s0", "s1"], f32>')
    assert run(root / "symbol_order", ordered)[2]["symbols"] == ["s1", "s0"]
    for dtype, name in (("f16", "float16"), ("i8", "int8")):
        assert run(root / dtype, identity.replace("f32", dtype))[2]["logical_source"]["dtype"] == name
    cases = {
        "merge": (function("4, 6", "24", '%0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>\n%1 = reloc.reshape %0 to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>\nreturn %1 : !sym.tensor<[24], f32>'), "fold_unsupported"),
        "pad_split": (function("6", "2, 4", '%0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>\n%1 = reloc.reshape %0 to [2, 4] : !sym.tensor<[8], f32> -> !sym.tensor<[2, 4], f32>\nreturn %1 : !sym.tensor<[2, 4], f32>'), "fold_unsupported"),
        "fills": (function("6", "10", '%0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>\n%1 = reloc.pad %0 axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[8], f32> -> !sym.tensor<[10], f32>\nreturn %1 : !sym.tensor<[10], f32>'), "fold_unsupported"),
        "unrelated": (identity.replace("%0 =", "%c = arith.constant 0 : i32\n%0 ="), "unsupported_operation"),
        "escape": (identity.replace("return %0", '%1 = reloc.transpose %0 perm [0, 1] : !sym.tensor<[4, 4], f32> -> !sym.tensor<[4, 4], f32>\nreturn %0'), "disconnected_chain"),
        "zero": (function("4", "4", "return %t : !sym.tensor<[4], f32>"), "empty_chain"),
        "multiple": (identity + identity.replace("@test", "@second"), "invalid_function_count"),
        "rank0": (function("", "", '%0 = reloc.transpose %t perm [] : !sym.tensor<[], f32> -> !sym.tensor<[], f32>\nreturn %0 : !sym.tensor<[], f32>'), "unsupported_descriptor"),
        "empty": (identity.replace("4, 4", "0, 4"), "unsupported_descriptor"),
        "dtype": (identity.replace("f32", "f64"), "unsupported_dtype"),
    }
    for name, (source, reason) in cases.items():
        run(root / name, source, 1 if name == "rank0" else 2, reason)
    folded = subprocess.run([str(Path(TOOL).with_name("sym-opt")), "--reloc-fold"], input=SOURCE, capture_output=True, text=True, check=True).stdout
    run(root / "prefolded", folded, 2, "prefolded_input")
    run(root / "malformed", "invalid MLIR", 1)
    run(root / "negative", identity.replace("4, 4", "-1, 4"), 2, "unsupported_descriptor")
    run(root / "wrong_return", identity.replace("return %0", "return %t"), 2, "disconnected_chain")
    run(root / "unsupported_divisor", SOURCE.replace("s0 floordiv 64", "s0 floordiv s1"), 2, "unsupported_expression")
    # Failure publishing the second file must roll back the first owned file.
    proc = subprocess.run([TOOL, str(root / "first/input.mlir"), "--output", str(root / "rollback.bin"), "--manifest", str(root / "absent/manifest")], capture_output=True, text=True)
    assert proc.returncode == 1 and proc.stderr.strip()
    assert not (root / "rollback.bin").exists()
    # Exact non-round-number f32 fill bits survive the public artifact path.
    padded = function("6", "8", '%0 = reloc.pad %t axis 0 lo 1 hi 1 value (0x3EAAAAAB : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>\nreturn %0 : !sym.tensor<[8], f32>')
    pad_blob = run(root / "pad_bits", padded)[0]
    # One pad on axis zero, two width expressions of 1, then f32 + u64 bits.
    pad_entry = struct.pack("<IIIBqIBqBIQ", 1, 0, 1, 1, 1, 1, 1, 1, 0, 32, 0x3EAAAAAB)
    assert pad_entry in pad_blob
    for args in ([], ["missing.mlir", "--output", str(root / "missing.bin"), "--manifest", str(root / "missing.json")], [str(root / "first/input.mlir"), "--output", str(root / "absent/plan"), "--manifest", str(root / "io.json")]):
        proc = subprocess.run([TOOL, *args], capture_output=True, text=True)
        assert proc.returncode == 1 and proc.stderr.strip(), proc
    # An existing artifact must never be overwritten, even on unsupported input.
    proc = subprocess.run([TOOL, str(root / "zero/input.mlir"), "--output", str(root / "first/plan.bin"), "--manifest", str(root / "stale.json")], capture_output=True, text=True)
    assert proc.returncode == 1 and (root / "first/plan.bin").read_bytes() == first[0]
    assert not (root / "stale.json").exists()
print("export contract passed")
