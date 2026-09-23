"""Typed artifact export contract (C3, issue #143); compiler-only, no Torch
or extension dependency. Decoding the produced v1 blobs with the runtime is
libreloc/python/tests/test_typed_bindings.py's job."""
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


def run(root, source, expected=0, reason=None, *, typed=False):
    root.mkdir()
    inp, plan, manifest = (root / n for n in ("input.mlir", "plan.bin", "manifest.json"))
    inp.write_text(source)
    args = [TOOL, str(inp), "--output", str(plan), "--manifest", str(manifest)]
    if typed:
        args.append("--typed")
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode == expected, (proc.returncode, proc.stderr, source)
    if expected == 1:
        assert proc.stderr.strip()
        assert not plan.exists() and not manifest.exists()
        return
    meta = json.loads(manifest.read_text())
    assert meta["compiler"]["name"] == "sym-reloc-export"
    assert meta["compiler"]["interface_version"] == 1
    if expected == 2:
        # Unsupported manifests describe no plan: always schema 1 / wire 0.
        assert meta["schema_version"] == 1 and meta["wire_version"] == 0
        assert meta["status"] == "unsupported" and meta["reason"] == reason and meta["detail"]
        assert not plan.exists()
        return meta
    blob = plan.read_bytes()
    assert meta["status"] == "ok" and meta["plan_count"] == 1
    assert meta["plan_sha256"] == hashlib.sha256(blob).hexdigest()
    assert meta["input_sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
    version, = struct.unpack_from("<I", blob, 4)
    assert blob[:4] == b"RPLN" and version == meta["wire_version"]
    count, = struct.unpack_from("<I", blob, 8)
    pos, symbols = 12, []
    for _ in range(count):
        size, = struct.unpack_from("<I", blob, pos)
        pos += 4
        symbols.append(blob[pos:pos + size].decode())
        pos += size
    assert meta["symbols"] == symbols
    return blob, manifest.read_bytes(), meta


def typed_keys(meta):
    return set(meta) == {
        "schema_version", "wire_version", "status", "compiler", "plan_count", "symbols",
        "logical_source", "logical_destination", "constraints", "stages", "fills",
        "parameters", "plan_sha256", "input_sha256",
    }


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # Without --typed: the R1 rejection, unchanged in shape and reason.
    run(root / "plain", SOURCE, 2, "typed_unsupported")
    # With --typed: a wire v1 blob and a schema-2 manifest, deterministic.
    first = run(root / "typed", SOURCE, typed=True)
    assert first[:2] == run(root / "typed_again", SOURCE, typed=True)[:2]
    blob, _, meta = first
    assert meta["schema_version"] == 2 and meta["wire_version"] == 1
    assert typed_keys(meta)
    assert meta["symbols"] == ["B"]
    assert meta["logical_source"] == {"shape": [["symbol", "B"], ["const", 3]], "strides": [["const", 3], ["const", 1]], "offset": ["const", 0], "dtype": "float32"}
    assert meta["logical_destination"] == {"shape": [["const", 3], ["symbol", "B"]], "strides": [["symbol", "B"], ["const", 1]], "offset": ["const", 0], "dtype": "int8"}
    assert meta["constraints"] == {"divisibility": []}
    stage, = meta["stages"]
    assert stage == {
        "transform": "quantize", "policy": "symmetric_rne",
        "input_dtype": "float32", "output_dtype": "int8",
        "shape": [["symbol", "B"], ["const", 3]],
        "scale": {"kind": "binding", "name": "s", "dtype": "float32", "extents": [["const", 3]]},
        "zero_point": None, "axis": 1,
        # Channel maps are affine over the logical RESULT coordinates: the
        # transpose moved the channel axis to result dim 0.
        "channel": {"dims": 2, "expr": ["dim", 0]},
    }
    assert meta["fills"] == []
    assert meta["parameters"] == [{"name": "s", "dtype": "float32", "extents": [["const", 3]]}]
    # The layout body is v0 verbatim: the v0 magic/version never appears
    # inside a v1 blob's header, and the header is the only version marker.
    assert blob.count(b"RPLN") == 1

    # Layout-only chains are byte-identical with and without the flag.
    identity = 'func.func @test(%t: !sym.tensor<[4, 4], f32>) -> !sym.tensor<[4, 4], f32> {\n%0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 4], f32> -> !sym.tensor<[4, 4], f32>\nreturn %0 : !sym.tensor<[4, 4], f32>\n}\n'
    plain = run(root / "layout_plain", identity)
    flagged = run(root / "layout_typed", identity, typed=True)
    assert plain[:2] == flagged[:2]
    assert plain[2]["schema_version"] == 1 and plain[2]["wire_version"] == 0
    assert struct.unpack_from("<I", plain[0], 4) == (0,)

    # Fills keep their original dtype and exact bits and record the stage
    # they entered at: after the cast (stage 1) here, before the quantize
    # (stage 0) below.
    cast_pad = 'func.func @test(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f16> {\n%0 = reloc.cast %t policy ieee_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], f16>\n%1 = reloc.pad %0 axis 0 lo 1 hi 1 value (0x3C00 : f16) : !sym.tensor<[6], f16> -> !sym.tensor<[8], f16>\nreturn %1 : !sym.tensor<[8], f16>\n}\n'
    meta = run(root / "cast_pad", cast_pad, typed=True)[2]
    assert meta["stages"][0]["transform"] == "cast" and meta["stages"][0]["policy"] == "ieee_rne"
    assert meta["stages"][0]["axis"] == -1 and meta["stages"][0]["channel"] is None
    assert meta["stages"][0]["scale"] is None and meta["stages"][0]["zero_point"] is None
    assert meta["fills"] == [{"dst_axis": 0, "stage": 1, "dtype": "float16", "bits": "3c00"}]
    assert meta["logical_source"]["dtype"] == "float32" and meta["logical_destination"]["dtype"] == "float16"
    pad_quant = 'func.func @test(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {\n%0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.5 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>\n%1 = reloc.quantize %0 scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>\nreturn %1 : !sym.tensor<[8], i8>\n}\n'
    meta = run(root / "pad_quant", pad_quant, typed=True)[2]
    assert meta["fills"] == [{"dst_axis": 0, "stage": 0, "dtype": "float32", "bits": "3f000000"}]
    assert meta["stages"][0]["scale"] == {"kind": "inline", "dtype": "float32", "shape": [], "bits": ["3f000000"]}

    # Inline per-channel parameters carry exact bits; i32 zero points are
    # two's complement bit patterns, never decimal text.
    dequant = 'func.func @test(%q: !sym.tensor<[2, 3, 4], i8>) -> !sym.tensor<[2, 3, 4], f32> {\n%0 = reloc.dequantize %q axis 2 scale(dense<[1.0, 2.0, 4.0, 8.0]> : tensor<4xf32>) zero_point(dense<-3> : tensor<i32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>\nreturn %0 : !sym.tensor<[2, 3, 4], f32>\n}\n'
    stage, = run(root / "dequant", dequant, typed=True)[2]["stages"]
    assert stage["transform"] == "dequantize" and stage["policy"] == "affine"
    assert stage["scale"] == {"kind": "inline", "dtype": "float32", "shape": [4], "bits": ["3f800000", "40000000", "40800000", "41000000"]}
    assert stage["zero_point"] == {"kind": "inline", "dtype": "int32", "shape": [], "bits": ["fffffffd"]}
    assert stage["channel"] == {"dims": 3, "expr": ["dim", 2]}

    # Runtime parameters are declared once each, in first-declaration order.
    runtime = 'func.func @test(%q: !sym.tensor<["N"], i8>) -> !sym.tensor<["N"], f32> {\n%0 = reloc.dequantize %q scale(#reloc.binding<"s" : [], f32>) zero_point(#reloc.binding<"zp" : [], i32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>\nreturn %0 : !sym.tensor<["N"], f32>\n}\n'
    meta = run(root / "runtime", runtime, typed=True)[2]
    assert meta["parameters"] == [{"name": "s", "dtype": "float32", "extents": []}, {"name": "zp", "dtype": "int32", "extents": []}]
    assert meta["symbols"] == ["N"]

    # Symbolic channel divisors use the channel-expression vocabulary
    # (two expressions), and every symbol they mention is in the table.
    flatten = 'func.func @test(%t: !sym.tensor<["B", "C"], f32>) -> !sym.tensor<["B" * "C"], i8> {\n%0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [C], f32>) policy symmetric_rne : !sym.tensor<["B", "C"], f32> -> !sym.tensor<["B", "C"], i8>\n%1 = reloc.reshape %0 to [B * C] : !sym.tensor<["B", "C"], i8> -> !sym.tensor<["B" * "C"], i8>\nreturn %1 : !sym.tensor<["B" * "C"], i8>\n}\n'
    meta = run(root / "flatten", flatten, typed=True)[2]
    assert meta["stages"][0]["channel"] == {"dims": 1, "expr": ["mod", ["dim", 0], ["symbol", "C"]]}
    assert meta["symbols"] == ["B", "C"]
    assert meta["parameters"] == [{"name": "s", "dtype": "float32", "extents": [["symbol", "C"]]}]

    # Typed fold bails are fold_unsupported, never a partial plan; a
    # pre-folded typed plan is prefolded_input like a layout plan.
    bail = 'func.func @test(%t: !sym.tensor<[6, 3], f32>) -> !sym.tensor<[8, 3], i8> {\n%0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6, 3], f32> -> !sym.tensor<[8, 3], f32>\n%1 = reloc.quantize %0 axis 1 scale(dense<[0.5, 0.25, 0.125]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[8, 3], f32> -> !sym.tensor<[8, 3], i8>\nreturn %1 : !sym.tensor<[8, 3], i8>\n}\n'
    run(root / "bail", bail, 2, "fold_unsupported", typed=True)
    folded = subprocess.run([str(Path(TOOL).with_name("sym-opt")), "--reloc-fold"], input=SOURCE, capture_output=True, text=True, check=True).stdout
    assert "reloc.typed_plan_result" in folded
    run(root / "prefolded", folded, 2, "prefolded_input", typed=True)
    run(root / "prefolded_plain", folded, 2, "prefolded_input")
    # A verifier-invalid typed op stays an error (exit 1) with the flag too.
    run(root / "invalid", cast_pad.replace("policy ieee_rne", "policy exact"), 1, typed=True)
    # The flag never relaxes the layout-only rejections.
    run(root / "typed_empty", 'func.func @test(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f32> {\nreturn %t : !sym.tensor<[4], f32>\n}\n', 2, "empty_chain", typed=True)
    run(root / "typed_dtype", identity.replace("f32", "f64"), 2, "unsupported_dtype", typed=True)
print("typed export contract passed")
