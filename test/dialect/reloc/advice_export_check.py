"""Cost advice (interface 2): every success manifest carries `advice`; --no-advice nulls it."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TOOL = shutil.which(sys.argv[1])
assert TOOL, "supported sym-reloc-export executable is missing"
KV = Path(sys.argv[2]).read_text()

STAGES = '''func.func @dlrm(%t: !sym.tensor<[26, "B", 64], f32>) -> !sym.tensor<["B", 26, 64], f16> {
  %0 = reloc.transpose %t perm [1, 0, 2] : !sym.tensor<[26, "B", 64], f32> -> !sym.tensor<["B", 26, 64], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<["B", 26, 64], f32> -> !sym.tensor<["B", 26, 64], f16>
  return %1 : !sym.tensor<["B", 26, 64], f16>
}
'''
DEQUANT_F16 = '''func.func @weight(%q: !sym.tensor<["out", "in"], i8>) -> !sym.tensor<["out", "in"], f16> {
  %0 = reloc.dequantize %q axis 0 scale(#reloc.binding<"scale" : [out], f32>) policy affine : !sym.tensor<["out", "in"], i8> -> !sym.tensor<["out", "in"], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<["out", "in"], f32> -> !sym.tensor<["out", "in"], f16>
  return %1 : !sym.tensor<["out", "in"], f16>
}
'''
DEQUANT_F32 = '''func.func @weight(%q: !sym.tensor<["out", "in"], i8>) -> !sym.tensor<["out", "in"], f32> {
  %0 = reloc.dequantize %q axis 0 scale(#reloc.binding<"scale" : [out], f32>) policy affine : !sym.tensor<["out", "in"], i8> -> !sym.tensor<["out", "in"], f32>
  return %0 : !sym.tensor<["out", "in"], f32>
}
'''
INDEXED = '''func.func @gather(%x: !sym.tensor<["N", 128], f32>, %i: !sym.tensor<["M"], i64>) -> !sym.tensor<["M", 128], f16> {
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<["N", 128], f32>, !sym.tensor<["M"], i64> -> !sym.tensor<["M", 128], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<["M", 128], f32> -> !sym.tensor<["M", 128], f16>
  return %1 : !sym.tensor<["M", 128], f16>
}
'''


def run(root, source, *flags):
    root.mkdir()
    inp, plan, manifest = (root / n for n in ("input.mlir", "plan.bin", "manifest.json"))
    inp.write_text(source)
    proc = subprocess.run([TOOL, str(inp), "--output", str(plan), "--manifest", str(manifest), *flags],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    meta = json.loads(manifest.read_text())
    assert meta["status"] == "ok" and meta["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert meta["compiler"]["interface_version"] == 2
    return plan.read_bytes(), meta


def names(advice):
    return [p["name"] for p in advice["paths"]]


def ev(expr, env):
    """The manifest expression vocabulary, evaluated (docs/reloc-export.md)."""
    tag = expr[0]
    if tag == "const":
        return expr[1]
    if tag == "symbol":
        return env[expr[1]]
    if tag == "add":
        return ev(expr[1], env) + ev(expr[2], env)
    if tag == "mul":
        return ev(expr[1], env) * ev(expr[2], env)
    if tag == "floordiv":
        return ev(expr[1], env) // expr[2]
    if tag == "mod":
        return ev(expr[1], env) % expr[2]
    raise AssertionError(expr)


TERMS = [["producer", "source_bytes"], ["link", "wire"], ["device", "wire"]]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # Layout-only: constant inner run of 64 inside a symbolic total, pipelined
    # + grouped, a "read" producer.
    _, kv = run(root / "kv", KV)
    a = kv["advice"]
    assert a["version"] == 1 and a["pattern"] == "blocked", a
    assert ev(a["run_elements"], {}) == 64
    assert ev(a["total_elements"], {"S": 5}) == 16 * 5 * 64
    assert ev(a["source_bytes"], {"S": 5}) == 16 * 5 * 64 * 4
    assert ev(a["result_bytes"], {"S": 5}) == 16 * 5 * 64 * 4
    assert a["producer"] == {"kind": "layout", "kernel": "read"} and a["receive"] == {"kernel": None}
    assert names(a) == ["pipelined", "grouped"]
    pipelined, grouped = a["paths"]
    assert pipelined == {"name": "pipelined", "form": "overlapped", "wire": "result",
                         "intercept": "overhead.dispatch.layout_ms", "chunks": "pipeline.chunks_per_buffer",
                         "terms": TERMS}, pipelined
    assert grouped == {"name": "grouped", "form": "grouped", "wire": "result", "intercept": "overhead.group_ms",
                       "member_intercept": "overhead.group_member.layout_ms", "terms": TERMS}, grouped
    # Typed host stages: convert producer, f16 result wire, no direct path.
    _, st = run(root / "stages", STAGES, "--typed")
    a = st["advice"]
    assert a["pattern"] == "blocked" and a["producer"] == {"kind": "stages", "kernel": "convert_f32_f16"}, a
    assert a["receive"]["kernel"] is None and names(a) == ["pipelined", "grouped"]
    assert a["paths"][0]["intercept"] == "overhead.dispatch.stages_ms" and a["paths"][0]["wire"] == "result"
    assert a["paths"][1]["member_intercept"] == "overhead.group_member.stages_ms"
    assert ev(a["result_bytes"], {"B": 7}) == 26 * 7 * 64 * 2
    assert ev(a["source_bytes"], {"B": 7}) == 26 * 7 * 64 * 4
    # Dequantize-first identity layout: raw int8 copy + the fused receive kernel.
    _, dq = run(root / "dequant", DEQUANT_F16, "--typed")
    a = dq["advice"]
    assert a["pattern"] == "symbolic" and a["producer"] == {"kind": "none", "kernel": None}, a
    assert a["receive"] == {"kernel": "dequant_s8_f16"}
    assert names(a) == ["direct", "pipelined", "grouped"]
    direct = a["paths"][0]
    # The direct path's fixed cost splits into one submission plus one per member.
    assert direct == {"name": "direct", "form": "serial", "wire": "source", "intercept": "overhead.direct.dequant_ms",
                      "member_intercept": "overhead.direct.dequant_member_ms",
                      "terms": [["link", "wire"], ["device", "wire"]]}, direct
    assert a["paths"][2]["member_intercept"] == "overhead.group_member.dequant_ms"
    assert a["paths"][1]["intercept"] == "overhead.dispatch.dequant_ms" and a["paths"][1]["wire"] == "source"
    assert ev(a["run_elements"], {"out": 3, "in": 5}) == ev(a["total_elements"], {"out": 3, "in": 5}) == 15
    assert ev(a["source_bytes"], {"out": 3, "in": 5}) == 15 and ev(a["result_bytes"], {"out": 3, "in": 5}) == 30
    _, dq32 = run(root / "dequant32", DEQUANT_F32, "--typed")
    assert dq32["advice"]["receive"] == {"kernel": "dequant_s8_f32"}
    # Indexed rows: direct gather into an f16 wire of M x 128.
    _, ix = run(root / "indexed", INDEXED, "--indexed", "--typed")
    a = ix["advice"]
    assert a["pattern"] == "indexed_rows" and a["producer"] == {"kind": "indexed_rows", "kernel": "gather_f16"}, a
    assert names(a) == ["direct", "pipelined", "grouped"]
    assert a["paths"][0]["intercept"] == "overhead.direct.indexed_rows_ms" and a["paths"][0]["wire"] == "result"
    assert a["paths"][0]["member_intercept"] == "overhead.direct.indexed_rows_member_ms"
    assert a["paths"][2]["member_intercept"] == "overhead.group_member.indexed_rows_ms"
    assert a["paths"][1]["intercept"] == "overhead.dispatch.indexed_rows_ms"
    assert ev(a["result_bytes"], {"M": 9, "N": 50}) == 9 * 128 * 2
    assert ev(a["source_bytes"], {"M": 9, "N": 50}) == 9 * 128 * 4  # the selected rows at f32
    assert a["paths"][0]["terms"] == TERMS
    # --no-advice: identical manifest except the block, identical plan bytes.
    blob, meta = run(root / "advised", KV)
    blob2, meta2 = run(root / "unadvised", KV, "--no-advice")
    assert blob == blob2 and meta2["advice"] is None
    assert {k: v for k, v in meta.items() if k != "advice"} == {k: v for k, v in meta2.items() if k != "advice"}
    _, ix2 = run(root / "indexed-unadvised", INDEXED, "--indexed", "--typed", "--no-advice")
    assert ix2["advice"] is None
    # Deterministic.
    assert run(root / "again", KV)[1] == meta
print("ok")
