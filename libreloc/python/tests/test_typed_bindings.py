"""Typed artifacts (C3, issue #143) at the Torch-free runtime surface.

The versioned decode contract (v0 and v1 never cross), typed binding with
its distinct footprints and declared parameters, the layout-only paths'
refusal of a typed layout, and the real exporter's ``--typed`` artifacts
loaded by pyreloc. The exporter is REQUIRED for the last part: an absent
``SYM_RELOC_EXPORT`` fails those tests instead of skipping them, so CI runs
them against the real binary.
"""
import json
import struct
import subprocess

import numpy as np
import pytest

import pyreloc
from conftest import corpus_entries
from typed_support import reloc_export_executable, typed_golden_hex

TYPED_GOLDENS = (
    "quantize_transpose",
    "pad_quantize",
    "quant_dequant_sym",
    "quantize_channel_sym",
    "dequant_binding",
    "dequant_cast",
)

QUANTIZE_CHANNEL_TRANSPOSE = (
    'func.func @test(%t: !sym.tensor<["B", 3], f32>) -> !sym.tensor<[3, "B"], i8> {\n'
    '%0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [3], f32>) policy symmetric_rne'
    ' : !sym.tensor<["B", 3], f32> -> !sym.tensor<["B", 3], i8>\n'
    '%1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<["B", 3], i8> -> !sym.tensor<[3, "B"], i8>\n'
    'return %1 : !sym.tensor<[3, "B"], i8>\n}\n'
)
IDENTITY = (
    'func.func @test(%t: !sym.tensor<[4, 4], f32>) -> !sym.tensor<[4, 4], f32> {\n'
    '%0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 4], f32> -> !sym.tensor<[4, 4], f32>\n'
    'return %0 : !sym.tensor<[4, 4], f32>\n}\n'
)


def _typed(name):
    return bytes.fromhex(typed_golden_hex(name))


def _exporter():
    tool = reloc_export_executable()
    assert tool.is_file(), (
        "the real sym-reloc-export is required for the typed artifact tests "
        f"(set SYM_RELOC_EXPORT or SYM_OPT); missing: {tool}"
    )
    return tool


def _export(tool, root, source, *, typed):
    root.mkdir()
    inp, plan, manifest = (root / n for n in ("input.mlir", "plan.bin", "manifest.json"))
    inp.write_text(source)
    args = [str(tool), str(inp), "--output", str(plan), "--manifest", str(manifest)]
    if typed:
        args.append("--typed")
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc, plan, manifest


@pytest.mark.parametrize("name", TYPED_GOLDENS)
def test_version_dispatch_is_explicit(name):
    blob = _typed(name)
    assert pyreloc.wire_version(blob) == 1
    # The layout-only decoder is exactly as strict as the pre-C3 runtime:
    # offset 4, "unsupported wire format version", never a reinterpretation.
    with pytest.raises(pyreloc.DecodeError, match="offset 4: unsupported wire format version"):
        pyreloc.load_plan(blob)
    plan = pyreloc.load_typed_plan(blob)
    assert plan.num_stages >= 1
    assert repr(plan).startswith("TypedPlanHandle(")


def test_typed_loader_rejects_v0_blobs_and_junk():
    entries = corpus_entries()
    assert entries, "the committed v0 corpus is required"
    for _name, blob, _meta in entries:
        assert pyreloc.wire_version(blob) == 0
        with pytest.raises(pyreloc.DecodeError, match="offset 4: unsupported wire format version"):
            pyreloc.load_typed_plan(blob)
        pyreloc.load_plan(blob)
    assert pyreloc.wire_version(b"junk") is None
    assert pyreloc.wire_version(b"RPLN\x01\x00\x00") is None
    with pytest.raises(pyreloc.DecodeError):
        pyreloc.load_typed_plan(b"RPLN\x01\x00\x00\x00")


def test_typed_bind_reports_distinct_footprints():
    plan = pyreloc.load_typed_plan(_typed("quant_dequant_sym"))
    assert (plan.source_dtype, plan.result_dtype, plan.num_stages) == ("float32", "float32", 2)
    assert plan.parameters == []
    bound = pyreloc.bind_typed(plan, {"N": 8})
    # Source, wire and destination are three different numbers: 8 f32 in,
    # 8 int8 between the stages, 8 f32 out; two inline scales are 8 bytes.
    assert (bound.source_bytes, bound.destination_bytes, bound.parameter_bytes) == (32, 32, 8)
    assert [(c["boundary"], c["dtype"], c["elements"], c["bytes"]) for c in bound.cuts] == [
        (0, "float32", 8, 32),
        (1, "int8", 8, 8),
        (2, "float32", 8, 32),
    ]
    assert bound.wire_bytes(1) == 8
    with pytest.raises(ValueError):
        bound.wire_bytes(3)
    assert bound.requirements == ["typed_execution_dispatch"]
    assert bound.layout.typed is True and bound.layout.total_bytes == 32
    assert (bound.source_extents, bound.result_extents) == ([8], [8])
    assert (bound.source_dtype, bound.result_dtype) == ("float32", "float32")
    assert [s["transform"] for s in bound.stages] == ["quantize", "dequantize"]
    assert bound.stages[0]["scale"] == {"dtype": "float32", "length": 1, "bytes": 4, "binding": None}
    assert repr(bound).startswith("TypedBoundPlan(source_bytes=32")
    for symbols in ({"N": 0}, {"N": -4}, {}, {"N": 8, "extra": 1}, {"M": 8}):
        with pytest.raises(pyreloc.BindError):
            pyreloc.bind_typed(plan, symbols)


def test_runtime_parameters_are_declared_snapshotted_and_validated():
    plan = pyreloc.load_typed_plan(_typed("quantize_channel_sym"))
    assert plan.parameters == [
        {"name": "s", "dtype": "float32", "rank": 1, "stage": 0, "role": "scale"}
    ]
    stage, = plan.stages
    assert (stage["transform"], stage["policy"]) == ("quantize", "symmetric_rne")
    assert (stage["input"], stage["output"], stage["output_signedness"]) == ("float32", "int8", "signed")
    assert (stage["axis"], stage["has_channel"], stage["rank"]) == (0, True, 2)
    scales = struct.pack("<2f", 0.5, 0.25)
    bound = pyreloc.bind_typed(plan, {"B": 2}, {"s": ("float32", [2], scales)})
    assert bound.parameter_bytes == 8
    assert bound.stages[0]["scale"] == {"dtype": "float32", "length": 2, "bytes": 8, "binding": "s"}
    assert bound.stages[0]["zero_point"] is None
    assert (bound.source_bytes, bound.destination_bytes) == (24, 6)
    with pytest.raises(pyreloc.BindError, match="unbound parameter: s"):
        pyreloc.bind_typed(plan, {"B": 2})
    with pytest.raises(pyreloc.BindError, match="length 3 but the channel extent is 2"):
        pyreloc.bind_typed(plan, {"B": 2}, {"s": ("float32", [3], struct.pack("<3f", 1, 1, 1))})
    rejected = (
        ("float16", [2], b"\0\0\0\0"),  # dtype
        ("float32", [2], scales[:-1]),  # byte size
        ("float32", [], scales),  # rank
        ("float32", [2], struct.pack("<2f", 0.5, 0.0)),  # non-positive scale
        ("float32", [2], struct.pack("<2f", 0.5, float("inf"))),  # non-finite scale
    )
    for value in rejected:
        with pytest.raises(pyreloc.BindError):
            pyreloc.bind_typed(plan, {"B": 2}, {"s": value})
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_typed(plan, {"B": 2}, {"s": ("float32", [2], scales), "t": ("float32", [2], scales)})
    with pytest.raises(ValueError, match="unknown parameter dtype"):
        pyreloc.bind_typed(plan, {"B": 2}, {"s": ("float", [2], scales)})
    with pytest.raises(ValueError):
        pyreloc.bind_typed(plan, {"B": 2}, {"s": ("float32", [2])})
    # Rebinding with another channel length re-validates the parameter.
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_typed(plan, {"B": 3}, {"s": ("float32", [2], scales)})
    assert pyreloc.bind_typed(plan, {"B": 3}, {"s": ("float32", [3], struct.pack("<3f", 1, 2, 4))}).parameter_bytes == 12


def test_dequantize_runtime_scale_and_zero_point():
    plan = pyreloc.load_typed_plan(_typed("dequant_binding"))
    assert [(p["name"], p["dtype"], p["rank"], p["role"]) for p in plan.parameters] == [
        ("s", "float32", 0, "scale"),
        ("zp", "int32", 0, "zero_point"),
    ]
    params = {"s": ("float32", [], struct.pack("<f", 0.5)), "zp": ("int32", [], struct.pack("<i", -3))}
    bound = pyreloc.bind_typed(plan, {"N": 5}, params)
    assert (bound.source_bytes, bound.destination_bytes, bound.parameter_bytes) == (5, 20, 8)
    assert [c["bytes"] for c in bound.cuts] == [5, 20]
    assert bound.stages[0]["zero_point"] == {"dtype": "int32", "length": 1, "bytes": 4, "binding": "zp"}
    with pytest.raises(pyreloc.BindError):  # outside the int8 range
        pyreloc.bind_typed(plan, {"N": 5}, {**params, "zp": ("int32", [], struct.pack("<i", 200))})
    with pytest.raises(pyreloc.BindError, match="unbound parameter: zp"):
        pyreloc.bind_typed(plan, {"N": 5}, {"s": params["s"]})


def test_layout_only_paths_refuse_the_typed_layout():
    plan = pyreloc.load_typed_plan(_typed("quant_dequant_sym"))
    layout = pyreloc.bind_typed(plan, {"N": 8}).layout
    src = np.zeros(8, dtype=np.float32)
    dst = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="layout of a typed plan"):
        pyreloc.relocate(layout, src.ctypes.data, src.nbytes, dst.ctypes.data, dst.nbytes)
    with pytest.raises(ValueError, match="layout of a typed plan"):
        pyreloc.relocate_inverse(layout, dst.ctypes.data, dst.nbytes, src.ctypes.data, src.nbytes)
    view = pyreloc.BufferView(src.ctypes.data, src.nbytes, 0, [8], [1], 4, "host")
    with pytest.raises(pyreloc.TransferError, match="typed_unsupported"):
        pyreloc.validate_transfer_source(layout, view, "h2d")
    assert dst.tobytes() == bytes(32)


def test_real_exporter_typed_artifact_loads_and_binds(tmp_path):
    tool = _exporter()
    proc, plan_path, manifest_path = _export(tool, tmp_path / "typed", QUANTIZE_CHANNEL_TRANSPOSE, typed=True)
    assert proc.returncode == 0, proc.stderr
    blob = plan_path.read_bytes()
    meta = json.loads(manifest_path.read_text())
    assert (meta["schema_version"], meta["wire_version"]) == (2, 1)
    assert pyreloc.wire_version(blob) == 1
    plan = pyreloc.load_typed_plan(blob)
    assert list(plan.symbols) == meta["symbols"] == ["B"]
    assert [(p["name"], p["dtype"], p["rank"]) for p in plan.parameters] == [
        (p["name"], p["dtype"], len(p["extents"])) for p in meta["parameters"]
    ] == [("s", "float32", 1)]
    assert plan.num_stages == len(meta["stages"]) == 1
    stage, = plan.stages
    assert (stage["transform"], stage["policy"], stage["input"], stage["output"], stage["axis"],
            stage["has_channel"]) == ("quantize", "symmetric_rne", "float32", "int8", 1, True)
    assert (plan.source_dtype, plan.result_dtype) == (meta["logical_source"]["dtype"], meta["logical_destination"]["dtype"])
    bound = pyreloc.bind_typed(plan, {"B": 4}, {"s": ("float32", [3], struct.pack("<3f", 0.5, 0.25, 0.125))})
    assert (bound.source_bytes, bound.destination_bytes, bound.parameter_bytes) == (48, 12, 12)
    assert [c["bytes"] for c in bound.cuts] == [48, 12]
    assert bound.layout.typed
    with pytest.raises(pyreloc.DecodeError, match="unsupported wire format version"):
        pyreloc.load_plan(blob)
    # Without --typed the same input is the declared rejection, no plan.
    proc, plan_path, manifest_path = _export(tool, tmp_path / "plain", QUANTIZE_CHANNEL_TRANSPOSE, typed=False)
    assert proc.returncode == 2 and not plan_path.exists()
    meta = json.loads(manifest_path.read_text())
    assert meta["status"] == "unsupported" and meta["reason"] == "typed_unsupported"
    assert (meta["schema_version"], meta["wire_version"]) == (1, 0)


def test_real_exporter_keeps_layout_only_artifacts_byte_identical(tmp_path):
    tool = _exporter()
    plain = _export(tool, tmp_path / "plain", IDENTITY, typed=False)
    flagged = _export(tool, tmp_path / "flagged", IDENTITY, typed=True)
    assert plain[0].returncode == 0 and flagged[0].returncode == 0, (plain[0].stderr, flagged[0].stderr)
    blob = plain[1].read_bytes()
    assert blob == flagged[1].read_bytes()
    assert json.loads(plain[2].read_text()) == json.loads(flagged[2].read_text())
    assert pyreloc.wire_version(blob) == 0
    bound = pyreloc.bind(pyreloc.load_plan(blob), {})
    assert bound.typed is False
    with pytest.raises(pyreloc.DecodeError, match="unsupported wire format version"):
        pyreloc.load_typed_plan(blob)
