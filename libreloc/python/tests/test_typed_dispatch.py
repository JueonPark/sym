"""R3 typed dispatch (issue #147) at the Torch-free surface.

Programs come from the real exporter (``--typed``); expected values from a
NumPy oracle that spells out C1's arithmetic (never from a runtime kernel).
Host-to-host execution through ``pyreloc.prepare_dispatch`` /
``execute_dispatch`` is the CI-visible form of the forced CPU baseline;
capability rows are pure and can be asked for a CUDA device end without one
being present. The exporter is REQUIRED (fails, never skips).
"""
import json
import struct
import subprocess

import numpy as np
import pytest

import pyreloc
from typed_support import reloc_export_executable

PROGRAMS = {
    "quantize_channel_transpose": (
        'func.func @t(%t: !sym.tensor<["B", 3], f32>) -> !sym.tensor<[3, "B"], i8> {\n'
        '  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [3], f32>) policy symmetric_rne'
        ' : !sym.tensor<["B", 3], f32> -> !sym.tensor<["B", 3], i8>\n'
        '  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<["B", 3], i8> -> !sym.tensor<[3, "B"], i8>\n'
        '  return %1 : !sym.tensor<[3, "B"], i8>\n}\n'
    ),
    "quant_dequant": (
        'func.func @t(%t: !sym.tensor<["N"], f32>) -> !sym.tensor<["N"], f32> {\n'
        '  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne'
        ' : !sym.tensor<["N"], f32> -> !sym.tensor<["N"], i8>\n'
        '  %1 = reloc.dequantize %0 scale(dense<0.5> : tensor<f32>) policy affine'
        ' : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>\n'
        '  return %1 : !sym.tensor<["N"], f32>\n}\n'
    ),
    "dequant_runtime": (
        'func.func @t(%q: !sym.tensor<["N"], i8>) -> !sym.tensor<["N"], f32> {\n'
        '  %0 = reloc.dequantize %q scale(#reloc.binding<"s" : [], f32>)'
        ' zero_point(#reloc.binding<"zp" : [], i32>) policy affine'
        ' : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>\n'
        '  return %0 : !sym.tensor<["N"], f32>\n}\n'
    ),
    "cast_pad": (
        'func.func @t(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f16> {\n'
        '  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], f16>\n'
        '  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (0x3C00 : f16) : !sym.tensor<[6], f16> -> !sym.tensor<[8], f16>\n'
        '  return %1 : !sym.tensor<[8], f16>\n}\n'
    ),
    "pad_quantize": (
        'func.func @t(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {\n'
        '  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>\n'
        '  %1 = reloc.quantize %0 scale(dense<0.5> : tensor<f32>) policy symmetric_rne'
        ' : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>\n'
        '  return %1 : !sym.tensor<[8], i8>\n}\n'
    ),
    "quantize_pad": (
        'func.func @t(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {\n'
        '  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne'
        ' : !sym.tensor<[6], f32> -> !sym.tensor<[6], i8>\n'
        '  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (1 : i8) : !sym.tensor<[6], i8> -> !sym.tensor<[8], i8>\n'
        '  return %1 : !sym.tensor<[8], i8>\n}\n'
    ),
    "witness": (
        'func.func @t(%t: !sym.tensor<[2, 3, 4], f32>) -> !sym.tensor<[3, 2, 4], i8> {\n'
        '  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 1.0, 2.0]> : tensor<3xf32>) policy symmetric_rne'
        ' : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>\n'
        '  %1 = reloc.transpose %0 perm [1, 0, 2] : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[3, 2, 4], i8>\n'
        '  return %1 : !sym.tensor<[3, 2, 4], i8>\n}\n'
    ),
}


#===----------------------------------------------------------------------===#
# Independent oracle: C1 §3 written in NumPy (binary32 throughout).
#===----------------------------------------------------------------------===#

def oracle_quantize(x, scale):
    inv = (np.float32(1) / np.asarray(scale, dtype=np.float32)).astype(np.float32)
    t = (np.asarray(x, dtype=np.float32) * inv).astype(np.float32)
    clamped = np.minimum(np.maximum(t, np.float32(-128)), np.float32(127))
    clamped = np.where(np.isnan(t), np.float32(-128), clamped)
    return np.rint(clamped).astype(np.int8)


def oracle_dequantize(q, zero_point, scale):
    d = (np.asarray(q, dtype=np.int32) - np.int32(zero_point)).astype(np.float32)
    return (d * np.float32(scale)).astype(np.float32)


def oracle_narrow(x):
    with np.errstate(over="ignore"):  # 65520 -> inf is the C1 rule, not a fault
        return np.asarray(x, dtype=np.float32).astype(np.float16)


def specials(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-70, 70, size=n).astype(np.float32)
    x[0], x[1], x[2], x[3] = np.nan, np.inf, -np.inf, -0.0
    return x


#===----------------------------------------------------------------------===#
# Fixtures: real artifacts, views, bindings.
#===----------------------------------------------------------------------===#

@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    tool = reloc_export_executable()
    assert tool.is_file(), f"the real sym-reloc-export is required (SYM_RELOC_EXPORT); missing: {tool}"
    root = tmp_path_factory.mktemp("typed-dispatch")
    out = {}
    for name, source in PROGRAMS.items():
        folder = root / name
        folder.mkdir()
        inp, plan, manifest = folder / "in.mlir", folder / "plan.bin", folder / "m.json"
        inp.write_text(source)
        proc = subprocess.run(
            [str(tool), str(inp), "--output", str(plan), "--manifest", str(manifest), "--typed"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (name, proc.stderr)
        meta = json.loads(manifest.read_text())
        assert meta["schema_version"] == 2 and meta["wire_version"] == 1
        out[name] = (plan.read_bytes(), meta)
    return out


def view(array, kind="host"):
    itemsize = array.dtype.itemsize
    return pyreloc.BufferView(
        array.ctypes.data, array.nbytes, 0, list(array.shape),
        [s // itemsize for s in array.strides], itemsize, kind,
    )


def f32(values):
    return ("float32", [len(values)], np.asarray(values, dtype=np.float32).tobytes())


def scalar_f32(value):
    return ("float32", [], np.float32(value).tobytes())


def scalar_i32(value):
    return ("int32", [], np.int32(value).tobytes())


def bound_for(artifacts, name, symbols, parameters=None):
    plan = pyreloc.load_typed_plan(artifacts[name][0])
    return pyreloc.bind_typed(plan, symbols, parameters or {})


def run_host(bound, source, result_shape, result_dtype, direction="h2d", **options):
    dst = np.zeros(result_shape, dtype=result_dtype)
    dst.view(np.uint8)[...] = 0xCD
    request = pyreloc.prepare_dispatch(bound, view(source), view(dst), direction, **options)
    report = pyreloc.execute_dispatch(request)
    return dst, report, request


def eligible(bound, direction, device):
    return [row["implementation"] for row in pyreloc.query_capability(bound, direction, device)["eligible"]]


def excluded(bound, direction, device):
    return {row["implementation"]: row["reason"] for row in pyreloc.query_capability(bound, direction, device)["excluded"]}


#===----------------------------------------------------------------------===#
# Capability is pure and only lists implemented, equivalent rows.
#===----------------------------------------------------------------------===#

def test_capability_rows_depend_on_program_direction_and_device(artifacts):
    qd = bound_for(artifacts, "quant_dequant", {"N": 16})
    assert eligible(qd, "h2d", "host") == ["cpu_reference"]
    assert excluded(qd, "h2d", "host")["cuda_relocate_f32"] == "no_cuda_device"
    rows = {r["implementation"]: r for r in pyreloc.query_capability(qd, "h2d", "cuda")["eligible"]}
    assert set(rows) == {"cpu_reference", "cpu_stages_cuda_stages@0", "cpu_stages_cuda_stages@1", "cuda_relocate_f32"}
    assert rows["cpu_reference"]["wire_bytes"] == 64 and rows["cpu_reference"]["method"] == "A"
    assert rows["cpu_stages_cuda_stages@1"]["wire_bytes"] == 16  # the s8 intermediate
    assert rows["cuda_relocate_f32"]["method"] == "B" and rows["cuda_relocate_f32"]["wire_bytes"] == 64
    assert excluded(qd, "h2d", "cuda")["cuda_dequant_relocate"] == "stage 0 is not a dequantize"
    assert eligible(qd, "d2h", "cuda") == ["cpu_reference", "cuda_stages_then_cpu@1", "cuda_stages_then_cpu@2"]

    witness = bound_for(artifacts, "witness", {})
    assert eligible(witness, "h2d", "cuda") == ["cpu_reference", "cpu_stages_cuda_stages@0", "cuda_relocate_f32"]
    assert eligible(witness, "d2h", "cuda") == ["cpu_reference"]
    assert excluded(witness, "d2h", "cuda")["cuda_stages_then_cpu@1"] == "stage 0: channel_not_source_outer_axis"

    cast_pad = bound_for(artifacts, "cast_pad", {})
    assert eligible(cast_pad, "h2d", "cuda") == ["cpu_reference"]
    assert excluded(cast_pad, "h2d", "cuda")["cpu_stages_cuda_stages@0"] == "pads_not_settled"
    assert excluded(cast_pad, "h2d", "cuda")["cuda_relocate_f32"] == "layout has pads"

    nonzero = bound_for(artifacts, "dequant_runtime", {"N": 8}, {"s": scalar_f32(0.3), "zp": scalar_i32(5)})
    assert eligible(nonzero, "h2d", "cuda") == ["cpu_reference"]
    assert excluded(nonzero, "h2d", "cuda")["cpu_stages_cuda_stages@0"] == "stage 0: no_cuda_kernel:nonzero_zero_point"
    zero = bound_for(artifacts, "dequant_runtime", {"N": 8}, {"s": scalar_f32(0.3), "zp": scalar_i32(0)})
    assert eligible(zero, "h2d", "cuda") == ["cpu_reference", "cpu_stages_cuda_stages@0"]


#===----------------------------------------------------------------------===#
# The forced CPU baseline, host to host, against the oracle.
#===----------------------------------------------------------------------===#

def test_layout_plus_quantize_matches_the_oracle_in_both_directions(artifacts):
    scales = [1.0, 0.5, 0.25]
    bound = bound_for(artifacts, "quantize_channel_transpose", {"B": 5}, {"s": f32(scales)})
    x = specials(15, 1).reshape(5, 3)
    expected = np.ascontiguousarray(oracle_quantize(x, np.asarray(scales, np.float32)[None, :]).T)
    for direction in ("h2d", "d2h"):
        dst, report, request = run_host(bound, x, (3, 5), np.int8, direction, policy="original_cpu")
        np.testing.assert_array_equal(dst, expected)
        assert report["implementation"] == "cpu_reference"
        assert report["policy"] == "original_cpu" and report["placement_reason"] == "forced"
        assert (report["source_bytes"], report["destination_bytes"], report["parameter_bytes"]) == (60, 15, 12)
        assert report["wire_bytes"] == (15 if direction == "h2d" else 60)
        assert report["payload_bytes_transferred"] == report["wire_bytes"]
        assert report["executed"] is True and report["artifact_version"] == 1
        assert request.consumed
        with pytest.raises(pyreloc.TransferError, match="already_executed"):
            pyreloc.execute_dispatch(request)


def test_layout_plus_dequantize_and_cast_match_the_oracle(artifacts):
    # s8 -> f32 with a runtime scale and zero point.
    q = (np.arange(200, dtype=np.int32) * 37 % 256 - 128).astype(np.int8)
    bound = bound_for(artifacts, "dequant_runtime", {"N": 200}, {"s": scalar_f32(0.3), "zp": scalar_i32(127)})
    dst, report, _ = run_host(bound, q, (200,), np.float32, policy="original_cpu")
    np.testing.assert_array_equal(dst.view(np.uint32), oracle_dequantize(q, 127, 0.3).view(np.uint32))
    assert dst.view(np.uint32)[np.where(q == -128)[0][0]] == 0xC2990000  # C1 §6
    # f32 -> f16 then a pad that enters after the cast: C1's binary16 vectors.
    x = np.array([2.0 ** -24, 2.0 ** -25, 1.5 * 2.0 ** -25, 2.0 ** -14, 65504.0, 65520.0], dtype=np.float32)
    bound = bound_for(artifacts, "cast_pad", {})
    dst, report, _ = run_host(bound, x, (8,), np.float16, policy="original_cpu")
    assert dst.view(np.uint16).tolist() == [0x3C00, 0x0001, 0x0000, 0x0001, 0x0400, 0x7BFF, 0x7C00, 0x3C00]
    np.testing.assert_array_equal(dst[1:7].view(np.uint16), oracle_narrow(x).view(np.uint16))
    assert (report["source_bytes"], report["wire_bytes"], report["destination_bytes"]) == (24, 16, 16)


def test_pad_order_is_semantic(artifacts):
    x = np.array([0.3, 0.75, -0.75, 2.0, -128.0, 100.0], dtype=np.float32)
    body = oracle_quantize(x, 0.5)
    before, _, _ = run_host(bound_for(artifacts, "pad_quantize", {}), x, (8,), np.int8, policy="original_cpu")
    after, _, _ = run_host(bound_for(artifacts, "quantize_pad", {}), x, (8,), np.int8, policy="original_cpu")
    np.testing.assert_array_equal(before[1:7], body)
    np.testing.assert_array_equal(after[1:7], body)
    assert before[0] == before[7] == 2, "f32 1.0 padded before the quantize at scale 0.5 is code 2"
    assert after[0] == after[7] == 1, "s8 1 padded after the quantize stays 1"


def test_two_lossy_stages_keep_the_intermediate_rounding(artifacts):
    x = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, -64.5, 64.0, np.nan, np.inf, -np.inf, 0.3], dtype=np.float32)
    bound = bound_for(artifacts, "quant_dequant", {"N": 12})
    dst, report, _ = run_host(bound, x, (12,), np.float32, policy="original_cpu")
    q = oracle_quantize(x, 0.5)
    assert q.tolist() == [-2, -2, 0, 0, 2, 2, -128, 127, -128, 127, -128, 1]
    np.testing.assert_array_equal(dst.view(np.uint32), oracle_dequantize(q, 0, 0.5).view(np.uint32))
    assert dst[-1] == np.float32(0.5), "0.3 -> code 1 -> 0.5: not the identity"
    rows = {r["implementation"]: r["wire_bytes"] for r in pyreloc.query_capability(bound, "h2d", "cuda")["eligible"]}
    assert rows["cpu_stages_cuda_stages@1"] == 12 and rows["cpu_reference"] == 48


def test_witness_channel_follows_the_result_coordinates(artifacts):
    x = (np.arange(24, dtype=np.float32) - 11.5).reshape(2, 3, 4)
    scales = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    expected = np.ascontiguousarray(oracle_quantize(x, scales[None, :, None]).transpose(1, 0, 2))
    dst, _, _ = run_host(bound_for(artifacts, "witness", {}), x, (3, 2, 4), np.int8, policy="original_cpu")
    np.testing.assert_array_equal(dst, expected)
    assert dst[:, 0, :].tolist() == [[-23, -21, -19, -17], [-8, -6, -6, -4], [-2, -1, -1, 0]]


#===----------------------------------------------------------------------===#
# Policies, selection and rejections.
#===----------------------------------------------------------------------===#

def test_policies_and_explicit_rows(artifacts):
    bound = bound_for(artifacts, "quant_dequant", {"N": 16})
    x = specials(16, 2)
    # auto on a host-only device end: the reference is the only qualified path.
    dst, report, _ = run_host(bound, x, (16,), np.float32, policy="auto")
    assert (report["implementation"], report["policy"], report["placement_reason"]) == (
        "cpu_reference", "auto", "only_qualified_path")
    # Pure selection for a CUDA device end without a calibration.
    chosen = pyreloc.select_dispatch(bound, "h2d", "cuda", policy="auto")
    assert (chosen["implementation"], chosen["placement_reason"]) == ("cpu_reference", "no_calibration")
    chosen = pyreloc.select_dispatch(bound, "h2d", "cuda", policy="original_cpu")
    assert (chosen["implementation"], chosen["policy"], chosen["placement_reason"]) == ("cpu_reference", "original_cpu", "forced")
    chosen = pyreloc.select_dispatch(bound, "h2d", "cuda", implementation="cpu_stages_cuda_stages@1")
    assert (chosen["policy"], chosen["wire_bytes"], chosen["wire_boundary"]) == ("explicit", 16, 1)
    with pytest.raises(pyreloc.TransferError, match="implementation_unavailable"):
        pyreloc.select_dispatch(bound, "h2d", "cuda", implementation="cuda_dequant_relocate")
    with pytest.raises(pyreloc.TransferError, match="implementation_unavailable"):
        pyreloc.prepare_dispatch(bound, view(x), view(np.zeros(16, np.float32)), "h2d",
                                 implementation="cpu_stages_cuda_stages@1")  # host views: no CUDA rows
    with pytest.raises(ValueError):
        pyreloc.select_dispatch(bound, "h2d", "cuda", policy="fastest")
    # Views must match the typed program, not the layout's element size.
    with pytest.raises(pyreloc.TransferError, match="plan_mismatch"):
        pyreloc.prepare_dispatch(bound, view(np.zeros(32, np.float16)), view(np.zeros(16, np.float32)), "h2d")
    with pytest.raises(pyreloc.TransferError, match="plan_mismatch"):
        pyreloc.prepare_dispatch(bound, view(x), view(np.zeros(8, np.float32)), "h2d")
    # The layout-only validators still refuse the typed layout.
    with pytest.raises(pyreloc.TransferError, match="typed_unsupported"):
        pyreloc.validate_transfer_source(bound.layout, view(x), "h2d")


def test_invalid_parameters_fail_at_binding_before_any_dispatch(artifacts):
    plan = pyreloc.load_typed_plan(artifacts["dequant_runtime"][0])
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_typed(plan, {"N": 8}, {"s": scalar_f32(0.3), "zp": scalar_i32(200)})
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_typed(plan, {"N": 8}, {"s": scalar_f32(0.0), "zp": scalar_i32(0)})
    with pytest.raises(pyreloc.BindError, match="unbound parameter"):
        pyreloc.bind_typed(plan, {"N": 8}, {"s": scalar_f32(0.3)})
    plan = pyreloc.load_typed_plan(artifacts["quantize_channel_transpose"][0])
    with pytest.raises(pyreloc.BindError, match="length 2"):
        pyreloc.bind_typed(plan, {"B": 4}, {"s": f32([1.0, 2.0])})


def test_prefold_spec_names_only_the_s8_variants(artifacts):
    bound = bound_for(artifacts, "quantize_channel_transpose", {"B": 8}, {"s": f32([0.5, 0.25, 0.125])})
    spec = pyreloc.typed_prefold_spec(bound)
    assert spec["output_spec"] == "s8_gather_quant" and spec["channels"] == 3
    assert struct.unpack("<3f", spec["inv_scales"]) == (2.0, 4.0, 8.0)
    with pytest.raises(pyreloc.TransferError, match="prefold_unavailable"):
        pyreloc.typed_prefold_spec(bound_for(artifacts, "quant_dequant", {"N": 8}))
    with pytest.raises(pyreloc.TransferError, match="prefold_unavailable"):
        pyreloc.typed_prefold_spec(bound_for(artifacts, "pad_quantize", {}))
