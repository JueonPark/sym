"""Typed conformance (C4, issue #144), Torch-free.

Every committed fixture in ``libreloc/test/corpus/typed`` came from the
public exporter and the independent NumPy oracle. Here the real runtime
must reproduce it: decode, bind (symbols and named parameters), the forced
CPU baseline in both directions, the footprints, the capability rows, the
auto policy's recorded reasons, rebinding under another symbol binding,
guard rejections before any dispatch, reuse from a fresh process, and
byte-identical re-export from the recorded MLIR. The exporter is REQUIRED.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import pyreloc
import typed_reference as ref
from typed_support import reloc_export_executable

CORPUS = pathlib.Path(__file__).resolve().parents[2] / "test" / "corpus" / "typed"
FIXTURES = sorted(p.stem for p in CORPUS.glob("*.json"))


def load(name):
    meta = json.loads((CORPUS / f"{name}.json").read_text())
    return meta, (CORPUS / f"{name}.bin").read_bytes()


def array(block):
    return ref.from_bits(block["dtype"], [int(b, 16) for b in block["bits"]], tuple(block["shape"]))


def parameters(meta):
    return {name: array(block) for name, block in meta["parameters"].items()}


def parameter_map(values):
    return {name: (v.dtype.name, list(v.shape), v.tobytes()) for name, v in values.items()}


def view(a):
    itemsize = a.dtype.itemsize
    return pyreloc.BufferView(a.ctypes.data, a.nbytes, 0, list(a.shape),
                              [s // itemsize for s in a.strides], itemsize, "host")


def bind(meta, blob, binding, values=None):
    plan = pyreloc.load_typed_plan(blob)
    symbols = {name: binding["symbols"][name] for name in meta["symbols"]}
    return pyreloc.bind_typed(plan, symbols, parameter_map(values if values is not None else parameters(meta)))


def dispatch(bound, source, expected, direction, **options):
    actual = np.zeros(expected.shape, dtype=expected.dtype)
    actual.view(np.uint8)[...] = 0xCD
    request = pyreloc.prepare_dispatch(bound, view(source), view(actual), direction, **options)
    report = pyreloc.execute_dispatch(request)
    return actual, report


def test_corpus_is_present_and_complete():
    assert len(FIXTURES) >= 12, "run libreloc/test/corpus/typed/generate_typed_corpus.py and commit"
    names = set(FIXTURES)
    # The rows the issue asks for by name.
    assert {"cast_transpose_f16", "widen_reshape", "quantize_channel_transpose_sym",
            "quant_dequant_padded", "dequant_channel_runtime_transpose",
            "pad_then_quantize", "quantize_then_pad", "witness_channel"} <= names
    for name in FIXTURES:
        meta, blob = load(name)
        assert (CORPUS / f"{name}.reloc").is_file()
        assert meta["manifest"]["schema_version"] == 2 and meta["manifest"]["wire_version"] == 1
        assert hashlib.sha256(blob).hexdigest() == meta["plan_sha256"] == meta["manifest"]["plan_sha256"]
        assert pyreloc.wire_version(blob) == 1
        with pytest.raises(pyreloc.DecodeError):
            pyreloc.load_plan(blob)  # v0 artifacts and readers are untouched: v1 is not theirs


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_reproduces_through_the_forced_cpu_baseline(name):
    meta, blob = load(name)
    plan = pyreloc.load_typed_plan(blob)
    assert list(plan.symbols) == meta["symbols"] == meta["manifest"]["symbols"]
    for binding in meta["bindings"]:
        source, expected = array(binding["source"]), array(binding["expected"])
        assert hashlib.sha256(expected.tobytes()).hexdigest() == binding["expected"]["sha256"]
        # The stored expectation is what the oracle says today, too.
        replayed = ref.replay(meta["recipe"], source, binding["symbols"], parameters(meta))
        assert replayed.tobytes() == expected.tobytes(), "stale fixture: oracle disagrees"
        bound = bind(meta, blob, binding)
        for direction in ("h2d", "d2h"):
            actual, report = dispatch(bound, source, expected, direction, policy="original_cpu")
            assert actual.tobytes() == expected.tobytes(), (name, direction)
            assert report["implementation"] == "cpu_reference" and report["placement_reason"] == "forced"
            footprints = binding["footprints"][direction]
            for key, value in footprints.items():
                assert report[key] == value, (name, direction, key)
            assert report["payload_bytes_transferred"] == report["wire_bytes"]
            assert report["executed"] is True
        for device in ("host", "cuda"):
            for direction in ("h2d", "d2h"):
                rows = [r["implementation"] for r in pyreloc.query_capability(bound, direction, device)["eligible"]]
                assert rows == binding["capability"][device][direction], (name, device, direction)
        # Auto without a calibration is the reference, with the reason recorded.
        actual, report = dispatch(bound, source, expected, "h2d", policy="auto")
        assert actual.tobytes() == expected.tobytes()
        assert report["policy"] == "auto" and report["placement_reason"] == "only_qualified_path"


def test_pad_order_witness():
    before, after = load("pad_then_quantize"), load("quantize_then_pad")
    expected_before = array(before[0]["bindings"][0]["expected"])
    expected_after = array(after[0]["bindings"][0]["expected"])
    assert expected_before[0] == expected_before[-1] == 2, "f32 1.0 before the quantize at scale 0.5 becomes code 2"
    assert expected_after[0] == expected_after[-1] == 1, "s8 1 after the quantize stays 1"
    np.testing.assert_array_equal(expected_before[1:-1], expected_after[1:-1])
    # The compiler kept the order: the fill entered at stage 0 in one artifact
    # and at stage 1 in the other.
    assert before[0]["manifest"]["fills"][0]["stage"] == 0 and before[0]["manifest"]["fills"][0]["bits"] == "3f800000"
    assert after[0]["manifest"]["fills"][0]["stage"] == 1 and after[0]["manifest"]["fills"][0]["bits"] == "1"


def test_c1_witness_tables_are_in_the_corpus():
    edges = array(load("quantize_edges")[0]["bindings"][0]["expected"]).tolist()
    assert edges[:6] == [-2, -2, 0, 0, 2, 2]
    assert edges[6:12] == [-128, -128, -127, 126, 127, 127]
    assert edges[12:16] == [-128, 127, -128, 0]
    bits = array(load("dequant_scale_witness")[0]["bindings"][0]["expected"]).view(np.uint32).tolist()
    assert bits == [0xC219999A, 0xBE99999A, 0x00000000, 0x3E99999A, 0x42186667]
    assert array(load("dequant_zp127")[0]["bindings"][0]["expected"]).view(np.uint32)[0] == 0xC2990000
    narrow = array(load("cast_transpose_f16")[0]["bindings"][0]["expected"]).view(np.uint16)
    source = array(load("cast_transpose_f16")[0]["bindings"][0]["source"])
    # The transpose moved the boundary vectors: read them back through it.
    flat = np.ascontiguousarray(narrow.reshape(6, 4).T).reshape(-1)[:13].tolist()
    assert flat == [0x0001, 0x0000, 0x0001, 0x0400, 0x7BFF, 0x7BFF, 0x7C00, 0x7C00, 0x3C00, 0x3C01, 0x8000, 0x7C00, 0xFC00]
    assert source.dtype == np.float32
    witness = array(load("witness_channel")[0]["bindings"][0]["expected"])
    assert witness[:, 0, :].tolist() == [[-23, -21, -19, -17], [-8, -6, -6, -4], [-2, -1, -1, 0]]
    reciprocal = array(load("reciprocal_formation")[0]["bindings"][0]["expected"])
    x = array(load("reciprocal_formation")[0]["bindings"][0]["source"])
    np.testing.assert_array_equal(reciprocal, ref.quantize_f32_s8(x, np.float32(0.3)))
    # C1 §6: the scaled products x * fl32(1/0.3) and x / 0.3 differ in 3 of 8
    # binary32 bit patterns; the contract is the former (formed once).
    product = (x * (np.float32(1) / np.float32(0.3)).astype(np.float32)).astype(np.float32)
    quotient = (x / np.float32(0.3)).astype(np.float32)
    assert int(np.sum(product.view(np.uint32) != quotient.view(np.uint32))) == 3


def test_symbolic_fixtures_rebind_and_footprints_scale():
    meta, blob = load("quant_dequant_padded")
    assert len(meta["bindings"]) == 2
    first, second = meta["bindings"]
    assert first["symbols"] != second["symbols"]
    for binding in (first, second):
        n = binding["symbols"]["s0"]
        assert binding["footprints"]["h2d"] == {
            "source_bytes": 4 * n, "wire_bytes": 4 * (n + 3), "destination_bytes": 4 * (n + 3),
            "parameter_bytes": 8, "wire_boundary": 2,
        }
        assert binding["footprints"]["d2h"]["wire_bytes"] == 4 * n
    # One artifact, both bindings, each against its own oracle output.
    for binding in meta["bindings"]:
        bound = bind(meta, blob, binding)
        actual, _ = dispatch(bound, array(binding["source"]), array(binding["expected"]), "h2d", policy="original_cpu")
        assert actual.tobytes() == array(binding["expected"]).tobytes()
    rows = {r["implementation"]: r["wire_bytes"] for r in pyreloc.query_capability(bind(meta, blob, first), "h2d", "cuda")["eligible"]}
    # A cut at the s8 intermediate exists only if the pad had entered by then: it enters last, so no cuda cut.
    assert set(rows) == {"cpu_reference"}


def test_guards_reject_before_any_dispatch():
    meta, blob = load("quantize_channel_transpose_sym")
    plan = pyreloc.load_typed_plan(blob)
    good = parameter_map(parameters(meta))
    for symbols in ({"s0": 0}, {"s0": -3}, {}, {"s0": 5, "extra": 1}):
        with pytest.raises(pyreloc.BindError):
            pyreloc.bind_typed(plan, symbols, good)
    with pytest.raises(pyreloc.BindError):  # channel length
        pyreloc.bind_typed(plan, {"s0": 5}, {"s": ("float32", [2], np.array([1, 1], np.float32).tobytes())})
    with pytest.raises(pyreloc.BindError):  # non-positive scale
        pyreloc.bind_typed(plan, {"s0": 5}, {"s": ("float32", [3], np.array([1, 0, 1], np.float32).tobytes())})
    with pytest.raises(pyreloc.BindError):  # missing parameter
        pyreloc.bind_typed(plan, {"s0": 5}, {})
    meta, blob = load("dequant_channel_runtime_transpose")
    plan = pyreloc.load_typed_plan(blob)
    values = parameters(meta)
    bad = dict(values)
    bad["zero_points"] = np.array([-128, 0, 200], dtype=np.int32)
    with pytest.raises(pyreloc.BindError):  # zero point range
        pyreloc.bind_typed(plan, {}, parameter_map(bad))
    bad["zero_points"] = np.array([-128, 0, 127], dtype=np.int64)
    with pytest.raises(pyreloc.BindError):  # declared dtype
        pyreloc.bind_typed(plan, {}, parameter_map(bad))
    # A wrong destination width or a too-small source is refused at preparation.
    bound = pyreloc.bind_typed(plan, {}, parameter_map(values))
    source = array(meta["bindings"][0]["source"])
    with pytest.raises(pyreloc.TransferError, match="plan_mismatch"):
        pyreloc.prepare_dispatch(bound, view(source), view(np.zeros((3, 2, 4), np.float16)), "h2d")
    with pytest.raises(pyreloc.TransferError, match="plan_mismatch"):
        pyreloc.prepare_dispatch(bound, view(source[:1]), view(np.zeros((3, 2, 4), np.float32)), "h2d")


def test_parameter_values_change_results_between_calls():
    meta, blob = load("dequant_channel_runtime_transpose")
    binding = meta["bindings"][0]
    source, expected = array(binding["source"]), array(binding["expected"])
    values = parameters(meta)
    actual, _ = dispatch(bind(meta, blob, binding, values), source, expected, "h2d", policy="original_cpu")
    assert actual.tobytes() == expected.tobytes()
    changed = dict(values)
    changed["scales"] = np.array([0.25, 2.0, 4.0], dtype=np.float32)
    fresh, _ = dispatch(bind(meta, blob, binding, changed), source, expected, "h2d", policy="original_cpu")
    assert fresh.tobytes() != expected.tobytes()
    assert fresh.tobytes() == ref.replay(meta["recipe"], source, binding["symbols"], changed).tobytes()


def test_fresh_process_reuses_the_committed_artifact():
    meta, _ = load("dequant_channel_runtime_transpose")
    script = r"""
import hashlib, json, pathlib, sys
import numpy as np
import pyreloc
sys.path.insert(0, sys.argv[2])
import typed_reference as ref
corpus = pathlib.Path(sys.argv[1])
meta = json.loads((corpus / "dequant_channel_runtime_transpose.json").read_text())
blob = (corpus / "dequant_channel_runtime_transpose.bin").read_bytes()
def arr(b): return ref.from_bits(b["dtype"], [int(x, 16) for x in b["bits"]], tuple(b["shape"]))
def vw(a): return pyreloc.BufferView(a.ctypes.data, a.nbytes, 0, list(a.shape), [s // a.dtype.itemsize for s in a.strides], a.dtype.itemsize, "host")
params = {n: (arr(b).dtype.name, list(arr(b).shape), arr(b).tobytes()) for n, b in meta["parameters"].items()}
bound = pyreloc.bind_typed(pyreloc.load_typed_plan(blob), {}, params)
b0 = meta["bindings"][0]
src, exp = arr(b0["source"]), arr(b0["expected"])
out = np.zeros(exp.shape, exp.dtype)
report = pyreloc.execute_dispatch(pyreloc.prepare_dispatch(bound, vw(src), vw(out), "h2d", policy="original_cpu"))
print(json.dumps({"sha": hashlib.sha256(out.tobytes()).hexdigest(), "implementation": report["implementation"]}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(CORPUS), str(pathlib.Path(__file__).parent)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"sha": meta["bindings"][0]["expected"]["sha256"], "implementation": "cpu_reference"}


@pytest.mark.parametrize("name", FIXTURES)
def test_re_export_is_byte_identical(name, tmp_path):
    tool = reloc_export_executable()
    assert tool.is_file(), f"the real sym-reloc-export is required; missing: {tool}"
    meta, blob = load(name)
    inp, plan, manifest = tmp_path / "in.mlir", tmp_path / "plan.bin", tmp_path / "m.json"
    inp.write_text(meta["mlir"])
    proc = subprocess.run([str(tool), str(inp), "--output", str(plan), "--manifest", str(manifest), "--typed"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert plan.read_bytes() == blob
    fresh = json.loads(manifest.read_text())
    stable = {k: v for k, v in fresh.items() if k != "compiler"}
    assert stable == {k: v for k, v in meta["manifest"].items() if k != "compiler"}
