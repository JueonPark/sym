#!/usr/bin/env python3
"""Reproduce the public compiler -> artifact -> load/bind -> execute paths (R4, issue #148).

Runs every accepted example as a named scenario and writes machine-readable
results (environment, command, outcome, dispatch/fallback evidence, bytes):

    compiler_runtime_handoff.py --device cpu  --output runtime-cpu.json
    compiler_runtime_handoff.py --device cuda --output runtime-cuda.json

Prerequisites are validated before any scenario runs: the real exporter
(``SYM_RELOC_EXPORT``, else the sibling of ``SYM_OPT``), the standalone C++
consumer ``reloc-run-artifact`` from the same build tree, an importable
``pyreloc`` extension and, for ``--device cuda``, a CUDA-enabled extension
plus a visible GPU. A missing prerequisite is an error (exit 2), never a skip.
Every scenario is required; any failure makes the exit status nonzero.
Expected exclusions (a reason-coded fallback to PyTorch) are scenarios in
their own right and are reported as ``fallback`` outcomes, never counted as
offloads. Latency figures are descriptive only; nothing here gates on them.

Environment (see docs/runtime-integration.md):
    export PYTHONPATH="$BUILD/python:$PWD/libreloc/python"
    export SYM_OPT="$BUILD/sym/tools/sym-opt"
    export SYM_RELOC_EXPORT="$BUILD/sym/tools/sym-reloc-export"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

REPO = pathlib.Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "libreloc" / "python" / "examples"
RECIPES = REPO / "libreloc" / "examples" / "recipes"
TYPED_CORPUS = REPO / "libreloc" / "test" / "corpus" / "typed"
V0_CORPUS = REPO / "libreloc" / "test" / "corpus"
SCHEMA = 1
CHILD_TIMEOUT = 900


class ScenarioFailure(Exception):
    """A required scenario did not produce its documented result."""


def check(condition, message):
    if not condition:
        raise ScenarioFailure(message)


#===----------------------------------------------------------------------===#
# Prerequisites and environment.
#===----------------------------------------------------------------------===#

def exporter_path(environ=os.environ):
    configured = environ.get("SYM_RELOC_EXPORT")
    if configured is None and environ.get("SYM_OPT"):
        configured = str(pathlib.Path(environ["SYM_OPT"]).with_name("sym-reloc-export"))
    return pathlib.Path(configured) if configured else None


def build_root(exporter, override=None):
    if override:
        return pathlib.Path(override)
    # <build>/sym/tools/sym-reloc-export
    return exporter.resolve().parents[2] if exporter else None


def prerequisites(device, build):
    """Return (problems, facts). Problems are fatal: the run is not acceptance."""
    problems, facts = [], {}
    exporter = exporter_path()
    if exporter is None or not exporter.is_file():
        problems.append(f"the real exporter is missing (SYM_RELOC_EXPORT/SYM_OPT): {exporter}")
    facts["exporter"] = str(exporter)
    root = build_root(exporter, build)
    consumer = root / "libreloc" / "examples" / "reloc-run-artifact" if root else None
    if consumer is None or not consumer.is_file():
        problems.append(f"the C++ consumer reloc-run-artifact is missing: {consumer} "
                        "(build the reloc-run-artifact target in the same tree)")
    facts["run_artifact"] = str(consumer)
    try:
        import pyreloc

        facts["pyreloc"] = pyreloc.__file__
        facts["pyreloc_cuda_enabled"] = bool(pyreloc.cuda_enabled)
    except ImportError as error:
        problems.append(f"pyreloc is not importable: {error}")
        return problems, facts
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda"] = torch.version.cuda
        cuda = torch.cuda.is_available()
        facts["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
    except ImportError:
        problems.append("torch is not importable: the handoff runner needs the qualified Torch environment")
        cuda = False
    if device == "cuda":
        if not facts.get("pyreloc_cuda_enabled"):
            problems.append("--device cuda needs a pyreloc built with RELOC_ENABLE_CUDA")
        if not cuda:
            problems.append("--device cuda needs a visible CUDA device")
    return problems, facts


def environment(facts):
    import numpy

    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    env = {
        "source_revision": run("git", "rev-parse", "HEAD"),
        "source_dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "python_build": sys.implementation.name + " " + (sys.implementation.cache_tag or ""),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        **facts,
    }
    driver = run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    if driver:
        env["nvidia_smi"] = driver.splitlines()
    return env


#===----------------------------------------------------------------------===#
# Helpers.
#===----------------------------------------------------------------------===#

def child(command, *, timeout=CHILD_TIMEOUT, expect_zero=True):
    """Run a child command; return (returncode, stdout, stderr, seconds)."""
    start = time.perf_counter()
    proc = subprocess.run([str(c) for c in command], capture_output=True, text=True, cwd=REPO, timeout=timeout)
    seconds = time.perf_counter() - start
    if expect_zero and proc.returncode != 0:
        raise ScenarioFailure(f"{' '.join(map(str, command))} exited {proc.returncode}: "
                              f"{(proc.stderr or proc.stdout).strip()[-800:]}")
    return proc.returncode, proc.stdout, proc.stderr, seconds


def child_json(stdout, where):
    """The last JSON document a child printed; malformed output is a failure."""
    text = stdout.strip()
    start = text.find("{")
    check(start >= 0, f"{where} printed no JSON result")
    try:
        value = json.loads(text[start:])
    except json.JSONDecodeError as error:
        raise ScenarioFailure(f"{where} printed malformed JSON: {error}") from None
    check(isinstance(value, dict), f"{where} printed a non-object JSON result")
    return value


def export(exporter, recipe, directory, *, typed=False):
    directory.mkdir(parents=True, exist_ok=True)
    plan, manifest = directory / "plan.bin", directory / "manifest.json"
    command = [exporter, recipe, "--output", plan, "--manifest", manifest] + (["--typed"] if typed else [])
    child(command)
    meta = json.loads(manifest.read_text())
    check(meta["status"] == "ok", f"export of {recipe.name} is {meta['status']}")
    check(meta["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest(), "plan digest mismatch")
    return plan, meta


def transpose_pad_expected(x):
    import numpy as np

    return np.pad(np.transpose(x, (1, 2, 0)), ((1, 2), (0, 0), (0, 0)), constant_values=np.float32(1.5))


def split_transpose_expected(x):
    return x.reshape(x.shape[0] // 64, 64).T.copy()


def run_consumer(consumer, plan, symbols, source, direction, tmp):
    import numpy as np

    inp, out = tmp / f"in-{direction}.bin", tmp / f"out-{direction}.bin"
    inp.write_bytes(np.ascontiguousarray(source).tobytes())
    if out.exists():
        out.unlink()
    binding = ",".join(f"{k}={v}" for k, v in symbols.items())
    command = [consumer, plan, "--input", inp, "--output", out, "--direction", direction]
    if binding:
        command[2:2] = ["--symbols", binding]
    _, stdout, _, seconds = child(command)
    report = child_json(stdout, "reloc-run-artifact")
    return out.read_bytes(), report, seconds


#===----------------------------------------------------------------------===#
# Scenarios. Each returns a details dict or raises ScenarioFailure.
#===----------------------------------------------------------------------===#

def s_export(ctx):
    details = {}
    for name in ("transpose_pad", "split_transpose"):
        plan, meta = export(ctx["exporter"], RECIPES / f"{name}.mlir", ctx["tmp"] / name)
        check((meta["schema_version"], meta["wire_version"]) == (1, 0), f"{name}: expected schema 1 / wire v0")
        ctx["plans"][name] = plan
        details[name] = {"plan_bytes": plan.stat().st_size, "symbols": meta["symbols"],
                         "constraints": meta["constraints"], "plan_sha256": meta["plan_sha256"]}
    return details


def _consumer_case(ctx, name, direction, sizes):
    import numpy as np

    rows = []
    for n in sizes:
        if name == "transpose_pad":
            x = (np.arange(24, dtype=np.float32) * np.float32(0.25) - np.float32(3)).reshape(2, 3, 4)
            expected, symbols = transpose_pad_expected(x), {}
        else:
            x = np.arange(n, dtype=np.float32) * np.float32(0.5)
            expected, symbols = split_transpose_expected(x), {"s0": n}
        data, report, seconds = run_consumer(ctx["consumer"], ctx["plans"][name], symbols, x, direction, ctx["tmp"])
        check(data == np.ascontiguousarray(expected).tobytes(), f"{name} {direction} {symbols}: bytes differ from the numpy replay")
        check(report["destination_bytes"] == expected.nbytes, "destination footprint mismatch")
        if direction == "h2d":
            check(report["verification_readback_bytes"] == expected.nbytes, "h2d readback must be reported separately")
        rows.append({"symbols": symbols, "report": report, "seconds": round(seconds, 4)})
    return {"artifact": str(ctx["plans"][name].name), "cases": rows}


def s_cpp_host(ctx):
    return {
        "transpose_pad": _consumer_case(ctx, "transpose_pad", "host", [None]),
        "split_transpose_rebind": _consumer_case(ctx, "split_transpose", "host", [64, 128, 192]),
    }


def s_cpp_cuda(ctx):
    return {
        f"{name}_{direction}": _consumer_case(ctx, name, direction, sizes)
        for name, sizes in (("transpose_pad", [None]), ("split_transpose", [64, 128, 192]))
        for direction in ("h2d", "d2h")
    }


def s_cpp_rejections(ctx):
    import numpy as np

    tmp, consumer, plan = ctx["tmp"], ctx["consumer"], ctx["plans"]["split_transpose"]
    source = tmp / "rej-in.bin"
    source.write_bytes(np.zeros(128, np.float32).tobytes())
    truncated = tmp / "truncated.bin"
    truncated.write_bytes(plan.read_bytes()[:-3])
    typed = TYPED_CORPUS / "cast_transpose_f16.bin"
    cases = {
        "unknown_symbol": ([plan, "--symbols", "s0=128,bogus=1"], "bind error"),
        "guard_violation": ([plan, "--symbols", "s0=100"], "bind error"),
        "short_input": ([plan, "--symbols", "s0=192"], "input holds"),
        "malformed_plan": ([truncated, "--symbols", "s0=128"], "decode error"),
        "typed_plan_refused": ([typed], "typed (wire v1)"),
    }
    details = {}
    for name, (args, needle) in cases.items():
        out = tmp / f"rej-{name}.bin"
        code, _, stderr, _ = child([consumer, *args, "--input", source, "--output", out, "--direction", "host"],
                                   expect_zero=False)
        check(code != 0, f"{name}: the consumer accepted invalid input")
        check(needle in stderr, f"{name}: expected diagnostic {needle!r}, got {stderr.strip()!r}")
        check(not out.exists(), f"{name}: output written despite the failure")
        details[name] = {"exit": code, "diagnostic": stderr.strip()}
    return details


FRESH_PROCESS = r"""
import json, sys
import numpy as np
import pyreloc
from reloc_torch.artifact import CompiledRecipe
compiled = CompiledRecipe.load(sys.argv[1])
plan = pyreloc.load_plan(compiled.plan_bytes)
out = []
for n in (64, 128, 192):
    bound = pyreloc.bind(plan, {name: n for name in compiled.symbols})
    x = np.arange(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    pyreloc.relocate(bound, x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes)
    out.append({"n": n, "exact": y.tobytes() == x.reshape(n // 64, 64).T.copy().tobytes(),
                "total_bytes": bound.total_bytes})
print(json.dumps({"symbols": list(compiled.symbols), "wire_version": compiled.wire_version, "bindings": out}))
"""


def s_portable_artifact(ctx):
    from reloc_torch.compiler import CompilerClient
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, FloorDiv, Symbol, dense_strides

    n = Symbol("s0")
    destination = (Const(64), FloorDiv(n, 64))
    recipe = Recipe(TensorSpec((n,), (Const(1),), Const(0), "float32"),
                    (Reshape((FloorDiv(n, 64), Const(64))), Transpose((1, 0))),
                    TensorSpec(destination, dense_strides(destination), Const(0), "float32"), "h2d")
    compiled = CompilerClient.from_environment().compile(recipe)
    path = ctx["tmp"] / "split_transpose.reloc"
    compiled.save(path)
    _, stdout, _, _ = child([sys.executable, "-c", FRESH_PROCESS, path])
    result = child_json(stdout, "fresh-process loader")
    check(all(b["exact"] for b in result["bindings"]), "fresh-process rebinding produced wrong bytes")
    check(result["wire_version"] == 0, "layout artifact must be wire v0")
    return {"format_version": json.loads(path.read_bytes())["format_version"], **result}


def s_wire_compat(ctx):
    import pyreloc

    v0 = sorted(V0_CORPUS.glob("*.bin"))
    v1 = sorted(TYPED_CORPUS.glob("*.bin"))
    check(v0 and v1, "the committed v0 and v1 corpora are required")
    for blob in (p.read_bytes() for p in v0):
        check(pyreloc.wire_version(blob) == 0, "v0 corpus entry has another version")
        pyreloc.load_plan(blob)
        try:
            pyreloc.load_typed_plan(blob)
            raise ScenarioFailure("the typed loader accepted a v0 plan")
        except pyreloc.DecodeError:
            pass
    for blob in (p.read_bytes() for p in v1):
        check(pyreloc.wire_version(blob) == 1, "v1 corpus entry has another version")
        pyreloc.load_typed_plan(blob)
        try:
            pyreloc.load_plan(blob)
            raise ScenarioFailure("the layout loader accepted a v1 plan")
        except pyreloc.DecodeError as error:
            check("unsupported wire format version" in str(error), f"unexpected v1 rejection: {error}")
    return {"v0_plans": len(v0), "v1_plans": len(v1),
            "pre_c3_runtime": "rejects every v1 blob at byte offset 4 (recorded baseline, docs/reloc-export.md)"}


def s_typed_corpus(ctx):
    child([sys.executable, REPO / "libreloc/test/corpus/typed/generate_typed_corpus.py", "--check"])
    return {"fixtures": len(list(TYPED_CORPUS.glob("*.json")))}


def s_typed_example(ctx):
    args = ["--cuda"] if ctx["device"] == "cuda" else []
    _, stdout, _, seconds = child([sys.executable, EXAMPLES / "torch_typed_relocation.py", *args])
    check("all outputs match the independent reference" in stdout, "typed example reported a mismatch")
    lines = [line for line in stdout.splitlines() if " via " in line]
    return {"mode": "cuda" if args else "host", "rows": lines, "seconds": round(seconds, 3)}


def s_weight_loading(ctx):
    device = "cuda:0" if ctx["device"] == "cuda" else "cpu"
    _, stdout, _, _ = child([sys.executable, EXAMPLES / "torch_weight_loading.py", "--device", device])
    report = child_json(stdout, "torch_weight_loading.py")
    return {"device": device, "checks": report.get("checks"), "counters": report.get("counters"),
            "quantized_preparation": report.get("quantized_preparation")}


def s_dynamic_transfers(ctx):
    out = {}
    for direction in ("h2d", "d2h"):
        _, stdout, _, _ = child([sys.executable, EXAMPLES / "torch_dynamic_transfers.py", "--direction", direction,
                                 "--sizes", "128", "192", "256"])
        report = child_json(stdout, "torch_dynamic_transfers.py")
        check(report.get("reused_one_plan") is True, f"{direction}: one artifact was not reused")
        check(all(s["exact"] for s in report["sizes"]), f"{direction}: a size was not exact")
        out[direction] = {"counters": report["counters"], "fallbacks": report["fallbacks"],
                          "sizes": [s["n"] for s in report["sizes"]]}
    return out


def s_dynamic_extent_one_boundary(ctx):
    """Starting at s0 = 64 makes the first capture see s0 // 64 == 1: Dynamo
    specializes size-one extents, so the transposed contiguous() is
    conditional there and that capture keeps the original region
    (conditional_materialization, a T2 boundary). Every size must still be
    exact, and the later capture (generic in s0) must reach the runtime."""
    out = {}
    for direction in ("h2d", "d2h"):
        _, stdout, _, _ = child([sys.executable, EXAMPLES / "torch_dynamic_transfers.py", "--direction", direction,
                                 "--sizes", "64", "128", "192"], expect_zero=False)
        report = child_json(stdout, "torch_dynamic_transfers.py")
        check(all(s["exact"] for s in report["sizes"]), f"{direction}: a size was not exact")
        check("conditional_materialization" in report["exclusions"],
              f"{direction}: expected the conditional_materialization exclusion, got {report['exclusions']}")
        check(report["counters"]["runtime_executions"] >= 1, f"{direction}: the generic capture never executed")
        out[direction] = {"outcome": "fallback for s0 // 64 == 1, runtime otherwise",
                          "counters": report["counters"], "exclusions": report["exclusions"]}
    return out


def s_expected_exclusion_quantize_import(ctx):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    from reloc_torch.compat import symbolic_capture
    from reloc_torch.fx_import import import_graph

    def fn(x):
        q = torch.ops.quantized_decomposed.quantize_per_tensor(x, 0.5, 0, -128, 127, torch.int8)
        return torch.ops.aten._to_copy.default(q, device=torch.device("cuda:0"))

    report = import_graph(symbolic_capture(fn, torch.ones(8)))
    reasons = sorted({e.reason for e in report.exclusions})
    check(not report.candidates and "quantize_semantics_unproved" in reasons, f"unexpected import result {reasons}")
    return {"outcome": "fallback", "reasons": reasons}


def s_expected_exclusion_nonblocking(ctx):
    import torch
    from reloc_torch import RelocBackend

    backend = RelocBackend()
    try:
        def fn(x):
            return x.t().contiguous().to("cuda", non_blocking=True)

        compiled = torch.compile(fn, backend=backend, dynamic=False)
        with torch.no_grad():
            x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
            actual = compiled(x)
        torch.cuda.synchronize()
        check(torch.equal(actual.cpu(), fn(x).cpu()), "fallback result differs from PyTorch")
        stats = backend.stats()
        check(stats["runtime_executions"] == 0, "a non-blocking transfer was offloaded")
        return {"outcome": "fallback", "exclusions": stats["exclusions"], "fallbacks": stats["fallbacks"],
                "runtime_executions": stats["runtime_executions"]}
    finally:
        backend.close()


def _median_ms(fn, sync, warmup=3, repeats=15):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return round(statistics.median(samples), 4)


def s_latency(ctx):
    """Descriptive end-to-end latency: original PyTorch vs the selected path.
    Includes neither compilation nor preparation (both happen in warmup);
    each sample ends with torch.cuda.synchronize(). No threshold is applied."""
    import torch
    from reloc_torch import RelocBackend, dispatch
    from reloc_torch.compiler import CompilerClient
    from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, dense_strides

    rows = []
    backend = RelocBackend()
    try:
        def layout(x):
            return x.reshape(x.shape[0] // 64, 64).t().contiguous().to("cuda")

        compiled = torch.compile(layout, backend=backend, dynamic=True)
        for n in (1 << 16, 1 << 20, 1 << 22):
            x = torch.arange(n, dtype=torch.float32)
            with torch.no_grad():
                t_reloc = _median_ms(lambda: compiled(x), torch.cuda.synchronize)
                t_torch = _median_ms(lambda: layout(x), torch.cuda.synchronize)
            rows.append({"scenario": "layout_h2d_split_transpose", "elements": n, "source_bytes": 4 * n,
                         "wire_bytes": 4 * n, "reloc_ms": t_reloc, "pytorch_ms": t_torch,
                         "implementation": "R2 forward H2D (compiled region)"})
        stats = backend.stats()
    finally:
        backend.close()
    shape = (Const(1024), Const(1024))
    cast = Recipe(TensorSpec(shape, dense_strides(shape), Const(0), "float32"),
                  (Transpose((1, 0)), Cast("float16", "ieee_rne")),
                  TensorSpec(shape, dense_strides(shape), Const(0), "float16"), "h2d")
    compiled_cast = CompilerClient.from_environment().compile(cast)
    x = torch.randn(1024, 1024)

    def typed():
        request = dispatch.prepare_typed_transfer(compiled_cast, x, "cuda", parameters={}, policy="original_cpu")
        return dispatch.execute_typed_transfer(request)

    report = dict(typed().report)
    rows.append({"scenario": "typed_h2d_transpose_cast_f16", "elements": 1 << 20,
                 "source_bytes": report["source_bytes"], "wire_bytes": report["wire_bytes"],
                 "payload_bytes_transferred": report["payload_bytes_transferred"],
                 "reloc_ms": _median_ms(typed, torch.cuda.synchronize),
                 "pytorch_ms": _median_ms(lambda: x.t().contiguous().to("cuda", torch.float16), torch.cuda.synchronize),
                 "implementation": report["implementation"], "placement_reason": report["placement_reason"],
                 "includes_preparation": True})
    return {"rows": rows, "backend_counters": {k: stats[k] for k in ("plan_compiles", "symbol_binds", "runtime_executions")},
            "note": "descriptive only; no speedup gate"}


def s_gpu_test_selection(ctx):
    """The gpu-marked pytest selection and the CUDA gtests on this device.
    Zero executed tests, any failure, or any skip caused by a missing device
    or runtime is a failed acceptance: such a skip never certifies a GPU
    row. Skips a test declares by design (a parametrization that exists for
    one dtype only, a configuration this host does not have) are recorded
    with their reasons as unqualified boundaries."""
    import xml.etree.ElementTree as ET

    junit = ctx["tmp"] / "gpu-junit.xml"
    code, stdout, stderr, seconds = child(
        [sys.executable, "-m", "pytest", REPO / "libreloc" / "python" / "tests", "-m", "gpu", "-q",
         "-p", "no:cacheprovider", f"--junitxml={junit}"], expect_zero=False)
    check(junit.is_file(), f"pytest wrote no result file: {(stderr or stdout)[-400:]}")
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    counts = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    executed = counts["tests"] - counts["skipped"]
    skipped = {}
    for case in suite.iter("testcase"):
        mark = case.find("skipped")
        if mark is not None:
            skipped[case.get("classname", "") + "::" + case.get("name", "")] = mark.get("message", "")
    check(code == 0 and counts["failures"] == 0 and counts["errors"] == 0, f"gpu pytest failed: {counts}")
    check(executed > 0, "the gpu pytest selection executed zero tests")
    missing = {k: v for k, v in skipped.items()
               if any(s in v.lower() for s in ("cuda unavailable", "cuda is unavailable", "adapter unavailable",
                                               "no cuda", "not built with"))}
    check(not missing, f"GPU tests skipped for a missing device or runtime: {missing}")
    gtest = ctx["build"] / "libreloc" / "test" / "libreloc-test"
    check(gtest.is_file(), f"libreloc-test is missing: {gtest}")
    report = ctx["tmp"] / "gtest.json"
    child([gtest, "--gtest_filter=Cuda*", f"--gtest_output=json:{report}"])
    data = json.loads(report.read_text())
    ran = [t for s in data["testsuites"] for t in s["testsuite"] if t.get("result") == "COMPLETED"]
    skipped_g = [t["name"] for s in data["testsuites"] for t in s["testsuite"] if t.get("result") == "SKIPPED"]
    check(ran and data["failures"] == 0 and not skipped_g, f"CUDA gtests: {len(ran)} ran, {data['failures']} failed, skipped {skipped_g}")
    return {"pytest_gpu": {**counts, "executed": executed, "seconds": round(seconds, 2),
                           "skipped_by_design": skipped},
            "gtest_cuda": {"executed": len(ran), "failures": data["failures"],
                           "suites": sorted({s["name"] for s in data["testsuites"]})}}


CPU_SCENARIOS = [
    ("export_example_recipes", s_export),
    ("cpp_host_layout", s_cpp_host),
    ("cpp_rejections", s_cpp_rejections),
    ("portable_artifact_fresh_process", s_portable_artifact),
    ("wire_version_compatibility", s_wire_compat),
    ("typed_corpus_fresh", s_typed_corpus),
    ("typed_example", s_typed_example),
    ("weight_loading", s_weight_loading),
    ("expected_exclusion_quantize_import", s_expected_exclusion_quantize_import),
]
CUDA_SCENARIOS = CPU_SCENARIOS + [
    ("cpp_cuda_layout", s_cpp_cuda),
    ("dynamic_transfers", s_dynamic_transfers),
    ("dynamic_extent_one_boundary", s_dynamic_extent_one_boundary),
    ("expected_exclusion_nonblocking", s_expected_exclusion_nonblocking),
    ("descriptive_latency", s_latency),
    ("gpu_test_selection", s_gpu_test_selection),
]


def run(device, output, build=None, only=None):
    problems, facts = prerequisites(device, build)
    result = {"schema": SCHEMA, "device": device, "environment": None, "prerequisites": problems, "scenarios": []}
    if problems:
        result["summary"] = {"passed": 0, "failed": 0, "required_failed": 0, "prerequisites_failed": len(problems)}
        write(result, output)
        for problem in problems:
            print(f"prerequisite failed: {problem}", file=sys.stderr)
        return 2
    result["environment"] = environment(facts)
    scenarios = CUDA_SCENARIOS if device == "cuda" else CPU_SCENARIOS
    if only:
        scenarios = [(name, fn) for name, fn in scenarios if name in only]
        # Scenarios that consume exported plans need the export first.
        if any(name.startswith("cpp_") for name, _ in scenarios) and "export_example_recipes" not in only:
            scenarios.insert(0, ("export_example_recipes", s_export))
    with tempfile.TemporaryDirectory(prefix="reloc-handoff-") as tmp:
        ctx = {"device": device, "tmp": pathlib.Path(tmp), "plans": {},
               "build": build_root(pathlib.Path(facts["exporter"]), build),
               "exporter": pathlib.Path(facts["exporter"]), "consumer": pathlib.Path(facts["run_artifact"])}
        for name, fn in scenarios:
            start = time.perf_counter()
            entry = {"name": name, "required": True}
            try:
                entry["details"] = fn(ctx)
                entry["status"] = "passed"
            except Exception as error:  # every failure is recorded, then fails the run
                entry["status"] = "failed"
                entry["error"] = f"{type(error).__name__}: {error}"
                if not isinstance(error, ScenarioFailure):
                    entry["traceback"] = traceback.format_exc(limit=6)
            entry["seconds"] = round(time.perf_counter() - start, 3)
            result["scenarios"].append(entry)
            print(f"[{entry['status']}] {name} ({entry['seconds']} s)"
                  + (f": {entry.get('error')}" if entry["status"] != "passed" else ""), file=sys.stderr)
    failed = [s for s in result["scenarios"] if s["status"] != "passed"]
    result["summary"] = {"passed": len(result["scenarios"]) - len(failed), "failed": len(failed),
                         "required_failed": sum(s["required"] for s in failed), "prerequisites_failed": 0}
    write(result, output)
    return 1 if result["summary"]["required_failed"] else 0


def write(result, output):
    text = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if output:
        pathlib.Path(output).write_text(text)
    else:
        sys.stdout.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", default=None, help="write the JSON results here (default: stdout)")
    parser.add_argument("--build", default=None, help="build tree holding reloc-run-artifact (default: the exporter's tree)")
    parser.add_argument("--only", nargs="+", default=None, help="run only these scenarios (debugging)")
    args = parser.parse_args(argv)
    return run(args.device, args.output, args.build, set(args.only) if args.only else None)


if __name__ == "__main__":
    sys.exit(main())
