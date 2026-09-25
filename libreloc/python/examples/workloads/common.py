"""Shared helpers for the workload examples (DLRM, GNN, LLM offload, MoE).

Every example runs its workload twice on the same inputs: a reference pass
with plain PyTorch transfers and a Sym pass. ``Report`` collects the checks
(bit-identical transfers and outputs, expected plan counts), byte totals,
dispatch rows and descriptive timings; nothing here gates on time. A missing
prerequisite raises ``PrerequisiteError``, which ``run_example`` turns into
exit status 2.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[4]
EXIT_PREREQUISITE = 2
# GPU-name marker -> the committed cost-model calibration of that box.
CALIBRATIONS = (
    ("2080 Ti", REPO / "calibration" / "epyc7351-2080ti.cal"),
    ("4070 Ti SUPER", REPO / "calibration" / "7800x3d-4070tis.cal"),
)
_BACKEND_COUNTERS = ("dynamo_compiles", "plan_compiles", "symbol_binds", "runtime_executions", "typed_executions",
                     "typed_payload_bytes")
_TORCH_HINT = "run with the qualified interpreter (/tmp/sym-torch-cuda/bin/python or $SYM_PYTHON)"


class PrerequisiteError(RuntimeError):
    """The environment cannot run the example (exit status 2)."""


def run_example(main):
    """Run ``main() -> int``; a ``PrerequisiteError`` becomes exit status 2."""
    try:
        return main()
    except PrerequisiteError as error:
        print(f"prerequisite: {error}", file=sys.stderr)
        return EXIT_PREREQUISITE


def import_torch():
    """Import torch for an example module, or exit 2 with a one-line
    prerequisite message instead of an ImportError traceback."""
    try:
        import torch
    except ImportError as error:
        print(f"prerequisite: torch is not importable ({error}); {_TORCH_HINT}", file=sys.stderr)
        raise SystemExit(EXIT_PREREQUISITE) from None
    return torch


#===----------------------------------------------------------------------===#
# Environment.
#===----------------------------------------------------------------------===#

def exporter_path(environ=None):
    """The exporter from SYM_RELOC_EXPORT, else the sibling of SYM_OPT."""
    environ = os.environ if environ is None else environ
    configured = environ.get("SYM_RELOC_EXPORT")
    if not configured and environ.get("SYM_OPT"):
        configured = str(pathlib.Path(environ["SYM_OPT"]).with_name("sym-reloc-export"))
    return pathlib.Path(configured) if configured else None


def check_environment(devices):
    """Every problem that prevents running on ``devices`` (empty when ready)."""
    try:
        import torch
    except ImportError as error:
        return [f"torch is not importable ({error}); {_TORCH_HINT}"]
    try:
        from reloc_torch import CompatibilityError, check_version
    except ImportError as error:
        return [f"reloc_torch is not importable ({error}); put $SYM_BUILD/python and libreloc/python on PYTHONPATH"]
    problems = []
    try:
        check_version()
    except CompatibilityError as error:
        problems.append(f"unsupported Python/PyTorch: {error}")
    try:
        import pyreloc
    except ImportError as error:
        problems.append(f"pyreloc is not importable: {error}")
    else:
        if not pyreloc.cuda_enabled:
            problems.append("pyreloc was built without RELOC_ENABLE_CUDA")
    if not torch.cuda.is_available():
        problems.append("no CUDA device is visible to PyTorch")
    else:
        count = torch.cuda.device_count()
        for name in devices:
            try:
                device = torch.device(name)
            except RuntimeError:
                problems.append(f"{name} is not a valid device name")
                continue
            if device.type != "cuda":
                problems.append(f"{name} is not a CUDA device")
            elif (device.index or 0) >= count:
                problems.append(f"{name} does not exist ({count} visible)")
    exporter = exporter_path()
    if exporter is None or not exporter.is_file():
        problems.append(f"the exporter is missing (SYM_RELOC_EXPORT/SYM_OPT): {exporter}")
    return problems


def require_environment(devices):
    problems = check_environment([str(device) for device in devices])
    if problems:
        raise PrerequisiteError("; ".join(problems))


def seed_everything(seed):
    """Seed PyTorch and keep float32 matmuls exact so both passes match bit for bit."""
    import torch

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


#===----------------------------------------------------------------------===#
# Measuring and comparing.
#===----------------------------------------------------------------------===#

class Clock:
    """Wall-clock milliseconds between synchronizations of the given CUDA devices."""

    def __init__(self, devices):
        self.devices = list(devices)

    def __call__(self, fn, *args):
        self._synchronize()
        start = time.perf_counter()
        result = fn(*args)
        self._synchronize()
        return result, (time.perf_counter() - start) * 1e3

    def _synchronize(self):
        if self.devices:
            import torch

            for device in self.devices:
                torch.cuda.synchronize(device)


def same_tensor(actual, expected):
    """Bit-identical values plus the same shape, dtype, device and strides
    (strides of extent-1 dimensions are ignored)."""
    import torch

    if (actual.shape, actual.dtype, actual.device) != (expected.shape, expected.dtype, expected.device):
        return False
    if any(extent > 1 and a != b for extent, a, b in zip(actual.shape, actual.stride(), expected.stride())):
        return False
    return bool(torch.equal(actual, expected))


def difference(actual, expected):
    """One line describing how two tensors differ (a failed check's detail)."""
    if (actual.shape, actual.dtype, actual.device) != (expected.shape, expected.dtype, expected.device):
        return (f"metadata differs: {tuple(actual.shape)} {actual.dtype} {actual.device} vs "
                f"{tuple(expected.shape)} {expected.dtype} {expected.device}")
    gap = (actual.double() - expected.double()).abs().max().item() if actual.numel() else 0.0
    return f"max abs diff {gap:.6g}; strides {actual.stride()} vs {expected.stride()}"


def mib(count):
    return f"{count / 2 ** 20:.2f} MiB"


#===----------------------------------------------------------------------===#
# Report.
#===----------------------------------------------------------------------===#

class Report:
    """Checks, counters, bytes, dispatch rows and timings of one example run."""

    def __init__(self, example, devices=(), **sizes):
        self.data = {
            "example": example, "devices": [str(device) for device in devices], "sizes": sizes, "checks": {},
            "counters": {}, "plan_compiles": 0, "bytes": {}, "dispatches": {}, "timings_ms": {}, "workload": {},
            "notes": [],
        }
        self._samples = {}

    def check(self, name, condition, detail=None):
        """Record one observation of check ``name`` and return it. A check
        passes when every observation passed. ``detail`` (a string, or a
        callable evaluated only on failure) explains the first failure."""
        entry = self.data["checks"].setdefault(name, {"passed": 0, "failed": 0})
        ok = bool(condition)
        if ok:
            entry["passed"] += 1
        else:
            entry["failed"] += 1
            if detail is not None and "first_failure" not in entry:
                entry["first_failure"] = str(detail() if callable(detail) else detail)
        return ok

    @property
    def ok(self):
        checks = self.data["checks"].values()
        return bool(checks) and all(entry["failed"] == 0 for entry in checks)

    def note(self, text):
        self.data["notes"].append(text)

    def add_bytes(self, kind, source, wire, destination, payload=None):
        """One Sym transfer's bytes; ``payload`` is the runtime's observed link
        traffic (wire plus parameter uploads) when it reports one."""
        entry = self.data["bytes"].setdefault(kind, {"transfers": 0, "source": 0, "wire": 0, "destination": 0})
        entry["transfers"] += 1
        entry["source"] += int(source)
        entry["wire"] += int(wire)
        entry["destination"] += int(destination)
        if payload is not None:
            entry["payload"] = entry.get("payload", 0) + int(payload)

    def add_dispatch(self, row, count=1):
        self.data["dispatches"][row] = self.data["dispatches"].get(row, 0) + int(count)

    def add_plan_compiles(self, count):
        self.data["plan_compiles"] += int(count)

    def set_backend_stats(self, stats):
        """Keep a RelocBackend's plan-reuse and fallback counters; count its
        plan compiles and typed dispatch rows."""
        counters = {key: int(stats[key]) for key in _BACKEND_COUNTERS}
        counters["fallbacks"] = dict(stats["fallbacks"])
        counters["exclusions"] = dict(stats["exclusions"])
        self.data["counters"] = counters
        self.add_plan_compiles(counters["plan_compiles"])
        for row, count in dict(stats["dispatches"]).items():
            self.add_dispatch(row, count)

    def time(self, path, kind, milliseconds):
        """One timing sample of transfer ``kind`` on ``path`` ("torch" or "sym")."""
        self._samples.setdefault(kind, {}).setdefault(path, []).append(float(milliseconds))

    def finish(self, output=None):
        """Summarize, write the JSON report (when ``output`` is given), print
        the summary and return the exit status: 0 if every check passed, else 1."""
        timings, steady = {}, {}
        for kind, paths in self._samples.items():
            for path, samples in paths.items():
                rest = samples[1:]
                timings.setdefault(kind, {})[path] = {
                    "count": len(samples), "first_call": round(samples[0], 3),
                    "median": round(statistics.median(rest), 3) if rest else None, "total": round(sum(samples), 3)}
                steady[path] = round(steady.get(path, 0.0) + sum(rest), 3)
        counters = self.data["counters"]
        byte_totals = list(self.data["bytes"].values())
        self.data["timings_ms"] = timings
        self.data["ok"] = self.ok
        self.data["summary"] = {
            "checks_passed": sum(1 for entry in self.data["checks"].values() if entry["failed"] == 0),
            "checks_total": len(self.data["checks"]),
            "plan_compiles": self.data["plan_compiles"],
            "sym_transfers": sum(entry["transfers"] for entry in byte_totals),
            "wire_bytes": sum(entry["wire"] for entry in byte_totals),
            "destination_bytes": sum(entry["destination"] for entry in byte_totals),
            "fallbacks": sum(counters.get("fallbacks", {}).values()) + sum(counters.get("exclusions", {}).values()),
            "steady_transfer_ms": steady,
        }
        if output:
            pathlib.Path(output).write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        print(self.summary_text())
        return 0 if self.ok else 1

    def summary_text(self):
        summary = self.data["summary"]
        lines = [f"{self.data['example']}: {'PASS' if self.ok else 'FAIL'} "
                 f"({summary['checks_passed']}/{summary['checks_total']} checks)"]
        for name, entry in self.data["checks"].items():
            if entry["failed"]:
                detail = f" -- {entry['first_failure']}" if "first_failure" in entry else ""
                lines.append(f"  FAILED {name}: {entry['failed']} of {entry['passed'] + entry['failed']}{detail}")
        lines.append(f"  devices: {', '.join(self.data['devices']) or '-'}; sizes: {self.data['sizes']}")
        rows = ", ".join(f"{row}={count}" for row, count in sorted(self.data["dispatches"].items())) or "none"
        lines.append(f"  sym transfers: {summary['sym_transfers']}; plan compiles: {summary['plan_compiles']}; "
                     f"typed dispatch rows: {rows}; fallbacks/exclusions: {summary['fallbacks']}")
        for kind, entry in self.data["bytes"].items():
            lines.append(f"  {kind}: {entry['transfers']} Sym transfers moved {mib(entry['wire'])} over the link "
                         f"for {mib(entry['destination'])} of destination tensors (sources {mib(entry['source'])})")
        for kind, paths in self.data["timings_ms"].items():
            medians = [f"{path} {timing['median']:.2f} ms" for path, timing in sorted(paths.items())
                       if timing["median"] is not None]
            if medians:
                lines.append(f"  {kind} median per transfer: {' / '.join(medians)} (descriptive only)")
        lines.extend(f"  note: {text}" for text in self.data["notes"])
        return "\n".join(lines)


def reference_transfer(report, clock, kind, fn):
    """Wrap a PyTorch transfer: timed as the reference path of ``kind``."""
    def call(*args):
        result, milliseconds = clock(fn, *args)
        report.time("torch", kind, milliseconds)
        return result
    return call


def sym_transfer(report, clock, kind, fn, reference, count_bytes=True):
    """Wrap a Sym transfer: time it, check ``<kind>_equal`` against
    ``reference`` applied to the same arguments (untimed) and, unless the
    transfer reports its bytes itself, count them: for layout-only transfers
    and the ``cpu_reference`` row the link carries exactly the destination."""
    def call(source, *rest):
        result, milliseconds = clock(fn, source, *rest)
        report.time("sym", kind, milliseconds)
        expected = reference(source, *rest)
        report.check(f"{kind}_equal", same_tensor(result, expected), lambda: difference(result, expected))
        if count_bytes:
            size = result.numel() * result.element_size()
            report.add_bytes(kind, source.numel() * source.element_size(), size, size)
        return result
    return call


#===----------------------------------------------------------------------===#
# Calibration and offline weight preparation.
#===----------------------------------------------------------------------===#

def calibration_path(device_name):
    """The committed calibration of the box with this GPU, or None."""
    for marker, path in CALIBRATIONS:
        if marker in device_name:
            return path
    return None


def resolve_calibration(value, device_name):
    """``--calibration`` value -> (pyreloc.Calibration or None, label for the report)."""
    import pyreloc

    if value == "none":
        return None, "none"
    path = calibration_path(device_name) if value == "auto" else pathlib.Path(value)
    if path is None:
        return None, f"none (no committed calibration matches {device_name!r})"
    try:
        return pyreloc.load_calibration(str(path)), str(path)
    except ValueError as error:
        raise PrerequisiteError(f"cannot load calibration {path}: {error}") from error


def quantize_per_channel(weight):
    """Offline checkpoint preparation, done once with PyTorch (not Sym):
    symmetric int8 per output channel of a float32 ``[in, out]`` weight.
    Returns ``(q, scale)``: int8 ``[in, out]`` and float32 ``[out]``, every
    scale strictly positive (an all-zero channel keeps the smallest normal)."""
    import torch

    scale = (weight.abs().amax(dim=0) / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(weight / scale).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float32).contiguous()


#===----------------------------------------------------------------------===#
# Offloaded int8 weights through one symbolic typed recipe.
#===----------------------------------------------------------------------===#

class WeightFetcher:
    """int8 ``[in, out]`` weight + float32 per-output-channel scales on the
    CPU -> float32 ``[out, in]`` (the ``nn.Linear`` layout) on a GPU.

    One symbolic typed recipe, compiled once, is bound for every matrix
    shape: ``int8[s0, s1] -> dequantize(scale[s1], axis 1) -> transpose ->
    float32[s1, s0]``. ``policy="auto"`` with a calibration lets the
    runtime's cost model choose where the dequantize runs; without one the
    runtime records ``no_calibration`` and takes the ``cpu_reference`` row.
    Bytes are recorded under ``kind``; ``implementation`` forces a row (tests)."""

    def __init__(self, report, calibration=None, kind="weights", implementation=""):
        from reloc_torch.compiler import CompilerClient
        from reloc_torch.recipe import BindingParam, Dequantize, Recipe, TensorSpec, Transpose
        from reloc_torch.symbolic import Const, Symbol, dense_strides

        rows, cols = Symbol("s0"), Symbol("s1")

        def spec(shape, dtype):
            return TensorSpec(shape, dense_strides(shape), Const(0), dtype)

        recipe = Recipe(
            spec((rows, cols), "int8"),
            (Dequantize("float32", BindingParam("scale", "float32", (cols,)), None, 1, "affine"), Transpose((1, 0))),
            spec((cols, rows), "float32"),
            "h2d",
        )
        self.compiled = CompilerClient.from_environment().compile(recipe)
        self.compiles = 1
        self.shapes = set()
        self.report = report
        self.calibration = calibration
        self.kind = kind
        self.implementation = implementation
        report.add_plan_compiles(1)

    def fetch(self, q, scale, device):
        import torch
        from reloc_torch import dispatch

        device = torch.device(device)
        request = dispatch.prepare_typed_transfer(
            self.compiled, q, device, parameters={"scale": scale}, policy="auto",
            calibration=self.calibration, implementation=self.implementation)
        # Workaround for a runtime bug: typed dispatch launches its CUDA
        # kernels on the caller's current device (libreloc/src/Dispatch.cpp has
        # no device scope around the launches), so a GPU row targeting another
        # device fails with "invalid resource handle". Remove once fixed.
        with torch.cuda.device(device):
            result = dispatch.execute_typed_transfer(request)
        report = result.report
        self.shapes.add(tuple(q.shape))
        self.report.add_dispatch(report["implementation"])
        self.report.add_bytes(self.kind, report["source_bytes"], report["wire_bytes"], report["destination_bytes"],
                              payload=report["payload_bytes_transferred"])
        return result.tensor

    @staticmethod
    def reference(q, scale, device):
        """The same conversion in PyTorch: move the int8 weight, dequantize and transpose on the GPU."""
        return (q.to(device).float() * scale.to(device)).t().contiguous()
