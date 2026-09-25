#!/usr/bin/env python3
"""Run the workload examples (DLRM, GNN, LLM offload, MoE) against a Sym build.

    python3 libreloc/python/examples/workloads/run_workloads.py --quick

Each example runs in its own process under the qualified CUDA interpreter,
with PYTHONPATH, SYM_OPT and SYM_RELOC_EXPORT derived from the build, and
writes <output-dir>/<name>.json and <name>.log; the runner then prints one
summary row per example and writes summary.json. Exit status: 2 when a
prerequisite is missing (here or in any example), 1 when any example
failed, crashed or timed out, else 0. Timings are descriptive only.

Defaults: --python $SYM_PYTHON, else /tmp/sym-torch-cuda/bin/python;
--build $SYM_BUILD, else <repo>/build/torch-cuda; a fresh temporary output
directory. Only the Python standard library is needed to run this script.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXAMPLES = {
    "dlrm": HERE / "dlrm_embeddings.py",
    "gnn": HERE / "gnn_minibatch.py",
    "llm": HERE / "llm_offload.py",
    "moe": HERE / "moe_experts.py",
}
USES_CALIBRATION = frozenset({"llm", "moe"})
MULTI_DEVICE = frozenset({"moe"})
EXIT_BY_STATUS = {"pass": 0, "fail": 1, "crash": 1, "timeout": 1, "prerequisite": 2}
DEFAULT_PYTHON = "/tmp/sym-torch-cuda/bin/python"
HEADER = ["example", "result", "checks", "plans", "sym transfers", "wire/dest MiB", "steady sym/torch ms",
          "fallbacks"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small sizes (seconds per example)")
    parser.add_argument("--only", nargs="+", choices=sorted(EXAMPLES), help="run only these examples")
    parser.add_argument("--devices", default="cuda:0",
                        help="comma-separated CUDA devices; dlrm, gnn and llm use the first, moe uses all")
    parser.add_argument("--calibration", default="auto", help="auto, none or a .cal path (llm and moe)")
    parser.add_argument("--build", help="Sym build tree (default: $SYM_BUILD or <repo>/build/torch-cuda)")
    parser.add_argument("--python", help="interpreter (default: $SYM_PYTHON or /tmp/sym-torch-cuda/bin/python)")
    parser.add_argument("--output-dir", help="reports and logs (default: a fresh temporary directory)")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per example (default 900)")
    return parser.parse_args(argv)


def resolve_python(value, environ=None):
    """The interpreter path; a bare name is looked up on PATH. Never resolves
    symlinks: a venv interpreter must keep its venv path."""
    environ = os.environ if environ is None else environ
    candidate = value or environ.get("SYM_PYTHON") or DEFAULT_PYTHON
    path = pathlib.Path(candidate)
    if not path.is_file():
        found = shutil.which(candidate)
        path = pathlib.Path(found) if found else path
    return path.absolute()


def resolve_build(value, environ=None):
    environ = os.environ if environ is None else environ
    return pathlib.Path(value or environ.get("SYM_BUILD") or REPO / "build" / "torch-cuda").absolute()


def preflight(python, build, scripts):
    """Problems that stop the run before any example starts (empty when ready)."""
    problems = []
    if not (python.is_file() and os.access(python, os.X_OK)):
        problems.append(f"interpreter not found or not executable: {python} (use --python or SYM_PYTHON)")
    if not (build / "python" / "pyreloc").is_dir():
        problems.append(f"no staged pyreloc package under {build / 'python'} "
                        "(build the pyreloc_ext target; use --build or SYM_BUILD)")
    exporter = build / "sym" / "tools" / "sym-reloc-export"
    if not exporter.is_file():
        problems.append(f"the exporter is missing: {exporter} (build the sym-reloc-export target)")
    problems.extend(f"example script missing: {script}" for script in scripts if not script.is_file())
    return problems


def child_environment(build, environ=None):
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = os.pathsep.join([str(build / "python"), str(REPO / "libreloc" / "python")])
    env["SYM_OPT"] = str(build / "sym" / "tools" / "sym-opt")
    env["SYM_RELOC_EXPORT"] = str(build / "sym" / "tools" / "sym-reloc-export")
    return env


def child_arguments(name, args, output):
    argv = ["--output", str(output)]
    if args.quick:
        argv.append("--quick")
    if name in MULTI_DEVICE:
        argv += ["--devices", args.devices]
    else:
        argv += ["--device", args.devices.split(",")[0]]
    if name in USES_CALIBRATION:
        calibration = args.calibration
        if calibration not in ("auto", "none"):
            calibration = str(pathlib.Path(calibration).absolute())
        argv += ["--calibration", calibration]
    return argv


def read_report(path):
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def classify(returncode, report):
    if returncode == 2:
        return "prerequisite"
    if report is None or returncode not in (0, 1):
        return "crash"
    return "pass" if returncode == 0 and report.get("ok") is True else "fail"


def overall_exit(statuses):
    return max((EXIT_BY_STATUS[status] for status in statuses), default=0)


def run_one(name, script, args, python, env, out_dir):
    output = out_dir / f"{name}.json"
    log = out_dir / f"{name}.log"
    output.unlink(missing_ok=True)                 # never read a stale report
    command = [str(python), str(script), *child_arguments(name, args, output)]
    result = {"name": name, "command": command, "log": str(log), "returncode": None, "report": None}
    with open(log, "w", encoding="utf-8") as handle:
        try:
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            return result
    result["returncode"] = proc.returncode
    result["report"] = read_report(output)
    result["status"] = classify(proc.returncode, result["report"])
    return result


def summary_row(result):
    summary = (result["report"] or {}).get("summary") or {}
    steady = summary.get("steady_transfer_ms") or {}

    def milliseconds(path):
        value = steady.get(path)
        return "-" if value is None else f"{value:.1f}"

    checks = f"{summary['checks_passed']}/{summary['checks_total']}" if "checks_total" in summary else "-"
    wire = (f"{summary['wire_bytes'] / 2 ** 20:.1f}/{summary['destination_bytes'] / 2 ** 20:.1f}"
            if "wire_bytes" in summary else "-")
    return [result["name"], result["status"], checks, str(summary.get("plan_compiles", "-")),
            str(summary.get("sym_transfers", "-")), wire, f"{milliseconds('sym')}/{milliseconds('torch')}",
            str(summary.get("fallbacks", "-"))]


def format_table(results):
    rows = [HEADER] + [summary_row(result) for result in results]
    widths = [max(len(row[column]) for row in rows) for column in range(len(HEADER))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    python = resolve_python(args.python)
    build = resolve_build(args.build)
    selected = {name: script for name, script in EXAMPLES.items() if not args.only or name in args.only}
    problems = preflight(python, build, selected.values())
    if problems:
        for problem in problems:
            print(f"prerequisite: {problem}", file=sys.stderr)
        return 2
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="sym-workloads-"))
    print(f"reports and logs: {out_dir}", flush=True)
    env = child_environment(build)
    results = []
    for name, script in selected.items():
        print(f"running {name} ...", flush=True)
        result = run_one(name, script, args, python, env, out_dir)
        print(f"  {name}: {result['status']} (log: {result['log']})", flush=True)
        results.append(result)
    print()
    print(format_table(results))
    print("steady = total transfer time without each transfer kind's first call; timings are descriptive only")
    summary = [{key: result[key] for key in ("name", "status", "returncode", "log", "command")}
               | {"summary": (result["report"] or {}).get("summary")} for result in results]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return overall_exit(result["status"] for result in results)


if __name__ == "__main__":
    raise SystemExit(main())
