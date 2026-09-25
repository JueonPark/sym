"""Diagnose the installed compiler, runtime and optional Torch frontend."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import __version__
from .tools import native_tool


def diagnose(require_cuda=False, require_torch=False):
    report = dict(
        schema_version=1,
        status="ok",
        variant=__version__.split("+")[-1],
        versions={"sym": __version__, "python": sys.version.split()[0]},
        checks={},
    )

    def check(name, action):
        try:
            detail = action()
            report["checks"][name] = dict(status="ok", detail=detail)
        except Exception as error:
            report["checks"][name] = dict(status="failed", detail=str(error))
            report["status"] = "failed"

    def build_info():
        data = json.loads(Path(__file__).with_name("_build_info.json").read_text())
        if (
            data.get("schema_version") != 1
            or data.get("version") != __version__
            or data.get("variant") != report["variant"]
        ):
            raise RuntimeError(
                "installed build identity does not match package version"
            )
        if data.get("wire_versions") != [0, 1]:
            raise RuntimeError("unexpected installed wire support")
        return data

    def tools():
        paths = {
            name: str(native_tool(name))
            for name in ("sym-opt", "sym-reloc-export", "reloc-run-artifact")
        }
        subprocess.run(
            [paths["sym-opt"], "--version"], check=True, capture_output=True, text=True
        )
        return paths

    def runtime():
        import pyreloc

        if bool(pyreloc.cuda_enabled) != (report["variant"] == "cu126"):
            raise RuntimeError(
                "native runtime variant does not match installed package"
            )
        return dict(cuda_enabled=bool(pyreloc.cuda_enabled), module=pyreloc.__file__)

    def torch_check():
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch" or require_torch or require_cuda:
                raise
            return "Torch not installed (optional for standalone runtime)"
        from reloc_torch import check_version

        check_version()
        if str(torch.__version__) != f"2.14.0+{report['variant']}":
            raise RuntimeError("Torch and Sym package variants differ")
        report["versions"]["torch"] = str(torch.__version__)
        if require_cuda:
            if report["variant"] != "cu126" or not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but the installed variant or NVIDIA device is unavailable"
                )
            # Force real allocation, execution and synchronization.
            if (torch.arange(4, device="cuda") + 1).sum().item() != 10:
                raise RuntimeError("CUDA execution returned an incorrect result")
        return dict(version=str(torch.__version__), cuda=torch.version.cuda)

    check("build", build_info)
    check("tools", tools)
    check("runtime", runtime)
    check("torch", torch_check)
    return report


def print_report(report, as_json):
    if as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"Sym {report['versions']['sym']} ({report['variant']}): {report['status']}"
        )
        for name, result in report["checks"].items():
            print(f"  {name}: {result['status']} — {result['detail']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = diagnose(args.require_cuda, args.require_torch)
    print_report(report, args.json)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
