"""Resolve and execute the compiler binaries belonging to this installation."""

import os
from pathlib import Path
import sys

TOOLS = frozenset({"sym-opt", "sym-reloc-export", "reloc-run-artifact"})


def native_tool(name: str) -> Path:
    if name not in TOOLS:
        raise ValueError(f"unknown Sym tool: {name}")
    path = Path(__file__).resolve().parent / "_native" / "bin" / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"packaged executable is absent or not executable: {path}; "
            "install a complete Sym wheel, or set SYM_RELOC_EXPORT/SYM_OPT for a source build"
        )
    return path


def _run(name):
    try:
        path = native_tool(name)
        os.execv(str(path), [str(path), *sys.argv[1:]])
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


def sym_opt_main():
    _run("sym-opt")


def export_main():
    _run("sym-reloc-export")


def run_artifact_main():
    _run("reloc-run-artifact")
