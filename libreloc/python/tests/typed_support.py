"""Helpers shared by the typed (C3/R3) Torch-free tests.

A module of its own (not ``conftest``) so the standalone and the
``torch_frontend`` suites can run in one pytest invocation without the two
``conftest`` modules shadowing each other.
"""
import os
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TYPED_SERIALIZE_MLIR = REPO_ROOT / "test" / "dialect" / "reloc" / "typed_serialize.mlir"
_TYPED_GOLDEN_RE = re.compile(r"typed_plan_hex\((\w+)\): ([0-9a-f]+)")


def typed_golden_hex(name):
    """Wire v1 hex of a frozen typed golden from typed_serialize.mlir (C3)."""
    for m in _TYPED_GOLDEN_RE.finditer(_TYPED_SERIALIZE_MLIR.read_text()):
        if m.group(1) == name:
            return m.group(2)
    raise KeyError(f"no typed golden plan named {name!r} in {_TYPED_SERIALIZE_MLIR}")


def reloc_export_executable():
    """The real R1/C3 exporter, as the documented environment selects it:
    ``SYM_RELOC_EXPORT``, else the sibling of ``SYM_OPT``, else the default
    build tree. Standalone typed tests fail (not skip) when it is absent."""
    configured = os.environ.get("SYM_RELOC_EXPORT")
    if configured is None and os.environ.get("SYM_OPT"):
        configured = str(pathlib.Path(os.environ["SYM_OPT"]).with_name("sym-reloc-export"))
    if configured is None:
        configured = str(REPO_ROOT / "build" / "sym" / "sym" / "tools" / "sym-reloc-export")
    return pathlib.Path(configured)
