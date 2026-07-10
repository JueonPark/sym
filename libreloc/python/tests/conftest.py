"""Shared pytest infrastructure for the pyreloc test-suite (issue #46).

Import strategy: pyreloc lives in <build>/python (wheel-less, PYTHONPATH
install path). If the caller did not set PYTHONPATH, fall back to the
default build dir so `pytest libreloc/python/tests` just works locally.
"""
import json
import os
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_BUILD_PY = REPO_ROOT / "build" / "sym" / "python"

try:
    import pyreloc  # noqa: F401  (PYTHONPATH already set)
except ImportError:
    if _DEFAULT_BUILD_PY.is_dir():
        sys.path.insert(0, str(_DEFAULT_BUILD_PY))

# Seed for every randomized test in this run (acceptance: seeded, printed).
SEED = int(os.environ.get("RELOC_SEED", "20260710"))

CORPUS_DIR = REPO_ROOT / "libreloc" / "test" / "corpus"
_SERIALIZE_MLIR = REPO_ROOT / "test" / "dialect" / "reloc" / "serialize.mlir"
_GOLDEN_RE = re.compile(r"plan_hex\((\w+)\): ([0-9a-f]+)")


def golden_hex(name):
    """Wire hex of a frozen golden blob from serialize.mlir (single source
    of truth: the lit test pins these bytes)."""
    for m in _GOLDEN_RE.finditer(_SERIALIZE_MLIR.read_text()):
        if m.group(1) == name:
            return m.group(2)
    raise KeyError(f"no golden plan named {name!r} in {_SERIALIZE_MLIR}")


def corpus_entries():
    """All committed corpus entries as (name, plan_bytes, meta_dict), sorted."""
    entries = []
    for meta_path in sorted(CORPUS_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        blob = meta_path.with_suffix(".bin").read_bytes()
        entries.append((meta_path.stem, blob, meta))
    return entries


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA-enabled pyreloc build + GPU + torch "
        "(skipped in CI, run locally)")


def pytest_report_header(config):
    return f"pyreloc harness seed: RELOC_SEED={SEED}"
