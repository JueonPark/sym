"""Use the installed wheel in a different process outside the checkout."""

import json
import os
import subprocess

import pytest


def test_installed_payload(tmp_path):
    python = os.environ.get("SYM_INSTALLED_PYTHON")
    if not python:
        pytest.skip("set SYM_INSTALLED_PYTHON to the wheel test interpreter")
    code = """import sys, json, subprocess
import pyreloc, reloc_torch, sym_reloc
from sym_reloc.tools import native_tool
assert 'torch' not in sys.modules
for name in ('sym-opt', 'sym-reloc-export', 'reloc-run-artifact'):
    assert native_tool(name).is_file()
assert reloc_torch.CompilerClient.from_environment().executable == native_tool('sym-reloc-export')
print(json.dumps({'version':sym_reloc.__version__, 'extension':pyreloc.__file__}))
"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "LD_LIBRARY_PATH", "SYM_OPT", "SYM_RELOC_EXPORT")
    }
    result = subprocess.run(
        [python, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"].endswith("+cpu")
