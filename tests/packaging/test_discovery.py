"""Exercise actual override paths and installed-tool wrappers."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "libreloc/python"))
from reloc_torch.compiler import CompilerClient  # noqa: E402


def tool_module():
    try:
        return importlib.import_module("sym_reloc.tools")
    except ModuleNotFoundError:
        pytest.fail("packaged tool discovery is not implemented")


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_packaged_default_and_override_precedence(tmp_path, monkeypatch):
    tools = tool_module()
    bundled = executable(tmp_path / "bundle/sym-reloc-export")
    monkeypatch.setattr(tools, "native_tool", lambda name: bundled)
    assert CompilerClient.from_environment({}).executable == bundled
    override = executable(tmp_path / "path with spaces/exporter")
    assert (
        CompilerClient.from_environment(
            {"SYM_RELOC_EXPORT": str(override), "SYM_OPT": "/missing/sym-opt"}
        ).executable
        == override
    )
    with pytest.raises(RuntimeError, match="absent|executable"):
        CompilerClient.from_environment({"SYM_RELOC_EXPORT": "/missing/exporter"})
    override.chmod(0o644)
    with pytest.raises(RuntimeError, match="executable"):
        CompilerClient.from_environment({"SYM_RELOC_EXPORT": str(override)})


def test_sym_opt_sibling(tmp_path):
    path = executable(tmp_path / "sym-reloc-export")
    assert (
        CompilerClient.from_environment(
            {"SYM_OPT": str(tmp_path / "sym-opt")}
        ).executable
        == path
    )


def test_native_tool_allowlist():
    with pytest.raises(ValueError):
        tool_module().native_tool("../escape")


def test_wrapper_preserves_arguments_and_exit_code(tmp_path):
    tool_module()
    binary = tmp_path / "fake exporter"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >&2\nexit 2\n')
    binary.chmod(0o755)
    code = (
        "from pathlib import Path; from sym_reloc import tools; "
        f"tools.native_tool=lambda name: Path({str(binary)!r}); tools.export_main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "arg with spaces", "--typed"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "libreloc/python")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr == "arg with spaces\n--typed\n"
