import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("command", ["sym-doctor", "sym-demo"])
def test_installed_command(command, tmp_path):
    python = os.environ.get("SYM_INSTALLED_PYTHON")
    if not python:
        pytest.skip("set SYM_INSTALLED_PYTHON to a wheel environment")
    executable = Path(python).parent / command
    assert executable.is_file(), f"{command} not installed"
    args = [str(executable), "--json"]
    if command == "sym-demo":
        args += ["--device", "cpu"]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "LD_LIBRARY_PATH", "SYM_RELOC_EXPORT", "SYM_OPT")
    }
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert report["variant"] == "cpu"
    if command == "sym-demo":
        assert {"layout", "typed", "weights"} <= report["checks"].keys()
        assert all(row["status"] == "ok" for row in report["checks"].values())


def test_cpu_install_rejects_cuda(tmp_path):
    python = os.environ.get("SYM_INSTALLED_PYTHON")
    if not python:
        pytest.skip("set SYM_INSTALLED_PYTHON")
    result = subprocess.run(
        [str(Path(python).parent / "sym-doctor"), "--require-cuda", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "failed"


@pytest.mark.parametrize(
    "report",
    [
        {"implementation": "cpu_reference", "wire_bytes": 24},
        {"implementation": "cpu_stages_cuda_stages@0", "wire_bytes": 0},
    ],
)
def test_cuda_witness_rejects_cpu_or_empty_execution(report):
    from sym_reloc.examples import typed

    assert hasattr(typed, "require_gpu_report"), "typed GPU evidence validator missing"
    with pytest.raises(RuntimeError, match="GPU"):
        typed.require_gpu_report(report)
