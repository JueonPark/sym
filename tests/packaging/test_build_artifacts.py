import importlib
import subprocess
import sys

import pytest


def test_helper_zipapp_runs_without_checkout(tmp_path):
    try:
        builder = importlib.import_module("build_tools.release.build_artifacts")
    except ModuleNotFoundError:
        pytest.fail("release artifact builder is not implemented")
    helper = tmp_path / "installer.pyz"
    builder.build_helper(helper)
    result = subprocess.run(
        [sys.executable, "-I", str(helper), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
