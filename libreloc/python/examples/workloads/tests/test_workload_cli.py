"""Example command lines that must fail cleanly before any GPU work (CPU)."""
import os
import pathlib
import subprocess
import sys

import pytest

WORKLOADS = pathlib.Path(__file__).resolve().parents[1]


def test_an_example_without_the_build_environment_exits_two(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "SYM_OPT", "SYM_RELOC_EXPORT")}
    proc = subprocess.run([sys.executable, str(WORKLOADS / "dlrm_embeddings.py"), "--quick"], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "prerequisite: reloc_torch is not importable" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("script", ["dlrm_embeddings.py", "gnn_minibatch.py", "llm_offload.py", "moe_experts.py"])
def test_an_example_under_an_interpreter_without_torch_exits_two(script):
    # -S skips site-packages, so torch is unimportable while the stdlib still loads.
    proc = subprocess.run([sys.executable, "-S", str(WORKLOADS / script), "--quick"],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "prerequisite: torch is not importable" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("devices, message", [
    ("cuda:0,cuda:0", "repeats a device"),
    ("cuda:0,cuda:1,cuda:2,cuda:3,cuda:4", "has only 4 experts"),     # the quick model has 4 experts
    ("gpu7", "invalid --devices"),
])
def test_moe_rejects_invalid_device_lists(devices, message):
    proc = subprocess.run([sys.executable, str(WORKLOADS / "moe_experts.py"), "--quick", "--devices", devices],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert message in proc.stderr
