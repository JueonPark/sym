"""Example command lines that must fail cleanly before any GPU work (CPU)."""
import os
import pathlib
import subprocess
import sys

WORKLOADS = pathlib.Path(__file__).resolve().parents[1]


def test_an_example_without_the_build_environment_exits_two(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "SYM_OPT", "SYM_RELOC_EXPORT")}
    proc = subprocess.run([sys.executable, str(WORKLOADS / "dlrm_embeddings.py"), "--quick"], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "prerequisite: reloc_torch is not importable" in proc.stderr
