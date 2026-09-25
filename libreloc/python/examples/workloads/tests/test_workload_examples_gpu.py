"""End to end: every workload example passes its own checks through the runner (GPU)."""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import run_workloads

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]

# The checks each example must report, all passing. Pinning the names keeps an
# example from passing by checking less.
EXPECTED_CHECKS = {
    "dlrm": {"pooled_embeddings_equal", "predictions_equal", "one_plan_for_every_batch_size",
             "one_execution_per_batch", "cpu_reference_row", "observed_payload_matches_wire", "no_fallbacks"},
    "gnn": {"node_features_equal", "logits_equal", "one_plan_for_every_row_count", "rows_change_between_batches",
            "one_execution_per_batch", "cpu_reference_row", "observed_payload_matches_wire", "no_fallbacks"},
}


def build_dir():
    if os.environ.get("SYM_BUILD"):
        return pathlib.Path(os.environ["SYM_BUILD"])
    if os.environ.get("SYM_RELOC_EXPORT"):
        return pathlib.Path(os.environ["SYM_RELOC_EXPORT"]).parents[2]
    pytest.skip("set SYM_BUILD (or SYM_RELOC_EXPORT) to a CUDA-enabled build")


def run_runner(tmp_path, *args):
    proc = subprocess.run([sys.executable, run_workloads.__file__, "--build", str(build_dir()), "--python",
                           sys.executable, "--output-dir", str(tmp_path), *args],
                          capture_output=True, text=True, timeout=3600)
    logs = "\n".join(path.read_text() for path in sorted(tmp_path.glob("*.log")))
    return proc, proc.stdout + proc.stderr + logs


def load(tmp_path, name):
    return json.loads((tmp_path / f"{name}.json").read_text())


@pytest.mark.parametrize("name", sorted(EXPECTED_CHECKS))
def test_each_example_passes_its_checks_at_quick_size(tmp_path, name):
    proc, output = run_runner(tmp_path, "--quick", "--only", name)
    assert proc.returncode == 0, output
    report = load(tmp_path, name)
    assert report["ok"] is True
    assert EXPECTED_CHECKS[name] <= set(report["checks"]), sorted(report["checks"])
    assert report["summary"]["checks_passed"] == report["summary"]["checks_total"]
