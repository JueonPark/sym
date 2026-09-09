import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/torch_transfer_inventory.py"


def test_cpu_cli(tmp_path):
    output = tmp_path / "inventory.json"
    result = subprocess.run([sys.executable, str(EXAMPLE), "--device", "cpu", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["versions"]["python"] == "3.14.7"
    assert {"to", "cpu", "cuda", "copy_", "module_parameter", "module_buffer", "load_state_dict", "same_device_noop", "cast", "layout", "mutation"} <= {s["name"] for s in data["scenarios"]}
    assert all(s["event_count"] or s["no_transfer_reason"] for s in data["scenarios"])
    assert all(r["execution_status"] == "excluded" for r in data["records"])
    assert data["classification_totals"]


def test_requested_unavailable_cuda(tmp_path):
    if torch.cuda.is_available():
        return
    output = tmp_path / "inventory.json"
    result = subprocess.run([sys.executable, str(EXAMPLE), "--device", "cuda", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(output.read_text())["failure"]["reason"] == "cuda_unavailable"


def test_scenario_failure_is_accounted():
    spec = importlib.util.spec_from_file_location("inventory_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fail():
        raise ValueError("same failure message")
    scenario, records = module.run_scenario("failing", "input", fail)
    assert scenario["status"] == "failed"
    assert scenario["failure_message"] == "same failure message"
