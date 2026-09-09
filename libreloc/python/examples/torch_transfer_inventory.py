"""Observe pinned Torch API scenarios; no transfer execution is replaced."""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sysconfig
import sys
import torch
from reloc_torch import check_version, classify, observe_transfers


def run_scenario(name, phase, operation):
    observer = observe_transfers()
    scenario = dict(name=name, phase=phase, status="observed")
    try:
        with observer, observer.phase(phase):
            operation()
    except Exception as error:
        scenario.update(status="failed", failure_type=type(error).__name__, failure_message=str(error))
    records = [{"scenario": name, **asdict(r), "classification": asdict(classify(r)), "execution_status": "excluded"} for r in observer.records]
    scenario["event_count"] = len(records)
    scenario["transfer_count"] = sum(r["source"]["device_type"] != r["destination"]["device_type"] for r in records)
    scenario["no_transfer_reason"] = None if scenario["transfer_count"] else "no_cross_device_transfer_observed"
    return scenario, records


def command_version(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inventory(device):
    check_version()
    versions = dict(python=platform.python_version(), implementation=platform.python_implementation(), full_python=sys.version, gil_enabled=sys._is_gil_enabled(), compiler=platform.python_compiler(), soabi=sysconfig.get_config_var("SOABI"), torch=str(torch.__version__), cuda=torch.version.cuda,
                    dependencies={n: installed_version(n) for n in ("torch", "numpy", "pytest", "pybind11")})
    try:
        import pyreloc._pyreloc as extension
        versions["extension"] = str(Path(extension.__file__).resolve())
    except ImportError:
        versions["extension"] = None
    data = dict(versions=versions, device=device, scenarios=[], records=[], gpu=None)
    if device == "cuda":
        if not torch.cuda.is_available():
            data["failure"] = dict(reason="cuda_unavailable")
            return data
        data["gpu"] = dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), arch_list=torch.cuda.get_arch_list(), driver=command_version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]), nvcc=command_version(["nvcc", "--version"]))
        try:
            probe = torch.arange(16, dtype=torch.float32).cuda()
            assert torch.equal((probe + 1).cpu(), torch.arange(16, dtype=torch.float32) + 1)
            torch.cuda.synchronize()
            data["gpu"]["smoke"] = "passed"
        except Exception as error:
            data["failure"] = dict(reason="cuda_smoke_failed", message=str(error))
            return data
    x = torch.arange(16, dtype=torch.float32)
    resident = x.to(device)
    module = torch.nn.Linear(4, 4).requires_grad_(False)
    module.register_buffer("inventory_buffer", torch.ones(4))
    buffers = torch.nn.Module()
    buffers.register_buffer("value", x.clone())
    operations = [
        ("to", "input", lambda: x.to(device)), ("cpu", "output", lambda: resident.cpu()),
        ("copy_", "input", lambda: torch.empty_like(resident).copy_(x)),
        ("copy_back", "output", lambda: torch.empty_like(x).copy_(resident)),
        ("same_device_noop", "input", lambda: x.to("cpu")),
        ("same_device_copy", "input", lambda: x.to("cpu", copy=True)),
        ("cast", "input", lambda: x.to(torch.float16)),
        ("layout", "input", lambda: x.view(4, 4).transpose(0, 1).contiguous().to(device)),
        ("mutation", "input", lambda: x.clone().add_(1).to(device)),
        ("module_parameter", "module_to", lambda: module.to(device)),
        ("module_buffer", "module_to", lambda: buffers.to(device)),
        ("load_state_dict", "load_state_dict", lambda: module.load_state_dict(dict(weight=torch.ones(4, 4), bias=torch.ones(4), inventory_buffer=torch.ones(4))))]
    if device == "cuda":
        operations.extend([( "cuda", "input", lambda: x.cuda()),
                           ("pinned", "input", lambda: x.pin_memory().to("cuda")),
                           ("nonblocking", "input", lambda: x.to("cuda", non_blocking=True))])
    else:
        data["scenarios"].append(dict(name="cuda", phase="input", status="not_run", event_count=0, transfer_count=0, no_transfer_reason="cuda_not_requested"))
    for name, phase, operation in operations:
        scenario, records = run_scenario(name, phase, operation)
        data["scenarios"].append(scenario)
        data["records"].extend(records)
    data["classification_totals"] = dict(Counter(r["classification"]["category"] for r in data["records"]))
    data["exclusions"] = dict(Counter(r["classification"]["reason"] for r in data["records"]))
    data["candidate_count"] = sum(r["classification"]["candidate"] for r in data["records"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = inventory(args.device)
    except Exception as error:
        data = dict(failure=dict(reason=getattr(error, "reason", type(error).__name__), message=str(error)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    return int("failure" in data or any(s["status"] == "failed" for s in data.get("scenarios", [])))


if __name__ == "__main__":
    raise SystemExit(main())
