#!/usr/bin/env python3
"""Measure the workload callers' retained/per-call policies in fresh processes.

Run serially, with a fixed CPU affinity and a current-main runtime build:
  taskset -c 4-7,20-23 python bench/typed_weight_resources.py \
      --build /path/to/build --output-dir /tmp/weight-resources --rounds 3
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
WORKLOADS = REPO / "libreloc/python/examples/workloads"
sys.path.insert(0, str(WORKLOADS))
import run_workloads


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(args):
    import torch
    import pyreloc
    import reloc_torch
    import common

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    module_name = {"llm": "llm_offload", "moe": "moe_experts"}[args.workload]
    original_finish = common.Report.finish

    def finish(report, output=None):
        report.data["raw_transfer_samples_ms"] = report._samples
        paths = [Path(__file__), WORKLOADS / "common.py", WORKLOADS / (module_name + ".py"),
                 Path(reloc_torch.__file__).with_name("dispatch.py"),
                 Path(reloc_torch.__file__).with_name("resources.py"),
                 Path(os.environ["SYM_RELOC_EXPORT"]), args.build / "libreloc/libreloc_runtime.so"]
        paths.extend(Path(pyreloc.__file__).parent.glob("_pyreloc*.so"))
        report.data["measurement"] = {
            "torch": torch.__version__, "python": sys.version,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "runtime_package": reloc_torch.__file__, "build": str(args.build),
            "source_and_binary_sha256": {str(path): digest(path) for path in paths},
            "scope": "Completed transfer calls including preparation and fresh outputs. Model compute, oracle checks, and final owner close are outside timers. First calls are preserved; remaining totals mix changing shapes.",
            "order": "Fresh process per policy/workload/round; Torch precedes Sym within the unchanged workload; process order is shuffled.",
            "memory": "Per-device typed retained scratch is reported before/after close. live_bytes=0 means no hard live cap; tensor inputs/outputs and allocator memory are outside this budget. Completed-call gauges are not peak live memory.",
        }
        return original_finish(report, output)

    common.Report.finish = finish
    device_flag = "--devices" if args.workload == "moe" else "--device"
    sys.argv = [module_name, device_flag, args.device, "--weight-resources", args.policy,
                "--output", str(args.output)]
    if args.quick:
        sys.argv.append("--quick")
    return common.run_example(importlib.import_module(module_name).main)


def summarize(records):
    summary = {}
    for workload in ("llm", "moe"):
        for policy in ("per-call", "retained"):
            rows = [entry["report"] for entry in records if entry["workload"] == workload and entry["policy"] == policy]
            values = {"sym_all_ms": [], "sym_after_first_ms": [], "torch_all_ms": [], "torch_after_first_ms": []}
            for report in rows:
                for path in ("sym", "torch"):
                    samples = [paths[path] for paths in report["raw_transfer_samples_ms"].values()]
                    values[f"{path}_all_ms"].append(sum(sum(block) for block in samples))
                    values[f"{path}_after_first_ms"].append(sum(sum(block[1:]) for block in samples))
            summary[f"{workload}/{policy}"] = {
                key: {"rounds": raw, "median": statistics.median(raw), "min": min(raw), "max": max(raw)}
                for key, raw in values.items()}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=("llm", "moe"), help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("retained", "per-call"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.build = args.build.resolve()
    if args.rounds < 1 or args.threads < 1:
        parser.error("rounds and threads must be positive")
    if args.child:
        return child(args)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(r, workload, policy) for r in range(args.rounds)
            for workload in ("llm", "moe") for policy in ("per-call", "retained")]
    random.Random(218).shuffle(jobs)
    records = []
    for r, workload, policy in jobs:
        stem = f"{r}-{workload}-{policy}"
        output = (args.output_dir / (stem + ".json")).resolve()
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--build", str(args.build),
                   "--workload", workload, "--policy", policy, "--output", str(output),
                   "--threads", str(args.threads), "--device", args.device]
        if args.quick:
            command.append("--quick")
        print(stem, flush=True)
        with (args.output_dir / (stem + ".log")).open("w") as log:
            subprocess.run(command, env=run_workloads.child_environment(args.build),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        report = json.loads(output.read_text())
        if not report["ok"]:
            raise RuntimeError(f"correctness failed: {output}")
        records.append({"round": r, "workload": workload, "policy": policy, "report": report})
    # Same workload semantics and dispatch rows in every policy/round.
    for workload in ("llm", "moe"):
        reports = [row["report"] for row in records if row["workload"] == workload]
        for key in ("bytes", "dispatches", "checks", "workload"):
            if any(report[key] != reports[0][key] for report in reports):
                raise RuntimeError(f"{workload}: policy comparison changed {key}")
    document = {"order": [{key: row[key] for key in ("round", "workload", "policy")} for row in records],
                "summary": summarize(records)}
    (args.output_dir / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
