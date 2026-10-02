#!/usr/bin/env python3
"""Trace configured-auto pinned H2D and pageable/single-buffer controls.

Build libreloc with RELOC_ENABLE_NVTX=ON and use Nsight Systems CUDA+NVTX capture.
bench/analyze_resource_overlap.py correlates native chunk submissions to actual
GPU memcpy activity. Profile timings are NOT headline benchmark samples.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from reloc_torch import TransferResources
from reloc_torch.transport import prepare_transfer, execute_transfer
from pinning_sweep import compile_transpose, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--buffers", type=int, choices=[1, 4], default=4)
    p.add_argument("--policy", choices=["configured", "unconfigured", "pageable"], default="configured")
    p.add_argument("--sqlite", type=Path, help="analyze a completed capture instead of running CUDA")
    args = p.parse_args()
    if args.sqlite:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from analyze_resource_overlap import analyze, chrome_trace
        result = analyze(args.sqlite)
        for request in result["requests"]:
            assert len(request["chunks"]) == 4
            assert all(chunk["bytes"] == 4 << 20 for chunk in request["chunks"])
        result["policy"] = args.policy
        result["qualification_scope"] = ("pinned overlap/control" if args.policy == "configured"
                                         else "pageable characterization; overlap not required")
        if args.policy != "configured":
            result["status"] = "characterized"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        args.output.with_suffix(".chrome.json").write_text(json.dumps(chrome_trace(result)) + "\n")
        if result["status"] == "failed":
            raise SystemExit("pinned overlap or single-buffer control failed")
        return
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    compiled = compile_transpose("h2d")
    source = torch.arange(4096 * 1024, dtype=torch.float32).reshape(4096, 1024)
    expected = source.t().contiguous()
    # 16 MiB gives four 4 MiB chunks for BOTH buffer controls. Analyzer checks sizes.
    options = dict(n_buffers=args.buffers, gather_threads=8,
        pinning="pageable" if args.policy == "pageable" else "auto",
        min_pinned_bytes=8 << 20 if args.policy == "configured" else None)
    reports = []
    with TransferResources() as owner:
        for index in range(8):
            if index == 5:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
            if index >= 5:
                torch.cuda.nvtx.range_push(f"reloc.benchmark/buffers={args.buffers}/request={index-5}")
            request = prepare_transfer(compiled, source, "cuda:0")
            out = execute_transfer(request, resources=owner, **options)
            if index >= 5:
                torch.cuda.nvtx.range_pop()
            assert torch.equal(out.cpu(), expected)
            reports.append(request.staging)
        torch.cuda.profiler.stop()
        stats = owner.stats()
    assert all(row[0]["memory_kind"] == ("pinned" if args.policy == "configured" else "pageable")
               for row in reports)
    args.output.write_text(json.dumps(dict(options=options, shape=list(source.shape), reports=reports,
        stats=stats, metadata=metadata(), correctness=True, scope=__doc__), indent=2) + "\n")


if __name__ == "__main__":
    main()
