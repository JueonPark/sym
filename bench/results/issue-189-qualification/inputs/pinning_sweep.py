#!/usr/bin/env python3
"""Completed transpose transfers, preserving first use and repeated allocations.

Compiler/CUDA initialization and input creation precede timing. Each sample
includes prepare, fresh output allocation, CPU transpose, copy and completion.
First means first use of a NEW Sym owner, not a cold process/CUDA/Torch allocator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
from reloc_torch import CompilerClient, TransferResources
from reloc_torch.recipe import Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Symbol, Const, dense_strides
from reloc_torch.transport import prepare_transfer, execute_transfer


VARIANTS = {"auto_default": ("auto", None), "auto_configured": ("auto", "configured"),
            "pinned": ("pinned", None), "pageable": ("pageable", None)}


def metadata():
    import pyreloc
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(pyreloc.__file__)]
    files += list(Path(pyreloc.__file__).resolve().parent.glob("*.so"))
    files += list(Path(pyreloc.__file__).resolve().parents[2].glob("libreloc/libreloc_runtime.so*"))
    files += [Path(os.environ[key]) for key in ("SYM_RELOC_EXPORT", "SYM_OPT") if key in os.environ]
    return dict(source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cpu=subprocess.check_output(["lscpu"], text=True),
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem", "--format=csv"], text=True),
        clocks="GPU clocks not locked; CPU governor not changed")


def compile_transpose(direction):
    n, m = Symbol("s0"), Symbol("s1")
    def spec(shape):
        return TensorSpec(shape, dense_strides(shape), Const(0), "float32")
    return CompilerClient.from_environment().compile(
        Recipe(spec((n, m)), (Transpose((1, 0)),), spec((m, n)), direction))


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(189)
    torch.cuda.set_device(args.device)
    compiled = compile_transpose(args.direction)
    torch.empty(1, device=args.device).cpu()  # initialize CUDA before all variants
    rows = []
    rng = random.Random(189)
    device = f"cuda:{args.device}"
    destination = device if args.direction == "h2d" else "cpu"
    result = dict(schema_version=2, direction=args.direction, min_pinned_bytes=args.min_pinned_bytes,
        samples=args.samples, warmup=args.warmup, rounds=args.rounds, rows=rows,
        metadata=metadata(), scope=__doc__, correctness="torch.equal on every completed output")
    for wire in args.sizes:
        if wire % (512 * 4):
            raise ValueError("sizes must be multiples of 2048 bytes")
        cpu = torch.randn(wire // (512 * 4), 512)
        expected = cpu.t().contiguous()
        source = cpu if args.direction == "h2d" else cpu.to(device)
        for round_id in range(args.rounds):
            variants = [(name, reuse) for name in VARIANTS for reuse in (False, True)] + [("torch", False)]
            rng.shuffle(variants)
            for order, (name, reuse) in enumerate(variants):
                owner = TransferResources() if reuse else None
                if name != "torch":
                    mode, threshold = VARIANTS[name]
                    if threshold == "configured":
                        threshold = args.min_pinned_bytes
                samples = []
                try:
                    for index in range(1 + args.warmup + args.samples):
                        phase = "first" if index == 0 else "warmup" if index <= args.warmup else "repeated"
                        torch.cuda.synchronize(args.device)
                        start = time.perf_counter_ns()
                        if name == "torch":
                            # Same CPU transpose and fresh contiguous output, not a GPU transpose.
                            out = (source.t().contiguous().to(device) if args.direction == "h2d"
                                   else source.to("cpu").t().contiguous())
                        else:
                            request = prepare_transfer(compiled, source, destination)
                            out = execute_transfer(request, resources=owner, pinning=mode,
                                min_pinned_bytes=threshold, gather_threads=args.threads)
                        torch.cuda.synchronize(args.device)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        assert torch.equal(out.cpu(), expected), (wire, name, reuse, round_id, index)
                        samples.append(dict(index=index, phase=phase, ms=elapsed,
                            staging=[] if name == "torch" else request.staging))
                        del out  # retirement outside timer for both paths
                    repeated = [row["ms"] for row in samples if row["phase"] == "repeated"]
                    rows.append(dict(bytes=wire, variant=name, reuse=reuse, round=round_id, order=order,
                        pinning=None if name == "torch" else mode,
                        min_pinned_bytes=None if name == "torch" else threshold,
                        calls=samples, samples_ms=repeated, median_ms=statistics.median(repeated),
                        stats=None if owner is None else owner.stats()))
                finally:
                    if owner is not None:
                        owner.close()
                Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(args.direction, wire, round_id, "checked all 9 variants", flush=True)
    result["correct"] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direction", choices=["h2d", "d2h"], default="h2d")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-pinned-bytes", type=int, default=8 << 20,
                        help="explicit experiment configuration, never a universal runtime default")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64 << 10, 256 << 10] +
                        [n << 20 for n in (1, 4, 7, 8, 9, 16, 32, 64)])
    args = parser.parse_args()
    if min(args.samples, args.rounds, args.threads, *args.sizes) <= 0 or args.warmup < 0:
        parser.error("sizes/samples/rounds/threads must be positive; warmup nonnegative")
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
