#!/usr/bin/env python3
"""Fused host torch.stack -> H2D: completed-call latency against PyTorch.

Every sample allocates a fresh output and includes completion; compilation,
input creation and correctness checks are outside the timer. Sym's methods
run on warm, retained transfer resources.

  B1   eager as written            torch.stack(xs, dim).to(dev)
  B2   eager copy-then-stack       torch.stack([x.to(dev) for x in xs], dim)
  B3   Inductor as written         torch.compile(B1)
  B4   Inductor copy-then-stack    torch.compile(B2)
  B5   pinned host stack           torch.stack(xs, dim, out=pinned).to(dev)
  S1   Sym single-source proxy     today's transfer over a pre-stacked [N, R, C]
  FD   Sym fused, direct adapter   prepare_stacked_transfer / execute_transfer
  FB   Sym fused, RelocBackend,    torch.compile(B1, backend=RelocBackend(min_stack_bytes=0,
       tuned options                 transfer_options={"pinning": "auto",
                                     "min_pinned_bytes": 8 << 20, "gather_threads": 8}))
  FBD  Sym fused, RelocBackend,    torch.compile(B1, backend=RelocBackend(min_stack_bytes=0)):
       default options               default transfer options (gather_threads 1, pageable staging)
  GB   below-threshold overhead    torch.compile(B1, backend=RelocBackend(min_stack_bytes=<one
                                     byte above the largest measured size>)): every call falls
                                     back to PyTorch; default transfer options

S1 and FD pass FB's tuned options to execute_transfer; min_pinned_bytes is
--min-pinned-bytes and gather_threads is --threads (8 MiB and 8 by default).
FB and FBD must execute every timed call through Sym and record no fallback;
GB must record below_stack_threshold for every timed call. Any other path
fails the run.
"""
import argparse
import hashlib
import inspect
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
from reloc_torch.backend import DEFAULT_MIN_STACK_BYTES, RelocBackend
from reloc_torch.recipe import Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides
from reloc_torch.transport import execute_transfer, prepare_stacked_transfer, prepare_transfer

MIB = 1 << 20
COLUMNS = 1024
PERMS = {0: (0, 1, 2), 1: (1, 0, 2), 2: (1, 2, 0)}  # result axis k <- logical axis perm[k]
INDUCTOR = ("B3", "B4")
# The execute_transfer keywords that transfer_options may set (gather_pool and
# resources are objects, not options): RelocBackend without transfer_options
# runs their defaults.
TRANSFER_KNOBS = ("n_buffers", "n_streams", "gather_threads", "pinning", "min_pinned_bytes")


class WrongResult(AssertionError):
    pass


def spec(shape):
    return TensorSpec(shape, dense_strides(shape), Const(0), "float32")


def recipe(logical, dim, stack_inputs=0):
    perm = PERMS[dim]
    operations = (Transpose(perm),) if dim else ()
    return Recipe(spec(logical), operations, spec(tuple(logical[p] for p in perm)), "h2d", stack_inputs)


def metadata(args, tuned, gb_min_stack_bytes):
    import pyreloc

    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(pyreloc.__file__)]
    files += list(Path(pyreloc.__file__).resolve().parent.glob("*.so"))
    files += [Path(os.environ[key]) for key in ("SYM_RELOC_EXPORT", "SYM_OPT") if key in os.environ]
    run = lambda *cmd: subprocess.check_output(cmd, cwd=root, text=True).strip()
    defaults = inspect.signature(execute_transfer).parameters
    return dict(
        source_revision=run("git", "rev-parse", "HEAD"),
        source_dirty=bool(run("git", "status", "--porcelain")),
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        inductor_compile_threads=os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        cpu=subprocess.check_output(["lscpu"], text=True),
        gpu=subprocess.check_output(
            ["nvidia-smi", "-i", str(args.device), "--query-gpu=name,driver_version,persistence_mode,"
             "pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"], text=True).strip(),
        clocks="GPU clocks not locked; CPU governor not changed", args=vars(args),
        default_min_stack_bytes=DEFAULT_MIN_STACK_BYTES,
        tuned_transfer_options=tuned,  # S1, FD and FB
        default_transfer_options={name: defaults[name].default for name in TRANSFER_KNOBS},  # FBD and GB
        gb_min_stack_bytes=gb_min_stack_bytes)


def build_methods(xs, stacked, dim, device, proxy, fused, owner, pinned, backends, tuned):
    sym = dict(tuned, resources=owner)
    torch._dynamo.reset()

    # One function per compiled method: distinct code objects keep Dynamo's caches apart.
    def b3_fn(*ts):
        return torch.stack(ts, dim).to(device)

    def b4_fn(*ts):
        return torch.stack([t.to(device) for t in ts], dim)

    def fb_fn(*ts):
        return torch.stack(ts, dim).to(device)

    def fbd_fn(*ts):
        return torch.stack(ts, dim).to(device)

    def gb_fn(*ts):
        return torch.stack(ts, dim).to(device)

    def b5():
        torch.stack(xs, dim, out=pinned)
        return pinned.to(device)

    b3 = torch.compile(b3_fn, dynamic=False)
    b4 = torch.compile(b4_fn, dynamic=False)
    fb = torch.compile(fb_fn, backend=backends["FB"], dynamic=True)
    fbd = torch.compile(fbd_fn, backend=backends["FBD"], dynamic=True)
    gb = torch.compile(gb_fn, backend=backends["GB"], dynamic=True)
    return {
        "B1": lambda: torch.stack(xs, dim).to(device),
        "B2": lambda: torch.stack([x.to(device) for x in xs], dim),
        "B3": lambda: b3(*xs),
        "B4": lambda: b4(*xs),
        "B5": b5,
        "S1": lambda: execute_transfer(prepare_transfer(proxy, stacked, device), **sym),
        "FD": lambda: execute_transfer(prepare_stacked_transfer(fused, xs, device), **sym),
        "FB": lambda: fb(*xs),
        "FBD": lambda: fbd(*xs),
        "GB": lambda: gb(*xs),
    }


def time_method(name, fn, expected, args):
    timed = []
    for index in range(args.warmup + args.samples):
        torch.cuda.synchronize(args.device)
        start = time.perf_counter_ns()
        out = fn()
        torch.cuda.synchronize(args.device)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not torch.equal(out.cpu(), expected):
            raise WrongResult(f"{name} produced a wrong result")
        if index >= args.warmup:
            timed.append(elapsed)
        del out
    return timed


def path_count(name, stats):
    """The backend counter one call of `name` advances: FB and FBD execute
    through Sym, GB falls back below its threshold."""
    if name == "GB":
        return stats["fallbacks"].get("below_stack_threshold", 0)
    return stats["stacked_executions"]


def check_path(name, backend, before, calls):
    """Fail the run unless each of `calls` calls of `name` took its path."""
    stats = backend.stats()
    if path_count(name, stats) - before != calls:
        if name == "GB":
            raise RuntimeError("GB fused or was excluded instead of falling back below its threshold")
        raise RuntimeError(f"{name} fell back instead of executing through Sym")
    if name != "GB" and stats["fallbacks"]:
        raise RuntimeError(f"{name} recorded fallbacks {stats['fallbacks']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 4, 8, 16, 24, 32, 64, 128])
    parser.add_argument("--counts", type=int, nargs="+", default=[2, 4, 16])
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-pinned-bytes", type=int, default=8 * MIB)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch._dynamo.config.recompile_limit = 1024
    device = f"cuda:{args.device}"
    torch.empty(1, device=device).cpu()
    client = CompilerClient.from_environment()
    proxies = {dim: client.compile(recipe((Symbol("s0"), Symbol("s1"), Symbol("s2")), dim)) for dim in args.dims}
    tuned = {"pinning": "auto", "min_pinned_bytes": args.min_pinned_bytes, "gather_threads": args.threads}
    gb_min_stack_bytes = max(args.sizes_mib) * MIB + 1  # every measured stack is below it
    backends = {
        "FB": RelocBackend(transfer_options=tuned, min_stack_bytes=0),
        "FBD": RelocBackend(min_stack_bytes=0),
        "GB": RelocBackend(min_stack_bytes=gb_min_stack_bytes),
    }
    result = dict(scope=__doc__, metadata=metadata(args, tuned, gb_min_stack_bytes), rows=[], correct=False,
                  complete=False)
    output = Path(args.output)
    rng = random.Random(2026)
    try:
        for total_mib in args.sizes_mib:
            for count in args.counts:
                rows = total_mib * MIB // (count * COLUMNS * 4)
                if rows * count * COLUMNS * 4 != total_mib * MIB:
                    raise ValueError(f"{total_mib} MiB does not split into {count} x [R, {COLUMNS}] fp32")
                generator = torch.Generator().manual_seed(total_mib * 1000 + count)
                xs = [torch.randn(rows, COLUMNS, generator=generator) for _ in range(count)]
                stacked = torch.stack(xs, 0).contiguous()
                for dim in args.dims:
                    fused = client.compile(recipe((Const(count), Symbol("s0"), Symbol("s1")), dim, count))
                    expected = torch.stack(xs, dim)
                    pinned = torch.empty(expected.shape, dtype=expected.dtype, pin_memory=True)
                    owner = TransferResources()
                    try:
                        methods = build_methods(xs, stacked, dim, device, proxies[dim], fused, owner,
                                                pinned, backends, tuned)
                        for name, fn in list(methods.items()):  # compile and allocate before any timing
                            try:
                                out = fn()
                                torch.cuda.synchronize(args.device)
                                if not torch.equal(out.cpu(), expected):
                                    raise WrongResult(f"{name} produced a wrong result")
                                del out
                            except WrongResult:
                                raise
                            except Exception as error:
                                if name not in INDUCTOR:
                                    raise
                                result["rows"].append(dict(total_mib=total_mib, count=count, dim=dim, method=name,
                                                           round=-1, error=f"{type(error).__name__}: {error}"[:500]))
                                del methods[name]
                        torch.cuda.synchronize(args.device)
                        for round_id in range(args.rounds):
                            names = list(methods)
                            rng.shuffle(names)
                            for order, name in enumerate(names):
                                backend = backends.get(name)
                                before = None if backend is None else path_count(name, backend.stats())
                                timed = time_method(name, methods[name], expected, args)
                                if backend is not None:
                                    check_path(name, backend, before, args.warmup + len(timed))
                                result["rows"].append(dict(total_mib=total_mib, count=count, dim=dim, method=name,
                                                           round=round_id, order=order, samples_ms=timed,
                                                           p50_ms=statistics.median(timed)))
                    finally:
                        owner.close()
                    del pinned
                    output.write_text(json.dumps(result, indent=1, default=str) + "\n")
                    print(f"done {total_mib} MiB N={count} dim={dim}", flush=True)
        result["correct"] = True
        result["complete"] = True
    finally:
        result["backend_stats"] = {name: backend.stats() for name, backend in backends.items()}
        for backend in backends.values():
            backend.close()
        output.write_text(json.dumps(result, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
