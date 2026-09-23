"""Guarded symbolic plan reuse: one compiled artifact, several shape bindings.

Compiles a split+transpose relocation once with ``RelocBackend`` and runs it at
every requested size, comparing each result exactly against plain PyTorch.
``--direction h2d`` starts from a CPU tensor and relocates onto the device;
``--direction d2h`` starts from a CUDA tensor and compiles the requested forward
computation from that root, independently of H2D. The report separates plan
compilation, symbol binding, Dynamo callbacks, runtime executions and
reason-coded fallbacks/exclusions.
"""
import argparse
import json
import sys

import torch

from reloc_torch import RelocBackend, check_version


COUNTERS = ("plan_compiles", "symbol_binds", "dynamo_compiles", "runtime_executions", "cache_hits", "replaced_regions")


def metadata(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
    )


def run(direction, sizes, device):
    check_version()
    device = torch.device(device)
    if device.type != "cuda":
        raise SystemExit("this example needs a CUDA device (--device cuda:0)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    backend = RelocBackend()

    if direction == "h2d":
        def fn(x):
            return x.reshape(x.shape[0] // 64, 64).t().contiguous().to(device)
    else:
        def fn(x):
            return x.reshape(x.shape[0] // 64, 64).t().contiguous().cpu()

    compiled = torch.compile(fn, backend=backend, dynamic=True)
    report = dict(direction=direction, device=str(device), sizes=[], torch=str(torch.__version__))
    ok = True
    with torch.no_grad():
        for n in sizes:
            x = torch.arange(n, dtype=torch.float32, device=device if direction == "d2h" else "cpu")
            actual = compiled(x)
            expected = fn(x)
            exact = (
                torch.equal(actual.cpu(), expected.cpu())
                and actual.stride() == expected.stride()
                and actual.storage_offset() == expected.storage_offset()
                and actual.device == expected.device
            )
            ok = ok and exact
            report["sizes"].append(dict(n=n, input=metadata(x), output=metadata(actual), exact=exact))
    stats = backend.stats()
    report["counters"] = {name: stats[name] for name in COUNTERS}
    report["fallbacks"] = stats["fallbacks"]
    report["exclusions"] = stats["exclusions"]
    # Evidence gate: one artifact, one bind and one execution per size, no fallback.
    report["reused_one_plan"] = (
        stats["plan_compiles"] == 1
        and stats["symbol_binds"] == len(sizes)
        and stats["runtime_executions"] == len(sizes)
        and not stats["fallbacks"]
    )
    backend.close()
    return report, ok and report["reused_one_plan"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("h2d", "d2h"), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 192, 256])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report, ok = run(args.direction, args.sizes, args.device)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
