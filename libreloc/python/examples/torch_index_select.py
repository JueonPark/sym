"""Completed CPU row-gather + FP16 H2D timing, with selection inside capture.

Uses the 128-column feature shape of the GNN example in PR #162. This measures
only feature selection/transfer, not graph sampling or end-to-end inference.
Compilation and warmup are excluded; eager and Sym use the same CPU threads.
"""
import argparse
import json
import statistics
import time

import torch

from reloc_torch import RelocBackend


def run(args):
    torch.manual_seed(162)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    source = torch.randn(args.rows, 128)
    backend = RelocBackend(transfer_options={'gather_threads': args.threads})

    def eager(features, nodes):
        return torch.index_select(features, 0, nodes).to(device=device, dtype=torch.float16)

    fused = torch.compile(eager, backend=backend, dynamic=True, fullgraph=True)
    results = []
    try:
        with torch.no_grad():
            for count in args.selected:
                index = torch.randint(args.rows, (count,))
                assert torch.equal(fused(source, index), eager(source, index))
                for _ in range(args.warmup):
                    eager(source, index)
                    fused(source, index)
                samples = {'torch': [], 'sym': []}
                for iteration in range(args.iterations):
                    paths = [('torch', eager), ('sym', fused)]
                    if iteration % 2:
                        paths.reverse()
                    for name, fn in paths:
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        output = fn(source, index)
                        torch.cuda.synchronize(device)
                        samples[name].append((time.perf_counter() - start) * 1000)
                        del output
                medians = {name: statistics.median(values) for name, values in samples.items()}
                results.append(dict(selected=count, median_ms=medians,
                                    speedup=medians['torch'] / medians['sym']))
        stats = backend.stats()
        assert stats['runtime_executions'] == len(args.selected) * (1 + args.warmup + args.iterations)
        assert not stats['fallbacks']
        return dict(torch=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
                    threads=args.threads, rows=args.rows, columns=128,
                    iterations=args.iterations, results=results, stats=stats)
    finally:
        backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rows', type=int, default=500_000)
    parser.add_argument('--selected', nargs='+', type=int, default=[4096, 16384, 65536])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if min(args.rows, args.threads, args.iterations, *args.selected) <= 0 or args.warmup < 0:
        parser.error('sizes, threads, iterations must be positive; warmup must be nonnegative')
    print(json.dumps(run(args), indent=2))
