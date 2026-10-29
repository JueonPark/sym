#!/usr/bin/env python3
"""Per-call overhead of the fast lane's entry points on one small transfer.

Compares, for the DLRM-shaped typed program (26 x B x 64 float32 -> [B, 26, 64]
float16 on the GPU) and the GraphSAGE gather (k rows of a [N, 128] table),
the completed latency of: the direct prepared call (``program.execute``), the
same call as a custom op in eager mode (``program.op``), the op inside a
``torch.compile(fullgraph=True)`` region, and the previous generic route
(``torch.compile(backend=RelocBackend())`` over the same region, which
validates every call from scratch). Medians of ``--samples`` completed calls
after warmup, microseconds.

    python bench/beat_torch/op_overhead.py [--batch 16] [--rows 256] [--samples 200]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'libreloc/python/tests/torch_frontend'))
from reloc_torch import RelocBackend
from reloc_torch.fast import FastTransfers
from test_fast import dlrm_recipe, gather_recipe


def completed_us(fn, samples):
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--rows', type=int, default=256)
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--json')
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    results = {}
    with torch.no_grad(), FastTransfers('cuda:0', gather_threads=8) as fast:
        source = torch.randn(26, args.batch, 64)
        typed = fast.typed(dlrm_recipe(), source)
        compiled_op = torch.compile(lambda x: typed.op(x), fullgraph=True, backend='inductor',
                                    options={'triton.cudagraphs': False})
        backend = RelocBackend(compute_backend='eager', transfer_options=dict(gather_threads=8, n_streams=1,
                                                                             n_buffers=2, pinning='pinned'))
        generic = torch.compile(lambda x: x.transpose(0, 1).contiguous().to('cuda:0', dtype=torch.float16),
                                backend=backend, fullgraph=True)
        torch_eager = lambda: source.transpose(0, 1).contiguous().to('cuda:0', dtype=torch.float16)
        rows = dict(execute=lambda: typed.execute(source), op_eager=lambda: typed.op(source),
                    op_compiled=lambda: compiled_op(source), reloc_backend=lambda: generic(source),
                    torch_eager=torch_eager)
        for fn in rows.values():
            for _ in range(10):
                fn()
        results[f'typed_26x{args.batch}x64'] = {name: completed_us(fn, args.samples) for name, fn in rows.items()}
        backend.close()

        table = torch.randn(100_000, 128)
        index = torch.randint(100_000, (args.rows,))
        indexed = fast.indexed(gather_recipe())
        compiled_idx = torch.compile(lambda x, i: indexed.op(x, i), fullgraph=True, backend='inductor',
                                     options={'triton.cudagraphs': False})
        backend = RelocBackend(compute_backend='eager', transfer_options=dict(gather_threads=8, n_streams=1,
                                                                             n_buffers=2, pinning='pinned'))
        generic_idx = torch.compile(lambda x, i: x.index_select(0, i).to('cuda:0', dtype=torch.float16),
                                    backend=backend, fullgraph=True)
        rows = dict(execute=lambda: indexed.execute(table, index), op_eager=lambda: indexed.op(table, index),
                    op_compiled=lambda: compiled_idx(table, index), reloc_backend=lambda: generic_idx(table, index),
                    torch_eager=lambda: table.index_select(0, index).to('cuda:0', dtype=torch.float16))
        for fn in rows.values():
            for _ in range(10):
                fn()
        results[f'indexed_{args.rows}x128'] = {name: completed_us(fn, args.samples) for name, fn in rows.items()}
        backend.close()
    for case, row in results.items():
        print(case, {k: round(v, 1) for k, v in row.items()})
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
