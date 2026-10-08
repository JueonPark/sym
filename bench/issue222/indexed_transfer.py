#!/usr/bin/env python3
"""Completed GNN producer region, or separate DLRM pooling opportunity.

Run each round in a fresh, CPU-affined process. No profiler during timings.
The baseline mode also runs on main before indexed artifacts existed.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from reloc_torch import RelocBackend

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'issue221'))
from typed_pipeline import metadata


def completed(fn, *args):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = fn(*args)
    torch.cuda.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e6


def distribution(values):
    ordered = sorted(values)
    return dict(p50_ms=statistics.median(values), p95_ms=ordered[int(.95 * (len(ordered)-1))])


def gnn(args):
    source = torch.randn(500_000, 128)
    def eager(x, i):
        return torch.index_select(x, 0, i).to('cuda:0', dtype=torch.float16)
    def upload(x):
        return x.to('cuda:0', dtype=torch.float16)
    def host(x, i):
        return torch.index_select(x, 0, i).half()
    def host_into(x, i, out):
        return out.copy_(torch.index_select(x, 0, i))
    def new_indices(count):
        if args.index_order == 'sorted_unique':
            return torch.randperm(source.shape[0])[:count].sort().values
        return torch.randint(source.shape[0], (count,))

    with ExitStack() as stack:
        methods, owners, pinned = {}, {}, {}
        for name in args.paths:
            if name in ('ring1', 'ring2', 'unfused'):
                backend = RelocBackend(transfer_options=dict(gather_threads=8, n_streams=1,
                    n_buffers=1 if name == 'ring1' else 2, pinning='pinned'))
                stack.callback(backend.close)
                owners[name] = backend
                compiled = torch.compile(upload if name == 'unfused' else eager,
                                         backend=backend, fullgraph=True, dynamic=True)
                methods[name] = ((lambda x, i, fn=compiled: fn(torch.index_select(x, 0, i)))
                                 if name == 'unfused' else compiled)
            elif name == 'torch':
                methods[name] = eager
            elif name == 'inductor':
                methods[name] = torch.compile(eager, fullgraph=True, dynamic=True)
            elif name == 'inductor_host':
                fn = torch.compile(host, fullgraph=True, dynamic=True)
                methods[name] = lambda x, i, fn=fn: fn(x, i).to('cuda:0')
            elif name in ('torch_pinned', 'inductor_pinned'):
                fn = host_into if name == 'torch_pinned' else torch.compile(host_into, fullgraph=True, dynamic=True)
                def into(x, i, fn=fn, name=name):
                    if name not in pinned or pinned[name].shape[0] != i.numel():
                        pinned[name] = torch.empty((i.numel(), 128), dtype=torch.float16, pin_memory=True)
                    fn(x, i, pinned[name])
                    return pinned[name].to('cuda:0', non_blocking=True)
                methods[name] = into
        results, samples = [], []
        for count in args.selected:
            indices = new_indices(count)
            oracle = host(source, indices)
            row = dict(selected=count, wire_bytes=count * 128 * 2,
                       gathered_fp32_bytes=count * 128 * 4, paths={})
            for name, fn in methods.items():
                out, elapsed = completed(fn, source, indices)
                assert torch.equal(out.cpu().view(torch.uint8), oracle.view(torch.uint8)), name
                row['paths'][name] = dict(first_completed_ms=elapsed)
            for _ in range(8):
                for fn in methods.values():
                    completed(fn, source, indices)
            times = {name: [] for name in methods}
            for iteration in range(args.samples):
                # Fresh values and selection each sample; mutation and oracle outside timing.
                source.neg_()
                indices.copy_(new_indices(count))
                oracle = host(source, indices)
                order = list(methods)
                shift = iteration % len(order)
                for name in order[shift:] + order[:shift]:
                    out, elapsed = completed(methods[name], source, indices)
                    assert torch.equal(out.cpu().view(torch.uint8), oracle.view(torch.uint8)), name
                    times[name].append(elapsed)
                    samples.append(dict(selected=count, iteration=iteration, path=name, ms=elapsed))
            for name, values in times.items():
                row['paths'][name].update(distribution(values))
                if name in pinned:
                    row['paths'][name]['pinned_logical_bytes'] = pinned[name].numel() * pinned[name].element_size()
            for name, backend in owners.items():
                stats = backend.stats()
                assert not stats['fallbacks'], stats
                assert stats['replaced_regions'] == 1, stats
                row['paths'][name]['native_executions'] = stats['runtime_executions']
                row['paths'][name]['resources'] = stats['transfer_resources']['typed']
            if args.trace:
                torch.cuda.profiler.start()
                for name, fn in methods.items():
                    for _ in range(5):
                        torch.cuda.nvtx.range_push('indexed222/' + name)
                        out = fn(source, indices)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        assert torch.equal(out.cpu(), oracle)
                torch.cuda.profiler.stop()
            if args.memory:
                row['allocation_diagnostic'] = {}
                for name, fn in methods.items():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                profile_memory=True) as prof:
                        out = fn(source, indices)
                        torch.cuda.synchronize()
                    allocations = [event for event in prof.events() if event.name == 'aten::index_select']
                    row['allocation_diagnostic'][name] = dict(
                        index_select_calls=len(allocations),
                        index_select_cpu_bytes=sum(event.cpu_memory_usage for event in allocations))
            results.append(row)
            print(count, {name: round(v['p50_ms'], 3) for name, v in row['paths'].items()}, flush=True)
        return results, samples


def dlrm(args):
    # PR #162 default: 26 FP32 tables, 100k rows x 64, sum bags of length 1..4.
    tables = [torch.nn.EmbeddingBag.from_pretrained(torch.randn(100_000, 64),
              freeze=True, mode='sum') for _ in range(26)]
    results, samples = [], []
    for batch in args.selected:
        lengths = [torch.randint(1, 5, (batch,)) for _ in tables]
        offsets = [torch.cat((torch.zeros(1, dtype=torch.int64), n.cumsum(0)[:-1])) for n in lengths]
        indices = [torch.randint(100_000, (int(n.sum()),)) for n in lengths]
        def pool():
            return [table(i, o) for table, i, o in zip(tables, indices, offsets)]
        def full():
            return torch.stack(pool()).transpose(0, 1).contiguous().to('cuda:0', dtype=torch.float16)
        first, cold = completed(full)
        for _ in range(8): completed(full)
        times = {name: [] for name in ('full', 'lookup_pool', 'stack', 'layout_cast_h2d')}
        for iteration in range(args.samples):
            for i in indices: i.random_(100_000)
            actual, elapsed = completed(full)
            times['full'].append(elapsed)
            # Separately instrumented phase diagnostics, never added into full timing.
            pooled, elapsed = completed(pool)
            times['lookup_pool'].append(elapsed)
            stacked, elapsed = completed(lambda: torch.stack(pooled))
            times['stack'].append(elapsed)
            expected, elapsed = completed(lambda: stacked.transpose(0, 1).contiguous().to('cuda:0', dtype=torch.float16))
            times['layout_cast_h2d'].append(elapsed)
            assert torch.equal(actual, expected)
            for name in times:
                samples.append(dict(selected=batch, iteration=iteration, path=name, ms=times[name][-1]))
        row = dict(batch=batch, fp32_pooled_bytes=26 * batch * 64 * 4,
                   wire_bytes=26 * batch * 64 * 2, first_completed_ms=cold,
                   paths={name: distribution(values) for name, values in times.items()})
        results.append(row)
        print(batch, row['paths'], flush=True)
    return results, samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--selected', nargs='+', type=int, default=[4096, 16384, 65536])
    p.add_argument('--paths', nargs='+', choices=['torch', 'torch_pinned', 'inductor', 'inductor_host',
                   'inductor_pinned', 'unfused', 'ring1', 'ring2'],
                   default=['torch', 'torch_pinned', 'inductor', 'inductor_host', 'inductor_pinned', 'ring1', 'ring2'])
    p.add_argument('--index-order', choices=['sorted_unique', 'random'], default='sorted_unique')
    p.add_argument('--dlrm', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--memory', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(222)
    torch.cuda.init()
    with torch.no_grad():
        results, samples = dlrm(args) if args.dlrm else gnn(args)
    with args.output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['selected', 'iteration', 'path', 'ms'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(samples)
    provenance = metadata()
    provenance['sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(dict(metadata=provenance,
        configuration=vars(args) | {'output': str(args.output)}, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
