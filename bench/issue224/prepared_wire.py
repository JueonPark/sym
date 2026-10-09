#!/usr/bin/env python3
"""Completed loads and total reuse cost, with equally prepacked Torch controls.

Run three independent processes with rotated path order. No overlapping GPU
jobs. Original checkpoint generation/pinning is common setup outside timing;
owned preparation, invalidation and first loads are reported separately. Reuse
sweeps include prepare, completed loads and invalidation (artifact/JIT setup
reported separately). Raw samples are compact arrays, not verbose execution logs.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'libreloc/python/examples/workloads'),
               str(ROOT / 'bench/issue221'), str(ROOT / 'bench/issue223')]
import common
from typed_pipeline import metadata, distribution
from model_prefetch import TorchQueue, Delivery, completed
from reloc_torch import TransferQueue

PATHS = ['sym_raw', 'sym_packed', 'torch_raw', 'torch_packed', 'inductor_raw', 'inductor_packed']
SHAPES = [(256, 1024), (1024, 4096), (4096, 1024)]
REUSE = [1, 2, 4, 8, 16, 32, 64]


def raw(q, s):
    return (q.float() * s).t().contiguous()


def packed(q, s):
    return q.float() * s[:, None]


def host_pack(q, s):
    value = torch.empty((q.shape[1], q.shape[0]), dtype=torch.int8, pin_memory=True)
    scale = torch.empty_like(s, pin_memory=True)
    value.copy_(q.t())
    scale.copy_(s)
    return value, scale


class LoadPath:
    def __init__(self, name, owner, fetcher, convert):
        self.name, self.owner, self.fetcher = name, owner, fetcher
        self.prepared = None
        self.queue = TransferQueue() if name == 'sym_raw' else TorchQueue(convert) if name != 'sym_packed' else None

    def prepare(self, q, s):
        if self.name == 'sym_packed':
            return dict(self.owner.prepare('w', q, s))
        if self.name.endswith('packed'):
            self.prepared = host_pack(q, s)
            return dict(prepared_bytes=q.numel()+s.numel()*4, pinned_bytes=q.numel()+s.numel()*4)

    def invalidate(self):
        if self.name == 'sym_packed':
            self.owner.invalidate()
        self.prepared = None

    def load(self, q, s):
        if self.name == 'sym_packed':
            return self.owner.load('w')
        if self.name == 'sym_raw':
            group = self.fetcher.prepare_many([(q, s)], 'cuda:0')
            with self.queue.submit(group) as handle:
                return handle.wait().tensors[0]
        handle = self.queue.submit([self.prepared or (q, s)])
        try:
            handle.wait()
            handle.stream = torch.cuda.current_stream()
            out, = handle.outputs
            out.record_stream(handle.stream)
            return out
        finally:
            handle.close()

    def stats(self):
        return self.owner.stats() if self.name == 'sym_packed' else self.queue.stats()

    def close(self):
        self.invalidate()
        if self.queue is not None:
            self.queue.close()


def measure(fn, samples):
    return distribution([completed(fn)[1] for _ in range(samples)])


def matrices(args, result, cleanup):
    start = time.perf_counter_ns()
    owner = None
    if any(name.endswith('packed') for name in args.paths):
        from reloc_torch import PreparedWireWeights
        owner = cleanup.enter_context(PreparedWireWeights())
    fetcher = cleanup.enter_context(common.WeightFetcher(common.Report('issue224'), implementation='cuda_dequant_relocate'))
    result['sym_artifact_setup_ms'] = (time.perf_counter_ns()-start)/1e6
    converters = {'torch_raw': raw, 'torch_packed': packed,
                  'inductor_raw': torch.compile(raw, fullgraph=True, dynamic=True),
                  'inductor_packed': torch.compile(packed, fullgraph=True, dynamic=True)}
    paths = [LoadPath(name, owner, fetcher, converters.get(name)) for name in args.paths]
    for path in paths:
        cleanup.callback(path.close)
    result['matrices'] = {}
    for shape in SHAPES:
        q = torch.randint(-128, 128, shape, dtype=torch.int8).pin_memory()
        s = torch.rand(shape[1]).add_(.001).pin_memory()
        expected = raw(q, s)
        row = result['matrices'][str(shape)] = dict(shape=shape,
            common_pinned_source_bytes=q.numel()+s.numel()*4,
            wire_bytes=q.numel(), parameter_wire_bytes=s.numel()*4,
            output_bytes=q.numel()*4, paths={})
        for path in paths:
            info, prep = completed(lambda: path.prepare(q, s))
            out, first = completed(lambda: path.load(q, s))
            assert common.same_tensor(out.cpu(), expected)
            record = row['paths'][path.name] = dict(first_prepare_ms=prep,
                first_completed_load_ms=first, preparation=info)
            for _ in range(5):
                completed(lambda: path.load(q, s))
        times = {p.name: [] for p in paths}
        if args.trace:
            torch.cuda.cudart().cudaProfilerStart()
        for i in range(args.samples):
            for path in paths[i % len(paths):]+paths[:i % len(paths)]:
                if args.trace:
                    torch.cuda.nvtx.range_push(f'issue224/{shape}/{path.name}/{i}')
                out, elapsed = completed(lambda: path.load(q, s))
                if args.trace:
                    torch.cuda.nvtx.range_pop()
                times[path.name].append(elapsed)
                assert common.same_tensor(out.cpu(), expected)
        if args.trace:
            torch.cuda.cudart().cudaProfilerStop()
        for path in paths:
            record = row['paths'][path.name]
            record['warm'] = distribution(times[path.name])
            record['resources'] = path.stats()
            if path.name.endswith('packed'):
                record['replace'] = measure(lambda: path.prepare(q, s), args.samples)
                record['invalidate_then_prepare'] = measure(lambda: (path.invalidate(), path.prepare(q, s)), args.samples)
            record['reuse_total'] = {}
            for count in REUSE:
                def cycle():
                    path.invalidate()
                    path.prepare(q, s)
                    for _ in range(count):
                        path.load(q, s)
                    path.invalidate()
                record['reuse_total'][str(count)] = measure(cycle, args.sweep_samples)
            path.invalidate()
        row['exact_checks'] = len(paths)*(args.samples+1)


def models(args, result, cleanup):
    from reloc_torch import PreparedWireWeights
    from llm_offload import OffloadedGPT, generate
    from moe_experts import OffloadedMoE
    result['models'] = {}
    for kind in ('llm_decode', 'moe_sparse'):
        generator = torch.Generator().manual_seed(224)
        device = torch.device('cuda:0')
        if kind == 'llm_decode':
            sizes = dict(vocab=2048, d_model=1024, heads=16, d_ff=4096, layers=4, prompt=8, new_tokens=4)
            model = OffloadedGPT(sizes, device, generator)
            containers = model.layers
            x = torch.randint(2048, (8,), device=device)
            def run(delivery):
                return generate(model, x, 4, None,
                    lambda t: t.transpose(0, 1).contiguous().cpu(),
                    lambda t: t.transpose(0, 1).contiguous().to(device), prefetch=delivery)
        else:
            sizes = dict(d_model=1024, d_ff=2048, experts=8, blocks=2, tokens=1)
            model = OffloadedMoE(sizes, [device], generator)
            containers = [expert for block in model.experts for expert in block]
            x = torch.randn(1, 1024, device=device)
            def run(delivery):
                routes = []
                return routes, model.forward(x, None, routes.append, prefetch=delivery)
        for item in containers:
            for key, (q, s) in item.items():
                item[key] = q.pin_memory(), s.pin_memory()
        pairs = [pair for item in containers for pair in item.values()]
        names = {id(q): str(i) for i, (q, s) in enumerate(pairs)}
        with ExitStack() as lifetime:
            owner = lifetime.enter_context(PreparedWireWeights())
            fetcher = lifetime.enter_context(common.WeightFetcher(common.Report('224model'), implementation='cuda_dequant_relocate'))
            _, prep_sym = completed(lambda: [owner.prepare(names[id(q)], q, s) for q, s in pairs])
            images, prep_torch = completed(lambda: {id(q): host_pack(q, s) for q, s in pairs})
            paths = {'sym_raw': fetcher.prefetch,
                     'sym_packed': lambda groups, d: owner.prefetch([names[id(q)] for q,s in g] for g in groups)}
            for name, fn in [('torch_raw', raw), ('torch_packed', packed), ('inductor_raw', raw), ('inductor_packed', packed)]:
                queue = TorchQueue(torch.compile(fn, fullgraph=True, dynamic=True) if name.startswith('inductor') else fn)
                lifetime.callback(queue.close)
                prepare = (lambda g,d: [images[id(q)] for q,s in g]) if name.endswith('packed') else (lambda g,d:g)
                paths[name] = Delivery(queue, prepare, True)
            row = result['models'][kind] = dict(sizes=sizes, sym_prepare_ms=prep_sym, torch_prepare_ms=prep_torch,
                common_source_bytes=sum(q.numel()+s.numel()*4 for q,s in pairs), paths={})
            expected = None
            for name in args.paths:
                out, first = completed(lambda: run(paths[name]))
                actual = out[0], out[1].cpu()
                if expected is None:
                    expected = actual
                assert actual[0] == expected[0] and common.same_tensor(actual[1], expected[1])
                row['paths'][name] = dict(first_completed_ms=first)
            times = {name: [] for name in args.paths}
            for _ in range(5):
                for name in args.paths:
                    completed(lambda: run(paths[name]))
            for i in range(args.samples):
                for name in args.paths[i % len(paths):] + args.paths[:i % len(paths)]:
                    out, elapsed = completed(lambda: run(paths[name]))
                    assert out[0] == expected[0] and common.same_tensor(out[1].cpu(), expected[1])
                    times[name].append(elapsed)
            for name in args.paths:
                row['paths'][name].update(distribution(times[name]))
            row['prepared_stats'] = owner.stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--sweep-samples', type=int, default=5)
    parser.add_argument('--paths', nargs='+', choices=PATHS, default=PATHS)
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    common.seed_everything(224)
    torch.cuda.init()
    result = dict(metadata=metadata(), configuration=vars(args) | {'output': str(args.output)},
        limits=dict(output_bytes=128 << 20, scratch_bytes=64 << 20, prepared_bytes=1 << 30, pinned_bytes=1 << 30))
    result['metadata']['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with torch.no_grad(), ExitStack() as cleanup:
        matrices(args, result, cleanup)
        if args.models:
            models(args, result, cleanup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
