#!/usr/bin/env python3
"""#225's matched workload harness plus the ``sym_fast`` arm.

``sym_fast`` keeps every model operation in eager PyTorch (exactly the
``torch`` control's compute) and moves only the transfers through
``reloc_torch.fast``: prepared programs for the DLRM layout/cast and the GNN
row gather, ``FastWeights`` prefetch windows for offloaded weights, and
``KVCacheSwap`` groups for the decode cache. The Torch/Inductor controls are
unchanged from bench/issue225. Run each round in a fresh process with an empty
Inductor cache and rotated path order.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'libreloc/python/examples/workloads'),
               str(ROOT / 'bench/issue221'), str(ROOT / 'bench/issue223'),
               str(ROOT / 'bench/issue224'), str(ROOT / 'bench/issue225')]
import common
from llm_offload import OffloadedGPT, generate, MATRICES
from moe_experts import OffloadedMoE
from model_prefetch import TorchQueue, Delivery, completed
from prepared_wire import host_pack, packed
from typed_pipeline import distribution, metadata
from reloc_torch import PreparedWireWeights, RelocBackend, compat
from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap
from reloc_torch.recipe import Cast, IndexSelect, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides
import models

PATHS = ('torch', 'inductor', 'sym', 'sym_inductor', 'sym_fast', 'torch_whole')
# torch_whole: the eager Torch control with sym_fast's whole-request weight prefetch
# window (a transparency arm; identical to 'torch' outside the offload regimes).
FAST_THREADS = {'dlrm': 8, 'gnn': 8}
CASES = ('dlrm_512', 'dlrm_2048', 'gnn_256', 'gnn_1024',
         'llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense')


class PathState:
    def __init__(self, name, lifetime, dynamic=False):
        self.name, self.compiles, self.compile_ms = name, 0, 0.
        self.dynamic = dynamic
        self.backend = None
        self.fast = None
        if name == 'sym_fast':
            self.fast = FastTransfers('cuda:0', gather_threads=8, pinning='pinned')
            lifetime.callback(self.fast.close)
        elif name == 'torch_whole':
            pass
        elif name.startswith('sym'):
            self.backend = RelocBackend(compute_backend='inductor' if name.endswith('inductor') else 'eager',
                transfer_options=dict(gather_threads=8, n_streams=1, n_buffers=2, pinning='pinned'))
            lifetime.callback(self.backend.close)

    def compile(self, fn):
        if self.name in ('torch', 'sym_fast', 'torch_whole'):
            return fn
        def counted(gm, inputs):
            start = time.perf_counter_ns()
            result = (self.backend(gm, inputs) if self.backend else
                      compat.compile_inductor(gm, inputs, {'triton.cudagraphs': False}))
            self.compiles += 1
            self.compile_ms += (time.perf_counter_ns() - start) / 1e6
            return result
        return torch.compile(fn, backend=counted, fullgraph=True, dynamic=self.dynamic)

    def stats(self):
        return dict(compile_callbacks=self.compiles, compile_ms=self.compile_ms,
                    backend=self.backend.stats() if self.backend else
                    self.fast.stats() if self.fast else None)


def check(actual, expected):
    # Native transfer bytes are exact; Inductor's compute fusion/reductions may
    # change floating point rounding. Keep tokens and selected experts exact.
    assert actual[0] == expected[0], (actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=2e-5)
    return (actual[1] - expected[1]).abs().max().item()


def recommendation(case, states, lifetime):
    gen = torch.Generator().manual_seed(225)
    count = int(case.split('_')[1])
    paths, transfer, extra = {}, {}, {}
    if case.startswith('dlrm'):
        model = models.DLRM(26, 100_000, 'cuda:0', gen)
        dense, indices, offsets = models.make_batch(count, 26, 100_000, gen)
        dense = dense.to('cuda:0')
        pooled = model.pooled(indices, offsets)
        def host_into(x, out):
            return out.copy_(x.transpose(0, 1))
        def region(x):
            return x.transpose(0, 1).contiguous().to('cuda:0', dtype=torch.float16)
        def whole(x, dense):
            return model.predict(dense, region(x))
        for state in states:
            name = state.name
            if name == 'sym_fast':
                # The producer itself is fused: bag sums -> f16 -> [B, T, 64] -> H2D in
                # one pipelined native call; the compute launches overlap the DMA tail.
                program = state.fast.pooled([bag.weight for bag in model.tables], dtype='float16')
                threads = FAST_THREADS['dlrm']
                def tx(program=program, threads=threads):
                    return program.submit(indices, offsets, gather_threads=threads).wait_stream()[0]
                paths[name] = lambda tx=tx: (None, model.predict(dense, tx()))
                transfer[name] = tx
            elif name.startswith('sym'):
                fn, tx = state.compile(whole), state.compile(region)
                paths[name] = lambda fn=fn: (None, fn(model.pooled(indices, offsets), dense))
                transfer[name] = lambda tx=tx: tx(pooled)
            else:
                wire = torch.empty((count, 26, 64), dtype=torch.float16, pin_memory=True)
                host, gpu = state.compile(host_into), state.compile(model.predict)
                def tx(x, host=host, wire=wire):
                    return host(x, wire).to('cuda:0', non_blocking=True)
                paths[name] = lambda tx=tx, gpu=gpu: (None, gpu(dense, tx(model.pooled(indices, offsets))))
                transfer[name] = lambda tx=tx: tx(pooled)
        def mutate():
            for index in indices:
                index.random_(100_000)
        extra.update(batch=count, tables=26, rows=100_000,
                     wire_bytes=count*26*64*2, common_source_bytes=26*100_000*64*4)
    else:
        graph = models.random_graph(500_000, 15, gen)
        source = torch.randn(500_000, 128, generator=gen)
        block = models.minibatch(graph, torch.randperm(500_000, generator=gen)[:count].sort().values,
                                [10, 5], gen)
        selected = block['nodes']
        # The graph and index transfers are common setup. They are not part of
        # the timed feature-store inference unit; feature lookup is included.
        layer1, layer2 = (tuple(t.to('cuda:0') for t in block[key]) for key in ('layer1', 'layer2'))
        model = models.GraphSAGE(gen, 'cuda:0')
        def host_into(x, i, out):
            return out.copy_(x.index_select(0, i))
        def region(x, i):
            return x.index_select(0, i).to('cuda:0', dtype=torch.float16)
        def whole(x, i):
            return model.forward(region(x, i), layer1, layer2)
        for state in states:
            name = state.name
            if name == 'sym_fast':
                n_sym, k_sym = Symbol('s0'), Symbol('k')
                recipe = Recipe(TensorSpec((n_sym, Const(128)), dense_strides((n_sym, Const(128))), Const(0), 'float32'),
                                (IndexSelect(TensorSpec((k_sym,), (Const(1),), Const(0), 'int64')),
                                 Cast('float16', 'ieee_rne')),
                                TensorSpec((k_sym, Const(128)), dense_strides((k_sym, Const(128))), Const(0), 'float16'),
                                'h2d')
                program = state.fast.indexed(recipe)
                threads = FAST_THREADS['gnn']
                def tx(program=program, threads=threads):
                    if os.environ.get('BEAT_TORCH_GNN_ASYNC', '0') == '1':
                        return program.submit(source, selected, gather_threads=threads).wait_stream()[0]
                    return program.execute(source, selected, gather_threads=threads)
                paths[name] = lambda tx=tx: (None, model.forward(tx(), layer1, layer2))
                transfer[name] = tx
            elif name.startswith('sym'):
                fn, tx = state.compile(whole), state.compile(region)
                paths[name] = lambda fn=fn: (None, fn(source, selected))
                transfer[name] = lambda tx=tx: tx(source, selected)
            else:
                wire = torch.empty((len(selected), 128), dtype=torch.float16, pin_memory=True)
                host, gpu = state.compile(host_into), state.compile(model.forward)
                def tx(host=host, wire=wire):
                    return host(source, selected, wire).to('cuda:0', non_blocking=True)
                paths[name] = lambda tx=tx, gpu=gpu: (None, gpu(tx(), layer1, layer2))
                transfer[name] = tx
        def mutate():
            source.neg_()
        extra.update(seeds=count, selected=len(selected), fanouts=[10, 5], nodes=500_000,
                     wire_bytes=len(selected)*128*2, common_source_bytes=source.numel()*4)
    return paths, transfer, mutate, extra


def offload(case, states, lifetime):
    gen = torch.Generator().manual_seed(225)
    device = torch.device('cuda:0')
    if case.startswith('llm'):
        sizes = dict(vocab=2048, d_model=1024, heads=16, d_ff=4096, layers=4,
                     prompt=256 if case == 'llm_prefill' else 8,
                     new_tokens=1 if case == 'llm_prefill' else 4)
        model = OffloadedGPT(sizes, device, gen)
        containers = model.layers
        x = torch.randint(2048, (sizes['prompt'],), device=device)
        kv_groups = [[torch.randn(16, sizes['prompt']+step, 64, device=device) for _ in range(8)]
                     for step in range(sizes['new_tokens']-1)]
    else:
        sizes = dict(d_model=1024, d_ff=2048, experts=8, blocks=2,
                     tokens=512 if case == 'moe_dense' else 1)
        model = OffloadedMoE(sizes, [device], gen)
        containers = [e for block in model.experts for e in block]
        x = torch.randn(sizes['tokens'], 1024, device=device)
    pairs = [pair for item in containers for pair in item.values()]
    names = {id(q): str(i) for i, (q, s) in enumerate(pairs)}
    paths, transfer, extra = {}, {}, dict(sizes=sizes, preparation={}, resources={}, kv_diagnostics={})
    for state in states:
        name = state.name
        start = time.perf_counter_ns()
        if name == 'sym_fast':
            owner = lifetime.enter_context(PreparedWireWeights())
            setup = (time.perf_counter_ns()-start)/1e6
            _, prep = completed(lambda: [owner.prepare(names[id(q)], q, s) for q, s in pairs])
            fast_weights = FastWeights(state.fast, owner)
            delivery = lambda groups, d, fw=fast_weights: fw.prefetch([names[id(q)] for q,s in g] for g in groups)
            extra['resources'][name] = owner.stats
        elif name.startswith('sym'):
            owner = lifetime.enter_context(PreparedWireWeights())
            setup = (time.perf_counter_ns()-start)/1e6
            _, prep = completed(lambda: [owner.prepare(names[id(q)], q, s) for q, s in pairs])
            delivery = lambda groups, d, owner=owner: owner.prefetch([names[id(q)] for q,s in g] for g in groups)
            extra['resources'][name] = owner.stats
        elif name == 'torch_whole':
            queue = TorchQueue(packed)
            lifetime.callback(queue.close)
            setup = (time.perf_counter_ns()-start)/1e6
            images, prep = completed(lambda: {id(q): host_pack(q, s) for q, s in pairs})
            delivery = Delivery(queue, lambda g,d,images=images: [images[id(q)] for q,s in g], True)
            extra['resources'][name] = queue.stats
        else:
            queue = TorchQueue(state.compile(packed))
            lifetime.callback(queue.close)
            setup = (time.perf_counter_ns()-start)/1e6
            images, prep = completed(lambda: {id(q): host_pack(q, s) for q, s in pairs})
            delivery = Delivery(queue, lambda g,d,images=images: [images[id(q)] for q,s in g], True)
            extra['resources'][name] = queue.stats
        extra['preparation'][name] = dict(artifact_and_owner_ms=setup, snapshot_ms=prep)
        if case.startswith('llm') and name in ('sym_fast', 'torch_whole'):
            layer, embed, finish = models.decoder_layer, models.embed, models.logits
            if name == 'sym_fast':
                swap = KVCacheSwap(state.fast, 16, 64)
                lifetime.callback(swap.drain)
                def evict_all(pairs_, swap=swap):
                    flat = swap.evict([t for pair in pairs_ for t in pair])
                    return [(flat[2*i], flat[2*i+1]) for i in range(len(pairs_))]
                def restore_all(pairs_, swap=swap):
                    flat = swap.restore([t for pair in pairs_ for t in pair])
                    return [(flat[2*i], flat[2*i+1]) for i in range(len(pairs_))]
            else:
                def evict_all(pairs_):
                    return [(k.transpose(0, 1).contiguous().cpu(), v.transpose(0, 1).contiguous().cpu())
                            for k, v in pairs_]
                def restore_all(pairs_):
                    return [(k.transpose(0, 1).contiguous().to(device), v.transpose(0, 1).contiguous().to(device))
                            for k, v in pairs_]
            if kv_groups:
                def kv_only(evict_all=evict_all, restore_all=restore_all):
                    for group in kv_groups:
                        pairs_ = [(group[2*i], group[2*i+1]) for i in range(len(group)//2)]
                        restored = restore_all(evict_all(pairs_))
                    return [t for pair in restored for t in pair]
                extra['kv_diagnostics'][name] = kv_only
            def run(delivery=delivery, layer=layer, embed=embed, finish=finish, evict_all=evict_all, restore_all=restore_all):
                tokens, start, offloaded, generated = x, 0, None, []
                # One prefetch window spans the whole request: weights are immutable,
                # so the next step's first layer streams in behind this step's tail.
                groups = (tuple(item[k] for k in MATRICES) for _ in range(sizes['new_tokens']) for item in model.layers)
                with delivery(groups, device) as loaded:
                    for step in range(sizes['new_tokens']):
                        cache = [None]*4 if offloaded is None else restore_all(offloaded)
                        h = embed(model.token, model.position, tokens, start)
                        next_cache = []
                        for past in cache:
                            h, kv = layer(h, past, next(loaded))
                            next_cache.append(kv)
                        output = finish(h, model.token)
                        token = output[-1].argmax().view(1)
                        generated.append(int(token))
                        start += len(tokens)
                        tokens = token
                        offloaded = evict_all(next_cache) if step < sizes['new_tokens']-1 else None
                return generated, output
            def transfer_only(delivery=delivery):
                groups = (tuple(item[k] for k in MATRICES) for _ in range(sizes['new_tokens']) for item in model.layers)
                with delivery(groups, device) as loaded:
                    for weights in loaded:
                        pass
                return weights
        elif case.startswith('llm'):
            layer, embed, finish = (state.compile(fn) for fn in (models.decoder_layer, models.embed, models.logits))
            evict = state.compile(lambda t: t.transpose(0, 1).contiguous().cpu())
            restore = state.compile(lambda t: t.transpose(0, 1).contiguous().to(device))
            if kv_groups:
                def kv_only(evict=evict, restore=restore):
                    for group in kv_groups:
                        offloaded = [evict(kv) for kv in group]
                        restored = [restore(kv) for kv in offloaded]
                    return restored
                extra['kv_diagnostics'][name] = kv_only
            def run(delivery=delivery, layer=layer, embed=embed, finish=finish, evict=evict, restore=restore):
                tokens, start, offloaded, generated = x, 0, None, []
                for step in range(sizes['new_tokens']):
                    cache = [None]*4 if offloaded is None else [(restore(k), restore(v)) for k,v in offloaded]
                    h = embed(model.token, model.position, tokens, start)
                    groups = (tuple(item[k] for k in MATRICES) for item in model.layers)
                    next_cache = []
                    with delivery(groups, device) as loaded:
                        for weights, past in zip(loaded, cache):
                            h, kv = layer(h, past, weights)
                            next_cache.append(kv)
                    output = finish(h, model.token)
                    token = output[-1].argmax().view(1)
                    generated.append(int(token))
                    start += len(tokens)
                    tokens = token
                    offloaded = [(evict(k), evict(v)) for k,v in next_cache] if step < sizes['new_tokens']-1 else None
                return generated, output
            def transfer_only(delivery=delivery):
                for _ in range(sizes['new_tokens']):
                    with delivery((tuple(item[k] for k in MATRICES) for item in model.layers), device) as loaded:
                        for weights in loaded:
                            pass
                return weights
        elif name in ('sym_fast', 'torch_whole'):
            route, expert, combine = models.route, models.expert, models.combine
            def run(delivery=delivery, route=route, expert=expert, combine=combine):
                h0, routes = x, []
                for router, experts in zip(model.routers, model.experts):
                    h, selected, gates = route(h0, router)
                    active = torch.unique(selected).tolist()
                    routes.append(active)
                    selections = [(selected == e).nonzero(as_tuple=True) for e in active]
                    out = torch.zeros_like(h0)
                    with delivery(((experts[e]['w1'], experts[e]['w2']) for e in active), device) as loaded:
                        for weights, (tokens, slots) in zip(loaded, selections):
                            out = combine(out, gates, tokens, slots, expert(h, tokens, weights))
                    h0 = h0 + out
                return routes, h0
            routes = []
            model.forward(x, lambda q,s,d: common.WeightFetcher.reference(q,s,d), routes.append)
            def transfer_only(delivery=delivery, routes=routes):
                for experts, active in zip(model.experts, routes):
                    with delivery(((experts[e]['w1'], experts[e]['w2']) for e in active), device) as loaded:
                        for weights in loaded:
                            pass
                return weights
        else:
            route, expert, combine = (state.compile(fn) for fn in (models.route, models.expert, models.combine))
            def run(delivery=delivery, route=route, expert=expert, combine=combine):
                h0, routes = x, []
                for router, experts in zip(model.routers, model.experts):
                    h, selected, gates = route(h0, router)
                    active = torch.unique(selected).tolist()
                    routes.append(active)
                    selections = [(selected == e).nonzero(as_tuple=True) for e in active]
                    out = torch.zeros_like(h0)
                    with delivery(((experts[e]['w1'], experts[e]['w2']) for e in active), device) as loaded:
                        for weights, (tokens, slots) in zip(loaded, selections):
                            out = combine(out, gates, tokens, slots, expert(h, tokens, weights))
                    h0 = h0 + out
                return routes, h0
            # Freeze the active set from an untimed eager reference. Routing
            # remains inside every end-to-end invocation above.
            routes = []
            model.forward(x, lambda q,s,d: common.WeightFetcher.reference(q,s,d), routes.append)
            def transfer_only(delivery=delivery, routes=routes):
                for experts, active in zip(model.experts, routes):
                    with delivery(((experts[e]['w1'], experts[e]['w2']) for e in active), device) as loaded:
                        for weights in loaded:
                            pass
                return weights
        paths[name], transfer[name] = run, transfer_only
    # Independent original-model oracle validates the factored compute regions.
    if case.startswith('llm'):
        oracle = generate(model, x, sizes['new_tokens'],
            lambda q,s: common.WeightFetcher.reference(q,s,device),
            lambda t: t.transpose(0,1).contiguous().cpu(),
            lambda t: t.transpose(0,1).contiguous().to(device))
    else:
        routes = []
        oracle = routes, model.forward(x, common.WeightFetcher.reference, routes.append)
    extra['oracle'] = oracle
    extra['common_source_bytes'] = sum(q.numel()+s.numel()*4 for q,s in pairs)
    extra['transfer_scope'] = 'request weight groups only; KV and routing excluded from transfer diagnostic'
    return paths, transfer, lambda: None, extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    p.add_argument('--paths', nargs='+', choices=PATHS, default=PATHS)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--dynamic', action='store_true', help='force symbolic compute shapes instead of fixed-regime specialization')
    p.add_argument('--trace', action='store_true', help='NVTX ranges around timed calls (run under nsys)')
    p.add_argument('--cache-state', choices=('rotated', 'cold', 'warm'), default='rotated',
                   help='rotated: #225 protocol (mutate once per sample, paths back to back); '
                        'cold: sweep the caches (mutate twice, an involution for the GraphSAGE source) before every '
                        'path; warm: run each path once untimed right before its timed run')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(225)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    torch.cuda.init()
    result = dict(metadata=metadata(), configuration=vars(args)|{'output':str(args.output)}, cases={})
    result['metadata']['thread_environment'] = {k:os.environ.get(k) for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OMP_WAIT_POLICY','GOMP_SPINCOUNT',
         'TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_CACHE_DIR')}
    result['metadata']['benchmark_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in Path(__file__).parent.glob('*.py')}
    with torch.no_grad():
        for case in args.cases:
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            with ExitStack() as lifetime:
                states = [PathState(n,lifetime,args.dynamic) for n in args.paths]
                paths, transfers, mutate, extra = (recommendation if case.startswith(('dlrm','gnn')) else offload)(case,states,lifetime)
                expected = extra.pop('oracle', None)
                resources = extra.pop('resources', {})
                kv_diagnostics = extra.pop('kv_diagnostics', {})
                row = result['cases'][case] = dict(configuration=extra, paths={})
                from torch._inductor import metrics
                for state in states:
                    before = metrics.generated_kernel_count
                    actual, elapsed = completed(paths[state.name])
                    if expected is None:
                        expected = actual
                    row['paths'][state.name] = dict(first_completed_ms=elapsed,
                        first_generated_kernels=metrics.generated_kernel_count-before,
                        first_max_abs_error=check(actual,expected), first_stats=state.stats())
                for _ in range(args.warmup):
                    for fn in paths.values():
                        completed(fn)
                before = {s.name:s.compiles for s in states}
                values, errors = {n:[] for n in paths}, {n:0. for n in paths}
                if args.trace:
                    torch.cuda.cudart().cudaProfilerStart()
                for i in range(args.samples):
                    mutate()
                    expected = None
                    order = list(paths)
                    order = order[i%len(order):]+order[:i%len(order)]
                    outputs = {}
                    for position, n in enumerate(order):
                        if args.cache_state == 'cold' and position:
                            mutate(); mutate()     # an involutive mutation: same values, caches swept twice
                        elif args.cache_state == 'warm':
                            completed(paths[n])    # the same path once untimed: its inputs are cache-hot
                        if args.trace:
                            torch.cuda.nvtx.range_push(f'beat_torch/{case}/{n}/{i}')
                        outputs[n], elapsed = completed(paths[n])
                        if args.trace:
                            torch.cuda.nvtx.range_pop()
                        values[n].append(elapsed)
                    expected = outputs.get('torch', next(iter(outputs.values())))
                    for n in paths:
                        errors[n] = max(errors[n],check(outputs[n],expected))
                for state in states:
                    n = state.name
                    row['paths'][n].update(distribution(values[n]), max_abs_error=errors[n],
                        warm_compile_callbacks=state.compiles-before[n], warm_stats=state.stats())
                    assert state.compiles == before[n], (case,n,'warm recompile')
                    if 'inductor' in n:
                        assert row['paths'][n]['first_generated_kernels'] > 0
                    # Compile and warm the isolated diagnostic AFTER headline samples.
                    completed(transfers[n])
                    row['paths'][n]['transfer'] = distribution([completed(transfers[n])[1] for _ in range(args.samples)])
                    if n in resources:
                        row['paths'][n]['resources'] = resources[n]()
                    if n in kv_diagnostics:
                        completed(kv_diagnostics[n])
                        row['paths'][n]['kv_roundtrips'] = distribution(
                            [completed(kv_diagnostics[n])[1] for _ in range(args.samples)])
                if args.trace:
                    torch.cuda.cudart().cudaProfilerStop()
                row['dynamo_counters'] = {k:dict(v) for k,v in torch._dynamo.utils.counters.items() if k in
                    ('stats','graph_break','unimplemented')}
                assert not row['dynamo_counters'].get('graph_break')
                print(case, {n:round(row['paths'][n]['p50_ms'],3) for n in paths}, flush=True)
                args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
