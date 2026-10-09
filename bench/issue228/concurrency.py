#!/usr/bin/env python3
"""Completed transfers, matched aggregate limits, isolated process rounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from reloc_torch import (CompilerClient, TransferQueue, TransferResources, dispatch,
                        execute_transfer_group, prepare_transfer_group)
from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench/issue221'))
from typed_pipeline import distribution, metadata

HOME = [4, 5, 6, 7, 20, 21, 22, 23]
LEFT, RIGHT, REMOTE = [4, 5, 20, 21], [6, 7, 22, 23], [12, 13, 28, 29]
LIMIT = 64 << 20
CONFIGS = {
    'small_single': dict(devices=[0], cpus=[HOME], family='small', threads=8),
    'cast_single': dict(devices=[0], cpus=[HOME], family='cast', threads=8),
    'chain_single': dict(devices=[0], cpus=[HOME], family='chain', threads=8),
    'alternating01': dict(devices=[0, 1], cpus=[HOME, HOME], family='cast', threads=4, serial=True),
    'pair01': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='cast', threads=4),
    'pair03_remote': dict(devices=[0, 3], cpus=[LEFT, REMOTE], family='cast', threads=4, remote=True),
    'pair03_local': dict(devices=[0, 3], cpus=[LEFT, REMOTE], family='cast', threads=4),
    'same_gpu': dict(devices=[0, 0], cpus=[LEFT, RIGHT], family='cast', threads=4),
    'oversubscribed': dict(devices=[0, 1], cpus=[HOME, HOME], family='cast', threads=8),
    'one_thread': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='cast', threads=1),
    'chain_whole': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='chain', threads=4),
    'chain_ring32k': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='chain', threads=4, chunk=32 << 10),
    'chain_ring1m': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='chain', threads=4, chunk=1 << 20),
    'chain_ring4m': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='chain', threads=4, chunk=4 << 20),
    'group_pair01': dict(devices=[0, 1], cpus=[LEFT, RIGHT], family='cast', threads=4, group=2),
}


def main(args):
    torch.set_num_interop_threads(1)
    torch._dynamo.config.recompile_limit = 256
    result = dict(protocol='sym228/1', round=args.round, variant=args.variant,
                  metadata=metadata(), rows=[])
    result['metadata']['sha256'][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['metadata']['topology'] = __import__('subprocess').check_output(['nvidia-smi', 'topo', '-m'], text=True)
    for name in args.configs or CONFIGS:
        config = CONFIGS[name]
        threads, count = config['threads'], len(config['devices'])
        torch.set_num_threads(threads)
        group_size = config.get('group', 1)
        shape = (65, 129) if config['family'] == 'small' else ((2097155,) if config['family'] == 'chain' else (2048, 1024))
        dims = tuple(Symbol('s' + str(i)) for i in range(len(shape)))
        operations = [] if len(shape) == 1 else [Transpose((1, 0))]
        operations += [Cast('float16', 'ieee_rne')]
        if config['family'] == 'chain': operations += [Cast('float32', 'exact')]
        dtype = 'float32' if config['family'] == 'chain' else 'float16'
        def spec(d, t): return TensorSpec(d, dense_strides(d), Const(0), t)
        compiled = CompilerClient.from_environment().compile(Recipe(spec(dims, 'float32'),
            tuple(operations), spec(dims[::-1], dtype), 'h2d'))
        def transform(x):
            y = x if x.ndim == 1 else x.t()
            y = y.half()
            return (y.float() if config['family'] == 'chain' else y).contiguous()
        # Separate persistent submitters make worker inheritance and first-touch explicit.
        with ExitStack() as stack:
            executors = [stack.enter_context(ThreadPoolExecutor(1)) for _ in range(count)]
            def init(i):
                os.sched_setaffinity(0, HOME if config.get('remote') else config['cpus'][i])
                # Fresh mappings avoid malloc arenas recycling pages from another node.
                sources, placements = [], []
                for j in range(group_size):
                    storage = mmap.mmap(-1, int(np.prod(shape))*4, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
                    array = np.ndarray(shape, dtype=np.float32, buffer=storage)
                    np.random.default_rng(228+i+j).standard_normal(shape, dtype=np.float32, out=array)
                    sources.append(torch.from_numpy(array))
                    address = array.ctypes.data
                    maps = Path('/proc/self/maps').read_text().splitlines()
                    mapping = next(line.split()[0] for line in maps if
                        int(line.split('-')[0],16) <= address < int(line.split()[0].split('-')[1],16))
                    start = int(mapping.split('-')[0], 16)
                    numa = next((line for line in Path('/proc/self/numa_maps').read_text().splitlines()
                                 if int(line.split()[0],16)==start), '')
                    placements.append(dict(mapping=mapping, source_address=hex(address), numa=numa))
                os.sched_setaffinity(0, config['cpus'][i])
                torch.cuda.set_device(config['devices'][i])
                stream = torch.cuda.Stream()
                return sources, stream, placements
            data = [ex.submit(init, i).result() for i, ex in enumerate(executors)]
            start = time.perf_counter_ns()
            fast = torch.compile(transform, fullgraph=True, options={'emulate_precision_casts': True})
            if not args.trace:
                for sources, _, _ in data:
                    for source in sources: fast(source)
            compile_ms = (time.perf_counter_ns() - start) / 1e6
            paths = args.paths or ['shared', 'independent', 'torch', 'inductor'] + (['queue'] if group_size > 1 else [])
            paths = paths[args.round % len(paths):] + paths[:args.round % len(paths)]
            row = dict(name=name, config=config, shape=shape, source_bytes=count*group_size*np.prod(shape).item()*4,
                       wire_bytes=count*group_size*np.prod(shape).item()*(4 if dtype=='float32' else 2),
                       compile_and_prime_ms=compile_ms, source_placement=[x[2] for x in data],
                       recipe_sha256=hashlib.sha256(compiled.to_bytes()).hexdigest(), paths={})
            for path in paths:
                with ExitStack() as owners_stack:
                    def resource(slots, budget):
                        options = dict(max_typed_retained_bytes=budget, max_typed_live_bytes=budget,
                                       max_typed_background_workers=(threads-1)*slots, max_typed_streams=slots)
                        if args.variant == 'candidate':
                            options.update(max_typed_contexts=slots, max_typed_contexts_per_device=slots if len(set(config['devices']))==1 else 1)
                        return owners_stack.enter_context(TransferResources(**options))
                    if path == 'shared' or (path == 'queue' and args.variant == 'candidate'):
                        owners = [resource(count, LIMIT)] * count
                    elif path == 'independent':
                        owners = [resource(1, LIMIT//count) for _ in range(count)]
                    else: owners = []
                    queues = []
                    if path == 'queue':
                        for i in range(count):
                            kwargs = dict(resources=owners[i]) if args.variant == 'candidate' else {}
                            queues.append(owners_stack.enter_context(TransferQueue(f'cuda:{config["devices"][i]}',
                                gather_threads=threads, max_scratch_bytes=LIMIT//count,
                                max_output_bytes=32 << 20, **kwargs)))
                        if args.variant == 'baseline': owners = [q._resources for q in queues]
                    def call(i):
                        sources, stream, _ = data[i]
                        with torch.cuda.stream(stream):
                            start = time.perf_counter_ns()
                            if args.trace: torch.cuda.nvtx.range_push(f'sym228/{name}/{path}/device{config["devices"][i]}')
                            if path in ('torch', 'inductor'):
                                outputs = tuple((transform if path == 'torch' else fast)(x).to(stream.device, non_blocking=True) for x in sources)
                            else:
                                requests = [dispatch.prepare_typed_transfer(compiled, x, stream.device,
                                    threads=threads, implementation='cpu_reference') for x in sources]
                                if group_size > 1:
                                    group = prepare_transfer_group(requests)
                                    if path == 'queue':
                                        handle = queues[i].submit(group)
                                        outputs = handle.wait().tensors
                                        handle.close()
                                    else:
                                        outputs = execute_transfer_group(group, resources=owners[i], gather_threads=threads,
                                            pinning='pinned', max_scratch_bytes=LIMIT//count).tensors
                                else:
                                    outputs = (dispatch.execute_typed_transfer(requests[0], resources=owners[i],
                                        gather_threads=threads, n_streams=1, pinning='pinned',
                                        pipeline=bool(config.get('chunk')), chunk_size=config.get('chunk')).tensor,)
                            stream.synchronize()
                            if args.trace: torch.cuda.nvtx.range_pop()
                            return outputs, (time.perf_counter_ns() - start)/1e6
                    def batch(check=True):
                        if args.trace: torch.cuda.nvtx.range_push(f'sym228/batch/{name}/{path}')
                        start = time.perf_counter_ns()
                        if config.get('serial'):
                            outputs = [ex.submit(call, i).result() for i, ex in enumerate(executors)]
                        else:
                            futures = [ex.submit(call, i) for i, ex in enumerate(executors)]
                            outputs = [f.result() for f in futures]
                        elapsed = (time.perf_counter_ns()-start)/1e6
                        if args.trace: torch.cuda.nvtx.range_pop()
                        if check:
                            for (tensors, _), (sources, _, _) in zip(outputs, data):
                                for tensor, source in zip(tensors, sources):
                                    assert torch.equal(tensor.cpu().view(torch.uint8), transform(source).view(torch.uint8)), (name, path)
                        return elapsed, [value[1] for value in outputs]
                    cold = []
                    for _ in range(3):
                        for owner in set(owners): owner.clear()
                        cold.append(batch()[0])
                    for _ in range(5): batch()
                    before = [o.stats()['typed'] for o in dict.fromkeys(owners)]
                    totals, calls = [], []
                    for _ in range(args.samples):
                        for sources, _, _ in data:
                            for x in sources: np.negative(x.numpy(), out=x.numpy())
                        total, latencies = batch(not args.trace)
                        totals.append(total)
                        calls.extend(latencies)
                    stats = [o.stats()['typed'] for o in dict.fromkeys(owners)]
                    if path == 'queue': stats = [q.stats() for q in queues]
                    row['paths'][path] = dict(batch=distribution(totals), calls=distribution(calls), cold_owner=distribution(cold),
                        completed_wire_GBs=row['wire_bytes']/distribution(totals)['p50_ms']/1e6,
                        resources=stats, resources_before=before, active_caller_budget=count, aggregate_background_budget=(threads-1)*count,
                        typed_scratch_budget=LIMIT, queue_output_budget=count*(32 << 20) if path=='queue' else 0)
            result['rows'].append(row)
            print(name, {p:round(v['batch']['p50_ms'],3) for p,v in row['paths'].items()}, flush=True)
    result['metadata']['loaded_runtime'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libreloc_runtime.so' in line})
    args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'candidate'], required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--configs', nargs='+', choices=CONFIGS)
    parser.add_argument('--paths', nargs='+')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--trace', action='store_true')
    main(parser.parse_args())
