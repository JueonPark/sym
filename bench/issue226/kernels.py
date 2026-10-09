#!/usr/bin/env python3
"""Typed CPU kernel and completed-transfer qualification (no model compute).

The build-only native helper times executeHost with prebound plans and buffers.
Completed calls include preparation, allocation, transform, DMA and completion.
Compilation, correctness checks and source mutations stay outside each sample.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from reloc_torch import CompilerClient, TransferResources, dispatch
from reloc_torch.recipe import BindingParam, Cast, Dequantize, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides
import _typed_host_test

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench/issue221'))
from typed_pipeline import distribution, metadata

CASES = {
    'cast_small': ((65, 129), (1, 0), 'cast'),
    'cast_medium': ((512, 512), (1, 0), 'cast'),
    'cast_large': ((2048, 1024), (1, 0), 'cast'),
    'cast_odd': ((2049, 1025), (1, 0), 'cast'),
    'cast_inner': ((128, 128, 128), (1, 0, 2), 'cast'),
    'chain_vector': ((2097155,), None, 'chain'),
    'chain_transpose': ((2049, 1025), (1, 0), 'chain'),
    'channel_outer': ((1025, 2049), (1, 0), 'outer'),
    'channel_inner': ((257, 513), (1, 0), 'inner'),
}
LIMITS = dict(max_typed_retained_bytes=64 << 20, max_typed_live_bytes=64 << 20)


def make_case(name, direction):
    shape, perm, kind = CASES[name]
    dims = tuple(Symbol('s' + str(i)) for i in range(len(shape)))
    parameters = {}
    ops = []
    quantized = kind in ('outer', 'inner')
    axis = 1 if kind == 'outer' else 0
    if quantized:
        parameters['scale'] = torch.arange(1, shape[axis] + 1, dtype=torch.float32) * .0137
        ops.append(Dequantize('float32', BindingParam('scale', 'float32', (dims[axis],)),
                              None, axis, 'affine'))
        host = torch.randint(-127, 128, shape, dtype=torch.int8)
    else:
        host = torch.randn(shape)
    if perm:
        ops.append(Transpose(perm))
    ops.append(Cast('float16', 'ieee_rne'))
    if kind == 'chain':
        ops.append(Cast('float32', 'exact'))
    result_dims = tuple(dims[i] for i in perm) if perm else dims
    def spec(ds, dtype):
        return TensorSpec(ds, dense_strides(ds), Const(0), dtype)
    recipe = CompilerClient.from_environment().compile(Recipe(
        spec(dims, 'int8' if quantized else 'float32'), tuple(ops),
        spec(result_dims, 'float32' if kind == 'chain' else 'float16'), direction))
    scale = parameters.get('scale')
    scale_shape = tuple(shape[i] if i == axis else 1 for i in range(len(shape)))
    def transform(x):
        if quantized:
            x = x.float() * scale.reshape(scale_shape)
        if perm:
            x = x.permute(perm)
        x = x.half()
        if kind == 'chain':
            x = x.float()
        return x.contiguous()
    return recipe, host, parameters, transform


def check(value, expected):
    assert value.dtype == expected.dtype and value.shape == expected.shape
    assert value.is_contiguous()
    assert torch.equal(value.cpu().view(torch.uint8), expected.view(torch.uint8))


def main(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(226)
    torch.cuda.set_device(0)
    torch._dynamo.config.recompile_limit = 256
    result = dict(protocol='sym226/1', metadata=metadata(), round=args.round,
                  inductor_options={'emulate_precision_casts': True},
                  limits=LIMITS, rows=[])
    result['metadata']['sha256'].update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(_typed_host_test.__file__))})
    result['metadata']['loaded_runtime'] = sorted({line.split()[-1]
        for line in Path('/proc/self/maps').read_text().splitlines() if 'libreloc_runtime.so' in line})
    names = args.cases or list(CASES)
    directions = ['h2d'] if args.sweep else ['h2d', 'd2h']
    def completed(fn):
        start = time.perf_counter_ns()
        value = fn()
        torch.cuda.current_stream().synchronize()
        return value, (time.perf_counter_ns() - start) / 1e6
    for name in names:
        for direction in directions:
            torch._dynamo.reset()
            recipe, host, params, transform = make_case(name, direction)
            source = host if direction == 'h2d' else host.cuda()
            target = 'cuda:0' if direction == 'h2d' else 'cpu'
            expected = transform(host)
            row = dict(case=name, direction=direction, shape=list(host.shape), threads=args.threads,
                       source_bytes=host.numel()*host.element_size(),
                       result_bytes=expected.numel()*expected.element_size(), paths={},
                       recipe_sha256=hashlib.sha256(recipe.to_bytes()).hexdigest())
            row['wire_bytes'] = row['result_bytes'] if direction == 'h2d' else row['source_bytes']
            with ExitStack() as stack:
                sym_paths = ['whole', 'ring']
                if args.sweep or name in ('cast_small', 'cast_medium', 'cast_large', 'chain_vector'):
                    sym_paths += ['ring_32k', 'ring_4m']
                owners = {p: stack.enter_context(TransferResources(**LIMITS,
                          max_typed_background_workers=args.threads-1)) for p in sym_paths}
                latest = {}
                def sym(path):
                    request = dispatch.prepare_typed_transfer(recipe, source, target, parameters=params,
                        threads=args.threads, implementation='cpu_reference')
                    size = {'ring_32k': 32768, 'ring_4m': 4 << 20}.get(path)
                    output = dispatch.execute_typed_transfer(request, resources=owners[path],
                        gather_threads=args.threads, n_buffers=2, n_streams=1, pinning='pinned',
                        pipeline=path != 'whole', chunk_size=size)
                    latest[path] = output.report
                    return output.tensor
                def transfer(fn):
                    if direction == 'h2d':
                        return fn(source).to(target, non_blocking=True)
                    return fn(source.cpu())
                compiled = torch.compile(transform, fullgraph=True, dynamic=True,
                                         options={'emulate_precision_casts': True})
                start = time.perf_counter_ns()
                check(compiled(host), expected)
                row['inductor_compile_and_prime_ms'] = (time.perf_counter_ns()-start)/1e6
                methods = {p: (lambda p=p: sym(p)) for p in sym_paths}
                methods.update(torch=lambda: transfer(transform), inductor=lambda: transfer(compiled))
                # Original cpu_reference used scalar strided/stage execution.
                # This helper calls only unchanged TypedBoundPlan/Program APIs,
                # so one build-only helper also measures the frozen old library.
                if direction == 'h2d':
                    bound = dispatch.prepare_typed_transfer(recipe, source, target, parameters=params,
                        threads=args.threads, implementation='cpu_reference').bound
                    destination = np.empty(tuple(expected.shape), dtype=expected.numpy().dtype)
                    samples = _typed_host_test.measure(bound, host.numpy(), destination,
                        threads=args.threads, samples=args.samples)
                    check(torch.from_numpy(destination), expected)
                    row['kernel'] = distribution(samples)
                    row['kernel']['effective_GBps'] = (row['source_bytes'] + row['result_bytes']) / (
                        row['kernel']['p50_ms'] * 1e6)
                order = list(methods)
                order = order[args.round % len(order):] + order[:args.round % len(order)]
                for path in order:
                    fn = methods[path]
                    value, first = completed(fn)
                    check(value, expected)
                    cold = []
                    if path in owners:
                        for _ in range(3):
                            owners[path].clear()
                            value, ms = completed(fn)
                            check(value, expected)
                            cold.append(ms)
                    for _ in range(5):
                        check(completed(fn)[0], expected)
                    row['paths'][path] = dict(first_completed_ms=first,
                        cold_owner=distribution(cold) if cold else None)
                samples = {p: [] for p in methods}
                for i in range(args.samples):
                    host.neg_()
                    if direction == 'd2h':
                        source.copy_(host)
                    expected = transform(host)
                    torch.cuda.synchronize()
                    order = list(methods)
                    shift = (i + args.round) % len(order)
                    for path in order[shift:] + order[:shift]:
                        value, elapsed = completed(methods[path])
                        check(value, expected)
                        samples[path].append(elapsed)
                for path, values in samples.items():
                    row['paths'][path]['warm'] = distribution(values)
                    if path in owners:
                        row['paths'][path]['report'] = dict(latest[path])
                        row['paths'][path]['resources'] = owners[path].stats()['typed']
                result['rows'].append(row)
                print(name, direction, 'kernel', round(row.get('kernel', {}).get('p50_ms', 0), 3),
                      {p: round(v['warm']['p50_ms'], 3) for p, v in row['paths'].items()}, flush=True)
                args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--cases', nargs='+', choices=CASES)
    parser.add_argument('--sweep', action='store_true')
    with torch.no_grad():
        main(parser.parse_args())
