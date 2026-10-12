#!/usr/bin/env python3
"""Pinning lifecycle qualification and matched completed raw-weight loads."""
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
sys.path.insert(0, str(ROOT))
from bench.issue221.typed_pipeline import distribution, metadata
from bench.issue219.request_overhead import Phases, completed
from reloc_torch import CompilerClient, TransferResources, dispatch, transport
from reloc_torch.index_select import prepare_index_select_transfer
from reloc_torch.recipe import Recipe, TensorSpec, Cast, Transpose, IndexSelect, Dequantize, BindingParam
from reloc_torch.symbolic import Const, Symbol, dense_strides

POLICIES = dict(pinned=dict(pinning='pinned'), pageable=dict(pinning='pageable'),
                auto=dict(pinning='auto'), gate8=dict(pinning='auto', min_pinned_bytes=8 << 20))
SIZES = dict(dlrm=[128, 512, 2048, 4096], gnn=[1024, 9085, 35180, 65539],
             kv_restore=[8, 33, 512, 2048], kv_evict=[8, 33, 512, 2048],
             weight=[(31, 33), (256, 1024), (1024, 1024), (1024, 4096), (4096, 1024), (2049, 1025)])


def qualified_metadata():
    result = metadata()
    files = {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
             if 'libreloc_runtime.so' in line}
    result['loaded_runtime_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    result['benchmark_sha256'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for directory in ('issue219', 'issue221', 'issue223', 'issue224', 'issue225', 'issue229')
        for f in (ROOT/'bench'/directory).glob('*.py')}
    result['thread_environment'] = {k: os.environ.get(k) for k in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT',
        'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHINDUCTOR_CACHE_DIR')}
    result['interop_threads'] = torch.get_num_interop_threads()
    result['python'] = sys.version
    return result


def owner():
    return TransferResources(max_retained_bytes=64 << 20, max_live_staging_bytes=64 << 20,
        max_background_workers=7, max_typed_retained_bytes=64 << 20,
        max_typed_live_bytes=64 << 20, max_typed_background_workers=7)


def spec(shape, dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


def artifact(kind):
    n, c = Symbol('N'), Symbol('C')
    if kind == 'dlrm':
        src, dst = (Const(26), n, Const(64)), (n, Const(26), Const(64))
        ops, dtype = (Transpose((1, 0, 2)), Cast('float16', 'ieee_rne')), 'float16'
    elif kind == 'gnn':
        src, dst = (Const(65536), Const(128)), (n, Const(128))
        ops, dtype = (IndexSelect(spec((n,), 'int64')), Cast('float16', 'ieee_rne')), 'float16'
    elif kind.startswith('kv_'):
        src = (n, Const(16), Const(64)) if kind == 'kv_restore' else (Const(16), n, Const(64))
        dst = (src[1], src[0], src[2])
        ops, dtype = (Transpose((1, 0, 2)),), 'float32'
    else:
        src, dst = (n, c), (c, n)
        ops, dtype = (Dequantize('float32', BindingParam('scale', 'float32', (c,)), None, 1, 'affine'), Transpose((1, 0))), 'float32'
    return CompilerClient.from_environment().compile(Recipe(
        spec(src, 'int8' if kind == 'weight' else 'float32'), ops, spec(dst, dtype),
        'd2h' if kind == 'kv_evict' else 'h2d'))


class Case:
    def __init__(self, kind, size, compiled):
        self.kind, self.size, self.compiled = kind, size, compiled
        self.device = 'cpu' if kind == 'kv_evict' else 'cuda:0'
        self.index, self.scale, self.report = None, None, None
        if kind == 'weight':
            self.source = torch.randint(-128, 128, size, dtype=torch.int8)
            self.scale = torch.rand(size[1]).add_(.01)
        elif kind == 'gnn':
            self.source = torch.randn(65536, 128)
            self.index = torch.randint(65536, (size,))
        else:
            shape = (26, size, 64) if kind == 'dlrm' else ((16, size, 64) if kind == 'kv_evict' else (size, 16, 64))
            self.source = torch.randn(shape, device='cuda:0' if kind == 'kv_evict' else 'cpu')

    def logical(self):
        if self.kind == 'weight':
            return (self.source.float() * self.scale).t().contiguous()
        if self.kind == 'gnn':
            return self.source.index_select(0, self.index).half()
        value = self.source.transpose(0, 1).contiguous()
        return value.half() if self.kind == 'dlrm' else value

    def mutate(self):
        self.source.neg_()
        if self.index is not None:
            self.index.add_(17).remainder_(65536)
        if self.scale is not None:
            self.scale.mul_(.5 if self.scale[0] > .01 else 2.)
        torch.cuda.synchronize()

    def sym(self, resources, policy):
        options = dict(resources=resources, gather_threads=8, n_buffers=2, n_streams=1, **POLICIES[policy])
        if self.kind.startswith('kv_'):
            request = transport.prepare_transfer(self.compiled, self.source, self.device)
            result = transport.execute_transfer(request, **options)
            self.report = dict(staging=list(request.staging))
            return result
        else:
            request = (prepare_index_select_transfer(self.compiled, self.source, self.index, self.device)
                       if self.kind == 'gnn' else dispatch.prepare_typed_transfer(
                           self.compiled, self.source, self.device, threads=8,
                           parameters={'scale': self.scale} if self.scale is not None else None,
                           implementation='cuda_dequant_relocate' if self.scale is not None else 'cpu_reference'))
            result = dispatch.execute_typed_transfer(request, **options)
        self.report = dict(result.report)
        return result.tensor

    def controls(self):
        if self.kind == 'weight':
            def convert(q, s):
                return (q.float() * s).t().contiguous()
            compiled = torch.compile(convert, fullgraph=True)
            def transfer(fn):
                return fn(self.source.to('cuda:0', non_blocking=True), self.scale.to('cuda:0', non_blocking=True))
            return dict(torch=lambda: transfer(convert), inductor=lambda: transfer(compiled))
        if self.kind == 'kv_evict':
            # Identical wire bytes: upload/download the source layout, then
            # finish the transpose on CPU after D2H completion.
            wire = torch.empty(self.source.shape, dtype=self.source.dtype, pin_memory=True)
            def transpose(x):
                return x.transpose(0, 1).contiguous()
            compiled = torch.compile(transpose, fullgraph=True)
            def transfer(fn):
                wire.copy_(self.source, non_blocking=True)
                torch.cuda.synchronize()
                return fn(wire)
            return dict(torch=lambda: transfer(transpose), inductor=lambda: transfer(compiled))
        expected = self.logical()
        wire = torch.empty(expected.shape, dtype=expected.dtype, pin_memory=True)
        if self.kind == 'gnn':
            def into(x, index, out):
                return out.copy_(x.index_select(0, index))
            args = lambda: (self.source, self.index, wire)
        else:
            def into(x, out):
                return out.copy_(x.transpose(0, 1))
            args = lambda: (self.source, wire)
        compiled = torch.compile(into, fullgraph=True)
        return dict(torch=lambda: into(*args()).to('cuda:0', non_blocking=True),
                    inductor=lambda: compiled(*args()).to('cuda:0', non_blocking=True))


def check(actual, expected):
    assert torch.equal(actual.cpu().contiguous().reshape(-1).view(torch.uint8),
                       expected.cpu().contiguous().reshape(-1).view(torch.uint8))


def rotate(items, n):
    items = list(items)
    n %= len(items)
    return items[n:] + items[:n]


def qualify(kind, args):
    compiled = artifact(kind)
    result = dict(recipe_sha256=hashlib.sha256(compiled.to_bytes()).hexdigest(), cases={}, growth={})
    with ExitStack() as lifetime:
        owners = {p: lifetime.enter_context(owner()) for p in POLICIES}
        for size in SIZES[kind]:
            case = Case(kind, size, compiled)
            expected = case.logical().cpu()
            row = result['cases'][str(size)] = dict(source_bytes=case.source.numel()*case.source.element_size(),
                wire_bytes=case.source.numel() if kind == 'weight' else expected.numel()*expected.element_size(),
                parameter_bytes=case.scale.numel()*4 if case.scale is not None else 0, paths={})
            methods = {p: (lambda p=p: case.sym(owners[p], p)) for p in POLICIES}
            methods.update(case.controls())
            # JIT, artifact preparation and allocator priming are outside cold-owner timing.
            for p in methods:
                out, first = completed(methods[p]); check(out, expected)
                row['paths'][p] = dict(first_including_compile_ms=first)
            for p in rotate(POLICIES, args.round):
                cold, ephemeral = [], []
                for _ in range(args.cold_samples):
                    owners[p].clear()
                    out, ms = completed(methods[p]); check(out, expected)
                    cold.append(ms)
                    def cycle():
                        with owner() as resources:
                            return case.sym(resources, p)
                    out, ms = completed(cycle); check(out, expected)
                    ephemeral.append(ms)
                row['paths'][p].update(cold_owner=distribution(cold), ephemeral=distribution(ephemeral))
            for _ in range(5):
                for fn in methods.values():
                    completed(fn)
            times = {p: [] for p in methods}
            for i in range(args.samples):
                case.mutate()
                expected = case.logical().cpu()
                for p in rotate(methods, i+args.round):
                    out, ms = completed(methods[p]); check(out, expected)
                    times[p].append(ms)
            for p in methods:
                row['paths'][p]['warm'] = distribution(times[p])
                if p in owners:
                    completed(methods[p])
                    row['paths'][p].update(resources=owners[p].stats(), report=case.report)
            if args.trace:
                row['phases'] = {}
                with Phases(True) as phases:
                    import pyreloc
                    phases.wrap(pyreloc, 'bind_indexed', 'binding')
                    phases.wrap(pyreloc, 'prepare_index_select_program', 'native_preparation')
                    for p in POLICIES:
                        row['phases'][p] = [phases.call(methods[p], f'{kind}/{size}/{p}')[1] for _ in range(5)]
            print(kind, size, {p: round(row['paths'][p]['warm']['p50_ms'], 3) for p in methods}, flush=True)
        # Independently retain each owner through small -> medium -> largest ->
        # small, then reuse each size. Request growth need not grow a 1 MiB ring.
        largest = max(SIZES[kind], key=lambda size: size[0]*size[1] if isinstance(size, tuple) else size)
        sequence = [SIZES[kind][0], SIZES[kind][1], largest, SIZES[kind][0]]
        for p in POLICIES:
            observations = result['growth'][p] = []
            for cycle in range(args.cold_samples):
                owners[p].clear()
                for position, size in enumerate(sequence):
                    case = Case(kind, size, compiled)
                    expected = case.logical().cpu()
                    before = owners[p].stats()
                    out, growth = completed(lambda: case.sym(owners[p], p)); check(out, expected)
                    after = owners[p].stats()
                    out, reuse = completed(lambda: case.sym(owners[p], p)); check(out, expected)
                    observations.append(dict(cycle=cycle, position=position, size=size,
                        request_ms=growth, reuse_ms=reuse, before=before, after=after, report=case.report))
    return result


def kernels(samples):
    import _cuda_kernel_test as harness
    result = {}
    for rows, cols in [(1, 65), (31, 33), (129, 517), *SIZES['weight'][1:]]:
        q = torch.randint(-128, 128, (rows, cols), device='cuda', dtype=torch.int8)
        scale = torch.rand(cols, device='cuda')
        out = torch.empty((cols, rows), device='cuda')
        expected = (q.float()*scale).t().contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        values = {p: [] for p in ('generic', 'tiled')}
        def measure(p, count):
            return harness.measure(rows, cols, q.data_ptr(), out.data_ptr(), scale.data_ptr(), stream, p == 'generic', count)
        for p in values:
            measure(p, 5)
        for i in range(samples):
            for p in rotate(values, i):
                values[p] += measure(p, 1)
                check(out, expected)
        result[str((rows, cols))] = {p: distribution(v) for p, v in values.items()}
    return result


def prepared_sources(args):
    """Owned immutable sources: pin_memory controls these; staging knobs do not."""
    from reloc_torch import PreparedWireWeights
    result = {}
    for shape in [(1024, 1024), (1024, 4096)]:
        q = torch.randint(-128, 128, shape, dtype=torch.int8)
        scale = torch.rand(shape[1]).add_(.01)
        expected = (q.float()*scale).t().contiguous()
        row = result[str(shape)] = {}
        with ExitStack() as lifetime:
            paths = {p: lifetime.enter_context(PreparedWireWeights(pin_memory=p == 'pinned'))
                     for p in ('pinned', 'pageable')}
            for p, weights in paths.items():
                prep, first, retire = [], [], []
                for _ in range(args.cold_samples):
                    _, ms = completed(lambda: weights.prepare('w', q, scale)); prep.append(ms)
                    out, ms = completed(lambda: weights.load('w')); first.append(ms); check(out, expected)
                    _, ms = completed(weights.invalidate); retire.append(ms)
                weights.prepare('w', q, scale)
                for _ in range(5):
                    completed(lambda: weights.load('w'))
                row[p] = dict(snapshot=distribution(prep), first_after_snapshot=distribution(first),
                              retirement=distribution(retire))
            times = {p: [] for p in paths}
            for i in range(args.samples):
                for p in rotate(paths, i+args.round):
                    out, ms = completed(lambda: paths[p].load('w')); check(out, expected)
                    times[p].append(ms)
            for p, weights in paths.items():
                row[p].update(warm=distribution(times[p]), resources=weights.stats(), reuse_total={})
                for count in (1, 4, 16):
                    def cycle():
                        weights.prepare('w', q, scale)
                        for _ in range(count):
                            weights.load('w')
                        weights.invalidate()
                    weights.invalidate()
                    row[p]['reuse_total'][str(count)] = distribution([
                        completed(cycle)[1] for _ in range(args.cold_samples)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--cold-samples', type=int, default=3)
    p.add_argument('--round', type=int, default=1)
    p.add_argument('--families', nargs='+', choices=list(SIZES), default=list(SIZES))
    p.add_argument('--trace', action='store_true')
    p.add_argument('--kernels', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.manual_seed(229)
    torch._dynamo.config.recompile_limit = 64
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    result = dict(metadata=qualified_metadata(), configuration=vars(args)|{'output': str(args.output)}, families={})
    result['metadata']['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with torch.no_grad():
        for kind in args.families:
            result['families'][kind] = qualify(kind, args)
            args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')
        if args.kernels:
            result['kernel_events'] = kernels(args.samples)
        if 'weight' in args.families:
            result['prepared_sources'] = prepared_sources(args)
    args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
