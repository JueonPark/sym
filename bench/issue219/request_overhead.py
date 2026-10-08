#!/usr/bin/env python3
"""Completed KV/weight calls and exclusive diagnostic phases for issue #219.

Run each revision/round in a fresh process with a fixed CPU affinity. Profiling
is a separate pass; its nested wrappers partition host wall time, not GPU work.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import cProfile
import functools
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'libreloc/python/examples/workloads'))


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def summary(values):
    return {'p50_ms': statistics.median(values), 'p95_ms': percentile(values, .95),
            'mean_ms': statistics.mean(values), 'samples_ms': values}


class Phases:
    def __init__(self, trace=False):
        self.trace = trace
        self.active = False
        self.undo = []
        self.wrappers = {}

    def account(self, now):
        self.times[self.category] += now - self.last
        self.last = now

    def wrap(self, module, name, category):
        if not hasattr(module, name):
            return
        original = getattr(module, name)
        key = id(original)
        if key in self.wrappers:
            wrapper = self.wrappers[key]
        else:
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                if not self.active:
                    return original(*args, **kwargs)
                self.account(time.perf_counter_ns())
                previous, self.category = self.category, category
                self.counts[name] += 1
                if self.trace:
                    self.torch.cuda.nvtx.range_push('phase219/' + category + '/' + name)
                try:
                    return original(*args, **kwargs)
                finally:
                    if self.trace:
                        self.torch.cuda.nvtx.range_pop()
                    self.account(time.perf_counter_ns())
                    self.category = previous
            self.wrappers[key] = wrapper
        self.undo.append((module, name, original))
        setattr(module, name, wrapper)

    def __enter__(self):
        import torch
        import pyreloc
        from reloc_torch import artifact, compat, dispatch, runtime, transport
        self.torch = torch
        for name in ('bind', 'bind_typed'):
            self.wrap(pyreloc, name, 'binding')
        for name in ('bind_values', 'parameter_extents'):
            self.wrap(artifact.CompiledRecipe, name, 'binding')
        for name in ('prepare_typed_program', 'query_capability', 'select_dispatch',
                     'prepare_dispatch_template', 'prepare_dispatch_from_template',
                     'prepare_dispatch', 'make_transfer', 'validate_transfer_source'):
            self.wrap(pyreloc, name, 'native_preparation')
        for name in ('execute_transfer', 'execute_dispatch'):
            self.wrap(pyreloc, name, 'native_including_completion')
        for mod in (runtime, transport, dispatch):
            self.wrap(mod, 'destination_descriptor', 'descriptors')
            self.wrap(mod, '_storage_view', 'fresh_validation')
        self.wrap(compat, 'storage_snapshot', 'fresh_validation')
        self.wrap(runtime, '_metadata_snapshot', 'fresh_validation')
        for mod in (runtime, transport, dispatch):
            self.wrap(mod, 'source_reason', 'fresh_validation')
        self.wrap(dispatch, '_parameter_snapshot', 'fresh_validation')
        for cls in (runtime.PreparedCall, transport.PreparedTransfer, dispatch.PreparedTypedTransfer):
            self.wrap(cls, 'recheck', 'fresh_validation')
        self.wrap(torch, 'empty_strided', 'output_allocation')
        self.wrap(torch.cuda, 'synchronize', 'explicit_completion')
        return self

    def __exit__(self, *exc):
        for mod, name, original in reversed(self.undo):
            setattr(mod, name, original)

    def call(self, fn, label):
        self.times, self.counts = Counter(), Counter()
        self.category = 'frontend_and_other_validation'
        if self.trace:
            self.torch.cuda.nvtx.range_push('request219/' + label)
        start = self.last = time.perf_counter_ns()
        self.active = True
        try:
            result = fn()
            self.torch.cuda.synchronize()
        finally:
            end = time.perf_counter_ns()
            self.account(end)
            self.active = False
            if self.trace:
                self.torch.cuda.nvtx.range_pop()
        assert sum(self.times.values()) == end - start
        return result, {'total_ms': (end - start) / 1e6,
                        'exclusive_ms': {k: v / 1e6 for k, v in self.times.items()},
                        'calls': dict(self.counts)}


def metadata():
    import os
    import pyreloc
    import reloc_torch
    import torch
    paths = [Path(__file__), REPO / 'libreloc/python/examples/workloads/common.py',
             REPO / 'calibration/epyc7351-2080ti.cal',
             *Path(reloc_torch.__file__).parent.glob('*.py'),
             *Path(pyreloc.__file__).parent.glob('*.py'),
             *Path(pyreloc.__file__).parent.glob('_pyreloc*.so')]
    for name in ('SYM_RELOC_EXPORT', 'SYM_OPT'):
        paths.append(Path(os.environ[name]))
    paths.extend(Path(pyreloc.__file__).parents[2].glob('libreloc/libreloc_runtime.so*'))
    return {'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'runtime_revision': os.environ.get('SYM_RUNTIME_REVISION', 'unspecified; see hashes'),
            'source_changes': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
            'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
            'gpu_driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'runtime_package': reloc_torch.__file__,
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def case(name, stack):
    import torch
    import common
    from reloc_torch import RelocBackend
    if name.startswith('kv_'):
        restore = name == 'kv_restore'
        src = torch.randn((33, 16, 64) if restore else (16, 33, 64), device='cpu' if restore else 'cuda:0')
        def reference(x):
            return x.transpose(0, 1).contiguous().to('cuda:0' if restore else 'cpu')
        backend = RelocBackend()
        stack.callback(backend.close)
        compiled = torch.compile(reference, backend=backend, dynamic=True)
        return src, lambda: compiled(src), lambda: reference(src), lambda: src.add_(1), backend.stats
    rows, cols = (1024, 1024) if name == 'weight_1m' else (1024, 4096)
    q = torch.randint(-127, 128, (rows, cols), dtype=torch.int8)
    scale = torch.ones(cols, dtype=torch.float32) / 128
    calibration, _ = common.resolve_calibration('auto', torch.cuda.get_device_name(0))
    report = common.Report(name)
    fetcher = stack.enter_context(common.WeightFetcher(report, calibration))
    def change():
        q.neg_()
        scale.mul_(.5 if scale[0] > .01 else 2)
    def stats():
        return {'resources': fetcher.resource_stats(), 'dispatch': report.data['dispatches'],
                'execution_cache': fetcher.compiled.execution_cache_info()
                if hasattr(fetcher.compiled, 'execution_cache_info') else None}
    return q, lambda: fetcher.fetch(q, scale, 'cuda:0'), lambda: fetcher.reference(q, scale, 'cuda:0'), change, stats


def completed(fn):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter_ns() - start) / 1e6


def main():
    import torch
    import common
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--profile-samples', type=int, default=30)
    p.add_argument('--case', nargs='+', choices=['kv_evict', 'kv_restore', 'weight_1m', 'weight_4m'],
                   default=['kv_evict', 'kv_restore', 'weight_1m', 'weight_4m'])
    p.add_argument('--trace', action='store_true')
    p.add_argument('--cprofile', action='store_true')
    args = p.parse_args()
    if args.samples < 1 or args.profile_samples < 1:
        p.error('sample counts must be positive')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(219)
    result = {'metadata': metadata(), 'cases': {}, 'scope': 'Completed calls include fresh allocation and completion; first Sym frontend KV call includes graph compilation. Weights compile before timing. Source mutations/oracles and owner close are outside timing. Instrumented phases are a separate diagnostic, exclusively partitioning host wall time. Native execution includes completion waits; GPU activity is not additive.'}
    with torch.no_grad():
        for name in args.case:
            with ExitStack() as stack:
                src, sym, ref, change, stats = case(name, stack)
                row = {'source_bytes': src.numel() * src.element_size(), 'first_completed_ms': {}, 'warm': {}}
                for path, fn in [('torch', ref), ('sym', sym)]:
                    actual, elapsed = completed(fn)
                    assert common.same_tensor(actual, ref())
                    row['first_completed_ms'][path] = elapsed
                for _ in range(8):
                    sym(); ref()
                # Interleave policies to limit drift; both get identical fresh values.
                times = {'sym': [], 'torch': []}
                old = None
                for i in range(args.samples):
                    change()
                    expected = ref()
                    for path in (('sym', 'torch') if i % 2 == 0 else ('torch', 'sym')):
                        actual, elapsed = completed(sym if path == 'sym' else ref)
                        assert common.same_tensor(actual, expected)
                        times[path].append(elapsed)
                    if old is not None:
                        assert common.same_tensor(*old)
                    out = sym()
                    old = (out, out.clone())
                row['warm'] = {path: summary(samples) for path, samples in times.items()}
                row['diagnostic_phases'] = []
                if args.trace:
                    torch.cuda.profiler.start()
                with Phases(args.trace) as phases:
                    for i in range(args.profile_samples):
                        actual, sample = phases.call(sym, name)
                        assert common.same_tensor(actual, ref())
                        row['diagnostic_phases'].append(sample)
                if args.trace:
                    torch.cuda.profiler.stop()
                if args.cprofile:
                    prof = cProfile.Profile()
                    prof.enable()
                    for _ in range(30):
                        sym()
                    torch.cuda.synchronize()
                    prof.disable()
                    with args.output.with_suffix('.' + name + '.cprofile.txt').open('w') as stream:
                        pstats.Stats(prof, stream=stream).strip_dirs().sort_stats('tottime').print_stats(35)
                    profile_path = args.output.with_suffix('.' + name + '.cprofile.txt')
                    profile_path.write_text(profile_path.read_text().rstrip() + '\n')
                row['stats'] = stats()
                row['correctness'] = True
                result['cases'][name] = row
                print(name, {k: (v['p50_ms'], v['p95_ms']) for k, v in row['warm'].items()}, flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
