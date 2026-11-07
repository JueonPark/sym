#!/usr/bin/env python3
"""Separate completed timing, CPU submission timing and construction diagnostics."""
import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.issue264.support import add_arguments, configure, metadata, policy, save


def memory(owner):
    import torch
    fields = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ('Rss:', 'Pss:', 'Private_Dirty:', 'Private_Clean:'):
            fields[parts[0][:-1].lower() + '_bytes'] = int(parts[1]) * 1024
    return dict(process=fields, torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(), transfer=owner.stats())


def run_case(name, args):
    import torch
    import pyreloc
    from reloc_torch import CompilerClient, PreparedWireWeights
    from reloc_torch.fast import FastTransfers, FastWeights
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, dense_strides
    torch.manual_seed(264)
    shapes = {'weights_small': [(64, 128)] * 2,
              'weights_decode': [(1024, 1024)] * 4 + [(1024, 4096), (4096, 1024)]}
    pairs = [(torch.randint(-127, 128, shape, dtype=torch.int8), torch.linspace(.001, .01, shape[1]))
             for shape in shapes.get(name, [])]
    source = torch.randn(64, 16, 32) if name == 'kv_restore' else None
    references = ([(q.float() * s).t().contiguous().half().cuda() for q, s in pairs] if pairs
                  else [source.transpose(0, 1).contiguous().cuda()] * 2)
    torch.cuda.synchronize()
    with ExitStack() as life:
        start = time.perf_counter_ns()
        compiler = CompilerClient.from_environment()
        fast = life.enter_context(FastTransfers(compiler=compiler, contexts=2, gather_threads=8,
                                               max_scratch_bytes=64 << 20))
        if pairs:
            weights = life.enter_context(PreparedWireWeights(compiler=compiler, dtype='float16'))
            names = [str(i) for i in range(len(pairs))]
            for key, (q, s) in zip(names, pairs): weights.prepare(key, q, s)
            loader = FastWeights(fast, weights)
            call = lambda: loader.submit(names)
            explicit_items = lambda: loader._items(names)
            extra = {'gather_threads': 1}
            source_bytes = sum(q.nbytes + s.nbytes for q, s in pairs)
        else:
            src, dst = tuple(map(Const, source.shape)), (Const(16), Const(64), Const(32))
            def spec(shape): return TensorSpec(shape, dense_strides(shape), Const(0), 'float32')
            program = fast.layout(Recipe(spec(src), (Transpose((1, 0, 2)),), spec(dst), 'h2d'), source)
            items = [(program, source)] * 2
            extra = dict(device_layout=True, gather_threads=1)
            call = lambda: fast.submit(items, **extra)
            explicit_items = lambda: items
            source_bytes = source.nbytes
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter_ns() - start) / 1e6
        before_first = memory(fast)
        held = deque(maxlen=2)
        def completed():
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            completion = call()
            outputs = completion.wait()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            held.append(outputs)
            return elapsed
        first = completed()
        assert all(torch.equal(a, b) for a, b in zip(held[-1], references))
        for _ in range(args.warmup): completed()
        before_warm = memory(fast)
        samples = [completed() for _ in range(args.samples)]
        after_warm = memory(fast)
        a, b = before_warm['transfer']['resources'], after_warm['transfer']['resources']
        assert all(a[k] == b[k] for k in ('context_creations', 'device_allocations', 'host_allocations'))
        assert all(torch.equal(a, b) for outputs in held for a, b in zip(outputs, references))
        # CPU submission timing has no construction wrappers or detailed reports.
        submit_samples = []
        for _ in range(args.submit_samples):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            completion = call()
            submit_samples.append((time.perf_counter_ns() - start) / 1e6)
            completion.wait()
            torch.cuda.synchronize()
        # Separate counts exercise actual construction APIs. Prepared native
        # constructions/rebinds come from the binder, rather than inferred hits.
        import reloc_torch.fast as module
        counts = dict(legacy_group_constructions=0, legacy_view_descriptors=0)
        undo = []
        def wrap(owner, key):
            original = getattr(owner, key)
            def counted(entries):
                counts['legacy_group_constructions'] += 1
                counts['legacy_view_descriptors'] += 2 * len(entries)
                return original(entries)
            undo.append((owner, key, original))
            setattr(owner, key, counted)
        if module._native_views is not None: wrap(module._native_views, 'group_from_tensors')
        wrap(pyreloc, 'prepare_dispatch_group')
        diagnostic_before = fast.stats()
        try:
            for _ in range(10): call().wait()
        finally:
            for owner, key, original in undo: setattr(owner, key, original)
        diagnostic_after = fast.stats()
        # Explicit preparation is a separate diagnostic, after all timing.
        explicit_ms, explicit_first = None, None
        if hasattr(fast, 'prepare_group'):
            items = explicit_items()
            start = time.perf_counter_ns()
            handle = fast.prepare_group([p for p, _ in items])
            explicit_ms = (time.perf_counter_ns() - start) / 1e6
            start = time.perf_counter_ns()
            handle.submit([s for _, s in items], **extra).wait()
            torch.cuda.synchronize()
            explicit_first = (time.perf_counter_ns() - start) / 1e6
            handle.close()
        row = dict(setup_ms=setup_ms, first_ms=first, samples_ms=samples, submit_samples_ms=submit_samples,
                   explicit_prepare_ms=explicit_ms, explicit_first_ms=explicit_first, source_bytes=source_bytes,
                   before_first=before_first, before_warm=before_warm, after_warm=after_warm,
                   diagnostics=dict(requests=10, before=diagnostic_before, after=diagnostic_after, **counts),
                   members=len(references), correctness=True)
    row['native_after_close'] = fast._resources.stats()['typed']
    assert row['native_after_close']['retained_bytes'] == row['native_after_close']['live_bytes'] == 0
    assert all(torch.equal(a, b) for outputs in held for a, b in zip(outputs, references))
    row['held_output_checks_after_close'] = len(held) * len(references)
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--submit-samples', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--cases', nargs='+', choices=('weights_small', 'weights_decode', 'kv_restore'),
                        default=['weights_small', 'weights_decode', 'kv_restore'])
    args = parser.parse_args()
    if min(args.samples, args.submit_samples) < 1 or args.warmup < 1:
        parser.error('positive sample/warmup counts required')
    old, runtime, artifact = configure(args)
    import torch
    torch.cuda.init()
    result = dict(configuration=vars(args), metadata=metadata(args, old, runtime, artifact),
                  cases={}, complete=False, protocol=dict(timing='unprofiled completed groups',
                    submit_timing='separate unwrapped submissions', diagnostics='separate ten requests',
                    dtype='float16 weights; float32 KV pairs', contexts=2, slots=2, scratch_bytes=128 << 20,
                    memory='process RSS/PSS includes allocators; native retained bytes separately scoped'))
    with torch.no_grad(), policy(args.policy) as selected:
        result['optimization'] = selected
        for case in args.cases:
            result['cases'][case] = run_case(case, args)
            save(args.output, result)
    result['complete'] = True
    save(args.output, result)


if __name__ == '__main__': main()
