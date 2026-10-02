#!/usr/bin/env python3
"""Annotate original workload transfers, untimed verification and model regions.

Trace collection starts after model/input initialization. Transfer NVTX ranges
exclude the pre-timing synchronization, include completion, and are indexed so
analysis can omit the first call exactly as in the earlier measurements.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--example', choices=['dlrm','gnn','llm','moe'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--typed-reuse', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo/'libreloc/python/examples/workloads'))
    import torch
    import common
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    from python_ranges import install
    install()
    from reloc_torch import TransferResources, dispatch
    owner = TransferResources() if args.typed_reuse else None
    original_execute = dispatch.execute_typed_transfer
    decisions = []
    def execute(request, **kwargs):
        if owner is not None:
            kwargs.setdefault('resources', owner)
        result = original_execute(request, **kwargs)
        decisions.append(dict(implementation=result.report['implementation'],
            source_bytes=result.report['source_bytes'], wire_bytes=result.report['wire_bytes'],
            destination_bytes=result.report['destination_bytes'], staging=request.staging))
        return result
    dispatch.execute_typed_transfer = execute
    counters = defaultdict(int)
    started = False

    @contextmanager
    def span(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def begin():
        nonlocal started
        if not started:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            started = True

    def timed(clock, path, kind, fn, *values):
        key = (path, kind)
        index = counters[key]
        counters[key] += 1
        clock._synchronize()
        with span(f'transfer/{path}/{kind}/{index}'):
            start = time.perf_counter()
            result = fn(*values)
            clock._synchronize()
            elapsed = (time.perf_counter() - start) * 1e3
        return result, elapsed

    def reference(report, clock, kind, fn):
        begin()
        def call(*values):
            result, elapsed = timed(clock, 'torch', kind, fn, *values)
            report.time('torch', kind, elapsed)
            return result
        return call

    def sym(report, clock, kind, fn, oracle, count_bytes=True):
        begin()
        def call(source, *values):
            result, elapsed = timed(clock, 'sym', kind, fn, source, *values)
            report.time('sym', kind, elapsed)
            with span(f'verification/sym/{kind}'):
                expected = oracle(source, *values)
                report.check(f'{kind}_equal', common.same_tensor(result, expected),
                             lambda: common.difference(result, expected))
            if count_bytes:
                size = result.numel() * result.element_size()
                report.add_bytes(kind, source.numel() * source.element_size(), size, size)
            return result
        return call

    common.reference_transfer = reference
    common.sym_transfer = sym
    names = {'dlrm':'dlrm_embeddings','gnn':'gnn_minibatch','llm':'llm_offload','moe':'moe_experts'}
    module = importlib.import_module(names[args.example])
    sizes = module.SIZES['default']
    if args.example == 'dlrm':
        target, method, calls = module.DLRM, 'predict', len(sizes['batches'])
    elif args.example == 'gnn':
        target, method, calls = module.GraphSAGE, 'forward', sizes['batches']
    elif args.example == 'llm':
        target, method, calls = module, 'generate', 1
    else:
        target, method, calls = module.OffloadedMoE, 'forward', len(sizes['batches'])
    original = getattr(target, method)
    model_calls = 0
    def model(*values, **keywords):
        nonlocal model_calls
        path = 'torch' if model_calls < calls else 'sym'
        index = model_calls % calls
        model_calls += 1
        with span(f'model/{path}/{args.example}/{index}'):
            return original(*values, **keywords)
    setattr(target, method, model)
    old_finish = common.Report.finish
    def finish(report, output=None):
        torch.cuda.synchronize()
        if started:
            torch.cuda.profiler.stop()
        report.data['raw_transfer_samples_ms'] = report._samples
        report.data['typed_decisions'] = decisions
        report.data['typed_reuse'] = args.typed_reuse
        if owner is not None:
            report.data['direct_typed_resources'] = owner.stats()['typed']
            owner.close()
        report.data['profiling'] = {'torch_threads':8, 'model_regions_per_path':calls,
                                   'transfer_range_includes_completion':True,
                                   'first_transfer_of_each_kind_excluded_in_analysis':True}
        return old_finish(report, output)
    common.Report.finish = finish
    sys.argv = [names[args.example], '--output', str(args.output),
                '--devices' if args.example == 'moe' else '--device', 'cuda:0']
    return common.run_example(module.main)


if __name__ == '__main__':
    raise SystemExit(main())
