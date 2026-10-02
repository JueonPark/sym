#!/usr/bin/env python3
"""Run an unchanged PR #162 example and preserve its raw completed-transfer samples.

This measures transfers within the original example, not whole-model throughput.
The example's Torch-first execution and validation remain unchanged. The optional
None policy is a matched control for the latest frontend's layout-resource cache.
"""
import argparse
from collections.abc import Mapping
import importlib
import os
from pathlib import Path
import sys


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--example', choices=['dlrm', 'gnn', 'llm', 'moe'], required=True)
    parser.add_argument('--policy', choices=['default', 'none'], default='default')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--typed-reuse', action='store_true', help='Pass an explicit owner to direct dispatch calls; frontend AUTO is unchanged.')
    parser.add_argument('--pinning', choices=['auto', 'pinned', 'pageable'])
    parser.add_argument('--min-pinned-bytes', type=int)
    parser.add_argument('--staged-upload', action='store_true')
    parser.add_argument('--cold-metadata', action='store_true')
    args = parser.parse_args()
    workloads = args.repo / 'libreloc/python/examples/workloads'
    sys.path.insert(0, str(workloads))
    import torch
    import reloc_torch
    import common

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from reloc_torch import dispatch, transport
    direct_owner = reloc_torch.TransferResources() if args.typed_reuse else None
    typed_execute = dispatch.execute_typed_transfer
    layout_execute = transport.execute_transfer
    def execute_typed(request, **kwargs):
        if args.typed_reuse:
            kwargs.setdefault('resources', direct_owner)
        if args.pinning: kwargs['pinning'] = args.pinning
        if args.min_pinned_bytes is not None: kwargs['min_pinned_bytes'] = args.min_pinned_bytes
        if args.staged_upload: kwargs['direct_dense_upload'] = False
        return typed_execute(request, **kwargs)
    def execute_layout(request, **kwargs):
        if args.pinning: kwargs['pinning'] = args.pinning
        if args.min_pinned_bytes is not None: kwargs['min_pinned_bytes'] = args.min_pinned_bytes
        return layout_execute(request, **kwargs)
    dispatch.execute_typed_transfer = execute_typed
    transport.execute_transfer = execute_layout
    if args.cold_metadata:
        from reloc_torch.artifact import CompiledRecipe
        original_bind = CompiledRecipe.bind_values
        def bind_cold(compiled, source):
            compiled.__dict__.pop('_metadata_binder', None)
            compiled.__dict__.pop('decoded_plan', None)
            compiled.__dict__.pop('_destination_metadata', None)
            return original_bind(compiled, source)
        CompiledRecipe.bind_values = bind_cold
    backends = []
    backend_class = reloc_torch.RelocBackend

    def backend(*positional, **keywords):
        if args.policy == 'none':
            keywords.setdefault('transfer_resources', None)
        owner = backend_class(*positional, **keywords)
        backends.append(owner)
        return owner

    reloc_torch.RelocBackend = backend
    old_stats = common.Report.set_backend_stats
    old_finish = common.Report.finish

    def save_stats(report, stats):
        report.data['resource_snapshot_before_close'] = plain(stats.get('transfer_resources'))
        old_stats(report, stats)

    def finish(report, output=None):
        report.data['candidate_options'] = dict(typed_reuse=args.typed_reuse,
            pinning=args.pinning, min_pinned_bytes=args.min_pinned_bytes, staged_upload=args.staged_upload, cold_metadata=args.cold_metadata)
        if direct_owner is not None:
            report.data['direct_typed_resources'] = direct_owner.stats()['typed']
            direct_owner.close()
        report.data['raw_transfer_samples_ms'] = report._samples
        report.data['measurement'] = {
            'policy': args.policy,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'torch': torch.__version__,
            'python': sys.version,
            'timing_scope': 'CPU wall time of transfer function through device completion; allocation, conversion, validation/binding and any first-call compile inside the function are included. Model computation, data preparation and byte-exact comparison are outside the transfer timer.',
            'order': 'Unchanged example: entire Torch pass precedes Sym pass; untimed Torch oracle checks follow each Sym transfer.',
            'summary_rule': 'Sum of all transfers after omitting the first call of each kind; this mixes changing shapes and is not a fixed-shape steady-state microbenchmark.',
        }
        report.data['backend_resources_after_close'] = [
            plain(owner.stats().get('transfer_resources')) for owner in backends]
        return old_finish(report, output)

    common.Report.set_backend_stats = save_stats
    common.Report.finish = finish
    modules = {'dlrm': 'dlrm_embeddings', 'gnn': 'gnn_minibatch',
               'llm': 'llm_offload', 'moe': 'moe_experts'}
    module = importlib.import_module(modules[args.example])
    sys.argv = [str(workloads / (modules[args.example] + '.py')), '--output', args.output]
    sys.argv += ['--devices' if args.example == 'moe' else '--device', args.devices]
    return common.run_example(module.main)


if __name__ == '__main__':
    raise SystemExit(main())
