#!/usr/bin/env python3
"""Apply shared host-phase definitions to Torch and Sym and audit native coverage.

Preparation/guards is an observed subset. Uninstrumented checks can remain in
operator/host residuals; these residuals must never be labeled zero guard cost.
TensorIterator fast_set_up includes output setup, so it is not pure guard time.
"""
import argparse
from collections import defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location('base', Path(__file__).parents[1]/'workload_profile/analyze.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PHASES = ['preparation_guards', 'output_setup_allocation', 'frontend_execution',
          'torch_operator_other', 'native_other', 'outer_host', 'cpu_transform',
          'staging_management', 'device_management', 'cuda_copy_api', 'cuda_wait_api', 'cuda_other_api']


def canonical(name):
    return name.split(', op_id =')[0]


def scope_phase(raw):
    name = canonical(raw)
    if name == 'torch.native.cpu_copy_operator':
        return 50, 'cpu_transform'
    if name in ('reloc.typed.host_program', 'reloc.gather.work', 'torch.native.cpu_loop'):
        return 70, 'cpu_transform'
    if name in ('reloc.staging.allocate', 'reloc.staging.free', 'reloc.scratch.host_acquire', 'reloc.scratch.host_release'):
        return 70, 'staging_management'
    if name in ('reloc.scratch.device_acquire', 'reloc.scratch.device_release'):
        return 70, 'device_management'
    if name.startswith('aten::empty') or name in ('torch.native.ti.allocate_or_resize_outputs', 'torch.native.ti.fast_set_up'):
        return 65, 'output_setup_allocation'
    if name.startswith('torch.native.ti.'):
        return 60, 'preparation_guards'
    if name in ('aten::t', 'aten::transpose', 'aten::as_strided', 'aten::detach', 'detach',
                'aten::resolve_conj', 'aten::resolve_neg', 'aten::is_pinned', 'aten::sym_size',
                'TorchDynamo Cache Lookup'):
        return 60, 'preparation_guards'
    if name.startswith('aten::'):
        return 30, 'torch_operator_other'
    if name in ('python.prepare_typed_transfer', 'python.layout.prepare_transfer',
                'python.runtime.bind_symbols', 'python.runtime.destination_descriptor',
                'python.adapter.preflight', 'python.recheck', 'python._parameter_snapshot',
                'python.native.prepare_dispatch', 'python.native.make_transfer',
                'python.runtime.verify_result'):
        return 40, 'preparation_guards'
    if name in ('python.native.execute_dispatch', 'python.native.execute_transfer'):
        return 20, 'native_other'
    if name in ('python.execute_typed_transfer', 'python.layout.execute_transfer', 'python.adapter.execute'):
        return 10, 'frontend_execution'
    return None


def exclusive(span, apis, scopes):
    # Priority 100 ensures CUDA API time is never double-counted inside a scope.
    pieces = [(span['start'], span['end'], 0, 'outer_host')]
    pieces += [(r['start'], r['end'], 100, base.api_phase(r['name'])) for r in apis]
    pieces += [(r['start'], r['end'], *scope_phase(r['name'])) for r in scopes if scope_phase(r['name'])]
    boundaries = sorted({t for row in pieces for t in row[:2]})
    totals = dict.fromkeys(PHASES, 0.0)
    timeline = []
    for begin, end in zip(boundaries, boundaries[1:]):
        label = max((p for p in pieces if p[0] <= begin and end <= p[1]), key=lambda p: p[2])[3]
        totals[label] += (end-begin)/1e6
        if timeline and timeline[-1]['phase'] == label and timeline[-1]['end_ns'] == begin:
            timeline[-1]['end_ns'] = end
        else:
            timeline.append(dict(phase=label, start_ns=begin, end_ns=end))
    assert abs(sum(totals.values())-(span['end']-span['start'])/1e6) < 1e-6
    for row in timeline:
        row['start_ms'] = (row.pop('start_ns')-span['start'])/1e6
        row['end_ms'] = (row.pop('end_ns')-span['start'])/1e6
    return totals, timeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--coarse', action='store_true', help='Only build/CPU-loop hooks, for perturbation control.')
    args = p.parse_args()
    base.PHASES = PHASES
    base.exclusive = exclusive
    summary = {}
    for file in sorted(args.directory.glob('*.sqlite')):
        result = base.analyze(file)
        # Exact symbol counts and durations, in addition to disjoint categories.
        result['scope_totals'] = {}
        for who in ('torch', 'sym'):
            aggregate = defaultdict(lambda: dict(count=0, inclusive_ms=0.0))
            for row in result['transfers']:
                if row['path'] != who or row['index'] == 0:
                    continue
                normalized = defaultdict(lambda: dict(count=0, ms=0.0))
                for name, value in row['inclusive_scopes'].items():
                    key = canonical(name)
                    normalized[key]['count'] += value['count']
                    normalized[key]['ms'] += value['ms']
                    aggregate[key]['count'] += value['count']
                    aggregate[key]['inclusive_ms'] += value['ms']
                row['inclusive_scopes'] = dict(normalized)
                if who == 'torch' and row['kind'] in ('pooled_embeddings', 'node_features', 'kv_restore'):
                    expected = 2 if row['kind'] == 'pooled_embeddings' else 1
                    assert normalized['torch.native.cpu_copy_operator']['count'] == expected, 'CPU-copy observer coverage'
            result['scope_totals'][who] = dict(aggregate)
        torch_scopes = result['scope_totals']['torch']
        assert torch_scopes['torch.native.ti.build']['count'] > 0, 'No TensorIterator interposition'
        for method in (() if args.coarse else ('compute_mem_overlaps', 'compute_shape', 'compute_types')):
            # All valid, non-empty workload iterators traverse these checks.
            assert torch_scopes['torch.native.ti.'+method]['count'] == torch_scopes['torch.native.ti.build']['count']
        with gzip.open(file.with_suffix('.analysis.json.gz'), 'wt') as stream:
            json.dump(result, stream)
        summary[file.stem] = {key: result[key] for key in ('totals', 'by_kind', 'model', 'scope_totals')}
        print(file.stem, {who: {key: round(value, 3) for key, value in row['exclusive_ms'].items()}
                          for who, row in result['totals'].items()}, flush=True)
    (args.directory/'shared-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
