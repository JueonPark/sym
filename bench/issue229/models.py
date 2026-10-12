#!/usr/bin/env python3
"""Reuse #225's models/controls with explicit per-path staging policies."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench/issue225'))
from bench.issue225 import compare
from bench.issue219.request_overhead import Phases
from bench.issue229.transfers import POLICIES, rotate
from reloc_torch import RelocBackend

PATHS = ['torch', 'inductor'] + ['sym_'+p+'_inductor' for p in POLICIES]


class State(compare.PathState):
    def __init__(self, name, lifetime):
        self.name, self.compiles, self.compile_ms, self.dynamic = name, 0, 0., False
        self.backend = None
        if name.startswith('sym'):
            policy = name.split('_')[1]
            self.backend = RelocBackend(compute_backend='inductor', transfer_options=dict(
                gather_threads=8, n_streams=1, n_buffers=2, **POLICIES[policy]))
            lifetime.callback(self.backend.close)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--round', type=int, default=1)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--cases', nargs='+', default=['dlrm_2048', 'gnn_1024', 'llm_decode', 'moe_sparse'])
    args = p.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.manual_seed(225)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    torch.cuda.init()
    result = dict(metadata=compare.metadata(), configuration=vars(args)|{'output': str(args.output)}, cases={})
    result['metadata']['harness_sha256'] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__), Path(compare.__file__), ROOT/'bench/issue225/models.py']}
    with torch.no_grad():
        for case in args.cases:
            torch._dynamo.reset()
            with ExitStack() as lifetime:
                states = [State(n, lifetime) for n in PATHS]
                factory = compare.recommendation if case.startswith(('dlrm', 'gnn')) else compare.offload
                paths, transfers, mutate, extra = factory(case, states, lifetime)
                expected = extra.pop('oracle', None)
                resources = extra.pop('resources', {})
                extra.pop('kv_diagnostics', None)
                row = result['cases'][case] = dict(configuration=extra, paths={})
                from torch._inductor import metrics
                for state in states:
                    kernels = metrics.generated_kernel_count
                    out, ms = compare.completed(paths[state.name])
                    if expected is None:
                        expected = out
                    row['paths'][state.name] = dict(first_including_compile_ms=ms,
                        generated_kernels=metrics.generated_kernel_count-kernels,
                        first_error=compare.check(out, expected))
                for _ in range(5):
                    for fn in paths.values():
                        compare.completed(fn)
                before = {s.name: s.compiles for s in states}
                values, errors = {n: [] for n in paths}, {n: 0. for n in paths}
                for i in range(args.samples):
                    mutate()
                    outputs = {}
                    for n in rotate(paths, i+args.round):
                        outputs[n], ms = compare.completed(paths[n])
                        values[n].append(ms)
                    for n in paths:
                        errors[n] = max(errors[n], compare.check(outputs[n], outputs['torch']))
                for state in states:
                    n = state.name
                    assert state.compiles == before[n], (case, n, 'warm recompile')
                    compare.completed(transfers[n])
                    tx = [compare.completed(transfers[n])[1] for _ in range(args.samples)]
                    row['paths'][n].update(warm=compare.distribution(values[n]), transfer=compare.distribution(tx),
                        max_abs_error=errors[n], warm_compile_callbacks=0, resources=state.stats())
                    if n in resources:
                        row['paths'][n]['weight_resources'] = resources[n]()
                if args.trace:
                    import pyreloc
                    row['phases'] = {}
                    with Phases(True) as phases:
                        for name in ('execute_dispatch_group', 'submit_dispatch_group'):
                            phases.wrap(pyreloc, name, 'native_including_completion')
                        for name in ('prepare_dispatch_group', 'prepare_index_select_program'):
                            phases.wrap(pyreloc, name, 'native_preparation')
                        phases.wrap(pyreloc, 'bind_indexed', 'binding')
                        for n in PATHS:
                            row['phases'][n] = [phases.call(paths[n], f'model/{case}/{n}')[1] for _ in range(5)]
                print(case, {n: round(row['paths'][n]['warm']['p50_ms'], 3) for n in paths}, flush=True)
                args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
