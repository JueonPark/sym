#!/usr/bin/env python3
"""Run the published #263 E2 path with production constructor options only."""
import argparse
import importlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.issue264.support import add_arguments, configure, metadata, policy, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--cases', nargs='+', default=['llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense'])
    parser.add_argument('--computes', nargs='+', choices=('eager', 'inductor'), default=['eager', 'inductor'])
    args = parser.parse_args()
    old, runtime, artifact = configure(args)
    if args.samples < 1 or args.warmup < 0 or any(case not in artifact.e2.CASES for case in args.cases):
        parser.error('invalid samples, warmup or case')
    frozen = importlib.import_module('bench.issue263.models')
    meter_type = importlib.import_module('bench.issue263.instrument').Instrumentation
    original = frozen.allocation_snapshot
    def snapshot(owner=None):
        result = original(owner)
        if owner is not None: result['transfer'] = owner.stats()
        return result
    frozen.allocation_snapshot = snapshot
    import torch
    torch.cuda.init()
    result = dict(configuration=vars(args), metadata=metadata(args, old, runtime, artifact),
        cases={}, complete=False, protocol=dict(dtype='float16', kernel='native', grouping='grouped',
        prefetch='whole', slots=2, scratch_limit=artifact.controls.SCRATCH_LIMIT, tf32=False,
        cudagraphs=False, timing='unprofiled; frozen #263 run_case unchanged'))
    try:
        with policy(args.policy) as selected, meter_type('off') as meter:
            result['optimization'] = selected
            for case in args.cases: frozen.run_case(case, args, artifact, meter, result)
    finally:
        frozen.allocation_snapshot = original
    result['complete'] = True
    save(args.output, result)


if __name__ == '__main__': main()
