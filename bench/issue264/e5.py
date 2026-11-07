#!/usr/bin/env python3
"""Frozen E5 lifecycle/burst programs, with identical constructor policy selection."""
import argparse
import importlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.issue264.support import add_arguments, configure_candidate, dependency, metadata, policy, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--kind', choices=('lifecycle', 'concurrency'), required=True)
    parser.add_argument('--case', choices=('llm_decode', 'moe_sparse'), required=True)
    parser.add_argument('--runtime', choices=('sym', 'torch'), required=True)
    parser.add_argument('--compute', choices=('eager', 'inductor'), default='eager')
    parser.add_argument('--callers', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--batches', type=int, default=30)
    args = parser.parse_args()
    for key in ('artifact', 'harness', 'build', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    sys.path.insert(0, str(args.build / 'python'))
    old = dependency(args.harness)
    frozen = importlib.import_module('bench.issue263.e5')
    forward = ['--artifact', str(args.artifact), '--build', str(args.build), '--output', str(args.output),
               '--round', str(args.round), '--variant', 'baseline', '--instrument', 'off',
               '--case', args.case, '--runtime', args.runtime]
    if args.policy != 'baseline': forward.append('--allow-new-build')
    if args.kind == 'lifecycle': forward += ['--compute', args.compute, '--requests', str(args.requests)]
    else: forward += ['--callers', str(args.callers), '--batches', str(args.batches)]
    argv = sys.argv
    original_configure = frozen.configure
    if args.policy != 'baseline':
        frozen.configure = lambda values, threads=None: configure_candidate(values, old, threads=threads)
    try:
        with policy(args.policy) as selected:
            sys.argv = [str(Path(frozen.__file__)), *forward]
            frozen.main(args.kind)
        result = json.loads(args.output.read_text())
        # The frozen loader has already checked inputs and native retirement.
        artifact = old.load_artifact(args.artifact, repo=REPO)
        previous = result['metadata']
        result.update(configuration=vars(args), optimization=selected,
                      metadata=metadata(args, old, previous, artifact))
        save(args.output, result)
    finally:
        sys.argv = argv
        frozen.configure = original_configure


if __name__ == '__main__': main()
