#!/usr/bin/env python3
"""Matched E2 requests with isolated interventions and separate trace modes."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.issue263.instrument import Instrumentation
from bench.issue263.interventions import Intervention
from bench.issue263.support import (add_arguments, allocation_snapshot, configure,
                                    load_artifact, provenance, save)


def run_case(case, args, artifact, meter, result):
    import torch
    from reloc_torch import CompilerClient, PreparedWireWeights
    from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap
    e2, controls = artifact.e2, artifact.controls
    e2.legacy.COMPUTE_DTYPE = 'float16'
    torch.manual_seed(225)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    with torch.no_grad(), ExitStack() as life:
        model, x, sizes, pairs, oracle = e2.make_model(case, torch.float16)
        names = {id(q): str(i) for i, (q, s) in enumerate(pairs)}
        images, torch_snapshot_ms = e2.legacy.completed(lambda: {
            names[id(q)]: e2.legacy.host_pack(q, s) for q, s in pairs})
        compiler = CompilerClient.from_environment()
        fast = life.enter_context(FastTransfers('cuda:0', compiler=compiler, gather_threads=8,
                                  contexts=2, max_scratch_bytes=controls.SCRATCH_LIMIT // 2, pinning='pinned'))
        weights = life.enter_context(PreparedWireWeights(compiler=compiler, dtype='float16'))
        _, sym_snapshot_ms = e2.legacy.completed(lambda: [weights.prepare(names[id(q)], q, s) for q, s in pairs])
        fw = FastWeights(fast, weights)
        sym_kv = KVCacheSwap(fast, 16, 64, dtype='float16')
        torch_kv = controls.TorchKV()
        life.callback(sym_kv.drain)
        life.callback(torch_kv.drain)
        states = {kind: e2.legacy.PathState('torch' if kind == 'eager' else 'inductor', life, dynamic=True)
                  for kind in args.computes}
        originals = ((e2.legacy.models.decoder_layer, e2.legacy.models.embed, e2.legacy.models.logits)
                     if case.startswith('llm') else
                     (e2.legacy.models.route, e2.legacy.models.expert, e2.legacy.models.combine))
        functions = {kind: tuple(meter.consumer(state.compile(fn), fn.__name__) for fn in originals)
                     for kind, state in states.items()}
        producers = {}
        for kind in args.computes:
            producer = controls.TorchProducer(images, lambda q, s: controls.fused_dequant(q, s, torch.float16), torch.float16)
            life.callback(producer.close)
            producers[kind] = producer
        layer_names = ([[names[id(item[k][0])] for k in e2.legacy.MATRICES] for item in model.layers]
                       if case.startswith('llm') else None)
        arms = [e2.Arm(runtime, compute) for compute in args.computes for runtime in ('torch', 'sym')]
        order = arms[args.round % len(arms):] + arms[:args.round % len(arms)]
        paths = {}
        for arm in arms:
            producer = fw if arm.runtime == 'sym' else producers[arm.compute]
            swap = controls.Swaps(sym_kv if arm.runtime == 'sym' else torch_kv, 'grouped')
            paths[arm.name] = (e2.gpt_path(model, x, sizes, producer, swap, functions[arm.compute], layer_names, 'whole')
                               if case.startswith('llm') else
                               e2.moe_path(model, x, producer, functions[arm.compute], names))
        row = result['cases'][case] = dict(sizes=sizes, arms={}, setup_order=[a.name for a in order],
                    torch_snapshot_ms=torch_snapshot_ms, sym_snapshot_ms=sym_snapshot_ms)

        def call(arm, phase, ordinal):
            label = f'e2/{case}/{arm.runtime}/{args.variant}/{arm.compute}/{phase}/{ordinal}'
            with meter.request(label):
                actual, elapsed = e2.legacy.completed(paths[arm.name])
            error = e2.legacy.check(actual, oracle)
            return elapsed, error, label

        for arm in order:
            elapsed, error, label = call(arm, 'first', 0)
            row['arms'][arm.name] = dict(factors=asdict(arm), first_ms=elapsed, first_error=error,
                                         samples_ms=[], labels=[], checks=0, max_abs_error=0.)
        for i in range(args.warmup):
            for arm in order:
                call(arm, 'warmup', i)
        before_compiles = {kind: state.compiles for kind, state in states.items()}
        for i in range(args.samples):
            rotated = order[i % len(order):] + order[:i % len(order)]
            for arm in rotated:
                elapsed, error, label = call(arm, 'warm', i)
                entry = row['arms'][arm.name]
                entry['samples_ms'].append(elapsed)
                entry['labels'].append(label)
                entry['checks'] += 1
                entry['max_abs_error'] = max(entry['max_abs_error'], error)
            if (i + 1) % 10 == 0:
                save(args.output, result)
        assert all(state.compiles == before_compiles[kind] for kind, state in states.items()), 'warm recompile'
        row['warm_compile_callbacks'] = 0
        row['compile_stats'] = {kind: state.stats() for kind, state in states.items()}
        for arm in order:
            sym_kv.drain()
            torch_kv.drain()
            torch.cuda.synchronize()
            before = allocation_snapshot(fast if arm.runtime == 'sym' else None)
            elapsed, error, label = call(arm, 'allocation_diagnostic', 0)
            after = allocation_snapshot(fast if arm.runtime == 'sym' else None)
            row['arms'][arm.name]['allocation_diagnostic'] = dict(before=before, after=after, completed_ms=elapsed,
                label=label, note='Separate completed request. Torch allocation requests include caching-allocator reuse; '
                'Python live blocks are a net count, not gross allocations. Native counters and Torch bytes have different scopes.')
        row['dynamo_counters'] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items() if k in ('stats', 'graph_break')}
        print(case, args.variant, args.instrument,
              {name: round(statistics.median(x['samples_ms']), 3) for name, x in row['arms'].items()}, flush=True)
    row['native_after_close'] = fast._resources.stats()['typed']
    assert row['native_after_close']['retained_bytes'] == row['native_after_close']['live_bytes'] == 0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    save(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--cases', nargs='+', default=['llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense'])
    parser.add_argument('--computes', nargs='+', choices=('eager', 'inductor'), default=['eager', 'inductor'])
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('samples must be positive and warmup nonnegative')
    runtime = configure(args)
    artifact = load_artifact(args.artifact)
    if any(case not in artifact.e2.CASES for case in args.cases):
        parser.error('unsupported E2 case')
    import torch
    torch.cuda.init()
    result = dict(configuration=vars(args), metadata=provenance(args, artifact, runtime), cases={}, complete=False,
        protocol=dict(dtype='float16', kernel='native', grouping='grouped', prefetch='whole', slots=2,
                      scratch_limit=artifact.controls.SCRATCH_LIMIT, tf32=False, cudagraphs=False,
                      timing='unprofiled when instrument=off; profiler use is additionally recorded by the runner'))
    with Intervention(args.variant, qualified_metadata=True) as intervention, Instrumentation(args.instrument) as meter:
        result['intervention'] = intervention.description()
        meter.install(artifact.controls)
        for case in args.cases:
            run_case(case, args, artifact, meter, result)
        result['host_phases'] = meter.records
    result['complete'] = True
    save(args.output, result)


if __name__ == '__main__':
    main()
