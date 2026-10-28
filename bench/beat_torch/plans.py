#!/usr/bin/env python3
"""Account for the compiler's contribution, recipe by recipe.

For every compiled recipe the fast-lane arms execute (GPT/MoE weight
dequantize, decode K/V evict and restore, the GraphSAGE gather), the typed
DLRM recipe the earlier ``sym``/``sym_inductor`` arms use, and one recipe
where the canonicalizer provably matters (an identity permutation that
becomes a pure view), compile with the canonicalizer on and off and report:

- what the compiler produced: wire version, stage count, plan rank as bound
  at the workload shape (canonical / raw), ``no_copy``;
- what it cost once: export (a fresh exporter process), decode + bind +
  native template preparation;
- what the runtime selected from the plan, and the completed per-call
  latency of the same transfer in both modes at the workload shape.

    python bench/beat_torch/plans.py [--samples 30] [--json out.json]

The DLRM pooled producer of ``sym_fast`` has no row here: it is synthesized
natively (``prepare_pooled_program``) and never passes through the compiler.
Both modes still run foldChainOp() and plan construction; this tests only
the subsequent canonicalization pass, not folding as a whole. A no-copy
layout proof does not eliminate the H2D transfer.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'libreloc/python/tests/torch_frontend'))
from reloc_torch import CompilerClient
from reloc_torch.artifact import UnsupportedRecipe
from reloc_torch.fast import FastTransfers, KVCacheSwap
from reloc_torch.recipe import BindingParam, Dequantize, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides
from test_fast import dlrm_recipe, gather_recipe


def ms(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1e6


def completed_us(fn, samples, warmup=5):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(values)


def weight_recipe():
    rows, cols = Symbol('out'), Symbol('in')
    spec = lambda dtype: TensorSpec((rows, cols), dense_strides((rows, cols)), Const(0), dtype)
    return Recipe(spec('int8'), (Dequantize('float32', BindingParam('scale', 'float32', (rows,)), None, 0,
                                            'affine'),), spec('float32'), 'h2d')


def identity_recipe():
    shape = (Symbol('s0'), Const(128))
    spec = TensorSpec(shape, dense_strides(shape), Const(0), 'float32')
    return Recipe(spec, (Transpose((0, 1)),), spec, 'h2d')


def plan_facts(compiled, program):
    """Exported plan axes (what the compiler wrote), bound axes (what the
    runtime coalesced at bind time), stage count and no_copy."""
    plan = compiled.decoded_plan
    if compiled.indexed:
        return dict(wire=compiled.wire_version, stages=1, exported_axes=None, axes=None, no_copy=None,
                    note='indexed plans are not canonicalized')
    if compiled.typed:
        bound = program.bound
        return dict(wire=compiled.wire_version, stages=len(bound.stages), exported_axes=plan.layout_axes,
                    axes=len(bound.layout.extents), no_copy=False)
    return dict(wire=compiled.wire_version, stages=0, exported_axes=plan.num_axes, axes=len(program.bound.extents),
                no_copy=bool(plan.no_copy))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--json')
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    gen = torch.Generator().manual_seed(225)
    q = torch.randint(-127, 128, (3072, 1024), dtype=torch.int8)
    scale = torch.rand(3072, generator=gen) * 0.05 + 1e-3
    caches = [torch.randn(16, 12, 64, device='cuda:0') for _ in range(8)]
    table = torch.randn(500_000, 128, generator=gen)
    index = torch.randint(500_000, (9000,), generator=gen).sort().values
    dlrm_source = torch.randn(26, 512, 64, generator=gen)
    ident = torch.randn(2048, 128, generator=gen)
    results = {}
    with torch.no_grad():
        for mode in ('canonical', 'raw'):
            rows = {}
            fresh = lambda: CompilerClient.from_environment(canonicalize=mode == 'canonical')
            fast = FastTransfers('cuda:0', compiler=fresh(), gather_threads=8, contexts=2)
            with fast:
                # One-time costs (CUDA context, pools, first artifact decode) are
                # absorbed by a throwaway program before anything is timed.
                fast.typed(dlrm_recipe(), dlrm_source).execute(dlrm_source)
                torch.cuda.synchronize()
                # GPT/MoE weights: int8 [out, in] + per-row scale -> f32 on the GPU.
                compiled, export = ms(lambda: fresh().compile(weight_recipe()))
                program, prepare = ms(lambda: fast.typed(weight_recipe(), q, parameters={'scale': scale},
                                                         implementation='cpu_stages_cuda_stages@0'))
                rows['gpt_weight_3072x1024'] = dict(plan_facts(compiled, program), export_ms=export, prepare_ms=prepare,
                                                    row=program.selection['implementation'],
                                                    per_call_us=completed_us(lambda: program.execute(q), args.samples))
                # Decode K/V: [16, S, 64] <-> [S, 16, 64], eight caches per group.
                swap = KVCacheSwap(fast, 16, 64)
                compiled, export = ms(lambda: fresh().compile(swap._evict_recipe))
                stored, prepare = ms(lambda: swap.evict(caches))
                evict_program = swap._program(swap._evict_recipe, caches[0])
                rows['kv_evict_16x12x64_x8'] = dict(plan_facts(compiled, evict_program), export_ms=export, prepare_ms=prepare,
                                                    row='layout_cuda (group_device_layout)',
                                                    per_call_us=completed_us(lambda: swap.evict(caches), args.samples))
                compiled, export = ms(lambda: fresh().compile(swap._restore_recipe))
                restored, prepare = ms(lambda: swap.restore(stored))
                torch.cuda.synchronize()
                restore_program = swap._program(swap._restore_recipe, stored[0])
                rows['kv_restore_12x16x64_x8'] = dict(plan_facts(compiled, restore_program), export_ms=export,
                                                      prepare_ms=prepare, row='layout_cuda (group_device_layout)',
                                                      per_call_us=completed_us(lambda: swap.restore(stored), args.samples))
                swap.drain()
                # GraphSAGE gather: 9k rows of a [500k, 128] table -> f16.
                compiled, export = ms(lambda: fresh().compile(gather_recipe()))
                program, prepare = ms(lambda: fast.indexed(gather_recipe()))
                program.execute(table, index)
                rows['gnn_gather_9000x128'] = dict(plan_facts(compiled, program), export_ms=export, prepare_ms=prepare,
                                                   row='cpu_reference (indexed rows)',
                                                   per_call_us=completed_us(lambda: program.execute(table, index),
                                                                            args.samples))
                # DLRM typed recipe (sym / sym_inductor arms): [26, B, 64] f32 -> [B, 26, 64] f16.
                compiled, export = ms(lambda: fresh().compile(dlrm_recipe()))
                program, prepare = ms(lambda: fast.typed(dlrm_recipe(), dlrm_source))
                rows['dlrm_layout_cast_26x512x64'] = dict(plan_facts(compiled, program), export_ms=export,
                                                          prepare_ms=prepare, row=program.selection['implementation'],
                                                          per_call_us=completed_us(lambda: program.execute(dlrm_source),
                                                                                   args.samples))
                # Identity layout still requires H2D. If no relocation program
                # is needed, time direct transport instead of recording zero.
                compiled, export = ms(lambda: fresh().compile(identity_recipe()))
                try:
                    program, prepare = ms(lambda: fast.layout(identity_recipe(), ident))
                except UnsupportedRecipe as error:
                    assert error.reason == 'no_copy_layout', error
                    direct = lambda: ident.to(fast.device)
                    # Check the same destination and values as the raw arm.
                    output = direct()
                    assert output.device == fast.device and torch.equal(output.cpu(), ident)
                    rows['identity_view_2048x128'] = dict(wire=compiled.wire_version, stages=0,
                                                          exported_axes=compiled.decoded_plan.num_axes, axes=None,
                                                          no_copy=bool(compiled.decoded_plan.no_copy), export_ms=export,
                                                          prepare_ms=0.0, row='direct H2D (identity layout)',
                                                          per_call_us=completed_us(direct, args.samples))
                else:
                    rows['identity_view_2048x128'] = dict(plan_facts(compiled, program), export_ms=export, prepare_ms=prepare,
                                                          row='layout (group)',
                                                          per_call_us=completed_us(
                                                              lambda: fast.submit([(program, ident)]).wait(), args.samples))
            results[mode] = rows
    names = list(results['canonical'])
    def pair(c, r, key):
        if c[key] is None and r[key] is None:
            return 'n/a'
        return ' / '.join('-' if v[key] is None else str(v[key]) for v in (c, r))
    print('| Recipe | wire | stages | exported axes (c / r) | bound axes (c / r) | no_copy (c / r) | export ms (c / r) | '
          'prepare ms (c / r) | row | per call µs (c / r) |')
    print('| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |')
    for name in names:
        c, r = results['canonical'][name], results['raw'][name]
        print(f"| {name} | v{c['wire']} | {c['stages']} | {pair(c, r, 'exported_axes')} | {pair(c, r, 'axes')} | "
              f"{pair(c, r, 'no_copy')} | {c['export_ms']:.1f} / {r['export_ms']:.1f} | "
              f"{c['prepare_ms']:.2f} / {r['prepare_ms']:.2f} | {c['row']} | {c['per_call_us']:.0f} / {r['per_call_us']:.0f} |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
