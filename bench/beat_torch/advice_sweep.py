#!/usr/bin/env python3
"""MISCLASS / REGRET of the cost advice over a per-recipe size sweep
(evidence-advice/REGISTERED.md).

    taskset -c 4-7,20-23 python bench/beat_torch/advice_sweep.py --out sweep.json [--samples 40]

Cells: the pooled DLRM producer (bags), the GraphSAGE row gather (rows), one
GPT/MoE weight (input columns of a 1024-row int8 weight) and the decode K/V
restore group (members); five sizes each, spanning the model's predicted
crossing between paths by x1/4 .. x4, or the workload size by x1/4 .. x4 when
the model predicts no crossing. Every admissible path is measured through the
fast lane itself, as completed latency, medians of --samples calls: direct
(forced by a synthetic calibration, as in calibrate_paths.py), pipelined and
grouped (FastTransfers(advice=False)), split (each member submitted as its own
group). The model's selection is what the advised fast lane would execute. A
cell is misclassified when the selected path is slower than the measured
fastest beyond either path's own spread (the relative distance between the
medians of the first and second half of its samples). REGRET =
(t_selected - t_best) / t_best.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'libreloc/python/tests/torch_frontend'))
sys.path.insert(0, str(ROOT / 'bench/beat_torch'))
import pyreloc
from reloc_torch import PreparedWireWeights
from reloc_torch.advice import pooled_advice, resolve, select, select_group
from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap, _load_native_views
from test_fast import gather_recipe
from calibrate_paths import bags_batch, forcing_calibration
from isolation import Watch

NATIVE = _load_native_views()
BARS = dict(misclass=0.15, regret_p90=0.20)
PREFERENCE = dict(advice_prefers_direct='direct', advice_prefers_pipelined='pipelined',
                  advice_prefers_grouped='grouped', advice_prefers_split='split')


def timings(fn, samples, warmup=10):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return values


def summary(values):
    """(median, spread): spread = relative distance between the medians of the
    two halves of the samples, the path's own noise."""
    half = len(values) // 2
    a, b = statistics.median(values[:half]), statistics.median(values[half:])
    return statistics.median(values), abs(a - b) / min(a, b)


def five(center):
    return [max(1, int(round(center * f))) for f in (0.25, 0.5, 1, 2, 4)]


def crossing(grid, decide):
    """The geometric mean of the first neighbouring grid sizes whose decisions
    differ, or None when the decision never flips on the grid."""
    previous = None
    for size in grid:
        current = decide(size)
        if previous is not None and current != previous[1]:
            return math.sqrt(previous[0] * size)
        previous = (size, current)
    return None


def cell(recipe, size, predicted, measured, selected, intercepts):
    best = min(measured, key=lambda name: measured[name][0])
    t_selected, t_best = measured[selected][0], measured[best][0]
    noise = max(measured[selected][1], measured[best][1])
    return dict(recipe=recipe, size=size, predicted=predicted,
                measured={name: value[0] for name, value in measured.items()},
                spread={name: value[1] for name, value in measured.items()},
                selected=selected, best=best,
                misclassified=selected != best and t_selected > t_best * (1 + noise),
                regret=(t_selected - t_best) / t_best,
                # The measured paths lie within the larger path intercept of each other.
                intercept_cell=max(v[0] for v in measured.values()) - t_best <= intercepts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--samples', type=int, default=40)
    p.add_argument('--calibration', default=str(ROOT / 'calibration/epyc7351-2080ti.cal'))
    args = p.parse_args()
    assert NATIVE is not None, 'needs the libtorch helper (RELOC_ENABLE_TORCH_EXT=ON)'
    torch.manual_seed(0)
    torch.set_num_threads(8)
    gen = torch.Generator().manual_seed(225)
    device = torch.device('cuda:0')
    cal = pyreloc.load_calibration(args.calibration)
    threads = 8
    cells = []
    with Watch() as watch, tempfile.TemporaryDirectory() as tmp, torch.no_grad(), \
            FastTransfers(device, gather_threads=threads, advice=False) as plain, \
            FastTransfers(device, gather_threads=threads,
                          calibration=forcing_calibration(Path(args.calibration), tmp)) as forced:
        # -- pooled DLRM producer: bags of 26 [100k, 64] tables.
        tables = [torch.randn(100_000, 64, generator=gen) for _ in range(26)]
        programs = dict(direct=forced.pooled(tables), pipelined=plain.pooled(tables))
        advice = pooled_advice()

        def decide_pooled(bags):
            return select(cal, resolve(advice, {'bags': bags, 'tables': 26, 'dim': 64}, threads),
                          direct_available=True)
        center = crossing([2 ** k for k in range(3, 17)], lambda bags: decide_pooled(bags).path) or 512
        intercepts = max(cal.value('overhead.direct.pooled_ms'), cal.value('overhead.dispatch.pooled_ms'))
        for bags in five(center):
            indices, offsets = bags_batch(bags, 26, 100_000, gen)
            decision = decide_pooled(bags)
            measured = {name: summary(timings(lambda program=program: program.execute(indices, offsets),
                                              args.samples))
                        for name, program in programs.items()}
            cells.append(cell('pooled_dlrm', bags, decision.predicted, measured, PREFERENCE[decision.reason],
                              intercepts))
        del tables, programs
        # -- GraphSAGE row gather: rows of a [500k, 128] f32 table -> f16.
        source = torch.randn(500_000, 128, generator=gen)
        programs = dict(direct=forced.indexed(gather_recipe()), pipelined=plain.indexed(gather_recipe()))
        advice = programs['pipelined'].compiled.advice

        def decide_gather(rows):
            return select(cal, resolve(advice, {'s0': 500_000, 'k': rows}, threads), direct_available=True)
        center = crossing([2 ** k for k in range(4, 21)], lambda rows: decide_gather(rows).path) or 9000
        intercepts = max(cal.value('overhead.direct.indexed_rows_ms'),
                         cal.value('overhead.dispatch.indexed_rows_ms'))
        for rows in five(center):
            index = torch.randint(500_000, (rows,), generator=gen).sort().values
            decision = decide_gather(rows)
            measured = {name: summary(timings(lambda program=program: program.execute(source, index),
                                              args.samples))
                        for name, program in programs.items()}
            cells.append(cell('gnn_gather', rows, decision.predicted, measured, PREFERENCE[decision.reason],
                              intercepts))
        del source, programs
        # -- one weight per group: [out=1024, in=cols] int8 -> f32 (the x4 point
        # is the workloads' largest matrix). A group of one is its own split,
        # so the measured paths are direct and grouped.
        intercepts = max(cal.value('overhead.direct.dequant_ms') + cal.value('overhead.direct.dequant_member_ms'),
                         cal.value('overhead.group_ms') + cal.value('overhead.group_member.dequant_ms'))
        with PreparedWireWeights(device, compiler=plain._compiler) as weights:
            advice = weights.advice

            def decide_weight(cols):
                return select_group(cal, [resolve(advice, {'out': 1024, 'in': cols}, threads)],
                                    direct_available=True)
            center = crossing([2 ** k for k in range(4, 15)], lambda cols: decide_weight(cols).path) or 1024
            loaders = dict(direct=FastWeights(forced, weights), grouped=FastWeights(plain, weights))
            for cols in five(center):
                q = torch.randint(-127, 128, (cols, 1024), dtype=torch.int8, generator=gen)
                weights.prepare(f'{cols}', q, torch.rand(1024, generator=gen) * 0.05 + 1e-3)
                decision = decide_weight(cols)
                measured = {name: summary(timings(lambda loader=loader: loader.load_many([f'{cols}']),
                                                  args.samples))
                            for name, loader in loaders.items()}
                selected = PREFERENCE[decision.reason]
                cells.append(cell('gpt_weight', cols, decision.predicted, measured,
                                  'grouped' if selected == 'split' else selected, intercepts))
                weights.invalidate(f'{cols}')
        # -- the decode K/V restore group: n caches [12, 16, 64] f32 -> [16, 12, 64].
        swap = KVCacheSwap(plain, 16, 64)
        stored = swap.evict([torch.randn(16, 12, 64, device=device) for _ in range(16)])
        member = resolve(swap._program(swap._restore_recipe, stored[0]).compiled.advice, {'s0': 12}, threads)
        intercepts = max(cal.value('overhead.group_ms') + cal.value('overhead.group_member.layout_ms'),
                         cal.value('overhead.dispatch.layout_ms'))
        for n in (1, 2, 4, 8, 16):
            decision = select_group(cal, [member] * n, direct_available=True)
            predicted = dict(grouped=decision.predicted['grouped'], split=decision.predicted['pipelined'])
            measured = dict(grouped=summary(timings(lambda: swap.restore(stored[:n]), args.samples)),
                            split=summary(timings(lambda: [swap.restore([t]) for t in stored[:n]], args.samples)))
            cells.append(cell('kv_restore_group', n, predicted, measured, PREFERENCE[decision.reason], intercepts))
        swap.drain()
    misclassified = [c for c in cells if c['misclassified']]
    regrets = sorted(c['regret'] for c in cells)
    regret_p90 = regrets[min(len(regrets) - 1, math.ceil(0.9 * len(regrets)) - 1)]
    verdict = dict(cells=len(cells), misclass=len(misclassified) / len(cells), regret_p90=regret_p90,
                   misclass_pass=len(misclassified) / len(cells) <= BARS['misclass'],
                   regret_pass=regret_p90 <= BARS['regret_p90'],
                   misclassified=[(c['recipe'], c['size']) for c in misclassified],
                   intercept_cells=[(c['recipe'], c['size']) for c in cells if c['intercept_cell']], bars=BARS)
    print('| Recipe | size | predicted ms | measured ms | selected | fastest | regret | intercept cell |')
    print('| --- | ---: | --- | --- | --- | --- | ---: | --- |')
    for c in cells:
        predicted = ', '.join(f'{k} {v:.3f}' for k, v in c['predicted'].items() if v is not None)
        measured = ', '.join(f'{k} {v:.3f}' for k, v in c['measured'].items())
        print(f"| {c['recipe']} | {c['size']} | {predicted} | {measured} | {c['selected']} | {c['best']} | "
              f"{c['regret']:+.3f} | {'yes' if c['intercept_cell'] else ''} |")
    print(f"MISCLASS {verdict['misclass']:.3f} (bar {BARS['misclass']}): "
          f"{'PASS' if verdict['misclass_pass'] else 'FAIL'}; REGRET p90 {regret_p90:.3f} "
          f"(bar {BARS['regret_p90']}): {'PASS' if verdict['regret_pass'] else 'FAIL'}")
    if watch.observed:
        raise SystemExit(f'contaminated measurement, nothing written: foreign work {watch.observed}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args.out.write_text(json.dumps(dict(source_revision=revision, samples=args.samples, calibration=args.calibration,
                                        verdict=verdict, cells=cells), indent=2) + '\n')


if __name__ == '__main__':
    main()
