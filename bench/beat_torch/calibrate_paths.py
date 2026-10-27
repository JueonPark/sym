#!/usr/bin/env python3
"""Measure the cost-advice calibration keys of this machine (docs/reloc-export.md
"Cost advice"): the per-call intercepts of the direct and dispatcher paths, the
group intercepts, the producer rates and the f16 dequantize multiplier. Writes
bench/results/paths_<machine>.json; bench/rtrack/make_calibration.py turns it
into calibration keys.

    taskset -c 4-7,20-23 python bench/beat_torch/calibrate_paths.py --machine epyc7351-2080ti \\
        --out bench/results/paths_epyc7351-2080ti.json [--samples 60]

Every path is the fast lane's own implementation, measured as completed latency
(synchronize before and after), the median of --samples calls after 10 warmups:
the dispatcher paths with FastTransfers(advice=False), the direct paths with a
synthetic calibration that forces them (so the intercepts include the decision
lookup, the wire ring and the event, everything the model must price). An
intercept is the two-point affine fit's intercept at a small and a large size,
clamped at 0. Each path's slope gives its producer rate: the direct path's
through the serial form (cpu.t8.*, the producer running alone), the
dispatcher's through the overlapped form (cpu_pipe.t8.*, the in-pipeline rate
of the issue #110 convention, which the pricing prefers on overlapped paths).
The model then reproduces both measured lines at the calibration sizes;
host-only kernel timings are recorded as measurements for comparison. Rates
are per wire byte for the pooled producer (a pooled call's source volume
changes with its bag lengths) and per selected source byte for the row
gather, at eight threads. Single-thread rates are not measured: the native
kernels share one worker pool whose size the first call fixes for the
process. The host must be idle.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'libreloc/python/tests/torch_frontend'))
import pyreloc
from reloc_torch import CompilerClient, PreparedWireWeights
from reloc_torch.advice import resolve
from reloc_torch.fast import FastTransfers, FastWeights, KVCacheSwap, _load_native_views
from test_fast import gather_recipe

NATIVE = _load_native_views()
ADVICE_PREFIXES = ('overhead.direct.', 'overhead.dispatch.', 'overhead.group_', 'cpu.t8.contiguous.pooled_sum',
                   'cpu.t8.indexed_rows.', 'cpu_pipe.t8.contiguous.pooled_sum', 'cpu_pipe.t8.indexed_rows.',
                   'recv.m.dequant_s8_f16', 'hbm.m.indexed_rows')
FORCE_DIRECT = """overhead.direct.pooled_ms 0
overhead.direct.indexed_rows_ms 0
overhead.direct.dequant_ms 0
overhead.dispatch.pooled_ms 1000
overhead.dispatch.indexed_rows_ms 1000
overhead.dispatch.dequant_ms 1000
overhead.dispatch.layout_ms 1000
overhead.dispatch.stages_ms 1000
overhead.group_ms 1000
overhead.group_member_ms 1000
cpu.t8.contiguous.pooled_sum_f16_gbps 1
cpu.t8.indexed_rows.gather_f16_gbps 1
recv.m.dequant_s8_f16 1
hbm.m.indexed_rows 1
"""


def completed_ms(fn, samples, warmup=10):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(values)


def host_ms(fn, samples, warmup=10):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(values)


def intercept(points):
    (s1, t1), (s2, t2) = points
    slope = (t2 - t1) / (s2 - s1)
    return max(0.0, t1 - slope * s1)


def direct_rate(cal, points, producer_per_wire):
    """The producer rate (GB/s per producer byte) at which the serial form
    reproduces the direct path's measured slope:
        slope = producer / R + link + device   (ms per wire byte)
    None when the slope leaves nothing for the producer."""
    (s1, t1), (s2, t2) = points
    slope = (t2 - t1) / (s2 - s1)
    producer = (slope - 1e-6 / cal.value('pcie.h2d_gbps')
                - 1e-6 * cal.value('hbm.m.contiguous') / cal.value('hbm.bw_gbps'))
    return None if producer <= 0 else producer_per_wire * 1e-6 / producer


def pipe_rate(cal, points, producer_per_wire):
    """The in-pipeline producer rate (GB/s per producer byte) that makes the
    overlapped form reproduce the dispatcher's measured slope, assuming the
    producer is the pipeline's largest stage:
        slope = producer / R_pipe + (link + device) / n   (ms per wire byte)
    None when the measured slope leaves the producer below the link (the
    form cannot absorb the difference; the isolated rate then stays)."""
    (s1, t1), (s2, t2) = points
    slope = (t2 - t1) / (s2 - s1)
    link = 1e-6 / cal.value('pcie.h2d_gbps')
    device = 1e-6 * cal.value('hbm.m.contiguous') / cal.value('hbm.bw_gbps')
    producer = slope - (link + device) / cal.value('pipeline.chunks_per_buffer')
    if producer <= max(link, device):
        return None
    return producer_per_wire * 1e-6 / producer


def bags_batch(bags, tables, rows, gen):
    indices, offsets = [], []
    for _ in range(tables):
        lengths = torch.randint(1, 5, (bags,), generator=gen)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, rows, (int(lengths.sum()),), generator=gen))
    return indices, offsets


def forcing_calibration(base, directory):
    """The machine calibration without any advice key, plus keys that make
    every admissible direct path the cheapest."""
    lines = [line for line in base.read_text().splitlines() if not line.startswith(ADVICE_PREFIXES)]
    path = Path(directory) / 'force-direct.cal'
    path.write_text('\n'.join(lines) + '\n' + FORCE_DIRECT)
    return pyreloc.load_calibration(str(path))


def check_paths(owner, label, count):
    dispatches = owner.stats()['dispatches']
    assert dispatches.get(label, 0) >= count, (label, dispatches)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--machine', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--samples', type=int, default=60)
    args = p.parse_args()
    assert NATIVE is not None, 'needs the libtorch helper (RELOC_ENABLE_TORCH_EXT=ON)'
    torch.manual_seed(0)
    torch.set_num_threads(8)
    gen = torch.Generator().manual_seed(225)
    device = torch.device('cuda:0')
    base = ROOT / 'calibration' / f'{args.machine}.cal'
    cal = pyreloc.load_calibration(str(base))
    measurements, fitted, notes = {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad(), \
            FastTransfers(device, gather_threads=8, advice=False) as plain, \
            FastTransfers(device, gather_threads=8, calibration=forcing_calibration(base, tmp)) as forced:
        # -- pooled DLRM producer: 64 and 2048 bags over 26 [100k, 64] tables.
        tables = [torch.randn(100_000, 64, generator=gen) for _ in range(26)]
        pooled = {'direct': forced.pooled(tables), 'pipelined': plain.pooled(tables)}
        points = {'direct': [], 'pipelined': []}
        for bags in (64, 2048):
            indices, offsets = bags_batch(bags, 26, 100_000, gen)
            nbytes = bags * 26 * 64 * 2
            row = dict(wire_bytes=nbytes)
            for path, program in pooled.items():
                row[f'{path}_ms'] = completed_ms(lambda: program.execute(indices, offsets), args.samples)
                points[path].append((nbytes, row[f'{path}_ms']))
            measurements[f'pooled_{bags}'] = row
            if bags == 2048:
                wire = torch.empty((bags, 26, 64), dtype=torch.float16, pin_memory=True)
                ms = host_ms(lambda: NATIVE.pooled_bags(tables, indices, offsets, wire, 8), args.samples)
                row['host_producer_gbps'] = nbytes / (ms * 1e-3) / 1e9
        check_paths(forced, 'direct', 1)
        for path in ('direct', 'pipelined'):
            key = f"overhead.{'direct' if path == 'direct' else 'dispatch'}.pooled_ms"
            fitted[key] = round(intercept(points[path]), 4)
            notes[key] = f'two-point fit of the fast lane\'s {path} path, 64 and 2048 bags'
        rate = direct_rate(cal, points['direct'], 1)
        if rate is not None:
            fitted['cpu.t8.contiguous.pooled_sum_f16_gbps'] = round(rate, 3)
            notes['cpu.t8.contiguous.pooled_sum_f16_gbps'] = (
                'producer rate (per wire byte) reproducing the direct path\'s slope, 64 and 2048 bags')
        rate = pipe_rate(cal, points['pipelined'], 1)
        if rate is not None:
            fitted['cpu_pipe.t8.contiguous.pooled_sum_f16_gbps'] = round(rate, 3)
            notes['cpu_pipe.t8.contiguous.pooled_sum_f16_gbps'] = (
                'in-pipeline producer rate (per wire byte) reproducing the dispatcher\'s slope, 64 and 2048 bags')
        del tables, pooled
        # -- indexed rows: 256 and 8192 rows of a [500k, 128] f32 table -> f16.
        source = torch.randn(500_000, 128, generator=gen)
        compiled = CompilerClient.from_environment().compile(gather_recipe())
        indexed = {'direct': forced.indexed(gather_recipe()), 'pipelined': plain.indexed(gather_recipe())}
        points = {'direct': [], 'pipelined': []}
        for rows in (256, 8192):
            index = torch.randint(500_000, (rows,), generator=gen).sort().values
            nbytes = rows * 128 * 2
            row = dict(wire_bytes=nbytes)
            for path, program in indexed.items():
                row[f'{path}_ms'] = completed_ms(lambda: program.execute(source, index), args.samples)
                points[path].append((nbytes, row[f'{path}_ms']))
            measurements[f'indexed_{rows}'] = row
            if rows == 8192:
                bindings = compiled.bind_values(source, index)
                wire = torch.empty((rows, 128), dtype=torch.float16, pin_memory=True)
                ms = host_ms(lambda: NATIVE.gather_rows(compiled.decoded_plan, bindings, source, index, wire, 8),
                             args.samples)
                row['host_producer_gbps'] = rows * 128 * 4 / (ms * 1e-3) / 1e9
        for path in ('direct', 'pipelined'):
            key = f"overhead.{'direct' if path == 'direct' else 'dispatch'}.indexed_rows_ms"
            fitted[key] = round(intercept(points[path]), 4)
            notes[key] = f'two-point fit of the fast lane\'s {path} path, 256 and 8192 rows'
        rate = direct_rate(cal, points['direct'], 2)  # f32 source bytes per f16 wire byte
        if rate is not None:
            fitted['cpu.t8.indexed_rows.gather_f16_gbps'] = round(rate, 3)
            notes['cpu.t8.indexed_rows.gather_f16_gbps'] = (
                'gather rate (per selected source byte) reproducing the direct path\'s slope, 256 and 8192 rows')
        rate = pipe_rate(cal, points['pipelined'], 2)  # f32 source bytes per f16 wire byte
        if rate is not None:
            fitted['cpu_pipe.t8.indexed_rows.gather_f16_gbps'] = round(rate, 3)
            notes['cpu_pipe.t8.indexed_rows.gather_f16_gbps'] = (
                'in-pipeline gather rate (per selected source byte) reproducing the dispatcher\'s slope, 256 and '
                '8192 rows')
        del source, indexed
        # -- dequantize-first weights: one [out, in] int8 weight per group.
        # direct: the forced direct group; pipelined (a member on its own):
        # a group of one through the dispatcher.
        points = {'direct': [], 'pipelined': []}
        with PreparedWireWeights(device, compiler=plain._compiler) as weights:
            loaders = {'direct': FastWeights(forced, weights), 'pipelined': FastWeights(plain, weights)}
            for rows, cols in ((256, 768), (1024, 3072)):
                q = torch.randint(-127, 128, (rows, cols), dtype=torch.int8, generator=gen)
                weights.prepare(f'{cols}', q, torch.rand(cols, generator=gen) * 0.05 + 1e-3)
                nbytes = rows * cols
                row = dict(wire_bytes=nbytes)
                for path, loader in loaders.items():
                    row[f'{path}_ms'] = completed_ms(lambda: loader.load_many([f'{cols}']), args.samples)
                    points[path].append((nbytes, row[f'{path}_ms']))
                measurements[f'dequant_{cols}x{rows}'] = row
                if cols == 3072:
                    _, packed, scale = weights.fast_entries([f'{cols}'])[0]
                    qd, s_dev = packed.to(device), scale.to(device)
                    out32 = torch.empty(packed.shape, dtype=torch.float32, device=device)
                    copy32 = torch.empty_like(out32)
                    out16 = torch.empty(packed.shape, dtype=torch.float16, device=device)
                    stream = torch.cuda.current_stream().cuda_stream
                    copy = completed_ms(lambda: copy32.copy_(out32), args.samples)
                    k16 = completed_ms(lambda: NATIVE.dequant_s8_f16(qd, s_dev, out16, stream), args.samples)
                    k32 = completed_ms(lambda: NATIVE.dequant_s8_f32(qd, s_dev, out32, stream), args.samples)
                    measurements['recv_3072x1024'] = dict(copy_f32_ms=copy, dequant_s8_f16_ms=k16,
                                                          dequant_s8_f32_ms=k32)
                    fitted['recv.m.dequant_s8_f16'] = round(k16 / copy, 3)
                    notes['recv.m.dequant_s8_f16'] = ('dequant_s8_f16 ms / device copy_f32 ms over the same '
                                                      'element count (3072x1024), same run')
        check_paths(forced, 'direct', 2)
        for path in ('direct', 'pipelined'):
            key = f"overhead.{'direct' if path == 'direct' else 'dispatch'}.dequant_ms"
            fitted[key] = round(intercept(points[path]), 4)
            notes[key] = f'two-point fit of the fast lane\'s {path} weight path, 256x768 and 1024x3072'
        # -- groups: K/V restores of N caches [16, 12, 64] f32 -> [12, 16, 64]
        # stored, N = 1, 2, 4, 8, 16, as one group each.
        swap = KVCacheSwap(plain, 16, 64)
        stored = swap.evict([torch.randn(16, 12, 64, device=device) for _ in range(16)])
        ns, ts = [], []
        for n in (1, 2, 4, 8, 16):
            ms = completed_ms(lambda: swap.restore(stored[:n]), args.samples)
            measurements[f'kv_restore_x{n}'] = dict(members=n, member_bytes=16 * 12 * 64 * 4, ms=ms)
            ns.append(n)
            ts.append(ms)
        mean_n, mean_t = statistics.mean(ns), statistics.mean(ts)
        slope = sum((n - mean_n) * (t - mean_t) for n, t in zip(ns, ts)) / sum((n - mean_n) ** 2 for n in ns)
        # What the advice itself charges per member besides the member
        # intercept: the member's grouped terms, priced with this calibration.
        layout = swap._program(swap._restore_recipe, stored[0])
        member = resolve(layout.compiled.advice, {'s0': 12}, plain.gather_threads).path('grouped')
        priced = pyreloc.price_advice(cal, [dict(member, form='serial', intercept='overhead.b_ms',
                                                 member_intercept=None, chunks=None)])[0]['ms']
        modelled = priced - cal.value('overhead.b_ms')
        fitted['overhead.group_ms'] = round(max(0.0, mean_t - slope * mean_n), 4)
        fitted['overhead.group_member_ms'] = round(max(0.0, slope - modelled), 4)
        notes['overhead.group_ms'] = 'intercept of ms vs N, K/V restore groups of N [12,16,64] f32 members, N=1..16'
        notes['overhead.group_member_ms'] = ('slope of ms vs N minus the advice\'s own per-member terms '
                                             f'({modelled:.4f} ms) under this calibration')
        # A layout member on its own: single-member restores at S=12 and S=192.
        big = swap.evict([torch.randn(16, 192, 64, device=device)])
        single = completed_ms(lambda: swap.restore(stored[:1]), args.samples)
        large = completed_ms(lambda: swap.restore(big), args.samples)
        swap.drain()
        measurements['kv_restore_single'] = dict(s12_ms=single, s192_ms=large)
        fitted['overhead.dispatch.layout_ms'] = round(intercept([(16 * 12 * 64 * 4, single),
                                                                 (16 * 192 * 64 * 4, large)]), 4)
        notes['overhead.dispatch.layout_ms'] = 'two-point fit over single-member K/V restores, S=12 and S=192'
        fitted['hbm.m.indexed_rows'] = cal.value('hbm.m.contiguous')
        notes['hbm.m.indexed_rows'] = 'gathered rows land densely on the device: hbm.m.contiguous'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args.out.write_text(json.dumps(dict(machine=args.machine, source_revision=revision, samples=args.samples,
                                        gpu=torch.cuda.get_device_name(0), measurements=measurements,
                                        fitted=fitted, notes=notes), indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(fitted=fitted, measurements=measurements), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
