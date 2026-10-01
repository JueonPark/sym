#!/usr/bin/env python3
"""Completed frontend transfers with fresh outputs; resource-reuse qualification (#171).

Run each thread budget in a separate process. Compilation, correctness checks,
input creation and cache inspection are outside samples; caller-stream completion
is inside. --trace emits NVTX request ranges, never performance evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SHAPES = [(4096, 1024), (1024, 4096), (256, 1024), (1009, 4093)]
CREATIONS = ('staging_allocations', 'stream_creations', 'worker_creations')
LIMITS = dict(max_retained_bytes=256 << 20, max_contexts=4,
              max_contexts_per_device=2, max_background_workers=64,
              max_live_staging_bytes=512 << 20, acquire_timeout_ms=30000)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def shape(value):
    try:
        rows, cols = map(positive, value.lower().split('x'))
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError('expected positive ROWSxCOLS') from None
    return rows, cols


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=positive, default=8)
    p.add_argument('--cpus', help='Linux affinity list, e.g. 4,5,6,7,20,21,22,23')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--warmup', type=positive, default=5)
    p.add_argument('--samples', type=positive, default=30)
    p.add_argument('--rounds', type=positive, default=3)
    p.add_argument('--callers', type=positive, default=2)
    p.add_argument('--shapes', type=shape, nargs='+', default=SHAPES)
    p.add_argument('--scenarios', nargs='+', choices=['steady', 'changing', 'd2h', 'concurrent', 'control'],
                   default=['steady', 'changing', 'd2h', 'concurrent', 'control'])
    p.add_argument('--seed', type=int, default=171)
    p.add_argument('--trace', action='store_true', help='profile retained H2D, fixed reference shape')
    p.add_argument('--trace-buffers', type=positive, choices=[1, 4], default=4)
    return p


def distribution(samples):
    ordered = sorted(samples)
    return dict(p50_ms=statistics.median(ordered), p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1],
                min_ms=ordered[0], max_ms=ordered[-1])


def transpose_schedule(shape, buffers):
    """Derived dense FP32 transpose schedule, not native instrumentation.

    Mirrors the documented clamp in ChunkSchedule.h for this benchmark's only
    recipe. The trace analyzer independently checks actual DMA sizes for the
    one/four-buffer control. Changes to that native policy require revalidation.
    """
    rows, cols = shape
    row_bytes = rows * 4  # destination rows
    target = min(64 << 20, max(4 << 20, rows * cols * 4 // (8 * buffers)))
    per_chunk = min(cols, max(1, target // row_bytes))
    chunks = [min(per_chunk, cols - begin) * row_bytes for begin in range(0, cols, per_chunk)]
    return dict(chunk_bytes=chunks, effective_buffers=min(buffers, len(chunks)),
                row_bytes=row_bytes, source='derived from dense-transpose chunk policy')


def command(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f'unavailable: {error}'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(args, torch, pyreloc):
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if line.endswith('/libreloc_runtime.so')})
    sources = [Path(__file__), ROOT / 'libreloc/src/ChunkSchedule.cpp',
               ROOT / 'libreloc/include/reloc/ChunkSchedule.h']
    nvtx = []
    for path in libraries:
        cache = Path(path).parents[1] / 'CMakeCache.txt'
        lines = cache.read_text().splitlines() if cache.exists() else []
        nvtx.append(next((line for line in lines if line.startswith('RELOC_ENABLE_NVTX:')), 'unrecorded'))
    return dict(revision=command(['git', 'rev-parse', 'HEAD']),
                git_status=command(['git', 'status', '--short']),
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                runtime_sha256={p: digest(p) for p in libraries},
                runtime_nvtx=nvtx,
                python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                pyreloc=str(pyreloc.__file__), platform=platform.platform(),
                cpu=command(['lscpu']), gpu=torch.cuda.get_device_name(args.device),
                gpu_status=command(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,utilization.gpu', '--format=csv']),
                affinity=sorted(os.sched_getaffinity(0)), threads_per_call=args.threads,
                concurrency_thread_budget='each caller has the full gather budget; CPU affinity is shared',
                limits=LIMITS, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                completion='blocking frontend plus caller CUDA stream synchronize inside each sample',
                first_call='first measured transfer after compilation, with retained cache cleared; process/CUDA/Torch allocators already initialized',
                allocations='fresh outputs for every method; pageable CPU inputs; Inductor allocates CPU transpose then H2D; Torch GPU path allocates H2D source then dense transpose',
                counters='native cache counters available only for retained Sym; None is not measured as zero',
                memory='CUDA allocator peak reset per method/round; process ru_maxrss is cumulative; native cache stats bound staging only',
                clocks_locked=False, input_cache_flush=False)


class Sym:
    def __init__(self, torch, compiler, target, threads, retained, buffers=4):
        from reloc_torch import RelocBackend, TransferResources
        self.owner = TransferResources(**LIMITS) if retained else None
        self.backend = RelocBackend(compiler=compiler, transfer_resources=self.owner,
                                    transfer_options=dict(n_buffers=buffers, n_streams=2, gather_threads=threads))
        def transpose_transfer(x):
            return x.t().contiguous().to(target)
        self.fn = torch.compile(transpose_transfer, backend=self.backend, dynamic=True)

    def __call__(self, src):
        return self.fn(src)

    def stats(self):
        return self.backend.stats()

    def clear(self):
        if self.owner:
            self.owner.clear()

    def close(self):
        self.backend.close()
        if self.owner:
            self.owner.close()


def verify(torch, out, src):
    expected = src.cpu().t().contiguous()
    actual = out.cpu()
    assert out.is_contiguous() and out.storage_offset() == 0
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), 'incorrect completed output'
    assert out.data_ptr() != src.data_ptr(), 'output aliases input'


def inputs(torch, shapes, device):
    # Two changing payloads per shape. Exact FP32 integers reveal transposition
    # and stale-cache mistakes without floating-point comparison tolerances.
    return [(torch.arange(r * c, dtype=torch.int32) % 251 + seed).float().reshape(r, c).to(device)
            for r, c in shapes for seed in (0, 17)]


def stats_of(fn):
    return fn.stats() if isinstance(fn, Sym) else None


def measure(torch, fn, sources, args):
    # Compile all shapes outside timing. Clearing invalidates native contexts,
    # not compiled artifacts or Torch's warmed allocators.
    for src in sources:
        verify(torch, fn(src), src)
    if isinstance(fn, Sym):
        fn.clear()
    stream = torch.cuda.current_stream(args.device)
    stream.synchronize()
    start = time.perf_counter_ns()
    held = fn(sources[0])
    stream.synchronize()
    cold = (time.perf_counter_ns() - start) / 1e6
    verify(torch, held, sources[0])
    for index in range(max(args.warmup, len(sources))):
        src = sources[index % len(sources)]
        out = fn(src)
        stream.synchronize()
        verify(torch, out, src)
    del out
    before = stats_of(fn)
    torch.cuda.reset_peak_memory_stats(args.device)
    samples = []
    for index in range(args.samples):
        src = sources[index % len(sources)]
        start = time.perf_counter_ns()
        out = fn(src)
        stream.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        # Correctness and deallocation are outside the measured interval.
        verify(torch, out, src)
        del out
    after = stats_of(fn)
    verify(torch, held, sources[0])
    if before:
        assert after['dynamo_compiles'] == before['dynamo_compiles'], 'compilation during timing'
        assert after['runtime_executions'] - before['runtime_executions'] == args.samples
    stable = None
    if isinstance(fn, Sym) and fn.owner:
        stable = {key: after['transfer_resources'][key] - before['transfer_resources'][key] for key in CREATIONS}
        assert not any(stable.values()), f'resources grew during warmed samples: {stable}'
    return dict(first_ms=cold, samples_ms=samples, **distribution(samples), stats_before=before,
                stats_after=after, warmed_creation_delta=stable, byte_exact=True,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def latency_case(torch, compiler, args, label, shapes, direction, rng, control=False):
    # Each case owns new backend objects. Avoid exhausting Dynamo's per-code
    # recompile limit across these intentionally distinct configurations.
    torch._dynamo.reset()
    source_device = 'cpu' if direction == 'h2d' else f'cuda:{args.device}'
    target = f'cuda:{args.device}' if direction == 'h2d' else 'cpu'
    sources = inputs(torch, shapes, source_device)
    if control:
        methods = {'sym_retained_4_buffers': Sym(torch, compiler, target, args.threads, True),
                   'sym_retained_1_buffer': Sym(torch, compiler, target, args.threads, True, 1)}
        assert transpose_schedule(shapes[0], 1)['chunk_bytes'] == transpose_schedule(shapes[0], 4)['chunk_bytes']
    else:
        methods = dict(sym_ephemeral=Sym(torch, compiler, target, args.threads, False),
                       sym_retained=Sym(torch, compiler, target, args.threads, True))
    if not control and direction == 'h2d':
        cpu_transpose = torch.compile(lambda x: x.t().contiguous(), backend='inductor', fullgraph=True, dynamic=True)
        methods['inductor_cpu_h2d'] = lambda src: cpu_transpose(src).to(target)
        methods['torch_h2d_gpu_transpose'] = lambda src: src.to(target).t().contiguous()
    elif not control:
        methods['torch_gpu_transpose_d2h'] = lambda src: src.t().contiguous().cpu()
    result = dict(scenario=label, shapes=shapes, direction=direction,
                  schedules={str(buffers): [transpose_schedule(s, buffers) for s in shapes] for buffers in ([1, 4] if control else [4])},
                  methods={name: [] for name in methods}, round_order=[])
    try:
        for _ in range(args.rounds):
            order = list(methods)
            rng.shuffle(order)
            result['round_order'].append(order)
            for name in order:
                result['methods'][name].append(measure(torch, methods[name], sources, args))
    finally:
        for fn in methods.values():
            if isinstance(fn, Sym):
                fn.close()
    return result


def concurrent_case(torch, compiler, args, retained):
    torch._dynamo.reset()
    if args.callers > LIMITS['max_contexts_per_device']:
        raise ValueError('concurrency measurement requires one permitted context per caller')
    fn = Sym(torch, compiler, f'cuda:{args.device}', args.threads, retained)
    sources = inputs(torch, [SHAPES[0]], 'cpu')
    for src in sources:
        verify(torch, fn(src), src)
    ready = threading.Barrier(args.callers + 1)
    launch = threading.Barrier(args.callers + 1)
    try:
        def worker(index):
            stream = torch.cuda.Stream(device=args.device)
            samples = []
            held = []
            with torch.no_grad(), torch.cuda.device(args.device), torch.cuda.stream(stream):
                for n in range(args.warmup):
                    src = sources[(index + n) % len(sources)]
                    verify(torch, fn(src), src)
                ready.wait(timeout=60)
                launch.wait(timeout=60)
                for n in range(args.samples):
                    src = sources[(index + n) % len(sources)]
                    start = time.perf_counter_ns()
                    out = fn(src)
                    stream.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    if n in (0, args.samples - 1):
                        held.append((out, src))
                    del out
                # Checking outside the throughput window avoids serializing
                # callers on the default stream during D2H validation.
                return samples, held
        with ThreadPoolExecutor(args.callers) as threads:
            pending = [threads.submit(worker, i) for i in range(args.callers)]
            ready.wait(timeout=60)
            before = fn.stats()
            torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter_ns()
            launch.wait(timeout=60)
            completed = [future.result(timeout=120) for future in pending]
            elapsed = (time.perf_counter_ns() - start) / 1e9
        raw = [samples for samples, _ in completed]
        for _, held in completed:
            for out, src in held:
                verify(torch, out, src)
        after = fn.stats()
        assert after['dynamo_compiles'] == before['dynamo_compiles']
        assert after['runtime_executions'] - before['runtime_executions'] == args.callers * args.samples
        stable = None
        if retained:
            stable = {key: after['transfer_resources'][key] - before['transfer_resources'][key] for key in CREATIONS}
            assert not any(stable.values()), stable
        for src in sources:
            verify(torch, fn(src), src)
        return dict(scenario='concurrent', policy='retained' if retained else 'ephemeral', shape=SHAPES[0],
                    callers=args.callers, samples_ms_by_caller=raw, elapsed_s=elapsed,
                    completed_calls_per_s=args.callers * args.samples / elapsed,
                    payload_gib_per_s=args.callers * args.samples * math.prod(SHAPES[0]) * 4 / elapsed / (1 << 30),
                    stats_before=before, stats_after=after, warmed_creation_delta=stable, byte_exact=True,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
    finally:
        fn.close()


def trace(torch, compiler, args):
    fn = Sym(torch, compiler, f'cuda:{args.device}', args.threads, True, args.trace_buffers)
    sources = inputs(torch, [SHAPES[0]], 'cpu')
    try:
        for _ in range(args.warmup):
            verify(torch, fn(sources[0]), sources[0])
        torch.cuda.synchronize(args.device)
        torch.cuda.profiler.start()
        try:
            for index in range(args.samples):
                source = sources[index % len(sources)]
                with torch.cuda.nvtx.range(f'reloc.benchmark/buffers={args.trace_buffers}/request={index}'):
                    out = fn(source)
                    torch.cuda.current_stream(args.device).synchronize()
                verify(torch, out, source)
        finally:
            torch.cuda.profiler.stop()
        return dict(scenario='trace', buffers=args.trace_buffers, shape=SHAPES[0],
                    schedule=transpose_schedule(SHAPES[0], args.trace_buffers), stats=fn.stats(),
                    byte_exact=True, warning='profiled execution; no latency claim')
    finally:
        fn.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cpus:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.cpus.split(',')})
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', f'/tmp/sym-resource-inductor-t{args.threads}')
    import torch
    import pyreloc
    from reloc_torch import CompilerClient

    if not pyreloc.cuda_enabled or not torch.cuda.is_available():
        raise SystemExit('CUDA frontend required; missing hardware is not a passing benchmark')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    compiler = CompilerClient.from_environment()
    report = dict(schema_version=1, metadata=metadata(args, torch, pyreloc), results=[])
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(result):
        report['results'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(scenario=result['scenario'], shape=result.get('shapes', result.get('shape')))), flush=True)

    with torch.no_grad():
        if args.trace:
            save(trace(torch, compiler, args))
        else:
            for scenario in args.scenarios:
                if scenario in ('steady', 'd2h'):
                    for item in args.shapes:
                        save(latency_case(torch, compiler, args, scenario, [item], 'h2d' if scenario == 'steady' else 'd2h', rng))
                elif scenario == 'changing':
                    for direction in ('h2d', 'd2h'):
                        save(latency_case(torch, compiler, args, scenario, args.shapes, direction, rng))
                elif scenario == 'control':
                    save(latency_case(torch, compiler, args, scenario, [SHAPES[0]], 'h2d', rng, control=True))
                else:
                    for round_id in range(args.rounds):
                        order = [False, True]
                        rng.shuffle(order)
                        for retained in order:
                            result = concurrent_case(torch, compiler, args, retained)
                            result['round'] = round_id
                            save(result)
    report['gpu_after'] = command(['nvidia-smi', '--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,utilization.gpu', '--format=csv'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
