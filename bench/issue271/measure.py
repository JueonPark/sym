"""Bounded CUDA region experiment. The catalog and capacity model require only Python."""
import argparse
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
ALIGN = 256
SEED = 271


def workloads():
    """The grid is fixed before calibration or timing."""
    specs = [
        ('train', 4, 524288, 1, 0, [7, 12], []),
        ('held_capacity', 4, 1048576, 1, 0, [11, 14, 18], []),
        ('held_tail', 6, 393216, 3, 1, [10.5, 14], [0.75, 1.5]),
        ('held_final', 6, 655360, 1, 1, [17, 23.5], [0.75, 1.5]),
    ]
    return [dict(id=name, groups=g, elements=n, passes=k, retained_outputs=h,
                 split='train' if name == 'train' else 'held_out',
                 budgets_bytes=[int(b * MIB) for b in budgets],
                 arrivals_ms=periods, paced_requests=40, slo_ms=3.0)
            for name, g, n, k, h, budgets, periods in specs]


def plans(work):
    for resident in (0, work['groups'] // 2):
        for placement, chunk, depth, reuse in itertools.product(
                ('cpu', 'gpu', 'forward', 'fused'), (1, 2), (1, 2), (False, True)):
            yield dict(placement=placement, resident=resident, chunk=chunk, depth=depth, reuse=reuse)
    yield dict(placement='cpu', resident=work['groups'], chunk=1, depth=1, reuse=False)


def aligned(size):
    return (size + ALIGN - 1) // ALIGN * ALIGN


def byte_terms(work, plan):
    """Disjoint aligned arena sections, including the previous caller's output."""
    n = work['elements']
    if n % plan['chunk']:
        raise ValueError('chunk count must divide each group')
    piece = n // plan['chunk']
    resident = plan['resident']
    output = aligned(2 * work['groups'] * n)
    prefix = aligned(2 * n) * resident
    inflight = min(plan['depth'], (work['groups'] - resident) * plan['chunk'])
    raw = aligned(4 * piece) if plan['placement'] != 'cpu' else 0
    compact = aligned(2 * piece) if plan['placement'] in ('cpu', 'gpu') else 0
    temporary = (raw + compact) * inflight
    retained = output * work['retained_outputs']
    return dict(output=output, retained=retained, resident=prefix, scratch=temporary,
                live_peak=output + retained + prefix + temporary,
                context_retained=prefix + (temporary if plan['reuse'] else 0),
                slot=raw + compact, raw_slot=raw, compact_slot=compact, inflight=inflight)


def select_transfer(work, torch, pyreloc):
    calibration = ROOT / 'calibration/7800x3d-4070tis.cal'
    model = pyreloc.load_calibration(str(calibration))
    source_bytes = work['elements'] * 4
    kwargs = dict(pattern='contiguous', src_bytes=source_bytes, r=0.5,
                  threads=torch.get_num_threads(), k=1, b_placement='serial')
    native = pyreloc.predict(model, **kwargs)
    # CostModel.h prices A's CPU pack and B's raw H2D plus GPU work.
    label = native['method']
    return dict(calibration=calibration.name, args=kwargs, native=native,
                placement={'a': 'cpu', 'b': 'gpu'}[label])


class AdmissionError(RuntimeError):
    pass


def now():
    return time.perf_counter_ns() / 1e6


def p95(xs):
    return sorted(xs)[math.ceil(0.95 * len(xs)) - 1]


def memory(torch):
    free, total = torch.cuda.mem_get_info()
    return dict(free=int(free), total=int(total), allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved())


class Arena:
    """One physical allocation. Every device tensor submitted by a plan aliases a lease."""
    def __init__(self, torch, size):
        self.torch = torch
        self.size = size
        self.data = torch.empty(size, dtype=torch.uint8, device='cuda')
        self.data.zero_()  # Commit and touch the complete physical partition.
        torch.cuda.current_stream().synchronize()
        self.live = {}
        self.counts = Counter()
        self.high = 0
        self.live_bytes = 0
        self.peak_bytes = 0

    def get(self, tag, offset, elements, dtype):
        length = aligned(elements * dtype.itemsize)
        if offset + length > self.size:
            raise AdmissionError(f'{tag} needs [{offset}, {offset + length}) of {self.size} bytes')
        assert tag not in self.live
        assert all(offset + length <= begin or end <= offset for begin, end in self.live.values())
        self.live[tag] = (offset, offset + length)
        self.live_bytes += length
        self.high = max(self.high, offset + length)
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.counts[tag.split('/')[0]] += 1
        return self.data.narrow(0, offset, elements * dtype.itemsize).view(dtype)

    def release(self, tag):
        start, end = self.live.pop(tag)
        self.live_bytes -= end - start


class Engine:
    def __init__(self, torch, sources, work, plan, budget):
        self.t = torch
        self.src, self.w, self.p = sources, work, plan
        self.terms = byte_terms(work, plan)
        if plan['placement'] == 'fused':
            self.cast_add = fused_module().cast_add
        self.arena = Arena(torch, budget)
        self.compute = torch.cuda.Stream()
        self.link = torch.cuda.Stream()
        self.resident = []
        self.slots = []
        self.retained = None
        self.sequence = 0
        self.host_buffers = 0
        self.host_peak = 0
        self.host_views = 0
        n = work['elements']
        if plan['resident']:
            host = torch.empty(n, dtype=torch.float16, pin_memory=True)
            self.host_peak = 2 * n
            for i in range(plan['resident']):
                dst = self.arena.get(f'resident/{i}', i * aligned(2 * n), n, torch.float16)
                host.copy_(sources[i])
                # Preparation owns a single compact host slot. Its DMA is complete before reuse.
                dst.copy_(host, non_blocking=False)
                self.resident.append(dst)
        for j in range(self.terms['inflight']):
            slot = dict(id=j, ready=torch.cuda.Event(), done=torch.cuda.Event(), busy=False,
                        raw=None, compact=None, host=None)
            self.slots.append(slot)
            if plan['reuse']:
                self.open_slot(slot)

    def open_slot(self, slot):
        t, terms = self.t, self.terms
        piece = self.w['elements'] // self.p['chunk']
        base = terms['resident'] + terms['output'] + terms['retained'] + slot['id'] * terms['slot']
        if terms['raw_slot']:
            slot['raw'] = self.arena.get(f"raw/{slot['id']}", base, piece, t.float32)
        if terms['compact_slot']:
            slot['compact'] = self.arena.get(f"compact/{slot['id']}", base + terms['raw_slot'],
                                             piece, t.float16)
        if self.p['placement'] == 'cpu':
            slot['host'] = t.empty(piece, dtype=t.float16, pin_memory=True)
            self.host_buffers += 2 * piece
            self.host_peak = max(self.host_peak, self.host_buffers)
            self.host_views += 1

    def clear_slot(self, slot):
        for name in ('raw', 'compact'):
            if slot[name] is not None:
                self.arena.release(f"{name}/{slot['id']}")
                slot[name] = None
        if slot['host'] is not None:
            self.host_buffers -= slot['host'].numel() * 2
            slot['host'] = None

    def finish_slot(self, slot):
        if slot['busy']:
            slot['done'].synchronize()
            slot['busy'] = False
            if not self.p['reuse']:
                self.clear_slot(slot)

    def request(self):
        t, p, w, terms = self.t, self.p, self.w, self.terms
        n, groups = w['elements'], w['groups']
        index = self.sequence % (1 + w['retained_outputs'])
        output_tag = f'output/{index}'
        out = self.arena.get(output_tag, terms['resident'] + index * terms['output'],
                             n * groups, t.float16).view(groups, n)
        increment = 1 + self.sequence % 2
        with t.cuda.stream(self.compute):
            for i, src in enumerate(self.resident):
                dst = out[i]
                dst.copy_(src)
                for _ in range(w['passes']):
                    dst.add_(increment)
        piece = n // p['chunk']
        submitted = 0
        for g in range(p['resident'], groups):
            for part in range(p['chunk']):
                slot = self.slots[submitted % len(self.slots)]
                self.finish_slot(slot)
                if not p['reuse']:
                    self.open_slot(slot)
                src = self.src[g].narrow(0, part * piece, piece)
                dst = out[g].narrow(0, part * piece, piece)
                if slot['host'] is not None:
                    slot['host'].copy_(src)
                with t.cuda.stream(self.link):
                    if p['placement'] == 'cpu':
                        slot['compact'].copy_(slot['host'], non_blocking=True)
                    else:
                        slot['raw'].copy_(src, non_blocking=True)
                    slot['ready'].record()
                with t.cuda.stream(self.compute):
                    self.compute.wait_event(slot['ready'])
                    if p['placement'] == 'fused':
                        self.cast_add(slot['raw'], dst, w['passes'], increment)
                    else:
                        if p['placement'] == 'gpu':
                            slot['compact'].copy_(slot['raw'])
                        if p['placement'] == 'forward':
                            dst.copy_(slot['raw'])
                        else:
                            dst.copy_(slot['compact'])
                        for _ in range(w['passes']):
                            dst.add_(increment)
                    slot['done'].record()
                slot['busy'] = True
                submitted += 1
        self.compute.synchronize()
        for slot in self.slots:
            self.finish_slot(slot)
        if self.retained is not None:
            self.arena.release(self.retained[0])
            self.retained = None
        if w['retained_outputs']:
            self.retained = (output_tag, out)
        else:
            self.arena.release(output_tag)
        self.sequence += 1
        return out

    def close(self):
        for slot in self.slots:
            self.finish_slot(slot)
            self.clear_slot(slot)
        for i in range(len(self.resident)):
            self.arena.release(f'resident/{i}')
        self.resident.clear()
        self.slots.clear()
        self.link = self.compute = None
        # The caller's last published output can keep the slab alive past context close.
        self.retained = None
        self.arena.data = None


def paced(engine, periods, count):
    traces = []
    for period in periods:
        t0 = now()
        arrivals, starts, finishes = [], [], []
        for i in range(count):
            target = i * period
            remaining = t0 + target - now()
            if remaining > 0:
                time.sleep(remaining / 1000)
            starts.append(now() - t0)
            result = engine.request()
            finishes.append(now() - t0)
            del result
            arrivals.append(target)
        traces.append(dict(period_ms=period, arrival_ms=arrivals, start_ms=starts, finish_ms=finishes,
                           p95_ms=p95([f - a for f, a in zip(finishes, arrivals)])))
    return traces


def source_fixture(torch, work):
    generator = torch.Generator().manual_seed(SEED)
    shape = work['groups'], work['elements']
    src = torch.empty(shape, dtype=torch.float32, pin_memory=True).normal_(generator=generator)
    # Exercise negative zero, subnormal-range values and rounding boundaries.
    src[:, :4] = torch.tensor([-0.0, 2.0 ** -24, 1.00048828125, -1.00048828125])
    expected = []
    for step in (1, 2):
        ref = src.half()
        for _ in range(work['passes']):
            ref.add_(step)
        expected.append(ref.view(torch.int16))
    return src, expected


def check(torch, value, expected, idx):
    bits = value.detach().cpu().view(torch.int16)
    if not torch.equal(bits, expected[idx % 2]):
        raise AssertionError(f'output bit mismatch at request {idx}')


def clean_allocator(torch):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.accelerator.memory.empty_host_cache()


def finite_lifecycle(torch, src, work, plan, budget):
    """Elapsed open -> five completed calls -> release last published result and context."""
    start = now()
    engine = Engine(torch, src, work, plan, budget)
    opened = now()
    requests = []
    for i in range(5):
        begin = now()
        result = engine.request()
        if i != 4:
            del result
        requests.append(now() - begin)
    begin = now()
    engine.close()
    del engine, result
    torch.cuda.synchronize()
    done = now()
    return done - start, dict(setup_ms=opened - start, request_ms=requests, close_ms=done - begin)


def run_plan(torch, work, plan, budget, fixture, samples, do_paced):
    """Run the selected plan with the same engine and admission checks as the oracle."""
    clean_allocator(torch)
    before = memory(torch)
    data = dict(workload=work['id'], budget_bytes=budget, plan=plan,
                status='ok', samples_ms=[], paced=[], memory=dict(before=before))
    src, expected = fixture
    phase = 'setup'
    try:
        start = now()
        engine = Engine(torch, src, work, plan, budget)
        data['setup_ms'] = now() - start
        data['memory']['after_setup'] = memory(torch)
        phase = 'first'
        start = now()
        last = engine.request()
        data['first_ms'] = now() - start
        # Keep the external output only over the deliberate held-tail boundary.
        prior = last if work['retained_outputs'] else None
        check(torch, last, expected, 0)
        if not work['retained_outputs']:
            del last
        phase = 'warm'
        for j in range(samples):
            start = now()
            value = engine.request()
            data['samples_ms'].append(now() - start)
            if j == 0 and prior is not None:
                assert value.data_ptr() != prior.data_ptr()
                check(torch, prior, expected, 0)
                del last
                prior = None
            if j == samples - 1:
                check(torch, value, expected, j + 1)
            if j != samples - 1:
                del value
        # Check correctness and lifetimes outside measured samples.
        data['correctness'] = dict(exact_int16=True, first_and_last=True,
                                   previous_output_checked=bool(work['retained_outputs']))
        data['p95_ms'] = p95(data['samples_ms'])
        data['warm_median_ms'] = statistics.median(data['samples_ms'])
        if do_paced and work['arrivals_ms']:
            del value
            data['paced'] = paced(engine, work['arrivals_ms'], work['paced_requests'])
            # Hold the last actually published caller view for the close observation.
            value = engine.retained[1]
        data['memory'].update(arena_bytes=budget, arena_peak_address=engine.arena.high,
                              peak_live_bytes=engine.arena.peak_bytes,
                              live_at_request_end=engine.arena.live_bytes,
                              pinned_staging_peak=engine.host_peak,
                              pinned_staging_retained=engine.host_buffers,
                              pinned_source_bytes=src.numel() * src.element_size(),
                              host_staging_allocations=engine.host_views,
                              allocations=dict(engine.arena.counts),
                              after_requests=memory(torch))
        phase = 'close'
        start = now()
        engine.close()
        del engine
        data['context_close_ms'] = now() - start
        data['memory']['after_close_with_output'] = memory(torch)
        start = now()
        del value
        torch.cuda.synchronize()
        data['output_release_ms'] = now() - start
        data['close_ms'] = data['context_close_ms'] + data['output_release_ms']
    except (AdmissionError, torch.OutOfMemoryError) as exc:
        torch.cuda.synchronize()  # Keep all owners alive through a failed partial submission.
        data.update(status='refused' if isinstance(exc, AdmissionError) else 'oom',
                    phase=phase, reason=str(exc))
    # An exception unwinds the partial Engine constructor; successful trials explicitly close.
    if 'engine' in locals():
        del engine
    prior = last = value = None
    clean_allocator(torch)
    data['memory']['after_release'] = memory(torch)
    if data['status'] == 'ok':
        data['lifecycle5_ms'], data['lifecycle5'] = finite_lifecycle(torch, src, work, plan, budget)
    return data


def fused_module():
    if __package__:
        from . import fused
    else:
        import fused
    return fused


def calibrate(t):
    """Independent microbenchmarks. GPU component times use device events over batches."""
    rows = []
    cast_add = fused_module().cast_add
    reps, batch = 11, 8
    sizes = (131072, 262144, 524288, 786432, 1310720, 2097152, 4194304)

    def add(n, name, fn, timer='host_wall', repeat=batch):
        fn()  # Warm code paths and allocator metadata.
        t.cuda.synchronize()
        xs = []
        for _ in range(reps):
            if timer == 'device_event':
                begin, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(repeat):
                    fn()
                end.record()
                end.synchronize()
                ms = begin.elapsed_time(end)
            else:
                start = now()
                for _ in range(repeat):
                    fn()
                ms = now() - start
            xs.append(ms / repeat)
        rows.append(dict(elements=n, component=name, samples_ms=xs, timer=timer, batch=repeat))

    for n in sizes:
        host = t.randn(n, pin_memory=True)
        compact = t.empty(n, dtype=t.float16, pin_memory=True)
        raw = t.empty(n, dtype=t.float32, device='cuda')
        half = t.zeros(n, dtype=t.float16, device='cuda')
        dst = t.zeros_like(half)
        slab = t.empty(n * 4, dtype=t.uint8, device='cuda')
        src = t.cuda.Stream()
        compute = t.cuda.Stream()
        t.cuda.synchronize()
        add(n, 'cpu_pack', lambda: compact.copy_(host))
        add(n, 'h2d_f16', lambda: half.copy_(compact, non_blocking=True), 'device_event')
        add(n, 'h2d_f32', lambda: raw.copy_(host, non_blocking=True), 'device_event')
        add(n, 'gpu_cast', lambda: half.copy_(raw), 'device_event')
        add(n, 'device_copy', lambda: dst.copy_(half), 'device_event')
        add(n, 'consumer', lambda: dst.add_(1), 'device_event')
        for passes in (1, 3):
            add(n, f'fused_consumer_{passes}', lambda: cast_add(raw, dst, passes, 1), 'device_event')
        add(n, 'allocate_f16', lambda: slab.narrow(0, 0, n * 2).view(t.float16))
        add(n, 'allocate_f32', lambda: slab.narrow(0, 0, n * 4).view(t.float32))
        add(n, 'sync', lambda: compute.synchronize())
        event = t.cuda.Event()
        def launch():
            with t.cuda.stream(compute):
                event.record()
        add(n, 'launch', launch)
        def preload():
            compact.copy_(host)
            half.copy_(compact, non_blocking=False)
        add(n, 'resident_preload', preload)
        def slot():
            slab.narrow(0, 0, n * 2).view(t.float16)
            t.empty(n, dtype=t.float16, pin_memory=True)
        add(n, 'slot_setup', slot)
        def request_fixed():
            slab.narrow(0, 0, n * 2).view(t.float16).view(1, n)
            compute.synchronize()
        add(n, 'request_fixed', request_fixed)
        # Fork/join both independent streams, and measure common GPU elapsed time.
        # Batch both lanes to amortize the host's submission time at small sizes.
        def paired(sample):
            gate = t.cuda.Event(enable_timing=True)
            done_a, done_b = t.cuda.Event(), t.cuda.Event()
            end = t.cuda.Event(enable_timing=True)
            gate.record()
            src.wait_event(gate)
            compute.wait_event(gate)
            with t.cuda.stream(src):
                for _ in range(batch):
                    raw.copy_(host, non_blocking=True)
                done_a.record()
            with t.cuda.stream(compute):
                for _ in range(batch):
                    dst.add_(1)
                done_b.record()
            t.cuda.current_stream().wait_event(done_a)
            t.cuda.current_stream().wait_event(done_b)
            end.record()
            end.synchronize()
            if sample:
                return gate.elapsed_time(end) / batch
        paired(False)
        rows.append(dict(elements=n, component='paired_link_compute', samples_ms=[paired(True) for _ in range(reps)],
                         timer='device_event_fork_join', batch=batch))
        # Physical arena binding and release costs are separated (including real page touch).
        setup, close = [], []
        for _ in range(reps):
            start = now()
            a = Arena(t, 2 * n)
            streams = t.cuda.Stream(), t.cuda.Stream()
            setup.append(now() - start)
            start = now()
            del a, streams
            t.cuda.synchronize()
            close.append(now() - start)
        for name, xs in (('setup_fixed', setup), ('close_fixed', close)):
            rows.append(dict(elements=n, component=name, samples_ms=xs, timer='host_wall', batch=1))
        del raw, half, dst, slab, compact, host
    return rows


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    tmp = path.with_name(path.name + '.tmp')
    with opener(tmp, 'wt', encoding='utf-8') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workload', choices=[w['id'] for w in workloads()])
    parser.add_argument('--plan-json', type=json.loads)
    parser.add_argument('--budget-bytes', type=int)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--calibration-only', action='store_true')
    args = parser.parse_args()
    if args.samples < 5:
        parser.error('at least five warm samples are needed for the verification and lifecycle windows')
    if any(x is not None for x in (args.workload, args.plan_json, args.budget_bytes)):
        if not all(x is not None for x in (args.workload, args.plan_json, args.budget_bytes)):
            parser.error('single-plan execution requires --workload, --plan-json and --budget-bytes')
    if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1':
        parser.error('set PYTORCH_NO_CUDA_MEMORY_CACHING=1 before starting Python to isolate device allocations')
    # Isolate compilation artifacts so the recorded first-launch includes a cold Triton cache.
    triton_cache = tempfile.TemporaryDirectory(prefix='sym-271-triton-')
    os.environ['TRITON_CACHE_DIR'] = triton_cache.name
    import torch
    import pyreloc
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    start = now()
    torch.cuda.init()
    torch.zeros(1, device='cuda').add_(1)
    torch.cuda.synchronize()
    init_ms = now() - start
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,'
                                   'memory.total,memory.free', '--format=csv,noheader'], text=True).strip()
    cpu = next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
               if s.startswith('model name'))
    if ('Ryzen 7 7800X3D' not in cpu
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 4070 Ti SUPER'):
        parser.error('the native control requires the measured 7800X3D/4070 Ti SUPER calibration')
    cal_path = ROOT / 'calibration/7800x3d-4070tis.cal'
    native_dir = Path(pyreloc.__file__).resolve().parent
    native_extension, = native_dir.glob('_pyreloc.*.so')
    native_runtime = native_dir.parents[1] / 'libreloc/libreloc_runtime.so'
    native_files = [cal_path, native_extension]
    if native_runtime.exists():
        native_files.append(native_runtime)
    ws = workloads()
    data = dict(version=1, environment=dict(platform=platform.platform(), python=sys.version,
                torch=str(torch.__version__), cuda=torch.version.cuda, gpu=smi, cpu=cpu,
                torch_threads=1, interop_threads=1, affinity=sorted(os.sched_getaffinity(0)),
                caching_disabled=True, initialization_ms=init_ms, harness_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest(), native=pyreloc.__file__,
                initial_memory=memory(torch), seed=SEED, sha256={f.name: hashlib.sha256(
                    f.read_bytes()).hexdigest() for f in native_files}), workloads=ws, calibration=[], rows=[])
    if args.plan_json is None or args.plan_json.get('placement') == 'fused':
        start = now()
        fused = fused_module()
        data['fused_qualification'] = fused.qualify(torch)
        data['fused_qualification']['import_and_qualification_ms'] = now() - start
        data['environment']['triton'] = fused.triton.__version__
        data['environment']['fused_sha256'] = hashlib.sha256(Path(fused.__file__).read_bytes()).hexdigest()
    for work in ws:
        work['selector'] = select_transfer(work, torch, pyreloc)
    if args.workload is not None:
        w = next(w for w in ws if w['id'] == args.workload)
        if args.plan_json not in list(plans(w)) or args.budget_bytes <= 0:
            parser.error('plan must be in the workload catalog and budget must be positive')
        data['rows'] = [run_plan(torch, w, args.plan_json, args.budget_bytes,
                                 source_fixture(torch, w), args.samples, True)]
        save(args.output, data)
        return
    start = now()
    data['calibration'] = calibrate(torch)
    data['calibration_wall_ms'] = now() - start
    save(args.output, data)
    if args.calibration_only:
        return
    # Randomize the full workload/budget/plan tuple. Source fixtures are immutable.
    jobs = [(w, b, p) for w in ws for b in w['budgets_bytes'] for p in plans(w)]
    random.Random(SEED).shuffle(jobs)
    fixtures = {w['id']: source_fixture(torch, w) for w in ws}
    print(f"Calibration {data['calibration_wall_ms']:.0f} ms; {len(jobs)} candidate executions", flush=True)
    for i, (w, b, p) in enumerate(jobs):
        row = run_plan(torch, w, p, b, fixtures[w['id']], args.samples, True)
        row['order'] = i
        data['rows'].append(row)
        if (i + 1) % 20 == 0 or i + 1 == len(jobs):
            save(args.output, data)
            print(f'{i + 1}/{len(jobs)} completed', flush=True)
    data['environment']['final_memory'] = memory(torch)
    save(args.output, data)


if __name__ == '__main__':
    main()
