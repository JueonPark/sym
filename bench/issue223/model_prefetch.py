#!/usr/bin/env python3
"""Actual offloaded models, matched two-slot queues, completed end-to-end timing.

Use separate fresh processes for rounds. Inputs/weight updates and exact checks
are outside timing. Trace collection is a separate invocation. The Torch queue
is a benchmark control, not a general asynchronous API (no failure injection).
"""
import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'libreloc/python/examples/workloads'))
sys.path.insert(0, str(ROOT / 'bench/issue221'))
import common
from llm_offload import OffloadedGPT, generate
from moe_experts import OffloadedMoE
from typed_pipeline import metadata

OUTPUT_LIMIT, SCRATCH_LIMIT = 128 << 20, 64 << 20


def dequant(q, scale):
    return (q.float() * scale).t().contiguous()


class TorchHandle:
    def __init__(self, queue, weights, size):
        self.queue, self.weights, self.size = queue, weights, size
        self.stream, self.retired, self.closed = None, None, False
        with torch.cuda.stream(queue.stream):
            self.outputs = tuple(queue.convert(q.to('cuda:0', non_blocking=True),
                s.to('cuda:0', non_blocking=True)) for q, s in weights)
            self.ready = torch.cuda.Event()
            self.ready.record()

    def wait(self):
        self.ready.synchronize()

    def wait_stream(self):
        self.stream = torch.cuda.current_stream()
        self.stream.wait_event(self.ready)
        for output in self.outputs:
            output.record_stream(self.stream)
        return self.outputs

    def release(self):
        if self.retired is None and self.stream is not None:
            self.retired = torch.cuda.Event()
            self.retired.record(self.stream)

    def close(self):
        if self.closed:
            return
        self.release()
        self.wait()
        if self.retired is not None:
            self.retired.synchronize()
        self.closed = True
        self.weights, self.outputs = (), ()
        self.queue.held.remove(self)
        self.queue.output_bytes -= self.size


class TorchQueue:
    def __init__(self, convert):
        self.convert = convert
        self.stream = torch.cuda.Stream()
        self.held, self.output_bytes = set(), 0
        self.peak_bytes = self.peak_handles = self.scratch_bound = 0

    def can_submit(self, group):
        return len(self.held) < 2 and self.output_bytes + sum(q.numel()*4 for q, _ in group) <= OUTPUT_LIMIT

    def submit(self, group):
        group = tuple(group)
        # At most int8 input, fp32 scale, and two fp32 intermediates per
        # member. Same-stream allocator reuse orders their storage lifetimes.
        scratch = max(q.numel()*9 + s.numel()*4 for q, s in group)
        if scratch > SCRATCH_LIMIT or not self.can_submit(group):
            raise BufferError('matched output/scratch budget exceeded')
        # Match Sym's one-producer scratch lease; consumers may still run.
        for prior in self.held:
            prior.wait()
        size = sum(q.numel()*4 for q, _ in group)
        handle = TorchHandle(self, group, size)
        self.held.add(handle)
        self.output_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.output_bytes)
        self.peak_handles = max(self.peak_handles, len(self.held))
        self.scratch_bound = max(self.scratch_bound, scratch)
        return handle

    def stats(self):
        return dict(held=len(self.held), peak_output_bytes=self.peak_bytes,
                    peak_in_flight=self.peak_handles, scratch_live_upper_bound=self.scratch_bound)

    def close(self):
        for handle in tuple(self.held):
            handle.close()


class Delivery:
    """Identical scheduling for Sym and Torch; serial has no cross-group overlap."""
    def __init__(self, queue, prepare, overlap):
        self.queue, self.prepare, self.overlap = queue, prepare, overlap
        self.diagnostic = False
        self.startup, self.drain = [], []

    @contextmanager
    def __call__(self, groups, device):
        groups = (self.prepare(tuple(g), device) for g in groups)
        if self.overlap:
            from reloc_torch.asynchronous import PrefetchWindow
            window = PrefetchWindow(self.queue, groups)
            original_close = window.close
            def close():
                was_closed = window._closed
                start = time.perf_counter_ns()
                original_close()
                if self.diagnostic and not was_closed:
                    self.drain.append((time.perf_counter_ns()-start)/1e6)
            window.close = close
            def values():
                start = time.perf_counter_ns()
                for i, outputs in enumerate(window):
                    if i == 0 and self.diagnostic:
                        window._last.wait()  # separate diagnostic run, never headline timing
                        self.startup.append((time.perf_counter_ns()-start)/1e6)
                    yield outputs
            with window:
                yield values()
        else:
            def values():
                for group in groups:
                    handle = self.queue.submit(group)
                    try:
                        yield handle.wait_stream()
                    finally:
                        # Wait for this consumer before submitting the next.
                        handle.close()
            iterator = values()
            try:
                yield iterator
            finally:
                iterator.close()


class BlockingDelivery:
    """Existing main API control; runnable against the pre-#223 build too."""
    overlap = False

    def __init__(self, fetcher):
        self.fetcher, self.queue, self.output_bound = fetcher, self, 0

    @contextmanager
    def __call__(self, groups, device):
        def values():
            for group in groups:
                group = tuple(group)
                # The Python loop may hold the preceding tuple while the next
                # is allocated. Both fit the same two-output-group allowance.
                self.output_bound = max(self.output_bound, 2*sum(q.numel()*4 for q, _ in group))
                if self.output_bound > OUTPUT_LIMIT:
                    raise BufferError('matched output budget exceeded')
                yield self.fetcher.fetch_many(group, device, max_scratch_bytes=SCRATCH_LIMIT)
        iterator = values()
        try:
            yield iterator
        finally:
            iterator.close()

    def stats(self):
        return dict(held=0, peak_output_bytes=self.output_bound, output_bytes_is_upper_bound=True,
                    resources=self.fetcher.resource_stats()['cuda:0'])


def completed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter_ns()-start)/1e6


def dist(values):
    ordered = sorted(values)
    index = .95*(len(values)-1)
    lo = int(index)
    return dict(p50_ms=statistics.median(values),
                p95_ms=ordered[lo]+(ordered[min(lo+1,len(values)-1)]-ordered[lo])*(index-lo))


def make_model(case):
    generator = torch.Generator().manual_seed(223)
    if case.startswith('llm'):
        sizes = dict(vocab=2048, d_model=1024, heads=16, d_ff=4096, layers=4,
                     prompt=256 if case == 'llm_prefill' else 8, new_tokens=1 if case == 'llm_prefill' else 4)
        if case == 'llm_one_layer':
            sizes.update(layers=1, prompt=1, new_tokens=1)
        model = OffloadedGPT(sizes, torch.device('cuda:0'), generator)
        containers = model.layers
        def new_input():
            return torch.randint(sizes['vocab'], (sizes['prompt'],), device='cuda:0')
        def run(x, delivery):
            return generate(model, x, sizes['new_tokens'], None,
                lambda t: t.transpose(0, 1).contiguous().cpu(),
                lambda t: t.transpose(0, 1).contiguous().to('cuda:0'), prefetch=delivery)
        units = sizes['prompt'] if sizes['new_tokens'] == 1 else sizes['new_tokens']
    else:
        sizes = dict(d_model=1024, d_ff=2048, experts=8, blocks=2,
                     tokens=512 if case == 'moe_dense' else 1)
        model = OffloadedMoE(sizes, [torch.device('cuda:0')], generator)
        containers = [expert for block in model.experts for expert in block]
        def new_input():
            return torch.randn(sizes['tokens'], sizes['d_model'], device='cuda:0')
        def run(x, delivery):
            routes = []
            out = model.forward(x, None, routes.append, prefetch=delivery)
            return routes, out
        units = sizes['tokens']
    # Both implementations receive the exact same pinned checkpoint; its bytes
    # are reported separately from queue output/scratch budgets.
    for item in containers:
        for name, (q, s) in item.items():
            item[name] = q.pin_memory(), s.pin_memory()
    weights = [pair for item in containers for pair in item.values()]
    source_bytes = sum(q.numel()+s.numel()*4 for q, s in weights)
    def mutate(i):
        q, s = weights[i % len(weights)]
        q.neg_()
        s.mul_(.999 if i % 2 else 1.001)
    return sizes, new_input, run, mutate, units, source_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['llm_prefill', 'llm_decode', 'moe_dense', 'moe_sparse', 'llm_one_layer'])
    parser.add_argument('--paths', nargs='+', default=['sym_blocking', 'sym_serial', 'sym_prefetch', 'torch_serial',
                        'torch_prefetch', 'inductor_serial', 'inductor_prefetch'])
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(223)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    result = dict(metadata=metadata(), limits=dict(slots=2, output_bytes=OUTPUT_LIMIT,
                  scratch_bytes=SCRATCH_LIMIT), cases={})
    result['configuration'] = vars(args) | {'output': str(args.output)}
    result['metadata']['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    samples = []
    with torch.no_grad(), ExitStack() as cleanup:
        start = time.perf_counter_ns()
        fetcher = cleanup.enter_context(common.WeightFetcher(common.Report('issue223'),
            implementation='cuda_dequant_relocate'))
        result['artifact_setup_ms'] = (time.perf_counter_ns()-start)/1e6
        compiled = torch.compile(dequant, fullgraph=True, dynamic=True)
        for case in args.cases:
            sizes, new_input, run, mutate, units, source_bytes = make_model(case)
            row = result['cases'][case] = dict(sizes=sizes, pinned_source_bytes=source_bytes,
                throughput_unit='input_tokens/s' if case != 'llm_decode' else 'generated_tokens/s', paths={})
            with ExitStack() as owners:
                paths = {}
                for path in args.paths:
                    kind, mode = path.split('_')
                    if mode == 'blocking' and kind == 'sym':
                        paths[path] = BlockingDelivery(fetcher)
                        continue
                    if kind == 'sym':
                        from reloc_torch import TransferQueue
                        queue = TransferQueue(gather_threads=8)
                        prepare = fetcher.prepare_many
                    else:
                        queue = TorchQueue(compiled if kind == 'inductor' else dequant)
                        prepare = lambda g, d: g
                    owners.callback(queue.close)
                    paths[path] = Delivery(queue, prepare, mode == 'prefetch')
                x = new_input()
                expected = None
                for name, delivery in paths.items():
                    out, elapsed = completed(lambda: run(x, delivery))
                    row['paths'][name] = dict(first_completed_ms=elapsed)
                    if expected is None:
                        expected = (out[0], out[1].cpu())
                    assert out[0] == expected[0] and common.same_tensor(out[1].cpu(), expected[1]), (case, name)
                for _ in range(5):
                    for delivery in paths.values():
                        completed(lambda: run(x, delivery))
                times = {name: [] for name in paths}
                if args.trace:
                    torch.cuda.cudart().cudaProfilerStart()
                for i in range(args.samples):
                    mutate(i)
                    x = new_input()
                    expected = None
                    # Rotate within each process as well as between rounds.
                    names = list(paths)
                    names = names[i % len(names):]+names[:i % len(names)]
                    for name in names:
                        if args.trace:
                            torch.cuda.nvtx.range_push(f'issue223/{case}/{name}/{i}')
                        out, elapsed = completed(lambda: run(x, paths[name]))
                        if args.trace:
                            torch.cuda.nvtx.range_pop()
                        actual = out[1].cpu()
                        if expected is None:
                            expected = out[0], actual
                        assert out[0] == expected[0] and common.same_tensor(actual, expected[1]), (case, name, i)
                        times[name].append(elapsed)
                        samples.append(dict(case=case, path=name, sample=i, completed_ms=elapsed))
                if args.trace:
                    torch.cuda.cudart().cudaProfilerStop()
                for name, delivery in paths.items():
                    record = row['paths'][name]
                    record.update(dist(times[name]))
                    record['tokens_per_second'] = units * 1000 / record['p50_ms']
                    if delivery.overlap:
                        delivery.diagnostic = True
                        completed(lambda: run(x, delivery))
                        record['startup_host_ready_ms'] = delivery.startup
                        record['drain_ms'] = delivery.drain
                        delivery.diagnostic = False
                    torch.cuda.reset_peak_memory_stats()
                    completed(lambda: run(x, delivery))
                    record['allocator_diagnostic'] = dict(
                        torch_peak_allocated=torch.cuda.max_memory_allocated(),
                        torch_reserved=torch.cuda.memory_reserved())
                    record['queue'] = delivery.queue.stats()
                    assert record['queue']['held'] == 0
                    assert record['queue']['peak_output_bytes'] <= OUTPUT_LIMIT
                row['exact_checks'] = len(paths)*(args.samples+1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['case', 'path', 'sample', 'completed_ms'])
        writer.writeheader()
        writer.writerows(samples)


if __name__ == '__main__':
    main()
