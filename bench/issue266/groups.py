"""Small/large dequant groups: packing, lifetime recycling and selected depth."""
import gc
import gzip
import json
import statistics
import time
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from reloc_torch import CompilerClient, PreparedWireWeights
from reloc_torch.fast import FastTransfers, FastWeights
from bench.issue266.policies import configuration, install
from bench.issue266.support import digest


def run(job, inputs, output):
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    torch.cuda.mem_get_info()
    sampler = inputs.memory.Sampler()
    before = sampler.device_info()
    reserve = inputs.memory.Reserve(job['budget_mib'] << 20)
    baseline = sampler.device_info()
    assert abs(baseline['used'] - before['used'] - reserve.bytes) <= 16 << 20
    assert abs(baseline['free'] - reserve.after['free']) <= 16 << 20
    result = dict(budget=reserve.stats(), nvml_after_reservation=baseline)
    config, observations = configuration(job), {}
    module = SimpleNamespace(FastTransfers=FastTransfers, FastWeights=FastWeights,
                             TorchProducer=inputs.controls.TorchProducer)
    install(module, inputs, config, observations)
    try:
        with torch.no_grad(), ExitStack() as life:
            torch.manual_seed(266)
            shapes = [(8, 16)] * 4 if job['size'] == 'small' else [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)]
            pairs = [(torch.randint(-127, 128, shape, dtype=torch.int8), torch.rand(shape[1]) / 100) for shape in shapes]
            expected = [(q.t().float() * scale[:, None]).half() for q, scale in pairs]
            names = list(map(str, range(len(pairs))))
            start = time.perf_counter_ns()
            if job['runtime'] == 'sym':
                compiler = CompilerClient.from_environment()
                fast = life.enter_context(module.FastTransfers('cuda:0', compiler=compiler, contexts=2,
                    max_scratch_bytes=64 << 20, gather_threads=1, pinning='pinned'))
                snapshots = life.enter_context(PreparedWireWeights(compiler=compiler, dtype='float16'))
                for name, (q, scale) in zip(names, pairs): snapshots.prepare(name, q, scale)
                producer = module.FastWeights(fast, snapshots)
                stats = lambda: dict(native=fast.stats(), snapshots=snapshots.stats())
                submit = lambda: producer.submit(names)
            else:
                images = {name: inputs.e2.legacy.host_pack(q, scale) for name, (q, scale) in zip(names, pairs)}
                producer = module.TorchProducer(images, lambda q, s: inputs.controls.fused_dequant(q, s, torch.float16), torch.float16)
                life.callback(producer.close)
                stats = producer.stats
                submit = lambda: producer.submit(names if config['packed'] else list(images.values()))
            torch.cuda.synchronize()
            result['preparation_ms'] = (time.perf_counter_ns() - start) / 1e6
            def check(values):
                for value, reference in zip(values, expected):
                    assert torch.equal(value.cpu().view(torch.int16), reference.view(torch.int16))
            start = time.perf_counter_ns()
            completion = submit()
            first = completion.wait()
            result['first_ms'] = (time.perf_counter_ns() - start) / 1e6
            check(first)
            del first, completion
            for _ in range(3):
                completion = submit()
                check(completion.wait())
                del completion
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            result['warm_before'] = dict(torch=inputs.memory.torch_stats(torch), resources=stats())
            result['warm_start_ms'] = (time.perf_counter_ns() - sampler.start_ns) / 1e6
            samples, previous = [], None
            with producer.prefetch([names] * 100, slots=2) as window:
                for _ in range(100):
                    start = time.perf_counter_ns()
                    values = next(window)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    check(values)
                    if previous is not None: check(previous)
                    previous = values
            result['warm_end_ms'] = (time.perf_counter_ns() - sampler.start_ns) / 1e6
            result['warm_after'] = dict(torch=inputs.memory.torch_stats(torch), resources=stats())
            result.update(samples_ms=samples, median_ms=statistics.median(samples), checks=len(samples), outcome='success')
            start = time.perf_counter_ns()
        result['retirement_ms'] = (time.perf_counter_ns() - start) / 1e6
        check(previous)
        result['held_output_checks_after_close'] = 1
        result['torch_with_outputs_after_close'] = inputs.memory.torch_stats(torch)
        result['optimization'] = dict(config, **observations)
    finally:
        torch.cuda.synchronize()
        trace = sampler.stop()
        warm = [r for r in trace['samples'] if result.get('warm_start_ms', float('inf')) <= r[0] <= result.get('warm_end_ms', 0)]
        result['physical_meter_valid'] = trace['observed_device_used_peak'] >= baseline['used'] - (16 << 20)
        result['observed_warm_app_increment_peak'] = max((r[1] for r in warm), default=baseline['used']) - baseline['used']
        reserve.close()
        path = output.with_suffix('.memory.json.gz')
        with gzip.open(path, 'wt') as stream: json.dump(trace, stream)
        result['physical_memory'] = dict(file=path.name, sha256=digest(path), note=trace['scope'])
    return result
