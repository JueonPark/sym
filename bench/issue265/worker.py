#!/usr/bin/env python3
"""One fresh process/cache for one arm, case and independent round."""
import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.issue265.support import (artifact_hashes, environment, harness_hashes,
                                   load_e3, runtime_hashes, save)


def memory(torch):
    rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    free, total = torch.cuda.mem_get_info()
    return dict(rss_bytes=rss, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        device_used_bytes=total - free, torch_allocated_bytes=torch.cuda.memory_allocated(),
        torch_reserved_bytes=torch.cuda.memory_reserved(), torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())


class PathRunner:
    def __init__(self, job, fn, e3, lifetime):
        import torch
        from reloc_torch import RelocBackend, compat
        self.backend = self.fast = None
        self.compiles, self.compile_ms = 0, 0.
        arm = job['arm']
        if arm == 'torch':
            self.fn = fn
        elif arm.startswith('sym_fast'):
            self.fast = e3.FastPath(job['case'], arm.endswith('_cache'), lifetime)
            self.fn = self.fast
        else:
            if arm != 'inductor':
                kwargs = dict(compute_backend='inductor' if arm.endswith('inductor') else 'eager',
                    transfer_options=dict(gather_threads=8, n_buffers=2, n_streams=1, pinning='pinned'))
                if job['policy'] is not None:
                    from reloc_torch import PreparedExecutionPolicy
                    kwargs['prepared_execution'] = PreparedExecutionPolicy(**job['policy'])
                self.backend = RelocBackend(**kwargs)
                lifetime.callback(self.backend.close)
            def compile_graph(gm, examples):
                started = time.perf_counter_ns()
                result = (self.backend(gm, examples) if self.backend else
                          compat.compile_inductor(gm, examples, {'triton.cudagraphs': False}))
                self.compiles += 1
                self.compile_ms += (time.perf_counter_ns() - started) / 1e6
                return result
            self.fn = torch.compile(fn, backend=compile_graph, fullgraph=True, dynamic=True)

    def stats(self):
        return dict(dynamo_callbacks=self.compiles, callback_ms=self.compile_ms,
            backend=self.backend.stats() if self.backend else None,
            explicit=self.fast.stats() if self.fast else None)

    def resources(self):
        if self.backend is not None:
            runtime = self.backend._runtime
            return None if runtime is None else runtime._resources
        return self.fast.owner._resources if self.fast else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('artifact', 'build', 'contract', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text())
    job = next(job for job in contract['jobs'] if job['name'] == args.job)
    sys.path.insert(0, str(args.build / 'python'))
    import torch
    from torch._inductor import config
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch._dynamo.config.recompile_limit = 64
    config.emulate_precision_casts = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    e3 = load_e3(args.artifact)
    if job['case'] in e3.CASES:
        fn, inputs = e3.FUNCTIONS[job['case']], e3.inputs
    else:
        from bench.issue265 import workloads
        fn, inputs = getattr(workloads, job['case']), workloads.inputs
    phases = dict(import_and_configure=(time.time_ns() - int(os.environ['EXPERIMENT_LAUNCH_NS'])) / 1e6)
    start = time.perf_counter_ns()
    torch.cuda.init()
    phases['cuda_init'] = (time.perf_counter_ns() - start) / 1e6
    measured_environment = environment()
    if measured_environment != contract['environment']:
        raise ValueError('environment differs from the frozen contract')
    snapshots = {'initialized': memory(torch)}
    lifetime = ExitStack()
    start = time.perf_counter_ns()
    runner = PathRunner(job, fn, e3, lifetime)
    phases['owner_setup'] = (time.perf_counter_ns() - start) / 1e6
    snapshots['owner_ready'] = memory(torch)
    torch.cuda.reset_peak_memory_stats()
    rows, retained = [], None
    input_ms = oracle_ms = 0.
    with torch.no_grad():
        try:
            for step, n in enumerate(contract['sequence'] * contract['epochs']):
                start = time.perf_counter_ns()
                values = inputs(job['case'], n, step)
                input_ms += (time.perf_counter_ns() - start) / 1e6
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                actual = runner.fn(*values)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                # The first oracle follows the first measured request. It cannot
                # warm that request's kernels or the compiler under test.
                start = time.perf_counter_ns()
                expected = fn(*values)
                e3.exact(actual, expected)
                if retained is None and actual.numel():
                    retained = actual, actual.clone()
                old, snapshot = retained
                e3.exact(old, snapshot)
                if actual.numel() and step:
                    assert old.data_ptr() != actual.data_ptr()
                torch.cuda.synchronize()
                oracle_ms += (time.perf_counter_ns() - start) / 1e6
                rows.append(dict(step=step, n=n, completed_ms=elapsed, correct=True, held_correct=True,
                    shapes=[list(t.shape) for t in values], strides=[list(t.stride()) for t in values],
                    stats=runner.stats()))
                if step == 0: snapshots['after_first'] = memory(torch)
                if step + 1 == len(contract['sequence']): snapshots['after_first_cycle'] = memory(torch)
            stats = runner.stats()
            resources = runner.resources()
            snapshots['after_sequence'] = memory(torch)
        finally:
            start = time.perf_counter_ns()
            lifetime.close()
            torch.cuda.synchronize()
            phases['owner_retirement'] = (time.perf_counter_ns() - start) / 1e6
        e3.exact(*retained)
    retired = resources.stats() if resources is not None else None
    snapshots['after_owner_close'] = memory(torch)
    del values, actual, expected, retained, old, snapshot, runner
    gc.collect()
    torch.cuda.empty_cache()
    snapshots['after_caller_release'] = memory(torch)
    phases.update(input_generation=input_ms, oracle_and_held_checks=oracle_ms)
    metadata = dict(harness_sha256=harness_hashes(), artifact_sha256=artifact_hashes(args.artifact),
        runtime_sha256=runtime_hashes(args.build), environment=measured_environment)
    assert metadata['harness_sha256'] == contract['harness_sha256']
    assert metadata['artifact_sha256'] == contract['artifact_sha256']
    assert metadata['runtime_sha256'] == contract['runtime_sha256'][job['runtime']]
    result = dict(job=job, complete=True, policy=job['policy'], metadata=metadata, requests=rows,
        final_stats=stats, resources_after_close=retired, phases_ms=phases, memory=snapshots,
        held_checks_after_close=1,
        memory_scope='boundary snapshots; process RSS peak includes compiler/oracles; device snapshots are lower bounds, not transient peaks',
        accounting='completed calls include compile/preparation/execution; input generation, oracle and diagnostics are separate')
    save(args.output, result)
    print(job['name'], 'complete', len(rows), 'exact outputs', flush=True)


if __name__ == '__main__': main()
