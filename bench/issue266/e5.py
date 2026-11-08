"""Original E5 burst and held-output checks; only producer factories change."""
import json
import sys
import time
from bench.issue266.policies import configuration, install
from bench.issue266.support import load_module


def run(job, inputs, output):
    config, observations = configuration(job), dict(workers=[], retained_check_calls=0)
    clock = {}
    def elapsed():
        return (time.perf_counter_ns() - clock['start_ns']) / 1e6
    install(inputs.workload, inputs, config, observations)
    original_snapshots = inputs.workload.Snapshots
    class Snapshots(original_snapshots):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observations['snapshot_preparation_ms'] = self.snapshot_prepare_ms
            observations['owner_preparation_ms'] = self.owner_artifact_ms
    inputs.workload.Snapshots = Snapshots
    original_worker = inputs.workload.Worker
    class Worker(original_worker):
        def __init__(self, *args, **kwargs):
            observations['worker_attempts'] = observations.get('worker_attempts', 0) + 1
            self.costs = dict(caller=len(observations['workers']), prepare_start_ms=elapsed(),
                              warmup_samples_ms=[])
            observations['workers'].append(self.costs)
            self.remaining_warmup = 24 // job['callers']
            try:
                super().__init__(*args, **kwargs)
            finally:
                self.costs['prepare_end_ms'] = elapsed()
                self.costs['preparation_ms'] = self.costs['prepare_end_ms'] - self.costs['prepare_start_ms']

        def request(self, *args, **kwargs):
            if not self.remaining_warmup:
                return super().request(*args, **kwargs)
            start = elapsed()
            output, service = super().request(*args, **kwargs)
            end = elapsed()
            if 'first_request_ms' not in self.costs:
                self.costs.update(first_request_ms=service, first_start_ms=start, first_end_ms=end)
            self.costs['warmup_samples_ms'].append(service)
            self.costs['warmup_end_ms'] = end
            self.remaining_warmup -= 1
            return output, service

        def close(self):
            if self.core is None:
                return
            arena = (getattr(self.fast, 'weight_arena', None) if self.fast else
                     getattr(self.producer, 'arena', None))
            self.costs['close_start_ms'] = elapsed()
            try:
                super().close()
            finally:
                self.costs['close_end_ms'] = elapsed()
                self.costs['retirement_ms'] = self.costs['close_end_ms'] - self.costs['close_start_ms']
                self.costs['arena_after_close'] = arena.stats() if arena is not None else None
    inputs.workload.Worker = Worker
    original_retained = inputs.workload.exact_retained
    def retained(*args):
        original_retained(*args)
        observations['retained_check_calls'] += 1
    inputs.workload.exact_retained = retained
    module = load_module('e5_266', inputs.root / '05_lifecycle_concurrency/concurrency.py')
    original_sampler = module.Sampler
    def sampler():
        value = original_sampler()
        clock['start_ns'] = value.start_ns
        return value
    module.Sampler = sampler
    sys.argv = [str(module.__file__), '--output', str(output), '--round', str(job['round']),
                '--case', job['case'], '--runtime', job['runtime'], '--callers', str(job['callers']), '--batches', '30']
    failure = None
    try:
        module.main()
    except Exception as error:
        failure = error
    result = json.loads(output.read_text()) if output.exists() else {}
    if failure is not None:
        # Original concurrency writes incomplete data on policy failure. Keep
        # actual allocation failures, but never promote corruption/other errors.
        import torch
        from reloc_torch.memory import MemoryBudgetError
        message = str(failure).lower()
        if not observations.get('worker_attempts'):
            raise failure  # Fixture, oracle and reservation failures are invalid.
        if isinstance(failure, MemoryBudgetError):
            result.update(outcome='admission', admission_phase='worker_or_request', error=repr(failure))
        elif isinstance(failure, torch.OutOfMemoryError) or ('out of memory' in message and 'cudamalloc' in message):
            result.update(outcome='oom', oom_phase='worker_or_request', error=repr(failure))
        else:
            raise failure
    else:
        result['outcome'] = 'success'
    result['optimization'] = dict(config, **observations)
    result['worker_costs'] = observations['workers']
    result['retained_check_calls'] = observations['retained_check_calls']
    result['original_protocol'] = result['protocol']
    result['protocol'] = dict(result['protocol'], native_contexts_per_caller=config['contexts'],
        aggregate_native_scratch_bytes=config['contexts'] * config['native_scratch_per_context'] * job['callers'])
    result['snapshot_preparation_ms'] = observations.get('snapshot_preparation_ms')
    return result
