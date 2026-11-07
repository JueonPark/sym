"""Original E5 burst and held-output checks; only producer factories change."""
import json
import sys
from bench.issue266.policies import configuration, install
from bench.issue266.support import load_module


def run(job, inputs, output):
    config, observations = configuration(job), {}
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
            super().__init__(*args, **kwargs)
    inputs.workload.Worker = Worker
    module = load_module('e5_266', inputs.root / '05_lifecycle_concurrency/concurrency.py')
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
    result['original_protocol'] = result['protocol']
    result['protocol'] = dict(result['protocol'], native_contexts_per_caller=config['contexts'],
        aggregate_native_scratch_bytes=config['contexts'] * config['native_scratch_per_context'] * job['callers'])
    result['snapshot_preparation_ms'] = observations.get('snapshot_preparation_ms')
    return result
