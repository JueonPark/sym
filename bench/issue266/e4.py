"""Execute the original E4 loop with factory and residency-window adapters."""
import json
import sys
from bench.issue266.policies import configuration, install, resident_preload
from bench.issue266.support import load_module


def run(job, inputs, output):
    module = load_module('e4_266', inputs.root / '04_memory_budget/benchmark.py')
    config, observations = configuration(job), {}
    install(module, inputs, config, observations)
    if config['packed']:
        original = module.window
        calls = 0
        def window(producer, groups):
            nonlocal calls
            calls += 1
            return resident_preload(producer, groups, observations) if calls == 1 else original(producer, groups)
        module.window = window
    path = 'resident' if job['placement'] == 'resident' else job['placement'] + '_' + job['runtime']
    sys.argv = [str(module.__file__), '--output', str(output), '--round', str(job['round']), '--path', path,
                '--layers', str(job['layers']), '--context', str(job['context']), '--budget-mib', str(job['budget_mib']),
                '--resident-layers', str(config['resident_groups']), '--samples', '20', '--warmup', '3']
    failure = None
    try:
        module.main()
    except Exception as error:
        failure = error
    result = json.loads(output.read_text()) if output.exists() else {}
    if failure is not None:
        from reloc_torch.memory import MemoryBudgetError
        if isinstance(failure, MemoryBudgetError) and result.get('active_stage') in (
                'preparation_start', 'residency_start', 'first_request', 'warmup_request', 'timed_request'):
            result.update(outcome='admission', admission_phase=result['active_stage'], complete=True)
        else:
            raise failure
    result['optimization'] = dict(config, **observations)
    return result
