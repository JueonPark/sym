"""Pure campaign membership, hashes and portable immutable workload loading."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
CPUS = (4, 5, 6, 7, 20, 21, 22, 23)


def environment_contract(job):
    return dict(python_version=[3, 14, 7], torch='2.14.0+cu126', torch_cuda='12.6',
                gpu='NVIDIA GeForce RTX 2080 Ti', capability=[7, 5], affinity=list(CPUS),
                threads=8 // job.get('callers', 1), interop_threads=1)


def executed_configuration(job, policy):
    if job['kind'] == 'e4':
        path = 'resident' if job['placement'] == 'resident' else job['placement'] + '_' + job['runtime']
        return dict(round=job['round'], path=path, layers=job['layers'], context=job['context'],
                    budget_mib=job['budget_mib'], resident_layers=policy['resident_groups'], samples=20, warmup=3)
    if job['kind'] == 'e5':
        return dict(round=job['round'], case=job['case'], runtime=job['runtime'], callers=job['callers'], batches=30)
    return dict(round=job['round'], size=job['size'], runtime=job['runtime'], policy=job['policy'],
                budget_mib=job['budget_mib'], samples=100, warmup=3, members=4)


def matrix():
    jobs = []
    for round_ in range(1, 4):
        for layers in (16, 32):
            for context in (16, 512):
                for budget in (512, 1024):
                    arms = [('resident', 'torch', 'fresh'), ('resident', 'torch', 'packed'),
                            ('partial', 'torch', 'fresh'), ('partial', 'sym', 'fresh'),
                            ('offload', 'torch', 'fresh'), ('offload', 'sym', 'fresh'),
                            ('partial', 'torch', 'adaptive'), ('partial', 'sym', 'adaptive')]
                    # Isolate fixed packing/recycling at the previously failing
                    # partial-residency cell and a smaller-operation control.
                    if budget == 512 and (layers, context) in ((32, 512), (16, 16)):
                        arms += [('partial', rt, policy) for rt in ('torch', 'sym')
                                 for policy in ('packed', 'recycle')]
                    for placement, runtime, policy in arms[round_ % len(arms):] + arms[:round_ % len(arms)]:
                        jobs.append(dict(kind='e4', round=round_, layers=layers, context=context,
                                         budget_mib=budget, placement=placement, runtime=runtime, policy=policy))
        for case in ('llm_decode', 'moe_sparse'):
            for callers in (1, 2, 4):
                policies = ('fresh', 'adaptive', 'packed', 'recycle') if callers == 1 else ('fresh', 'adaptive')
                for policy in policies:
                    for runtime in (('torch', 'sym') if round_ % 2 else ('sym', 'torch')):
                        jobs.append(dict(kind='e5', round=round_, case=case, callers=callers,
                                         budget_mib=1024, runtime=runtime, policy=policy))
        for size in ('small', 'large'):
            for policy in ('fresh', 'packed', 'recycle', 'adaptive'):
                for runtime in ('torch', 'sym'):
                    jobs.append(dict(kind='groups', round=round_, size=size,
                                     budget_mib=512, runtime=runtime, policy=policy))
    for job in jobs:
        job['id'] = '-'.join(str(v) for v in job.values())
    return jobs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def provenance(artifact, build):
    artifact, build = Path(artifact), Path(build)
    sources = [p for directory in ('bench/issue266', 'bench/beat_torch', 'bench/issue221',
        'bench/issue223', 'bench/issue224', 'bench/issue225', 'libreloc/python/examples/workloads')
        for p in (ROOT / directory).glob('*.py') if p.name not in ('analyze.py', 'test_analysis.py')]
    sources += [ROOT / 'bench/issue227/run.py']
    sources += [ROOT / 'libreloc/python' / name for name in
                ('PyTorchViews.cpp', 'PyTorchStorage.cpp', 'CMakeLists.txt')]
    sources += list((ROOT / 'libreloc/python/reloc_torch').glob('*.py'))
    for source in (ROOT / 'libreloc/python/reloc_torch').glob('*.py'):
        if digest(source) != digest(build / 'python/reloc_torch' / source.name):
            raise ValueError('candidate Python differs from reviewed source: ' + source.name)
    baseline_path = Path(__file__).with_name('baseline-inputs.json')
    baseline = json.loads(baseline_path.read_text())
    for name, expected in baseline['repository_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('original workload source changed: ' + name)
    sources.append(baseline_path)
    inputs = [p for directory in ('00_protocol', '02_fp16_attribution', '04_memory_budget',
        '05_lifecycle_concurrency') for p in (artifact / directory).glob('*')
        if p.suffix in ('.py', '.so') and p.name != 'analyze.py']
    inputs += [artifact / '00_protocol/provenance.json']
    inventory = {row['path']: row['sha256'] for row in
                 json.loads((artifact / 'SHA256-MANIFEST.json').read_text())['files']}
    for path in inputs:
        if digest(path) != inventory[str(path.relative_to(artifact))]:
            raise ValueError('original artifact input changed: ' + str(path))
    runtime = [p for p in build.rglob('*') if p.is_file() and
        (p.suffix in ('.py', '.so') or p.name in ('sym-opt', 'sym-reloc-export'))]
    return dict(source={str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
                inputs={str(p.relative_to(artifact)): digest(p) for p in sorted(inputs)},
                runtime={str(p.relative_to(build)): digest(p) for p in sorted(runtime)})


def load_module(name, path, *, repository=None):
    """Only replace the historical absolute REPO assignment, never model code."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    tree = ast.parse(path.read_text(), filename=str(path))
    if repository is not None:
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'REPO' for t in n.targets)]
        if len(assignments) != 1: raise ValueError('expected exactly one repository path assignment')
        assignments[0].value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                       args=[ast.Constant(str(repository))], keywords=[])
        ast.fix_missing_locations(tree)
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def load_inputs(root):
    root = Path(root)
    sys.path[:0] = [str(root / p) for p in ('02_fp16_attribution', '05_lifecycle_concurrency', '00_protocol')]
    controls = load_module('controls', root / '02_fp16_attribution/controls.py')
    e2 = load_module('benchmark', root / '02_fp16_attribution/benchmark.py', repository=ROOT)
    workload = load_module('workload', root / '05_lifecycle_concurrency/workload.py')
    memory = load_module('observed_memory', root / '00_protocol/observed_memory.py')
    return SimpleNamespace(root=root, controls=controls, e2=e2, workload=workload, memory=memory)


def policy_configuration(job, planner=None):
    if planner is None:
        from libreloc.python.reloc_torch.memory import plan_transfer_memory
    else:
        plan_transfer_memory = planner
    kind = job['kind']
    if kind == 'e4':
        count, output, scratch, small = job['layers'], 24 * MIB, 16 * MIB, 8 * MIB
        workspace = 128 * MIB + job['layers'] * job['context'] * 4096 + job['context'] * 1024 * 16 + 32 * job['context'] ** 2
        available, safety, prefix = job['budget_mib'] * MIB, 32 * MIB, 8
    elif kind == 'e5':
        llm = job['case'] == 'llm_decode'
        count, output, scratch, small = (4, 24 * MIB, 16 * MIB, 8 * MIB) if llm else (16, 8 * MIB, 8 * MIB, 6 * MIB)
        # One 256 MiB aggregate transfer allowance; remaining physical capacity
        # covers shared fixture, compute/KV, held request outputs and libraries.
        available, workspace, safety, prefix = 256 * MIB // job['callers'], 0, 8 * MIB, 0
    else:
        count, output = 16, (8 * 64 * 2 if job['size'] == 'small' else 24 * MIB)
        scratch, small = (2 * MIB, MIB) if job['size'] == 'small' else (16 * MIB, 8 * MIB)
        available, workspace, safety, prefix = 2 * output + scratch + 8 * MIB, 0, 8 * MIB, 0
    plan = plan_transfer_memory([output] * count, [scratch] * count,
        available_bytes=available, workspace_bytes=workspace, safety_bytes=safety,
        max_resident_groups=prefix, subgroup_scratch_bytes=[small] * count)
    adaptive = job['policy'] == 'adaptive'
    slots = plan.slots if adaptive else 2
    native = plan.native_scratch_bytes if adaptive else (64 * MIB // job.get('callers', 1))
    return dict(policy=job['policy'], plan=asdict(plan), plan_applied=adaptive,
                contexts=plan.contexts if adaptive else 2, slots=slots,
                native_scratch_per_context=native, subgroup_members=plan.subgroup_members if adaptive else None,
                output_capacity=plan.output_bytes if adaptive else (slots + 1) * output,
                resident_groups=plan.resident_groups if adaptive else
                    (job['layers'] if job.get('placement') == 'resident' else 8 if job.get('placement') == 'partial' else 0),
                packed=job['policy'] != 'fresh', recycle=job['policy'] in ('recycle', 'adaptive'))
