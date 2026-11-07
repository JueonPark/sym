"""Pure campaign membership, hashes and portable immutable workload loading."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
CPUS = (4, 5, 6, 7, 20, 21, 22, 23)


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
