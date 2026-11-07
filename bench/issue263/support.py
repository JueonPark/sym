"""Explicit artifact/build paths, immutable imports, and campaign provenance."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

BASELINE = '2617ba166f98a7476a5f472d829a573317fc0cfb'
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root, expected):
    actual = {name: digest(Path(root) / name) for name in expected}
    wrong = [name for name in expected if actual[name] != expected[name]]
    if wrong:
        raise ValueError('provenance mismatch: ' + ', '.join(wrong))
    return actual


def load_module(name, path, *, repository=None):
    """Override only the historical REPO assignment in memory before execution."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    tree = ast.parse(path.read_text(), filename=str(path))
    if repository is not None:
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'REPO' for t in node.targets)]
        if len(assignments) != 1:
            raise ValueError(f'expected one top-level REPO assignment in {path}')
        assignments[0].value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                       args=[ast.Constant(str(repository))], keywords=[])
        ast.fix_missing_locations(tree)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def load_artifact(root, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    inventory = {x['path']: x['sha256'] for x in json.loads((root / 'SHA256-MANIFEST.json').read_text())['files']}
    files = ['02_fp16_attribution/benchmark.py', '02_fp16_attribution/controls.py',
             '05_lifecycle_concurrency/workload.py', '00_protocol/observed_memory.py']
    files += [str(p.relative_to(root)) for p in (root / '02_fp16_attribution').glob('*.so')]
    if not any(p.endswith('.so') for p in files):
        raise ValueError('artifact native control helper is missing; rebuild it for this Python/Torch environment')
    hashes = verify_files(root, {p: inventory[p] for p in files})
    sys.path[:0] = [str(root / '02_fp16_attribution'), str(root / '05_lifecycle_concurrency'),
                   str(root / '00_protocol'), str(repo)]
    controls = load_module('controls', root / '02_fp16_attribution/controls.py')
    e2 = load_module('benchmark', root / '02_fp16_attribution/benchmark.py', repository=repo)
    e5 = load_module('workload', root / '05_lifecycle_concurrency/workload.py')
    memory = load_module('observed_memory', root / '00_protocol/observed_memory.py')
    return SimpleNamespace(root=root, controls=controls, e2=e2, e5=e5, memory=memory, hashes=hashes)


def add_arguments(parser):
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--variant', choices=('baseline', 'no_report', 'no_check', 'both'), default='baseline')
    parser.add_argument('--instrument', choices=('off', 'counters', 'nvtx'), default='off')
    parser.add_argument('--allow-new-build', action='store_true', help='record different runtime hashes; not the original-baseline qualification')


def configure(args, *, threads=8):
    args.artifact, args.build = args.artifact.resolve(), args.build.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.build / 'python'))
    os.environ.update(SYM_RELOC_EXPORT=str(args.build / 'sym/tools/sym-reloc-export'),
                      SYM_OPT=str(args.build / 'sym/tools/sym-opt'))
    original = json.loads((args.artifact / '00_protocol/final-runtime-verification.json').read_text())
    recorded = {x['path'].split('/build/perf/', 1)[1]: x['sha256'] for x in original['files']}
    actual = {name: digest(args.build / name) for name in recorded}
    mismatches = [name for name in recorded if recorded[name] != actual[name]]
    if mismatches and not args.allow_new_build:
        raise ValueError('runtime differs from published baseline: ' + ', '.join(mismatches))
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    from torch._inductor import config
    config.emulate_precision_casts = True
    return dict(baseline_runtime_commit=BASELINE, runtime_sha256=actual, baseline_hash_mismatches=mismatches)


def provenance(args, artifact, runtime):
    import torch
    import pyreloc
    import reloc_torch
    return dict(runtime, artifact_sha256=artifact.hashes, python=sys.version,
                torch=torch.__version__, torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                runtime_packages=[pyreloc.__file__, reloc_torch.__file__],
                harness_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                harness_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH',
                    'TORCHINDUCTOR_CACHE_DIR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT',
                    'TORCHINDUCTOR_COMPILE_THREADS')})


def save(path, data):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)


def allocation_snapshot(owner=None):
    import torch
    stats = torch.cuda.memory_stats()
    return dict(torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),
                torch_allocation_requests=stats.get('allocation.all.allocated', 0),
                python_live_allocated_blocks=sys.getallocatedblocks(),
                native=owner.stats()['resources'] if owner else None)
