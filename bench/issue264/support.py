"""Frozen workload reuse, runtime-policy selection and explicit provenance."""
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
POLICIES = {'baseline': None, 'legacy-eager': (False, 'eager'), 'legacy-native': (False, 'native'),
            'prepared-eager': (True, 'eager'), 'prepared-native': (True, 'native')}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def code_hashes(directory):
    root = Path(directory)
    return {p.name: digest(p) for p in sorted(root.glob('*')) if p.suffix in ('.py', '.json')}


def measurement_hashes():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in ('support.py', 'groups.py', 'models.py', 'e5.py', 'run.py')}


def runtime_hashes(build):
    build = Path(build)
    files = [build / 'libreloc/libreloc_runtime.so', build / 'sym/tools/sym-reloc-export',
             build / 'sym/tools/sym-opt']
    for package in ('pyreloc', 'reloc_torch', 'sym_reloc'):
        files += [p for p in (build / 'python' / package).glob('*') if p.suffix in ('.py', '.so')]
    return {str(p.relative_to(build)): digest(p) for p in sorted(files)}


def dependency(harness):
    root = Path(harness).resolve()
    if not (root / 'bench/issue263/support.py').is_file():
        raise ValueError('--harness must name the extracted #263 source repository')
    sys.path.insert(0, str(root))
    return importlib.import_module('bench.issue263.support')


def add_arguments(parser):
    for key in ('artifact', 'harness', 'build', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--round', type=int, default=1)


def configure(args, *, threads=8):
    for key in ('artifact', 'harness', 'build', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    sys.path.insert(0, str(args.build / 'python'))
    old = dependency(args.harness)
    args.allow_new_build = args.policy != 'baseline'
    args.variant, args.instrument = 'baseline', 'off'
    runtime = (old.configure(args, threads=threads) if args.policy == 'baseline'
               else configure_candidate(args, old, threads=threads))
    artifact = old.load_artifact(args.artifact, repo=REPO)
    return old, runtime, artifact


def configure_candidate(args, old, *, threads=8):
    # The baseline build contains historical, unused Python modules that are
    # absent from current sources. Record absence instead of copying stale code.
    args.artifact, args.build, args.output = args.artifact.resolve(), args.build.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.build / 'python'))
    os.environ.update(SYM_RELOC_EXPORT=str(args.build / 'sym/tools/sym-reloc-export'),
                      SYM_OPT=str(args.build / 'sym/tools/sym-opt'))
    original = json.loads((args.artifact / '00_protocol/final-runtime-verification.json').read_text())
    recorded = {x['path'].split('/build/perf/', 1)[1]: x['sha256'] for x in original['files']}
    recorded.update(old.baseline_sources()['extra_runtime_sha256'])
    actual = runtime_hashes(args.build)
    mismatches = [name for name in recorded if actual.get(name) != recorded[name]]
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    from torch._inductor import config
    config.emulate_precision_casts = True
    return dict(baseline_runtime_commit=old.BASELINE, runtime_sha256=actual,
                baseline_hash_mismatches=mismatches,
                absent_baseline_files=[name for name in recorded if name not in actual])


def metadata(args, old, runtime, artifact):
    result = old.provenance(args, artifact, runtime)
    result.update(runtime_sha256=runtime_hashes(args.build),
                  dependency_sha256=code_hashes(args.harness / 'bench/issue263'),
                  harness_sha256=measurement_hashes(),
                  harness_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  torch_cuda=__import__('torch').version.cuda)
    return result


@contextmanager
def policy(name):
    """Only constructor options change; no validator/report/consumer bypasses."""
    from reloc_torch.fast import FastTransfers
    original = FastTransfers.__init__
    configured = []
    values = POLICIES[name]
    def initialize(owner, *args, **kwargs):
        if values is not None:
            kwargs.update(prepared_groups=values[0], report_mode=values[1])
        original(owner, *args, **kwargs)
        configured.append(dict(prepared_groups=getattr(owner, 'prepared_groups', None),
                               report_mode=getattr(owner, 'report_mode', None)))
    FastTransfers.__init__ = initialize
    result = dict(policy=name, owners=configured, validators=True, accounting=True,
                  source_values='live', outputs='fresh unless explicitly supplied')
    try:
        yield result
    finally:
        FastTransfers.__init__ = original
