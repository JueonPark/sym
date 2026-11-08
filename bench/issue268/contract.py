"""Frozen matrix and content-addressed dependencies (no Torch import)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPUS = [4, 5, 6, 7, 20, 21, 22, 23]
CALLER_CPUS = [4, 20, 5, 21, 6, 22, 7, 23]
MODES = ('plain', 'off', 'on')


def cells():
    result = []
    for round_ in (1, 2, 3):
        for compute in ('eager', 'inductor'):
            for case in ('llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense'):
                result.append(dict(kind='e2', case=case, compute=compute, callers=1, round=round_))
            for case in ('llm_decode', 'moe_sparse'):
                for callers in ((1, 2, 4) if compute == 'eager' else (1,)):
                    result.append(dict(kind='e5', case=case, compute=compute, callers=callers, round=round_))
    for cell in result:
        cell['id'] = '-'.join(str(v) for v in cell.values())
    return result


def arms(cell):
    choices = [(rt, mode) for mode in MODES for rt in ('torch', 'sym')]
    shift = cell['round'] * 2 % len(choices)
    return [cell | dict(runtime=rt, mode=mode, id=cell['id'] + '-' + rt + '-' + mode)
            for rt, mode in choices[shift:] + choices[:shift]]


def jobs():
    return [job for cell in cells() for job in arms(cell)]


def protocol(job):
    return dict(dtype='float16', kernel='native', grouping='grouped', prefetch='whole',
                slots=2, tf32=False, samples=60 if job['kind'] == 'e2' else 30,
                warmup_requests_per_caller=5 if job['kind']=='e2' else 24//job['callers'])


def environment(torch, job):
    import os
    import sys
    return dict(torch=torch.__version__, python=list(sys.version_info[:3]),
                gpu=torch.cuda.get_device_name(0), affinity=sorted(os.sched_getaffinity(0)),
                threads=torch.get_num_threads(), interop=torch.get_num_interop_threads())


def expected_environment(job):
    return dict(torch='2.14.0+cu126', python=[3, 14, 7], gpu='NVIDIA GeForce RTX 2080 Ti',
                affinity=CPUS, threads=8 // job['callers'], interop=1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def provenance(runtime, artifact, compiler):
    directories = ('bench/issue268', 'bench/issue266', 'bench/issue227', 'bench/beat_torch',
                   'bench/issue221', 'bench/issue223', 'bench/issue224', 'bench/issue225',
                   'libreloc/python/examples/workloads', 'libreloc/python/reloc_torch')
    source = {str(p.relative_to(ROOT)): sha(p) for d in directories for p in (ROOT / d).glob('*.py')
              if not p.name.startswith('test_') and p.name != 'analyze.py'}
    inputs = {str(p.relative_to(artifact)): sha(p) for d in
              ('00_protocol', '02_fp16_attribution', '05_lifecycle_concurrency')
              for p in (artifact / d).glob('*') if p.suffix in ('.py', '.so')}
    inventory = {r['path']: r['sha256'] for r in json.loads((artifact / 'SHA256-MANIFEST.json').read_text())['files']}
    if any(inventory.get(p) != digest for p, digest in inputs.items()):
        raise ValueError('historical artifact input changed')
    native = {str(p.relative_to(runtime)): sha(p) for d in ('python/pyreloc', 'python/reloc_torch', 'libreloc')
              for p in (runtime / d).glob('*') if p.suffix in ('.py', '.so')}
    for p in (runtime / 'python/reloc_torch').glob('*.py'):
        if sha(p) != sha(ROOT / 'libreloc/python/reloc_torch' / p.name):
            raise ValueError('runtime Python differs from source')
    return dict(source=source, inputs=inputs, runtime=native, compiler=sha(compiler))
