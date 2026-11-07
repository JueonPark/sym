"""Frozen inputs, portable paths and explicit runtime/source provenance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'ce0bbe3b2451737ec35765bb1051841f5d19fc3d'
E3_SEQUENCE = (128, 256, 128, 33, 512, 33, 1, 128, 0, 65, 256, 65)
SELECTION_SEQUENCE = (4, 24, 96, 384, 1536, 24, 4, 384, 96, 1536)
HELDOUT_SEQUENCE = (7, 41, 7, 113, 257, 41, 1, 7, 0, 53, 113, 53)
E3_CASES = ('layout_cast', 'kv_restore', 'dequant_live', 'index_live')
SELECTION_CASES = ('selection_cast', 'selection_layout', 'selection_dequant', 'selection_index')
HELDOUT_CASES = ('held_cast_chain', 'held_layout', 'held_dequant_axis0', 'held_index_rank3')
E3_ARMS = ('torch', 'inductor', 'sym_auto_eager', 'sym_auto_inductor',
           'sym_prepared_eager', 'sym_prepared_inductor', 'sym_fast_reprepare', 'sym_fast_cache')
POLICIES = {
    'always_device': dict(max_entries=32, max_parameter_bytes=256 << 10, min_bytes=0, min_reuses=0, row_policy='device'),
    'always_cpu': dict(max_entries=32, max_parameter_bytes=256 << 10, min_bytes=0, min_reuses=0, row_policy='cpu'),
    'admit_4k': dict(max_entries=32, max_parameter_bytes=256 << 10, min_bytes=4096, min_reuses=1, row_policy='device'),
    'admit_64k': dict(max_entries=32, max_parameter_bytes=256 << 10, min_bytes=65536, min_reuses=1, row_policy='device'),
}


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


def load_e3(artifact):
    path = Path(artifact) / '03_dynamic_shapes/benchmark.py'
    spec = importlib.util.spec_from_file_location('frozen_e3_265', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SEQUENCE == E3_SEQUENCE and module.CASES == E3_CASES
    return module


def harness_hashes():
    return {name: digest(Path(__file__).parent / name) for name in
            ('support.py', 'workloads.py', 'worker.py', 'run.py', '../issue227/run.py')}


def artifact_hashes(artifact):
    return {name: digest(Path(artifact) / name) for name in
            ('03_dynamic_shapes/benchmark.py', '00_protocol/provenance.json')}


def runtime_hashes(build):
    build = Path(build)
    files = [p for p in build.rglob('*') if p.is_file() and
             (p.suffix == '.so' or p.name in ('sym-opt', 'sym-reloc-export', 'reloc-run-artifact') or
              (p.suffix == '.py' and 'python' in p.parts))]
    return {str(p.relative_to(build)): digest(p) for p in sorted(files)}


def environment():
    import torch
    return dict(affinity=sorted(os.sched_getaffinity(0)), torch=torch.__version__, torch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
