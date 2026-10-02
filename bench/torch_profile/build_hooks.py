#!/usr/bin/env python3
"""Build a diagnostic preload library against the active Torch headers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--nvtx-include', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
include = Path(torch.__file__).parent/'include'
lib = Path(torch.__file__).parent/'lib'
command = ['g++', '-std=c++20', '-O2', '-DNDEBUG', '-shared', '-fPIC', '-Wall', '-Wextra',
           '-I'+str(include), '-I'+str(a.nvtx_include), str(Path(__file__).with_name('native_hooks.cpp')),
           '-o', str(a.output), '-L'+str(lib), '-Wl,-rpath,'+str(lib), '-ltorch_cpu', '-lc10', '-ldl']
subprocess.run(command, check=True)
a.output.with_suffix('.json').write_text(json.dumps(dict(
    command=command, torch=torch.__version__, torch_git=torch.version.git_version,
    source_sha256=hashlib.sha256(Path(__file__).with_name('native_hooks.cpp').read_bytes()).hexdigest(),
    sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()), indent=2)+'\n')
