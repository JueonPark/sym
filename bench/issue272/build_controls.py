"""Build matched CUDA runtimes with/without the bounded typed-kernel changes."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pybind11

REPO = Path(__file__).resolve().parents[2]
KERNEL_COMMIT = 'b319a0ab94dbc8c8264e4fa093ffbbb796e2c046'


def build(root, source, python_bindings, arch, jobs):
    subprocess.run([
        'cmake', '-S', str(Path(__file__).resolve().parent), '-B', str(root), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={arch}',
        f'-DRUNTIME_SOURCE={source}', f'-DPython_EXECUTABLE={sys.executable}',
        f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
        f'-DSYM_BUILD_PYTHON={"ON" if python_bindings else "OFF"}',
    ], check=True)
    subprocess.run(['cmake', '--build', str(root), '--parallel', str(jobs)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compiler-bin', required=True, type=Path)
    parser.add_argument('--cuda-arch', required=True)
    parser.add_argument('--jobs', default=2, type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error('choose a new output directory')
    tools = [args.compiler_bin.resolve() / name for name in ('sym-opt', 'sym-reloc-export')]
    if not all(path.is_file() for path in tools):
        parser.error('--compiler-bin must contain sym-opt and sym-reloc-export')
    root.mkdir(parents=True)
    on, off = root / 'on', root / 'off'
    control = root / 'source-off'
    shutil.copytree(REPO / 'libreloc', control / 'libreloc')
    # Undo just the introduced native kernel changes on a copy of current
    # sources. All later pipeline, resource and binding changes remain present.
    patch = subprocess.check_output([
        'git', 'diff', KERNEL_COMMIT + '^', KERNEL_COMMIT, '--',
        'libreloc/CMakeLists.txt', 'libreloc/src', 'libreloc/include',
    ], cwd=REPO)
    # Apply relative to the copy even when the output lives inside a worktree.
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '--reverse'],
                   cwd=control, input=patch, check=True)
    build(on, REPO, True, args.cuda_arch, args.jobs)
    build(off, control, False, args.cuda_arch, args.jobs)
    # Both arms load byte-identical extensions. Their origin-relative RUNPATH
    # selects the runtime in the corresponding build tree.
    shutil.copytree(on / 'python', off / 'python')
    for tree in (on, off):
        for package in ('pyreloc', 'reloc_torch'):
            shutil.copytree(REPO / 'libreloc/python' / package, tree / 'python' / package,
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
        target = tree / 'sym/tools'
        target.mkdir(parents=True)
        for path in tools:
            (target / path.name).symlink_to(path)


if __name__ == '__main__':
    main()
