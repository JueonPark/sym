"""Build the preserved E1 control runner against an explicit runtime."""
import argparse
from pathlib import Path
import subprocess
import sysconfig
import pybind11

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.runtime.resolve() / 'libreloc'
    command = ['c++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fvisibility=hidden',
        '-mavx2', '-mf16c', '-ffp-contract=off', '-fno-fast-math', str(ROOT / 'native.cpp'),
        '-I' + str(REPO / 'libreloc/include'), '-I' + str(REPO / 'libreloc/src'),
        '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'],
        '-L' + str(library), '-lreloc_runtime', '-Wl,-rpath,' + str(library),
        '-o', str(args.output / ('_folding_native' + sysconfig.get_config_var('EXT_SUFFIX')))]
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
