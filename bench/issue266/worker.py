"""One fresh-process job from an immutable campaign manifest."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue266.support import CPUS, load_inputs, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    job, = [j for j in manifest['jobs'] if j['id'] == args.job]
    assert sorted(os.sched_getaffinity(0)) == list(CPUS)
    build = Path(manifest['build'])
    sys.path.insert(0, str(build / 'python'))
    import torch
    import pyreloc
    import reloc_torch
    assert Path(pyreloc.__file__).is_relative_to(build)
    assert Path(reloc_torch.__file__).is_relative_to(build)
    inputs = load_inputs(manifest['artifact'])
    output = args.manifest.parent / (job['id'] + '.raw.json')
    if output.exists(): raise FileExistsError(output)
    module = importlib.import_module('bench.issue266.' + job['kind'])
    result = module.run(job, inputs, output)
    record = dict(job=job, result=result, provenance=manifest['provenance'], complete=True,
                  environment=dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                    affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(), packages=[pyreloc.__file__, reloc_torch.__file__]))
    save(args.manifest.parent / (job['id'] + '.record.json'), record)


if __name__ == '__main__': main()
