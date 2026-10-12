#!/usr/bin/env python3
"""Rerun all eight #225 regimes with a separately calibrated placement path."""
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench/issue225'))
from bench.issue225 import compare
from bench.issue227.placement import CONTEXT, LIMITS, OPTIONS
from bench.issue229.transfers import qualified_metadata
from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend, TransferResources

PATHS = (*compare.PATHS, 'sym_placement_inductor')


class State(compare.PathState):
    def __init__(self, name, lifetime, dynamic=False):
        super().__init__(name, lifetime, dynamic)
        if name == 'sym_placement_inductor':
            self.backend.close()
            resources = lifetime.enter_context(TransferResources(**LIMITS))
            self.backend = RelocBackend(compute_backend='inductor',
                placement=PlacementPolicy(
                    PlacementProfile.load(os.environ['SYM_PLACEMENT_PROFILE']),
                    context=CONTEXT, capacity=512),
                transfer_resources=resources, transfer_options=OPTIONS)
            lifetime.callback(self.backend.close)


def metadata():
    result = qualified_metadata()
    sources = [*Path(__file__).parent.glob('*.py'),
               ROOT / 'bench/issue227/placement.py', ROOT / 'bench/issue227/run.py']
    result['benchmark_sha256'].update({str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    profile = Path(os.environ['SYM_PLACEMENT_PROFILE'])
    result['placement_profile'] = dict(path=str(profile),
        sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), context=CONTEXT,
        resource_limits=LIMITS, transfer_options=OPTIONS)
    return result


if __name__ == '__main__':
    compare.PathState = State
    compare.PATHS = PATHS
    compare.metadata = metadata
    compare.main()
