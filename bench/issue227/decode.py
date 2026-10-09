#!/usr/bin/env python3
"""Reuse #225's exact decode workload with an additional placement path."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'issue225'))
import compare
from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend, TransferResources

# Import benchmark settings without shadowing reloc_torch.placement.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from placement import CONTEXT, LIMITS, OPTIONS

OriginalState = compare.PathState


class State(OriginalState):
    def __init__(self,name,lifetime,dynamic=False):
        super().__init__(name,lifetime,dynamic)
        if name == 'sym_placement_inductor':
            self.backend.close()
            resources = lifetime.enter_context(TransferResources(**LIMITS))
            profile = PlacementProfile.load(os.environ['SYM_PLACEMENT_PROFILE'])
            self.backend = RelocBackend(compute_backend='inductor',
                placement=PlacementPolicy(profile,context=CONTEXT,capacity=512),
                transfer_resources=resources,transfer_options=OPTIONS)
            lifetime.callback(self.backend.close)


compare.PathState = State
compare.PATHS += ('sym_placement_inductor',)

if __name__ == '__main__':
    compare.main()
