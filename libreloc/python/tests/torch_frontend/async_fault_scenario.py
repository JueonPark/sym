"""Separate process: completion failures must drain or quarantine every owner."""
import ctypes
import gc
import json
import os
import sys
import weakref

import torch
import pyreloc
from reloc_torch import CompilerClient, TransferQueue
from test_async_transfers import artifact, group

mode = int(sys.argv[1])
wire = len(sys.argv) > 2 and sys.argv[2] == 'wire'
shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
compiled = artifact(CompilerClient.from_environment())
weights = None
if wire:
    from reloc_torch import PreparedWireWeights
    weights = PreparedWireWeights(gather_threads=1)
    weights.prepare('w', torch.ones((256, 256), dtype=torch.int8), torch.ones(256))
    queue = weights._transfer_queue()
    source = None
    references = []
else:
    queue = TransferQueue(gather_threads=1)
    source = torch.ones(65536)
    references = [weakref.ref(source)]
submit = pyreloc.submit_dispatch_group

def inject(request, **options):
    references.extend(weakref.ref(tensor) for tensor in options['owners'][1])
    if wire:
        references.extend(weakref.ref(tensor) for tensor in options['owners'][0])
    options['caller_stream'] = None  # inject after payload, not at producer order
    if mode <= 4:
        shim.sym_dispatch_fault_mode(mode)
    return submit(request, **options)

pyreloc.submit_dispatch_group = inject
handle = None
try:
    # Three copies make the earlier member genuinely in-flight for modes 3/4.
    from reloc_torch import prepare_transfer_group
    prepared = (weights._group(['w'] * 3) if wire else
                prepare_transfer_group(tuple(group(compiled, source).requests[0] for _ in range(3))))
    handle = queue.submit(prepared)
    assert mode in (5, 6)
    shim.sym_dispatch_fault_mode(mode)
    handle.wait()
except (RuntimeError, pyreloc.TransferError) as error:
    assert ('completion_unknown' if mode in (2, 4, 6) else 'backend_failure') in str(error), error
else:
    raise AssertionError('fault did not reach asynchronous submission/completion')
finally:
    shim.sym_dispatch_fault_mode(0)
try:
    (weights or queue).close()
except (RuntimeError, pyreloc.TransferError):
    pass  # the stored failure must still be observable during close
del source, prepared, handle
gc.collect()
unknown = mode in (2, 4, 6)
stats = queue.stats()
assert stats['held'] == 0
assert stats['resources']['quarantined'] == unknown
assert all((ref() is not None) == unknown for ref in references)
if wire:
    assert (weights.stats()['prepared_bytes'] > 0) == unknown
    assert (weights.stats()['pinned_bytes'] > 0) == unknown
torch.cuda.synchronize()
print(json.dumps({'passed': True, 'mode': mode, 'unknown': unknown}))
