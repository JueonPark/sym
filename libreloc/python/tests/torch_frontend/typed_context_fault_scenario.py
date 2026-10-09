"""One unknown-completion context must not release pending sibling owners."""
import ctypes
import gc
import json
import os
import weakref

import torch
import pyreloc
from reloc_torch import CompilerClient, dispatch
from reloc_torch.group import _execute_transfer_group
from test_typed_contexts import artifact, group, owner, request

shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
compiled = artifact(CompilerClient.from_environment())
resources = owner()
good = torch.full((65539,), 7.)
good_ref = weakref.ref(good)
native, outputs, _ = _execute_transfer_group(group(compiled, good, 1),
    resources=resources, gather_threads=1, submit=True)
del good
bad = torch.full((65539,), 3.)
bad_ref = weakref.ref(bad)
prepared = request(compiled, bad, 0)
execute = pyreloc.execute_dispatch
def inject(req, **options):
    options['caller_stream'] = None
    shim.sym_dispatch_fault_mode(2)
    try: return execute(req, **options)
    finally: shim.sym_dispatch_fault_mode(0)
pyreloc.execute_dispatch = inject
try:
    dispatch.execute_typed_transfer(prepared, resources=resources,
        gather_threads=1, n_streams=1, pinning='pinned')
except RuntimeError as error:
    assert 'completion_unknown' in str(error), error
else:
    raise AssertionError('missing injected failure')
del prepared, bad
assert good_ref() is not None and bad_ref() is not None
assert resources.stats()['typed']['quarantined']
native.wait()
del native
gc.collect()
assert good_ref() is None and bad_ref() is not None
assert torch.equal(outputs[0].cpu(), torch.full((65539,), 7., dtype=torch.float16))
try: resources.close()
except pyreloc.TransferError as error: assert 'completion_unknown' in str(error)
else: raise AssertionError('quarantined close succeeded')
stats = resources.stats()['typed']
assert stats['active_contexts'] == 0 and stats['peak_active_contexts'] == 2
assert 0 < stats['live_bytes'] <= stats['limits']['live_bytes']
torch.cuda.synchronize()
print(json.dumps({'passed': True, 'resources': stats}))
