"""A failed chunk upload must retain the ring and all tensor owners until drained."""
import ctypes
import gc
import json
import os
import sys
import weakref

import torch
import pyreloc
from reloc_torch import CompilerClient, TransferResources, dispatch
from reloc_torch.recipe import Cast, IndexSelect, Recipe, TensorSpec
from reloc_torch.symbolic import Const

mode = int(sys.argv[1])
shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
def spec(dtype):
    return TensorSpec((Const(65539),), (Const(1),), Const(0), dtype)
recipe = Recipe(spec('float32'), (Cast('float16', 'ieee_rne'),), spec('float16'), 'h2d')
indexed = len(sys.argv) > 2 and sys.argv[2] == '1'
if indexed:
    recipe = Recipe(spec('float32'), (IndexSelect(spec('int64')), Cast('float16', 'ieee_rne')),
                    spec('float16'), 'h2d')
compiled = CompilerClient.from_environment().compile(recipe)
owner = TransferResources(max_typed_live_bytes=32768)
source = torch.ones(65539)
if indexed:
    from reloc_torch.index_select import prepare_index_select_transfer
    indices = torch.arange(65539).remainder_(17)
    request = prepare_index_select_transfer(compiled, source, indices, 'cuda:0')
else:
    request = dispatch.prepare_typed_transfer(compiled, source, 'cuda:0', policy='original_cpu')
refs = []
execute = pyreloc.execute_dispatch
def inject(request, **options):
    refs.extend(weakref.ref(tensor) for tensor in options['owners'])
    options['caller_stream'] = None  # inject after payload submission
    shim.sym_dispatch_fault_mode(mode)
    try:
        return execute(request, **options)
    finally:
        shim.sym_dispatch_fault_mode(0)
pyreloc.execute_dispatch = inject
try:
    dispatch.execute_typed_transfer(request, resources=owner, n_buffers=2,
        chunk_size=16384, gather_threads=1, pinning='pinned')
except RuntimeError as error:
    assert ('completion_unknown' if mode in (2, 4) else 'backend_failure') in str(error), error
else:
    raise AssertionError('fault did not reach the typed pipeline')
assert request.consumed
del source, request
gc.collect()
snapshot = owner.stats()['typed']
unknown = mode in (2, 4)
assert snapshot['quarantined'] == unknown
assert all((ref() is not None) == unknown for ref in refs)
assert snapshot['copy_calls'] >= (3 if mode in (3, 4) else 1)
if unknown:
    assert snapshot['retained_bytes'] == 32768
    try:
        owner.close()
    except pyreloc.TransferError as error:
        assert 'completion_unknown' in str(error)
    else:
        raise AssertionError('quarantined close succeeded')
else:
    assert snapshot['retained_bytes'] == 0
    owner.close()
torch.cuda.synchronize()
print(json.dumps({'passed': True, 'mode': mode, 'resources': snapshot}))
