"""Subprocess fault qualification after one or more group payloads submit."""
import ctypes
import gc
import json
import os
import sys
import weakref

import torch
import pyreloc
from reloc_torch import CompilerClient, TransferResources, prepare_transfer_group, execute_transfer_group
from reloc_torch import dispatch
from reloc_torch.recipe import Dequantize, InlineParam, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

mode = int(sys.argv[1])
shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
shape = (Const(128), Const(128))
def spec(dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
recipe = Recipe(spec('int8'), (Dequantize('float32', InlineParam('float32', (), (0x3f000000,)),
                    None, None, 'affine'), Transpose((1, 0))), spec('float32'), 'h2d')
compiled = CompilerClient.from_environment().compile(recipe)
owner = TransferResources()
sources = [torch.ones(128, 128, dtype=torch.int8) for _ in range(3)]
members = [dispatch.prepare_typed_transfer(compiled, source, 'cuda:0',
    implementation='cuda_dequant_relocate', threads=1) for source in sources]
group = prepare_transfer_group(members)
refs = []
execute = pyreloc.execute_dispatch_group

def inject(request, **options):
    refs.extend(weakref.ref(tensor) for part in options['owners'] for tensor in part)
    options['caller_stream'] = None
    shim.sym_dispatch_fault_mode(mode)
    try:
        return execute(request, **options)
    finally:
        shim.sym_dispatch_fault_mode(0)
pyreloc.execute_dispatch_group = inject
try:
    execute_transfer_group(group, resources=owner, gather_threads=1)
except RuntimeError as error:
    assert ('completion_unknown' if mode in (2, 4) else 'backend_failure') in str(error), error
else:
    raise AssertionError('fault did not reach group execution')
assert group.consumed and all(member.consumed for member in members)
assert group.native.report['copy_calls'] >= 2
del sources, members, group
# No accidental Python reference hides missing native quarantine ownership.
gc.collect()
snapshot = owner.stats()['typed']
unknown = mode in (2, 4)
assert snapshot['quarantined'] == unknown, snapshot
assert all((ref() is not None) == unknown for ref in refs)
if unknown:
    assert snapshot['retained_bytes'] > 0
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
print(json.dumps({'mode': mode, 'passed': True, 'resources': snapshot}))
