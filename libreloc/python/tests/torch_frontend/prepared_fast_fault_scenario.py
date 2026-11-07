"""Real partial-submit failures must retain unsafe owners without reusing slots."""
import ctypes
import gc
import json
import os
import sys
import weakref

import torch
import pyreloc
from reloc_torch import CompilerClient
from reloc_torch.fast import FastTransfers
from reloc_torch.recipe import Dequantize, InlineParam, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

mode = int(sys.argv[1])
finish = sys.argv[2] if len(sys.argv) > 2 else 'wait'
shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
shape = (Const(128), Const(128))
def spec(dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
recipe = Recipe(spec('int8'), (Dequantize('float32', InlineParam('float32', (), (0x3f000000,)),
    None, None, 'affine'), Transpose((1, 0))), spec('float32'), 'h2d')
runtime = FastTransfers(compiler=CompilerClient.from_environment(), gather_threads=1)
sources = [torch.ones(128, 128, dtype=torch.int8) for _ in range(3)]
program = runtime.typed(recipe, sources[0], implementation='cuda_dequant_relocate')
handle = runtime.prepare_group([program] * 3)
refs = [weakref.ref(source) for source in sources]
submit = pyreloc.submit_dispatch_group

def inject(group, **options):
    options['caller_stream'] = None
    shim.sym_dispatch_fault_mode(mode)
    try:
        return submit(group, **options)
    finally:
        shim.sym_dispatch_fault_mode(0)
if mode <= 4:
    pyreloc.submit_dispatch_group = inject
completion = None
try:
    completion = handle.submit(sources)
    if mode >= 5:
        shim.sym_dispatch_fault_mode(mode)
    if finish == 'close':
        handle.close()
    else:
        completion.wait()
except (RuntimeError, pyreloc.TransferError) as error:
    assert ('completion_unknown' if mode in (2, 4, 6) else 'backend_failure') in str(error), error
else:
    raise AssertionError('fault did not reach submission/completion')
finally:
    shim.sym_dispatch_fault_mode(0)
if completion is not None:
    assert completion.report is None
    try:
        completion.wait()
    except (RuntimeError, pyreloc.TransferError):
        pass  # a failed completion never becomes a successful group later
    else:
        raise AssertionError('failed completion was accepted')
del completion
assert runtime.stats()['groups'] == 0
assert runtime.stats()['prepared_groups']['live_slots'] == 0
handle.close()
del sources
gc.collect()
unknown = mode in (2, 4, 6)
assert runtime.stats()['resources']['quarantined'] == unknown
assert all((ref() is not None) == unknown for ref in refs)
if unknown:
    try:
        runtime.close()
    except pyreloc.TransferError as error:
        assert 'completion_unknown' in str(error)
    else:
        raise AssertionError('quarantined close succeeded')
else:
    pyreloc.submit_dispatch_group = submit
    healthy = runtime.prepare_group([program])
    out, = healthy.submit([torch.ones(128, 128, dtype=torch.int8)]).wait()
    assert torch.equal(out, torch.full((128, 128), .5, device='cuda'))
    runtime.close()
torch.cuda.synchronize()
print(json.dumps({'mode': mode, 'passed': True}))
