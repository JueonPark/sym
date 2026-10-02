"""Subprocess-only CUDA event/completion fault qualification for typed owners."""
import ctypes, gc, json, os, sys, weakref
import torch, pyreloc
from reloc_torch import CompilerClient, TransferResources, dispatch
from reloc_torch.recipe import Dequantize, InlineParam, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

mode=int(sys.argv[1]);shim=ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
shim.sym_dispatch_fault_mode.argtypes=[ctypes.c_int]
shape=(Const(1024),Const(1024))
spec=lambda dtype:TensorSpec(shape,dense_strides(shape),Const(0),dtype)
recipe=Recipe(spec('int8'),(Dequantize('float32',InlineParam('float32',(),(0x3f000000,)),None,None,'affine'),Transpose((1,0))),spec('float32'),'h2d')
compiled=CompilerClient.from_environment().compile(recipe)
resources=TransferResources()
source=torch.ones((1024,1024),dtype=torch.int8)
request=dispatch.prepare_typed_transfer(compiled,source,'cuda:0',implementation='cuda_dequant_relocate',threads=1)
refs=[];native=pyreloc.execute_dispatch

def fail_after_enqueue(request,**options):
    refs.extend(weakref.ref(owner) for owner in options['owners'])
    # Skip the producer event so the fault follows the actual dense upload.
    options['caller_stream']=None
    shim.sym_dispatch_fault_mode(mode)
    try:return native(request,**options)
    finally:shim.sym_dispatch_fault_mode(0)
pyreloc.execute_dispatch=fail_after_enqueue
try:
    dispatch.execute_typed_transfer(request,resources=resources)
except RuntimeError as error:
    assert ('completion_unknown' if mode==2 else 'backend_failure') in str(error), str(error)
else:raise AssertionError('fault did not reach typed dispatch')
assert request.consumed
del source,request
gc.collect()
snapshot=resources.stats()['typed']
assert snapshot['quarantined']==(mode==2),snapshot
assert all((ref() is not None)==(mode==2) for ref in refs)
if mode==2:
    assert snapshot['retained_bytes']>=1024*1024
    try:resources.close()
    except pyreloc.TransferError as error:assert 'completion_unknown' in str(error)
    else:raise AssertionError('quarantined close succeeded')
else:
    assert snapshot['retained_bytes']==0
    resources.close()
# The injected API failures do not poison the real GPU; drain it after proving
# the runtime retained every owner when its own completion proof was unknown.
torch.cuda.synchronize()
print(json.dumps(dict(mode=mode,passed=True,resources=snapshot)))
