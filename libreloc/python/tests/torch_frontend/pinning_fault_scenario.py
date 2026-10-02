"""Separate process: completion-unknown intentionally quarantines allocations."""
import ctypes
import gc
import os
import sys
import weakref

import pyreloc
import torch
from reloc_torch import CompilerClient, TransferResources, dispatch, transport
from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

path, pinning, fault = sys.argv[1:]
mode = int(fault)
shim = ctypes.CDLL(os.environ["SYM_DISPATCH_FAULT_SHIM"])
shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
shape = (Const(1024), Const(1024))
def spec(dtype):
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
operations = (Transpose((1, 0)),)
if path == "typed":
    operations += (Cast("float16", "ieee_rne"),)
recipe = Recipe(spec("float32"), operations,
                spec("float16" if path == "typed" else "float32"), "h2d")
compiled = CompilerClient.from_environment().compile(recipe)
owner = TransferResources()
source = torch.ones((1024, 1024), dtype=torch.float32)
if path == "typed":
    request = dispatch.prepare_typed_transfer(compiled, source, "cuda:0", policy="original_cpu")
    execute = dispatch.execute_typed_transfer
    native_name = "execute_dispatch"
else:
    request = transport.prepare_transfer(compiled, source, "cuda:0")
    execute = transport.execute_transfer
    native_name = "execute_transfer"
native = getattr(pyreloc, native_name)
refs = []

def inject(request, **options):
    refs.extend(weakref.ref(tensor) for tensor in options["owners"])
    options["caller_stream"] = None  # fail after the copy, not on a producer event
    shim.sym_dispatch_fault_mode(mode)
    try:
        return native(request, **options)
    finally:
        shim.sym_dispatch_fault_mode(0)

setattr(pyreloc, native_name, inject)
try:
    execute(request, resources=owner, pinning=pinning)
except RuntimeError as error:
    assert ("completion_unknown" if mode == 2 else "backend_failure") in str(error), str(error)
else:
    raise AssertionError("fault did not reach transfer")
assert request.consumed
assert request.staging[0]["memory_kind"] == pinning
assert request.staging[0]["reason"] == "forced_" + pinning
del request, source
gc.collect()
assert refs and all((ref() is not None) == (mode == 2) for ref in refs)
stats = owner.stats()
stats = stats["typed"] if path == "typed" else stats
assert bool(stats["quarantined"]) == (mode == 2), stats
if mode == 2:
    try:
        owner.close()
    except pyreloc.TransferError as error:
        assert "completion_unknown" in str(error)
    else:
        raise AssertionError("unknown completion must not free allocations")
else:
    owner.close()
    assert (stats["retained_bytes"] if path == "typed" else stats["allocated_staging_bytes"]) == 0
# Faults were injected only at the API boundary; the GPU itself is healthy.
torch.cuda.synchronize()
print(path, pinning, mode, "passed")
