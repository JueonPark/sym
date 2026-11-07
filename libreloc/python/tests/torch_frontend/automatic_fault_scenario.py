"""Automatic lowering inherits native failure quarantine and PID boundaries."""
import ctypes
import os
import sys
import torch
import torch.ao.quantization.fx._decomposed
import pyreloc
from reloc_torch import RelocBackend, PreparedExecutionPolicy
from reloc_torch.runtime import ExecutionError

owner = RelocBackend(prepared_execution=PreparedExecutionPolicy(min_bytes=0, min_reuses=0),
    transfer_options=dict(pinning='pinned', n_streams=1, gather_threads=1))
def fn(q):
    return torch.ops.quantized_decomposed.dequantize_per_tensor(q, .5, 0, -128, 127, torch.int8).t().contiguous().to('cuda:0')
x = torch.ones(128, 128, dtype=torch.int8)
run = owner(torch.fx.symbolic_trace(fn), (x,))
entry = owner._registry.lookup(run.handles[0])
mode = sys.argv[1]
if mode == 'fork':
    # Hold inherited locks: the child must reject before taking any of them.
    owner.runtime._prepared._lock.acquire()
    owner.runtime._lock.acquire()
    pid = os.fork()
    if pid == 0:
        try:
            owner.runtime._prepared.stats()
        except RuntimeError as error:
            os._exit(0 if 'process_mismatch' in str(error) else 2)
        os._exit(3)
    _, status = os.waitpid(pid, 0)
    owner.runtime._lock.release()
    owner.runtime._prepared._lock.release()
    assert status == 0
    owner.close()
else:
    mode = int(mode)
    shim = ctypes.CDLL(os.environ['SYM_DISPATCH_FAULT_SHIM'])
    shim.sym_dispatch_fault_mode.argtypes = [ctypes.c_int]
    submit = pyreloc.submit_dispatch_group
    def inject(group, **options):
        options['caller_stream'] = None
        shim.sym_dispatch_fault_mode(mode if mode < 5 else 0)
        try:
            result = submit(group, **options)
            return result
        finally:
            shim.sym_dispatch_fault_mode(mode if mode >= 5 else 0)
    pyreloc.submit_dispatch_group = inject
    try:
        run(x)
    except ExecutionError as error:
        assert ('completion_unknown' if mode in (2, 6) else 'backend_failure') in str(error), error
    else:
        raise AssertionError('fault did not reach automatic submission')
    finally:
        shim.sym_dispatch_fault_mode(0)
    assert entry.fallback_calls == 0
    stats = owner.stats()
    assert stats['runtime_executions'] == 1 and stats['prepared_execution']['executions'] == 0
    assert stats['prepared_execution']['native']['logical_transfers'] == 0
    assert stats['prepared_execution']['native']['prepared_groups']['live_slots'] == 0
    assert stats['transfer_resources']['typed']['quarantined'] == (mode in (2, 6))
    try:
        owner.close()
    except pyreloc.TransferError as error:
        assert mode in (2, 6) and 'completion_unknown' in str(error)
    else:
        assert mode in (1, 5)
print('passed', mode)
