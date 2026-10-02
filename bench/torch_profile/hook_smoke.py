#!/usr/bin/env python3
"""Prove interposition is active and preserves values / exception behavior."""
import ctypes
import json
import torch

hooks = ctypes.CDLL(None)
hooks.symprof_native_calls.restype = ctypes.c_ulonglong
hooks.symprof_cpu_copies.restype = ctypes.c_ulonglong
torch.set_num_threads(8)
hooks.symprof_enable_native(1)
a = torch.arange(32*64, dtype=torch.float32).reshape(32, 64)
b = (a+2).transpose(0, 1).contiguous()
assert torch.equal(b, (torch.arange(32*64).reshape(32, 64)+2).T.float())
checks = 0
for fn in (lambda: torch.empty(2, 3)+torch.empty(4, 3),
           lambda: torch.arange(8).expand(2, 8).copy_(torch.ones(2, 8))):
    try:
        fn()
    except RuntimeError:
        checks += 1
    else:
        raise AssertionError('Torch error was lost')
assert checks == 2
calls = hooks.symprof_native_calls()
assert calls > 0
copies = hooks.symprof_cpu_copies()
assert copies > 0
hooks.symprof_enable_native(0)
print(json.dumps(dict(correct=True, exceptions=checks, native_calls=calls, cpu_copy_callbacks=copies)))
