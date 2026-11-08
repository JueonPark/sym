"""A real asynchronous device fault, isolated from the parent's CUDA context."""
import gc
import json

import torch

from reloc_torch.memory import PackedOutputArena
from reloc_torch.runtime import ConcreteDescriptor

arena = PackedOutputArena("cuda:0", max_bytes=1 << 20)
spec = ConcreteDescriptor((1 << 18,), (1,), "float32", torch.device("cuda:0"))
outputs = arena.allocate([spec])
torch.cuda.synchronize()
torch._assert_async(torch.zeros((), dtype=torch.bool, device="cuda:0"))
try:
    arena.close()
except RuntimeError as error:
    assert "completion" in str(error)
else:
    raise AssertionError("the asynchronous fault must prevent retirement")
stats = arena.stats()
assert stats["quarantined_bytes"] == 1 << 20
try:
    arena.allocate([spec])
except RuntimeError:
    pass
else:
    raise AssertionError("a failed arena must never submit again")
del outputs, arena
gc.collect()  # The native destructor must quarantine, not throw or free unsafe bytes.
print(json.dumps(dict(quarantined_bytes=stats["quarantined_bytes"])))
