# Explicit GPU-region replay

`GPUReplay` is opt-in inference execution for a **pure GPU function**. It works
with Torch eager or an Inductor-compiled callable, including consumers of Sym's
prepared transfers. CPU transforms, host routing, H2D/D2H and transfer owners stay
outside capture. `RelocBackend` still rejects `triton.cudagraphs=True` for whole
graphs containing native transfers.

```python
from reloc_torch import GPUReplay, GPUReplayPolicy

# Weights are arguments: changed values and replacement tensors are refreshed.
def compute(x, weight):
    return (x @ weight).relu()

with torch.no_grad(), GPUReplay(compute, device='cuda:0',
        policy=GPUReplayPolicy(slots=2, max_buckets=8, max_bytes=256 << 20)) as region:
    region.prepare(example_x, example_weight)  # all slots for one exact shape
    region.freeze()  # required before concurrent serving; unseen shapes fall back
    result = region(new_x, new_weight)
```

For backend-owned lifetime, use
`backend.prepare_gpu_region(compute, example_x, example_weight, policy=...)`.
The returned object has the same API. Backend stats include `gpu_replay`, and
backend close closes these owners. Preparation uses the caller's no-grad or
inference mode; changing that mode or relevant math flags selects another bucket.

## Callable and stream contract

All changing state, including weights, must be explicit arguments. No mutable
module/closure state, RNG, tensor mutation, CPU side effects, CPU transfers or
host synchronization is supported inside a region. Input mutation detected in
warmup excludes the bucket. This guard is not a proof of purity. Mutable routing
and in-place accumulation belong outside the boundary. Output leaves must be
CUDA tensors; outputs are independent copies, with no input/output alias promise.

Arguments may be builtin pytrees of ordinary contiguous, nonempty CUDA tensors
on the owner's device and immutable scalars. Each shape, dtype, stride, scalar
value and structure is guarded. Replaced tensor addresses are supported by
refreshing every input. Unsupported metadata/devices or exhausted shape/slot
capacity runs the original callable. Every such reason is visible in `stats()`.
Grad-enabled execution raises. Autocast executes the original callable.

Prepare only while GPU activity in the process is quiescent, as required by
[PyTorch CUDA Graph capture](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
A global lock serializes Sym capture preparations, but cannot fence unrelated
application GPU work. Calls may lazily prepare before `freeze`; frozen serving
never captures. Prepare all desired buckets before starting serving threads.

The caller orders its producer streams before its current stream, just as with
ordinary Torch operations. Refresh, graph launch and output copies enqueue on
that current stream. Inputs are recorded on it for allocator lifetime. A slot's
event follows output copies; only completed slots may be reused. Busy slots use
the original callable. Held outputs/views remain valid across replay and close.
A clean capture failure disables the bucket; execution errors propagate and
poison the owner, without retrying a possibly submitted request.

`invalidate()` waits for slots, retires captures, and unfreezes the owner. Use it
when changing the callable's immutable revision; prepare/freeze again in a
quiescent phase. Parameter tensor changes need no invalidation. `close()` retires
slots and forbids later calls. Owners cannot be reused after fork.

## Memory and costs

The default is two slots, eight exact-shape buckets, and 256 MiB per owner.
`max_bytes` admits persistent input payload plus the complete reserved segments
of each graph-private pool. It is checked before input allocation and after
capture. Pool accounting requires the native Torch CUDA caching allocator's
segment metadata; missing accounting disables capture. Pools are never shared
between slots. Graph capture may transiently exceed the retained limit or fail
with OOM. This is **not** a physical-device allocation cap.

Caller inputs, returned clones (including held aliases), transfer buffers,
ordinary allocator reservation, library/context state and transient compilation
are outside that owner limit. Include these in the application physical budget;
use Torch peak counters and a physical-device meter. Closing drops owner storage,
not caller outputs or the global allocator cache.

`stats()` separates preparation, warmup and capture time, input payload, private
pool reservation, replay counts and exclusions. `enabled=False` retains the same
slot input copies/output copies and callable, while launching its ordinary
kernels. It is a replay attribution control; unwrapped execution remains useful
as a deployment comparison. Slot storage, copying and capture have costs; replay
is not a blanket performance guarantee.
