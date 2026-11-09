# Consumer-aware layout lowering

`RelocBackend(consumer_layout=True)` enables a conservative FX rule that removes
a private permutation materialization when every user is functional `cat`.
It is off by default. The resulting graph contains ordinary Torch operators,
so Inductor can optimize their representation; no opaque fused custom op is added.

```python
backend = RelocBackend(consumer_layout=True, compute_backend="inductor")
compiled = torch.compile(append_cache, backend=backend, fullgraph=True, dynamic=False)
# append_cache(wire, current) returns:
# torch.cat((wire.transpose(0, 1).contiguous(), current), dim=1)
with torch.no_grad():
    cache = compiled(wire, current)
backend.close()
```

The rule supports dense rank-two/rank-three CUDA inputs and transpose/permute
chains. Packed views with valid nonzero storage offsets are supported. It requires
known static FX metadata, a contiguous materialized intermediate and fresh
contiguous same-dtype/device cat outputs. Cat reads logical values directly from
the permuted view; its output owns new storage. Held cat outputs survive changes
to later inputs. Existing downstream arithmetic and dequantization are unchanged.

Intermediate graph outputs, view/alias users, CPU-visible or mixed consumers,
noncontiguous source inputs, casts, padding, channel maps, symbolic/unknown
metadata, mutations and opaque operators preserve the original materialization.
The conservative graph-wide effect check can reject an otherwise safe candidate.
No optimization is promised for an excluded graph. Training and active transforms
are outside qualification; the backend retains its original autograd fallback.

`ConsumerLayoutPolicy(min_saved_bytes=N)` sets a logical-byte threshold.
`lower_consumer_layout(gm, example_inputs, policy=...)` exposes the rule directly
for a compiler/planner and returns the graph and immutable decisions. Each row
records eligibility, selection/reason, source/materialization/consumer node names,
removed logical allocation bytes and twice those bytes for the eliminated copy's
read/write traffic. An already-contiguous view reports zero saved bytes and is
excluded as `already_contiguous`; a real clone still allocates. These estimates
do not predict latency or physical peaks.
Direct compiler users must maintain valid metadata/guards; Dynamo supplies runtime
specialization guards when using the backend. The pass never executes real inputs
or edits the supplied graph. Backend stats expose decisions as plain records.

Inductor may already remove the same copy. Equivalent controls must receive that
opportunity; the existence of this rule alone is not a new Sym performance claim.
The first target is inverse KV layout conversion before cat. Dequantization within
GEMM and automatic whole-program representation discovery remain outside scope.

Reference contracts: [Torch cat](https://docs.pytorch.org/docs/2.14/generated/torch.cat.html),
[contiguous](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.contiguous.html),
and [FX graph rewriting](https://docs.pytorch.org/docs/2.14/fx.html). Exact removed
work, compiled behavior and numerical guarantees are verified by the qualification
rather than inferred from these API descriptions.


## Explicit wire restore

`KVCacheSwap.restore_wire(stored)` returns fresh dense CUDA `[S,H,D]` tensors.
It is a separate, explicitly selected representation: ordinary `restore` still
returns materialized `[H,S,D]`. A transparent consumer can transpose the wire view
and let the qualified cat rule eliminate its private copy. The method checks the
declared KV dtype/head dimensions before submission.

`PackedWireRestore(device, max_bytes=128<<20, max_inflight=8)` exposes the same
copy helper to independent controls. It accepts ordinary dense rank-three CPU
FP32/FP16/BF16 tensors of one dtype without gradients. Adjacent pinned views of
one storage use one packed H2D copy; other legal dense inputs use separate copies.
All inputs are checked and all outputs allocated before copies begin. Every call
reads fresh values and returns independently owned output storage. Nothing is
prepacked, quantized or numerically converted by this helper.

The KV-owned helper inherits `contexts * max_scratch_bytes` as its per-request
logical output limit and `max_inflight` as its pending source-group limit. This
is a separate admission bound, not pooled native scratch, a combined physical
quota or a cap on caller-held outputs. Stats distinguish logical payload/pending
bytes, referenced host storage and allocation requests. These scopes overlap.
A rejected request raises `MemoryBudgetError` before any transfer; applications
can select their original materialized path when its own contract permits it.

Copies use the current stream on the selected device. The helper retains host
sources until completion; callers must not modify them until `drain()` or other
copy completion synchronization. Normal Torch stream rules apply to returned
tensors: another consumer stream must wait and record tensor use on that stream.
`drain()` waits pending copies; `close()` also rejects future work. Dropped helpers
fence pending copies in the owning process. FastTransfers.close owns cleanup of
its KV helpers. Copied outputs remain valid after helper/owner close.

Partial submission failures fence already submitted work and poison the helper;
no automatic retry can duplicate work. Explicit close reports cleanup failures.
Active forward AD/torch.func, tensor subclasses and training inputs are rejected.

The purity gate rejects `out=` writes, unknown operands/receivers, custom tensor
subclasses and attribute access that could hide a Python side effect. This can
exclude otherwise safe graphs; it preserves the original graph rather than
assuming purity. A rewritten eager callable, like an Inductor callable, rejects
execution after its backend closes.
