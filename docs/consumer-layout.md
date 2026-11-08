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
read/write traffic. These estimates do not predict latency or physical peaks.
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
