# Cross-boundary relocation regions

A per-transfer artifact tells the dispatcher how to produce a tensor. It stops
at the transfer boundary. A caller may need the producer's snapshot, or a
consumer may immediately copy the result into a fresh allocation. A region adds the
uses, storage ownership and ordering facts needed to answer those questions.
The executable case for [issue 270](https://github.com/JueonPark/sym/issues/270)
removes both allocations around an existing typed transfer. Its result is a
mechanism measurement on small inputs. A separate decoder experiment shows why
a removed copy need not yield a model-level speedup.

## A finite model

Take one invocation with bound positive extents, disjoint host and device
address spaces, a fixed number of streams and a finite, acyclic event graph.
A tensor has a logical array and a representation given by its element type, extents,
strides, device, allocation identity, byte offset and accessible span. Distinct
views may refer to the same storage. Inputs include the live tensors, shape
bindings and a validated immutable revision of each scale and zero point
(name, dtype, extents and exact bytes). Results include every returned tensor
or view and every intermediate exposed to an external user.

Each operation declares the byte ranges it reads and writes, the storage it
owns or borrows, and the events bounding its accesses. Stream order, recorded
events and waits, and blocking host completion generate the happens-before
relation by transitivity. Unordered overlapping accesses with a writer are
outside the model. A caller holding a storage alias can enqueue future work.
An event for work already enqueued cannot discharge that ownership.

For every admitted invocation and legal event ordering, equivalence requires
the same logical values, public representations, alias relationships, mutations
visible through aliases, and readiness at the promised completion points. Fresh
results must stay independently writable across calls. Allocation identities
may be consistently renamed. A private temporary has no external observer.
Time, allocator counters, and OOM timing are outside the equivalence relation.
The argument assumes successful allocation and covers inference with ordinary
tensors. A harness that inspects addresses can observe work removed by the pass.

### Element semantics

For a result coordinate `d`, the layout pullback yields either a source
coordinate or a pad fill with its entry stage `j`. Start with the gathered
source at stage 0, or with the fill in its declared type at stage `j`.
Evaluate the remaining value stages in program order. A per-channel stage
reads its bound parameter at its own channel map `cᵢ(d)`. Composing a consumer
map `f` substitutes `f(d)` in **both** the layout and channel maps. In the
executable suffix of `cat(prefix, t)`, subtract the prefix length from the
leading coordinate first. Passing the suffix as a dense view gives the
unchanged artifact exactly its original local coordinates.

The artifact owns rounding, clamp order, signed zero, subnormal handling,
NaN policy and exact integer interpretation. For example, narrowing
`1 + 2⁻¹¹` to FP16 and widening returns `1`. Quantizing a NaN with
`symmetric_rne` gives signed code −128. An F32 pad of 1 before quantization
at scale 0.5 gives code 2. An S8 pad of 1 after the stage gives code 1.
Moving storage cannot cancel narrowing/widening or quantize/dequantize,
recompute the reciprocal in another precision, move a fill across a stage,
or index a scale by the flat address in the containing allocation. The cast
contract compares non-NaNs by bits and NaNs by class. The committed fixtures
also pin the observed output bytes on the qualified rows. A downstream bitcast
that exposes NaN payloads needs a stronger bitwise guarantee to move a cut.
[Typed semantics](reloc-typed-semantics.md) · [Folding and channel-map examples](reloc-typed-folding.md)

## Transformation obligations

| Transformation | Sufficient proof and implementation guard | Counterexample and disposition |
| --- | --- | --- |
| Fuse or forward a producer | Its pure computation can supply exactly the original values in transfer read order, with the same stage policies, parameters and source version. Every elided result is private. Source owners and input readiness cover the extended read interval. The executable rule forwards a dense private host clone. | A write through a view between the clone and transfer changes the later read. A returned clone promises independent storage. Keep the snapshot. A concurrent external writer would violate the host input contract. |
| Move the transfer boundary | Split after exactly stage `k`. The sender produces the specified layout and dtype at that cut, the link preserves its bits, and the receiver executes the exact suffix. Shapes, parameter revisions, channel coordinates, pad entry points, device capability and execution order must all match. Select from the bound invocation's eligible rows. | A pad entering after a proposed split can leave the wire missing cells. A nonzero bound zero point can exclude a device dequant kernel. An explicitly forced unavailable row fails before dispatch. The ordinary selector can use the qualified reference row. |
| Fuse a consumer | Prove its coordinate map and fresh output contract. Each destination has exactly one writer, addresses are in bounds, and all users of the removed allocation accept the replacement. In the executable rule, the prefix and transfer suffix partition a fresh dense leading-axis concatenation on the result device. | Returning the transfer result separately would expose a new alias with the concatenation. Keep its allocation. A mixed consumer, unknown operator, dtype promotion or unsupported layout retains the source graph. |
| Reuse storage | Match device, size, alignment and representation. Before the first overwrite, **every** previous access to overlapping bytes must have completed or be ordered before that write, and all external leases capable of future access must have ended. Preserve independently owned outputs and all subviews. | If copy completion precedes two independent consumers, waiting for the copy or for only one consumer permits a race with the other. A retained tensor, detached view or storage handle permits a future read even after both consumers finish. Defer reuse, allocate within budget, or refuse before launch. |

**Why the rules preserve observations.** Fix the input versions and a legal
execution. For a private clone, the source value at each read equals the
original snapshot because no overlapping write can intervene. Induct on the
ordered stages for each gathered or padded cell. Equal inputs, policy and
parameter index give equal results at the next stage. A qualified cut inserts
a bit-preserving copy between two induction steps. For a concatenation, the
prefix is copied unchanged and the suffix writes those same proven cells at
their shifted positions. Fresh storage and disjoint ranges prevent a write
from changing any live input or returned intermediate. Finally, erasing an
unobserved allocation or recycling a dead range cannot affect a later read
when all preceding accesses happen before the overwrite and no future use can
be submitted through an old lease. These arguments compose for this bounded
region. They assume complete use/effect information and a truthful completion
backend.

## Executable scope and fallback

The research pass recognizes one closed FX graph with three inputs, a dense
host snapshot, a typed relocation and a functional leading-axis `cat`.
An output edge counts as a use. Returning just the snapshot or just the
transfer result blocks that edge's forwarding independently. Extra operators,
including mutations and other consumers, fail the structural match. Runtime
guards require a naturally aligned, pageable host input, a rank-one through
rank-three result, and compatible dense tensors, dtype, device and execution
modes. The destination is fresh on every call. The prototype lives under
`bench/issue270`. The deployed `RelocBackend` has its own consumer rule.
[Region lowering](../bench/issue270/region.py) · [Deployed consumer pass](consumer-layout.md)

The executable assumes the host source is ready and that no external alias
writes to it until the blocking dispatch completes. FX cannot
establish that fact about another thread. A native binding captures parameter
bytes. A new revision requires rebinding. The prepared Torch adapter also
rechecks live parameter values before use. The dispatcher validates actual
storage spans and executes the selected row on streams ordered after the
caller. An unsupported rewrite runs the original graph. Invalid artifact
bindings and failures after submission propagate to the caller. Uncertain
completion requires retaining affected owners and precludes replay.
[Prepared execution](automatic-prepared-execution.md) · [Dispatch](runtime-dispatch.md)

The prototype always allocates its public result. For reuse, the deployed
`PackedOutputArena` supplies a conservative implementation of the last rule. It checks for external storage owners and uses a device-wide barrier before
recycling. A transfer completion can release a borrowed source while the
returned destination is still live. The CUDA tests exercise a delayed reader
on a second stream, reuse of the same address after that reader completes,
held aliases across close, and quarantine on uncertain completion.
[Arena contract](budgeted-transfer-memory.md) · [Lifetime tests](../libreloc/python/tests/torch_frontend/test_memory.py)

## Same-engine ablation

Run the captured graph with the pass disabled and enabled. Each pair uses
the same input, prefix, bound artifact, revision, native runtime and explicitly
selected dispatch row. The runner checks the returned bytes against the existing
independent oracle fixtures, including the prefix. It checks the actual source
and destination pointers at dispatch, implementation, wire cut, host kernel
family and completed payload. Temporary bytes count only separately materialized
producer and transfer tensors. The prefix copy, fresh final result, parameter
traffic and internal staging still have their own costs.

On an RTX 4070 Ti SUPER, Python 3.14.7, Torch 2.14.0+cu126, with the native
runtime built from `23d48d5`, all **44 CUDA row/binding pairs** agreed on the
output and payload. The host control passed all **19 binding pairs** from the
same 16 artifacts using the host copy backend. Each enabled pair forwarded
both addresses. Three representative first bindings are shown below. Byte
counts are per invocation.

| Artifact | What the control materializes | Temporary bytes, off → on | Native payload in both arms for the named row |
| --- | --- | ---: | ---: |
| `cast_transpose_f16` | 96-byte F32 snapshot and 48-byte transposed FP16 result | 144 → 0 | 48 B, `cpu_reference` |
| `quantize_channel_transpose_sym` | 60-byte snapshot and 15-byte S8 result with remapped channel | 75 → 0 | 72 B including scale upload, `cpu_stages_cuda_stages@0` |
| `quant_dequant_padded` | 52-byte snapshot and 64-byte padded F32 result after quantization and dequantization | 116 → 0 | 64 B, `cpu_reference` |

On the 64-byte input and 16-byte result of `quantize_edges`, returning only
the snapshot keeps 64 temporary bytes. Returning only the transfer result
keeps 16. Returning both keeps all 80. Mutating the separately returned
tensor preserves the final concatenation in each case.

The negative cases independently exercise a source-alias write, escaping
producer and transfer outputs, mixed readers, dtype promotion, rebinding,
a rejected cut before late padding, held results and a delayed consumer on a
second stream. Existing C4 conformance also checks rounding boundaries, NaNs,
different pad orders, wrong parameter lengths and all eligible GPU rows in
both transfer directions.
[Region tests](../libreloc/python/tests/torch_frontend/test_relocation_region.py) ·
[C4 corpus and guards](../libreloc/python/tests/test_typed_conformance.py) ·
[CUDA row conformance](../libreloc/python/tests/torch_frontend/test_typed_conformance_gpu.py)

From the repository root with the matching Torch environment and native
bindings on `PYTHONPATH`, run the commands below. Each ablation emits one JSON
record per binding and eligible implementation. The host command needs a CPU
build or a working CUDA build. The device commands need a CUDA build and GPU.
The arena tests additionally need the native Torch bridge.
[Build instructions](torch-integration.md)

```sh
python -m bench.issue270.ablate --device cpu
python -m bench.issue270.ablate --device cuda
python -m pytest -q libreloc/python/tests/torch_frontend/test_relocation_region.py \
  libreloc/python/tests/torch_frontend/test_typed_conformance_gpu.py \
  libreloc/python/tests/torch_frontend/test_memory.py
```

## Research claim

The additional expressiveness relative to a per-transfer descriptor is the
ability to name external uses, ownership and storage destinations around the
transfer. The stage policies, channel maps, legal cuts and native kernels
already existed. The experiment attributes two local materializations to
region visibility. It measures neither elapsed time nor physical peak memory.
In the independent issue 269 decoder ablation, eager Sym removed 24 kernels
and 442,368 logical allocation bytes per request. Inductor executed 283 kernels
with identical allocation counts in both variants. Zero of 24 parent latency
targets passed. [Decoder controls and results](../bench/issue269/RESULTS.md)

Layout propagation and fusion already eliminate intermediates in XLA. MLIR
bufferization analyzes read-after-write conflicts for destination reuse. IREE
Stream represents transfers, resource ranges and asynchronous timepoints.
TEMPI also canonicalizes noncontiguous movement and selects packing strategies.
These facts and the matched results support using this model as the
correctness basis for joint planning and code generation in issue 262.
An independent full-paper novelty or performance claim would require a
broader automatic lowering and a workload advantage over those opportunities
and matched framework controls. [XLA GPU architecture](https://openxla.org/xla/gpu_architecture) ·
[MLIR bufferization](https://mlir.llvm.org/docs/Bufferization/) ·
[IREE Stream](https://iree.dev/reference/mlir-dialects/Stream/) ·
[TEMPI, HPDC 2021](https://carlpearson.net/pdf/20210624_pearson_hpdc.pdf)

The [joint region planner experiment](../bench/issue271/README.md) uses these
ownership and destination constraints to compare executable schedules under a
declared device allocation budget. It measures completed requests, five-call
lifecycles and paced arrivals against a finite GPU oracle.
