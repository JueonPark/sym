# Fast transfers: prepared programs with minimal per-call work

`reloc_torch.fast` is the explicit, low-overhead way to run Sym transfers
from eager PyTorch code. It exists because the general paths (the
`torch.compile` custom op and `dispatch.prepare_typed_transfer`) validate
every call from scratch, which costs several hundred microseconds per
transfer and dominated the model-level comparisons in
[bench/beat_torch](../bench/beat_torch/README.md). The fast lane binds a
recipe to one source shape signature once and then executes each call
through a single native entry point.

## Contract

This is an explicit inference API combining prepared transfer programs,
handwritten fused producer/dequantization kernels and grouped execution.
Its workload speedups do not by themselves establish an AST-folding gain.
The pooled EmbeddingBag producer is synthesized natively, outside the recipe
compiler, and changes the CPU pooling implementation while preserving its math.

Fast custom ops are opaque nodes to Inductor: surrounding compute can be
compiled without a graph break, but Inductor does not fuse through their bodies.
Keep the prepared program and its owner alive while executing a captured graph.
The handle registry holds programs weakly; program collection or owner `close()`
releases the handle. A stale captured handle raises an error at execution.

- A program is bound to a **shape signature** (shape, strides, dtype, device
  type) and, for typed recipes, to **parameter values** snapshotted at
  preparation. Different shapes need different programs; changed parameters
  need a new program. Values are never cached: every call reads the live
  source and writes a fresh output.
- The native validator still proves that the declared views fit the plan
  before anything launches; a tensor of another shape, stride or dtype is
  rejected (`ValueError`/`RuntimeError`), never silently reinterpreted.
- Blocking entry points (`execute`, `FastCompletion.wait`) return when the
  transfer completed. Asynchronous ones (`submit`, `FastCompletion.wait_stream`)
  return GPU outputs ordered on the caller's current stream; the completion
  stays registered with its `FastTransfers` owner until the native work is
  done, so dropping it never turns into a host wait. D2H groups always host
  complete. CPU sources and index/offset operands must stay unchanged until
  the completion finished; the owner holds references to them.
- `FastTransfers` owns bounded native resources: `max_scratch_bytes` per
  context, `contexts` independent contexts (each with its own stream and
  scratch share, so that many groups can be in flight), a pinned staging ring
  of `n_buffers` chunk slots for CPU-producing uploads, and `gather_threads`
  workers. `close()` (or the context manager) drains in-flight work.

## Programs

| Constructor | Recipe | Per call |
| --- | --- | --- |
| `typed(recipe, example, parameters=...)` | value transform (cast, dequantize, ...) | `execute(source)` blocking, chunked CPU/DMA pipeline or GPU row |
| `layout(recipe, example)` | layout only (transpose, reshape, pad) | used inside groups; `execute(source)` is a one-item group |
| `indexed(recipe)` | `IndexSelect` (+ cast), H2D | `execute(source, index)` / `submit(source, index)`; indices bind natively from the tensor's memory |
| `pooled(tables, dtype)` | fused `EmbeddingBag(mode="sum")` over several tables + cast + H2D | `execute(indices, offsets)` / `submit(...)`; bit-exact with Torch's sum in index order and IEEE RNE cast |

`submit(items, outputs=None, device_layout=False, ...)` enqueues a list of
`(program, source)` pairs as one native group with one completion barrier.
`device_layout=True` lets float32 layout-only items relocate on the GPU:
items sharing one plan run as a single batched permute kernel, and host
buffers that are adjacent views of one allocation share a single copy.

## Conveniences

- `KVCacheSwap(owner, heads, head_dim)`: `evict(caches)` moves `[H, S, Dh]`
  GPU caches to `[S, H, Dh]` pinned CPU tensors (views of one block) in one
  group; `restore(stored)` brings them back asynchronously.
- `FastWeights(owner, PreparedWireWeights)`: grouped loads and a
  `prefetch(name_groups, slots=2)` window over prepared INT8 snapshots. The
  window opens with one submission so that its DMA starts before the next is
  prepared, keeps `slots` groups in flight, and performs any slot wait at
  the start of the next iteration rather than between handing out a group
  and its first use.
- `FastPrefetch(owner, groups, slots)`: the same window over arbitrary
  group item lists.

## Inside `torch.compile`

A prepared program is also a `torch.compile` primitive. `TypedProgram.op(source)`
(H2D programs), `IndexedProgram.op(source, index)` and
`PooledProgram.op(indices, offsets)` call the functional custom ops
`reloc_torch::fast_typed`, `reloc_torch::fast_indexed` and
`reloc_torch::fast_pooled` (`reloc_torch.fast_ops`). Each op carries the
program handle and the output metadata as explicit arguments:

- the **fake kernel** returns `empty_strided` of the program's destination
  descriptor and never consults the registry, so Dynamo captures the call as
  one opaque node (`fullgraph=True` holds, no graph break) and Inductor
  schedules the surrounding compute around it as a fallback kernel; the
  indexed op derives its row count from the index tensor (`SymInt`-capable
  under dynamic shapes), the trailing extents are the recipe's constants;
- the **real kernel** runs `execute` (typed, indexed) or
  `submit(...).wait_stream()` (pooled: the output is ordered on the caller's
  current stream and the DMA tail overlaps the compute that follows), then
  verifies the produced metadata against the declared one;
- handles are registered when the program is prepared (never inside a
  trace) and die with the program or its `FastTransfers` owner; a graph that
  still carries one fails with `unknown or closed` instead of fabricating
  output. Autograd is rejected like the other `reloc_torch` ops.

What is **not** an op: `FastPrefetch`, `FastWeights.prefetch` and
`KVCacheSwap` carry completions across calls, so they sit between compiled
regions, where the Torch controls keep their own prefetch queues.

Per-call cost of the entry points (`bench/beat_torch/op_overhead.py`,
microseconds, median of 200 completed calls, EPYC 7351 + RTX 2080 Ti):

| Transfer | `execute` | `op` eager | `op` inside a compiled region | previous generic route (`RelocBackend` custom op) | eager Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| typed 26×16×64 f32 → [16, 26, 64] f16 (106 KB) | 128 | 204 | 294 | 489 | 85 |
| gather 256 rows × 128 f32 → f16 (64 KB) | 154 | 225 | 339 | 618 | 79 |

The dispatcher wrapper adds about 70 µs to an eager op call and the
compiled-function call another 90–115 µs; the generic route's per-call
rebinding and descriptor rebuild (260–360 µs) are gone. At these sizes eager
Torch is faster than any Sym entry point; the fast lane pays off once a
transfer is large enough for the chunked pipeline or the fused producer to
matter (bench/beat_torch).

## Attribution knobs and kernel-only entry points

For the ablation in [bench/beat_torch](../bench/beat_torch/README.md#ablation-kernels-versus-planruntime)
the runtime exposes switches that turn one optimization off without changing
any result (every disabled path is the bit-identical reference the
optimization replaced): `pyreloc.set_ablation(name, enabled)` /
`pyreloc.ablations()`, or `RELOC_ABLATE="name,..."` in the environment at
startup. Knobs: `scalar_dequant` (the one-element-per-thread dequantize
kernel instead of the `char4`/`float4` one). `IndexedProgram.execute` and
`PooledProgram.execute` take `pipeline=False` (gather the whole wire, then
one copy; no chunk overlap).

The libtorch helper also exposes the fast lane's kernels on plain tensors,
without plans, dispatch or pipeline, so that a Torch program can move the
bytes itself: `pooled_bags(tables, indices, offsets, out, threads)` (the fused
EmbeddingBag-sum producer into a host tensor), `gather_rows(plan, symbols,
source, index, out, threads)` (the fused row gather + cast of an
`index_select` recipe into a host tensor) and `dequant_s8_f32(q, scale, out,
stream)` (the per-channel dequantize kernel on a CUDA stream handle). They
exist for attribution measurements, not as a public transfer API.

## Build options

`RELOC_ENABLE_TORCH_EXT=ON` builds `pyreloc._pyreloc_torch`, a small
libtorch-linked helper that reads tensor metadata directly (storage base and
capacity, offset, sizes, strides, dtype) and builds the native requests in
one call. It binds the module to the configured Torch; wheels and Torch-free
installs omit it, and the fast lane falls back to the pure-Python descriptor
path. The helper must be compiled against the project's pybind11 (the one
`_pyreloc` uses), not Torch's bundled copy, so that both modules share one
type registry.

## Native additions behind the fast lane

- Typed host windows use a 64 KiB per-worker floor (the layout pipeline's
  1 MiB floor left most of a retained pool idle on 1 MiB chunks) and prefetch
  upcoming rows of an indexed gather.
- `TransferOptions.deviceLayout` and the batched `relocateBatchF32` kernel
  (grouped float32 layout items on the device).
- `HostKernel::StridedRuns`: rank-3 pad-free layouts with unit inner strides
  run as SIMD inner runs iterated so the source streams sequentially.
- `dispatch::preparePooledBags`: the fused embedding-bag producer (AVX2
  sums in index order, range-checked indices, f16 cast), executed through the
  ordinary chunked `cpu_reference` upload.
- Asynchronous groups can upload CPU-stage members through the chunk ring
  without draining it (`chunked_members`); the group's completion event
  covers every chunk.
- Pad-free wires of at most 4 MiB stream through one worker dispatch
  (`host_pipeline = "streamed"`): the whole wire is staged at once, rows are
  partitioned once across the pool, and the worker that finishes a chunk's
  last rows enqueues that chunk's copy, so small transfers overlap their CPU
  work with DMA without a per-chunk barrier or slot wait.
- The indexed gather prefetches 32 rows ahead (`RELOC_INDEXED_PREFETCH`
  overrides the distance for experiments); with the libtorch helper one call
  (`index_select_request`) binds the live indices, prepares the program and
  selects the row straight from the tensors.
- `GatherPool` schedules sub-range *p* on worker *p* (range 0 on the caller),
  so repeated dispatches with the same partition find a worker's rows in its
  own L2, like an OpenMP static schedule.
- `bindIndexed` over a caller-owned span (`bind_indexed_from_pointer`).
- A vectorized per-channel dequantize kernel (`char4` → `float4`, one scale
  per quad) with bit-identical results.
