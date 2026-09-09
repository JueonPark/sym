# Project finalization: high-level plan

Date: 2026-09-09. Status: proposed scope and issue breakdown; parent issues #131–#133 are filed, and detailed Torch subissue plans are available below. Implementation acceptance remains open.

## Objective and boundaries

Finish the original compiler/runtime architecture: capture a tensor's relocation recipe, fold it into a reusable symbolic plan, bind concrete dimensions, and execute through libreloc. Completion depends on semantic correctness, integration, reproducibility, and documented coverage. A speedup or a conference-paper contribution is not an acceptance condition.

Preserve the CPU relocation plus pipelined transfer implementation and the existing compiler/runtime separation. Existing placement models and GPU kernels remain reusable components. Any automatic placement must choose among implementations of the same requested computation; it must not silently introduce a cast or quantization to improve transfer time.

The proposed initial integration scope is inference and weight loading with dense CPU/CUDA tensors, matching the original P5 scope. Training, distributed offload, additional accelerator backends, automatic quantization policy, and arbitrary data-dependent shapes are extensions beyond this finalization. Unsupported cases execute through the original PyTorch path with a recorded reason.

The original motivation and its corrections remain historical evidence. In particular, issue #63's original performance condition is not a condition for completing these engineering tasks. Its final comment records that the experimental closure criterion was not met. [Issue #63 disposition](https://github.com/JueonPark/sym/issues/63#issuecomment-5267443274).

## What already exists

| Area | Current implementation | Remaining integration gap |
| --- | --- | --- |
| Compiler | `reloc.transpose`, `reloc.reshape`, `reloc.pad`, chain folding, plan verification, and serialization | PyTorch importer; cast/quantize/dequantize semantics; a supported plan-export entry point |
| Runtime | Wire decoding, symbolic binding, CPU executors, H2D/D2H pipelines, buffer/gather pools, CUDA kernels, prefolding, and cost-model decisions | One supported execution path connecting compiler artifacts, typed transforms, and framework buffer/stream ownership |
| Python | `pyreloc` load/bind/execute functions and a contiguous tensor-to-pointer helper | Automatic capture/replacement, symbolic metadata integration, safe framework stream interoperation, and weight lifecycle handling |

The runtime handoff is already demonstrated by compiler-generated plan corpora and tests. R6 additionally demonstrates portable binding and placement decisions against committed measurements; it is not an automatic PyTorch capture-and-execute integration. References: [runtime surface](../libreloc/README.md), [R6 scope](r6-crossbox-bind.md), [wire format](reloc-plan-format.md).

The three workstreams meet at these boundaries:

```text
PyTorch capture and eligible subgraph selection                 [1]
    -> symbolic relocation recipe
Compiler folding, typed semantics, and serialized plan          [2]
    -> plan artifact + symbols/constraints + tensor descriptors
Runtime load, bind, buffer/stream binding, and execution        [3]
    -> PyTorch tensor with the requested values and metadata
```

## 1. Capture and compile PyTorch CPU–GPU relocation

**Parent issue:** [#131 — Capture CPU–GPU relocation and lower it to symbolic plans](https://github.com/JueonPark/sym/issues/131)

**Detailed plans:** [T1–T4 implementation plans and dependency map](torch-finalization/README.md). Their selected qualification baseline is CPython 3.14.7 (standard GIL build) and PyTorch 2.14.0 with CPU/cu126 wheels; the index records the rationale, dependencies, and validation requirements.

**Outcome:** An opt-in PyTorch integration redirects supported tensor and weight transfers to the existing compilation stack, including supported adjacent layout operations and dynamic dimensions.

**Approach:** Use a `torch.compile` backend/FX pass as the main place to capture and fold complete operation chains. Add a scoped dispatch observer to inventory eager transfers, including model loading outside compiled `forward`. Route supported eager transfer boundaries through the same runtime adapter. Observation alone cannot remove layout operations that have already executed; eager fusion requires a captured recipe or an explicit preparation wrapper.

This follows PyTorch's documented [custom backend interface](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html) and [dispatch modes](https://docs.pytorch.org/docs/2.14/notes/extending.html). Pin and test the actual PyTorch release before relying on its internal operator names or dispatch hooks. Deep Inductor layout-assignment integration can remain a later extension: the first frontend covers transformations visible in the captured graph, not hidden backend layouts.

| Child | Work and deliverable | Acceptance |
| --- | --- | --- |
| T1 — Transfer inventory and eligibility | Observe `.to()`, `.cpu()`, `.cuda()`, cross-device `copy_`, and parameter/buffer transfers. Record the actual operator, direction, shapes, strides, storage offset, dtype, copy/mutation semantics, and nearby layout operations. Publish an eligibility matrix for the pinned PyTorch version. | Representative input and weight-loading examples account for every observed CPU–GPU transfer as supported or excluded with a reason. Same-device casts and metadata-only views are classified separately. |
| T2 — FX-to-reloc import and symbolic guards | Import supported transpose/permute, reshape/view, materialization, and constant-pad chains around transfer boundaries. Map symbolic sizes/strides to plan symbols and constraints. Retain the original subgraph for fallback. | Static and shape-varying examples produce verified compiler plans; no premature conversion of symbolic sizes to Python integers. Existing compiler bail cases remain unsupported rather than being assumed foldable. |
| T3 — Custom transfer op and graph replacement | Register the runtime call as a custom op with matching fake metadata, then replace eligible subgraphs. Connect supported eager transfer interception to the same adapter. Preserve mutation/alias behavior or fall back before replacement. Add plan-cache and fallback diagnostics. | Real and fake outputs agree in shape, strides, dtype, device, and relevant storage offset. Custom-op registration checks and independent numerical comparisons pass. Intermediates with other users or intervening mutations are not incorrectly removed. |
| T4 — Dynamic inputs and weight lifecycle | Demonstrate reuse of one symbolic plan across valid shape bindings. Handle parameter/buffer relocation outside `forward`; reuse existing prefolding for explicitly stable weights. Invalidate prepared data on weight replacement/mutation or quantization-parameter changes. Document installation and supported usage. | Examples cover input H2D, output D2H, and repeated weight loading. Weight updates cannot return stale data. Counters distinguish plan compilation, symbol binding, and Dynamo graph compilation. |

**Dynamic-shape contract:** Reuse is guaranteed only within the same supported recipe, rank/layout/dtype family, and validated shape constraints. Existing limits include constant split factors and compiler restrictions on padded reshapes. A failed correctness guard routes to the original graph before executing an invalid plan; the standalone runtime continues to report a bind error. This does not promise zero recompilations for an arbitrary enclosing PyTorch model.

**Numerical and metadata contract:** Ordinary relocation preserves the requested dtype and values. A view is not automatically a physical relocation. `copy_` requires mutation semantics; an out-of-place custom op cannot silently replace it. Gradient-requiring execution falls back until autograd is explicitly supported. Fake kernels must match the real output metadata, and `opcheck` does not replace numerical testing. [PyTorch custom-operator guidance](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html).

**Likely code areas:** A separate optional torch frontend beside [pyreloc](../libreloc/python/pyreloc/__init__.py), [torch interop](../libreloc/python/pyreloc/torch_interop.py), and [Python bindings](../libreloc/python/PyReloc.cpp); frontend tests and runnable examples. Keep torch dependencies outside the core runtime.

**Dependencies:** T1 and T2 can begin immediately. T3 requires R1/R2 below. Layout-only integration does not wait for item 2; importing the new typed operations follows their compiler/runtime support.

## 2. Extend plans with cast, quantize, and dequantize

**Parent issue:** [#132 — Add typed value transforms to relocation plans](https://github.com/JueonPark/sym/issues/132)

**Outcome:** The compiler explicitly represents and lowers requested value conversions alongside index relocation, with a versioned runtime contract.

**Approach:** Extend the existing relocation representation with typed value-transform information. Keep address mapping and value conversion distinct, and preserve operation order. Fold into a fused mapping/value program where equivalence can be proved; otherwise retain staged execution or fall back. The existing affine layout map alone cannot describe rounding, saturation, or dequantization.

| Child | Work and deliverable | Acceptance |
| --- | --- | --- |
| C1 — Typed operation semantics | Define `reloc.cast`, `reloc.quantize`, and `reloc.dequantize`, including input/output types, rounding, saturation, signedness, scales, zero points, and channel axis. Start with f32↔f16 and f32↔s8 using supplied per-tensor/per-channel parameters; expand only where the kernel contract is implemented. | Verifiers reject invalid type/parameter combinations. Numerical behavior is defined for ties, range limits, and non-finite values. Kernel/PyTorch semantic differences are either reconciled or explicitly unsupported. |
| C2 — Folding and canonicalization | Compose supported value transforms with transpose/reshape/pad, carry channel indexing through permutations and splits, and preserve fill-value semantics. Add typed capability checks and fallback diagnostics. | Tests distinguish pad→quantize from quantize→pad and cover channel-axis relocation. Lossy cast sequences and quantize→dequantize are not incorrectly canceled. No-copy is legal only when the requested value transformation is also an identity. |
| C3 — Versioned artifact and binding | Extend the compiler encoder and runtime decoder/binder together. Represent source/destination types, ordered transforms, parameter bindings, and constraints. Define source, wire, and destination byte footprints separately, including parameter storage where relevant. | A new wire version supports typed plans while v0 layout-only artifacts still load. Old runtimes reject the new version clearly. Malformed metadata, invalid scales/channel lengths, and byte-count overflow fail before execution. |
| C4 — Conformance and frontend enablement | Add scalar reference recipes, compiler folding tests, serialization goldens, and a support matrix by operation/dtype/layout/direction. Enable frontend import only for compiler/runtime combinations verified together. | Layout-only regression corpus remains exact. Cast and quantization outputs match the declared reference semantics; dequantization uses a specified tolerance where appropriate. At least one mixed layout+cast and one layout+quantize/dequantize recipe reaches the runtime. |

**Quantization scope:** Parameters are supplied by the program or an explicit configuration; this work does not add calibration/training or silently choose lower precision. Runtime-supplied scale/zero-point tensors need declared bindings, validation, and lifetime handling. Existing int4 pack/unpack kernels may be exposed later, after sub-byte storage and packing order have an explicit compiler/runtime contract.

**Direction and invertibility:** Layout inversion and value inversion are different. Dequantization is not a mathematical inverse that restores the original floats, and a narrowing cast has no general inverse. A D2H transfer compiles its requested forward computation; it uses the existing inverse-layout API only when that API's semantics actually match.

**Likely code areas:** [operation definitions](../sym/dialect/reloc/IR/RelocOps.td), [plan attributes](../sym/dialect/reloc/IR/RelocAttrs.td), [folding](../sym/dialect/reloc/Transforms/PlanBuilder.cpp), [serialization](../sym/dialect/reloc/IR/RelocSerialization.cpp), [runtime plan types](../libreloc/include/reloc/Plan.h), [binding](../libreloc/src/Bind.cpp), and their compiler/runtime tests.

**Dependencies:** C1/C2 can proceed alongside the layout-only integration. C3 owns the shared compiler/runtime representation; R3 consumes it. C4 finishes jointly with typed execution and frontend import.

## 3. Complete compiler-to-runtime execution

**Parent issue:** [#133 — Connect compiled plans to supported libreloc execution](https://github.com/JueonPark/sym/issues/133)

**Outcome:** A user can compile a supported recipe, load its artifact, bind dimensions and buffers, and execute it correctly from Python or C++ without benchmark-specific setup.

**Approach:** Promote the existing compiler→wire→decode→bind→execute path into a supported interface. Reuse the current executor, pools, prefolding, kernels, and backend abstraction. libreloc stays MLIR/LLVM/torch-free; compilation belongs in a separate tool or optional compiler binding.

| Child | Work and deliverable | Acceptance |
| --- | --- | --- |
| R1 — Supported compilation/artifact API | Expose deterministic plan export through a normal compiler tool or compiler-side binding, reusing `encodePlan`. Return plan bytes plus the symbol/descriptor information the frontend needs. Provide load/bind interfaces that do not depend on test-pass diagnostics or hand-authored blobs. | A generated layout-only recipe executes through the public path and survives save/load in a fresh process. Runtime library dependency checks remain satisfied. |
| R2 — Tensor, stream, and lifetime adapter | Bind validated tensor storage, capacity, offsets, strides, and device identity to runtime requests. Connect caller streams to libreloc streams with explicit producer/consumer ordering. Own staging allocations, tensor references, events, and cleanup for the duration of each operation. | Default-stream and non-default-stream H2D/D2H examples are correct, including immediate consumers and repeated allocation/reuse. The documented blocking path works first; nonblocking calls are offloaded only when completion and lifetime guarantees are implemented. |
| R3 — Plan-driven transform dispatch | Route supported typed plans to existing quant/cast/CUDA kernels and add missing reference paths where needed. Derive wire bytes from the requested computation. Separate execution capability from optional cost-model advice, and expose an explicit way to run the original CPU-transform pipeline. | The executed path is observable and matches the plan. Forced original-path execution works for its supported cases. Automatic placement uses only semantically equivalent, implemented alternatives; absent calibration or unsupported kernels cannot silently change precision or skip required operations. |
| R4 — Integration validation and handoff | Ship input-transfer, output-transfer, and weight-loading examples. Exercise symbolic rebinding, artifact compatibility, fallback, typed conversions, resource cleanup, and prefold invalidation. Document build/install steps, the support matrix, and reproduction commands. | A fresh checkout can reproduce all supported examples using generated plans. CPU CI covers compilation/binding/reference execution; CUDA validation covers actual transfers and stream ordering. Performance is reported descriptively, without a speedup gate. |

**Current gaps to address explicitly:** `as_ptr` currently accepts contiguous tensors only. `h2d`/`d2h` currently host-block on libreloc's own streams and do not order earlier work on other streams. A placement result stored in `BoundPlan.decision` is not itself a unified execution dispatcher. The adapter must preserve tensor storage semantics rather than merely pass `data_ptr()` and `numel()` for arbitrary views. [Current runtime contract](../libreloc/README.md).

**Transfer semantics:** A same-layout CPU→GPU transfer still requires data movement; a layout `no_copy` flag cannot turn it into a cross-device view. Distinguish ordinary D2H copying from inverse-layout scatter. Start by routing unsupported ranks, empty tensors, overlapping views, and layouts through PyTorch until their complete descriptor/execution contracts are supported. Stream ordering and allocator lifetime are separate obligations. [PyTorch CUDA stream guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

**Likely code areas:** [compiler serialization API](../sym/dialect/reloc/IR/RelocSerialization.h), [runtime execution](../libreloc/include/reloc/Execute.h), [pipeline](../libreloc/src/Pipeline.cpp), [backend interface](../libreloc/include/reloc/Backend.h), [CUDA backend](../libreloc/cuda/CudaBackend.cu), [CPU transforms](../libreloc/include/reloc/Quant.h), [CUDA kernels](../libreloc/include/reloc/CudaKernels.h), and [Python bindings](../libreloc/python/PyReloc.cpp).

**Dependencies:** R1/R2 start against v0 immediately. R3 depends on C1–C3. R4 closes the combined work with T4/C4.

## Delivery order

| Milestone | Work | Reviewable result |
| --- | --- | --- |
| M1 — Layout-only integration | T1/T2 + R1/R2, then T3 | A real PyTorch transfer and supported layout chain compile and execute through the original stack, with guards and fallback. |
| M2 — Typed relocation | C1–C3 + R3, then typed import/C4 | Explicit cast and quantize/dequantize recipes survive compilation, serialization, binding, and execution. |
| M3 — Final handoff | T4 + R4 and remaining C4 coverage | Dynamic-input and weight-loading examples, documented support boundaries, reproducible validation, and project disposition. |

C1's semantic/interface design can start during M1, but M1 must remain executable using existing layout-only plans. This gives an early complete integration path and keeps the typed wire-format change off its critical path. Estimate and assign dates after T1 and the R1/R2 interface audit; the old paper freeze date does not govern this plan.

## GitHub organization and historical closeout

The three parent issues above contain their goals, boundaries, child checklists, dependencies, and acceptance criteria. T1–T4, C1–C4, and R1–R4 are proposed child issue titles, not assigned GitHub issue numbers. Use one milestone, `Project finalization`, to group the work. Create individual child issues when scheduling them.

If a board is useful, add the same issues to one Project named `Sym finalization`, grouped by parent issue, with Status (`Backlog`, `Ready`, `In progress`, `Review`, `Done`) and delivery milestone. Issues remain the source of task descriptions and completion evidence. GitHub supports displaying parent/sub-issue relationships in Projects, so this organization can begin with issues and gain a board later. [GitHub sub-issues documentation](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues).

During M3, review the remaining historical issues rather than automatically inheriting their research goals:

- [#55](https://github.com/JueonPark/sym/issues/55) and [#56](https://github.com/JueonPark/sym/issues/56): related decoder and end-to-end pad coverage; attach them to the relevant runtime work if still applicable.
- [#57](https://github.com/JueonPark/sym/issues/57) and [#71](https://github.com/JueonPark/sym/issues/71): optional refactoring/sanitizer work; decide their disposition separately from the three core deliverables.
- [#88](https://github.com/JueonPark/sym/issues/88): reuse the relevant weight-loading/compute-overlap scenario in R4, while keeping its experimental acceptance conditions distinct from this integration plan.
- [#63](https://github.com/JueonPark/sym/issues/63) and [#73](https://github.com/JueonPark/sym/issues/73): preserve the experimental record and failed/withdrawn claims. Any eventual closure should state that the research track is concluded or superseded, without claiming a failed performance condition was achieved.

The final documentation should link the published [claim ledger](claim-ledger.md). Historical source records—the patent description, initial motivation, 0704 proposal/plan, and 0730 proposal/plan—remain local files under `human_history/` and are not part of this documentation set. Decide whether to publish those records during the final handoff; add links once they have repository locations. Preserve historical text and add current interpretation separately.

**Project completion:** All three parent issues satisfy their functional acceptance criteria, the documented examples run from a fresh checkout, unsupported cases have defined behavior, and the remaining historical issues have an explicit disposition. Performance measurements describe the resulting implementation; they do not determine whether it is finished.
