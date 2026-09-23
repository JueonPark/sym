# Reloc typed folding (C2, issue #142)

How the compiler folds the typed value transforms of
[reloc-typed-semantics.md](reloc-typed-semantics.md) (C1) together with the
layout chain into one `#reloc.typed_plan`, what that plan contains, which
chains are rejected and why. This is the compiler representation C3
([#143](https://github.com/JueonPark/sym/issues/143)) encodes and R3
([#147](https://github.com/JueonPark/sym/issues/147)) executes; C2 changes no
runtime dispatch and enables no execution. A `reloc.typed_plan_result` is not
an artifact: `sym-reloc-export` still answers `typed_unsupported`
([reloc-export.md](reloc-export.md)), the v0 encoder refuses a layout plan
whose element type changes, and the frontend keeps its
`typed_transform_unavailable` / `typed_artifacts_unavailable` rows.

## 1. The typed plan

```mlir
#reloc.typed_plan<
  source = tensor<[B, C], f32>,                 // logical source descriptor
  result = tensor<[B * C], i8>,                 // logical result descriptor (logical rank)
  symbols = ["C"],                              // affine symbol positions of the channel maps
  layout = #reloc.plan<src = tensor<[B, C], f32>, dst = tensor<[B * C], i8>, ...>,
  stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C],
                         scale = #reloc.binding<"s" : [C], f32>, axis = 1,
                         channel = (d0)[s0] -> (d0 mod s0)>],
  fills = [{dst_axis = 0, stage = 0, value = 1.0 : f32}]>
```

| Part | Meaning |
| --- | --- |
| `source`, `result` | The chain's logical source and result descriptors: extents, dense row-major strides, dtype. `result` keeps the logical rank even when the layout's axes coalesce; every channel map is written over it. |
| `layout` | ONE `#reloc.plan` for the whole chain's index mapping: `src` has the source dtype, `dst` the result dtype, and every `pad_fill` value is the **fused** fill in the result dtype (what the executor writes). `no_copy` is always `false`. Everything else (`perm`, `axes`, `inverse`, `divisible`, `contiguous`, `runtime_pad_check`) has its v0 meaning. |
| `stages` | The value stages in program order. Each one records the transform, its C1 policy, its input and output element types, the **logical shape of its operand** (the shape `axis` refers to), the C1 parameters (constants or `#reloc.binding` declarations, unchanged), and for per-channel stages the channel axis plus a `channel` map from result coordinates `(d0, ..., d_{r-1})` and plan symbols `[s0, ...]` to the parameter index. |
| `fills` | For every layout pad, the fill **as written** and the stage index at which it entered (`0` = before every stage, `k` = after stage `k`). |

The fused program a typed plan denotes, for every logical result index `d`
of the result descriptor:

1. If `d` lies in a pad region, write the layout's fused fill for that axis.
2. Otherwise gather the source element the layout maps to `d`, then apply the
   stages in order; stage `i` with a channel map uses parameter entry
   `channel_i(d, symbols)`, a per-tensor stage its single parameter.

Verifier invariants (`test/dialect/reloc/typed_invalid_plan.mlir`):

- at least one stage (layout-only chains stay `#reloc.plan`);
- the layout's element types and extents match the descriptors (the
  `plan_result` rank relaxation applies to a canonicalized `dst`);
- stage 0 consumes the source dtype, stage `k+1` consumes what stage `k`
  produces, the last stage produces the result dtype;
- every channel map has the result rank and the declared symbol count, and
  exactly one result;
- every fill resolves to a layout pad and vice versa, enters at a stage that
  exists, has the dtype of that stage's input, and folds through the later
  stages to exactly the fused fill (C1 reference arithmetic; a per-channel or
  runtime parameter after the entry point is rejected);
- the layout's `no_copy` is `false`: a value program is never a view.

Plans that share descriptors but differ in stage order, policy or parameter
values are different attributes, so they never share a canonical identity.

## 2. How chains fold

`--reloc-fold` treats `reloc.cast`, `reloc.quantize` and `reloc.dequantize` as
chain members alongside `transpose`, `reshape` and `pad`. A chain with at
least one typed op materializes as `reloc.typed_plan_result`; without one it
materializes as before. Folding is all-or-nothing per chain.

**Index maps commute with element-wise stages.** `transpose` and `reshape`
only move elements, so they fold into the single layout plan whatever the
stages around them; each stage's channel expression is rewritten so that it
still names the operand coordinate the parameter was declared over:

| Layout after the stage | Original channel (axis `a` of the stage operand) |
| --- | --- |
| `[B, C] -> transpose [C, B]` | output axis 0: `(d0, d1) -> (d0)` |
| `[B, 12] -> reshape [B, 4, 3]` | `3 * c_outer + c_inner`: `(d0, d1, d2) -> (d1 * 3 + d2)` |
| `[B, C] -> flatten [B * C]` | `flat_index mod C`: `(d0)[s0] -> (d0 mod s0)` with `symbols = ["C"]` |
| `[B, C] -> reshape [B, C floordiv 3, 3]` (symbolic split) | `(d0, d1, d2) -> (d1 * 3 + d2)`, `divisible(C, 3)` retained, the scale still declared over `[C]` (`parameter_length == C` stays C3's guard) |
| permutation then coalescing (`[2, 3, 4]` channel 0 → `transpose [1, 2, 0]` → `[12, 2]`) | `(d0, d1) -> (d1)`; the merged layout is not a standalone transpose |

The rules behind the table: a kept axis maps to its new position; a split
axis becomes the row-major recombination of its new run; a merged run's
axes become `floordiv`/`mod` of the merged coordinate (only the axes a
channel expression references are converted, so the symbol list stays
minimal). A layout op *before* the stage needs no rewriting: the stage's
channel is the operand coordinate, `(d0, d1) -> (d0)` for a quantize on axis
0 of a transposed operand. Later canonical axis merging touches the layout
only; the result descriptor and the channel maps keep the logical rank
(`TypedPlanBuilderTest.ChannelSurvivesTransposeThenSplit`,
`test/dialect/reloc/typed_folding.mlir`).

**Pads do not commute.** A fill enters at a definite stage and is folded
through every later stage with the C1 reference arithmetic:

| Order | Fill in the plan | Rule |
| --- | --- | --- |
| `pad(f32, 1.0)` then `quantize(scale 0.5)` | fused `2 : i8`, original `1.0 : f32` at stage 0 | per-tensor constant parameters fold (`TypedSemantics` §3.3 arithmetic) |
| `quantize(scale 0.5)` then `pad(i8, 1)` | fused `1 : i8`, original `1 : i8` at stage 1 | a fill entering after the stage is already in the result dtype |
| `pad(f32, 1.5)` then `cast ieee_rne` | fused `1.5 : f16` | casts always fold (the fill converts under the cast's table) |
| `quantize axis 1` then `pad axis 1 lo 1` | fused `0 : i8`, channel `(d0, d1) -> (d1 - 1)` | new channels take the fill and never index a parameter; the channel map shifts by the leading width |
| `pad` then a **per-channel** quantize/dequantize | — | **bail** `fill_not_foldable`: every padded position would take its own channel's code, whether the pad is on the channel axis (the parameters cover the padded operand) or not |
| `pad` then a stage with a **runtime** scale or zero point | — | **bail** `fill_not_foldable`: no compile-time fill |
| two pads of one axis entering at different stages | — | **bail** `pad_stage`: one fill per axis in the plan format |

The two order witnesses above are different plans in structure (entry stage,
stage operand shape `[8]` versus `[6]`) and in the reference fill (`2` versus
`1`).

**Transactional transfer functions.** A failing fold leaves the builder as it
was: `foldValueStage` computes every fused fill before it commits,
`foldReshape` and `foldPad` rewrite channel expressions only on their commit
path, and a layout bail after a folded stage leaves the stage, its channel
and the element type untouched
(`TypedPlanBuilderTest.LateLayoutBailLeavesStagesIntact`). The pass then
marks the whole original chain; no partial fold is ever published.

## 3. Rejection boundaries

Every op of a rejected chain that contains a typed op carries
`reloc.fallback` and `reloc.fallback_reason = "<reason>"`; layout-only
chains keep the v0 marking (`reloc.fallback` alone). Reasons are stable
strings (`test/dialect/reloc/typed_fold_bail.mlir`):

| Reason | When |
| --- | --- |
| `structural` | an intermediate value escapes the chain, or a non-reloc op interrupts it (both segments bail) |
| `marked` | the chain already carried `reloc.fallback` (manual opt-out or an earlier run); an existing reason is never rewritten |
| `layout_bail` | a layout transfer function refused: non-contiguous merge, split or merge of a padded axis, conflicting fill values on one axis, non-positive constant extents, undecidable symbolic counts (the v0 rules, unchanged) |
| `fill_not_foldable` | a pad precedes a per-channel stage or a stage with runtime parameters |
| `pad_stage` | two pads of the same axis enter at different stages |
| `type_chain`, `channel_axis` | defensive: the op verifier already rules these out |

Unsupported operations and policies never reach the fold: the C1 verifiers
reject them.

## 4. Canonicalization and capability

`canonicalizeTypedPlan` runs `canonicalizePlan` on the layout (axis merging,
constant folding, canonical pad/constraint order), then clears `no_copy`
whatever the index map proves, constant-folds the descriptors and stage
shapes, simplifies the channel maps, and sorts the fills by destination axis
without touching their bits. It removes **no** stage: no C1 policy is an
identity (there is no same-type cast), `f32 -> f16 -> f32` stays two lossy
stages, `quantize -> dequantize` stays two stages
(`test/dialect/reloc/typed_canonicalize.mlir`,
`TypedPlanBuilderTest.LossyPairsAreNotCancelled`). It is idempotent, and
`--reloc-fold` over its own output is a no-op.

`isPureView(layout)` still answers the *layout* question (R3 may use it to
pick a contiguous kernel); the plan as a whole is never a view.

**Capability (representation and semantics, not kernels).** The typed plan
represents any verifier-valid chain of transposes, reshapes the v0 folder
accepts, pads, and C1 value transforms, subject to §3. In particular it
represents per-channel parameters on any operand axis, symbolic extents and
channel lengths, runtime bindings, and interleavings of layout ops and
stages. What executes is a separate question: R3's kernels take the channel
on the **outermost coalesced plan axis** (`gatherQuantizeF32S8`,
`dequantRelocateS8F32`), assert **no pad regions**, need rank ≥ 2 and a unit
innermost destination stride, and there is no CPU dequantize or f16 → f32
kernel yet ([reloc-typed-semantics.md §7](reloc-typed-semantics.md#7-compatibility-and-characterization)).
A typed plan whose channel map is not a single plan-axis coordinate, or whose
layout has pads, is representable here and unsupported there until R3 adds a
reference path.

## 5. Handoff to C3

The typed artifact must carry: the source and result descriptors; the symbol
list (channel-map symbol positions, bound by name like plan symbols); the
layout plan with its fused fills in the result dtype; the ordered stages
with transform, policy, in/out types, operand shape, parameters (constants
by value, bindings by name/type/extents) and channel maps (affine over
result coordinates and symbols, encodable with the existing `PUSH_DIM` /
`PUSH_SYM` stack vocabulary); and the original fills with entry stages, so a
decoder can re-verify the fusion. Bind-time obligations are C1 §4.4's plus:
bound symbols must satisfy `divisible` constraints as before, channel maps
evaluate over logical result coordinates (dense row-major, so the flat plan
destination offset recovers them), and a runtime parameter's length is
checked against the **stage operand** extent, not a plan axis extent.
