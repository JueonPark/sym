# Reloc typed relocation semantics (C1, issue [#141](https://github.com/JueonPark/sym/issues/141))

Normative contract for the typed value transforms of the `reloc` dialect:
`reloc.cast`, `reloc.quantize`, `reloc.dequantize`. It is the single authority
that C2 (folding, [#142](https://github.com/JueonPark/sym/issues/142)), C3
(versioned artifact and binding, [#143](https://github.com/JueonPark/sym/issues/143)),
C4 (conformance, [#144](https://github.com/JueonPark/sym/issues/144)) and R3
(runtime dispatch, [#147](https://github.com/JueonPark/sym/issues/147)) test
against. Parent: [#132](https://github.com/JueonPark/sym/issues/132),
[project plan §2](project-finalization-plan.md#2-extend-plans-with-cast-quantize-and-dequantize).

**What this document claims.** IR validity (the verifiers in
`sym/dialect/reloc/IR/RelocOps.cpp`) and numerical meaning (the tables below).
**What it does not claim.** Executable runtime capability or a typed artifact:
wire format v0, manifest schema 1 and `sym-reloc-export` interface 1 stay
layout-only, and a verifier-valid typed module is a declared unsupported
export (`typed_unsupported`, [reloc-export.md](reloc-export.md)). The frontend
keeps reporting `typed_transform_unavailable` and `typed_artifacts_unavailable`
([torch-support.md](torch-support.md)) until C4 enables verified rows.

**Versioning rule.** Every policy name below is bound to exactly one table.
A different rounding, saturation, NaN or parameter interpretation is a new
policy with its own name and reference; an existing name is never
reinterpreted (Global Constraints of [#141](https://github.com/JueonPark/sym/issues/141)).

## 1. Vocabulary

| Term | Meaning |
| --- | --- |
| `f32`, `f16` | IEEE 754-2008 binary32 and binary16. |
| `s8` | Signed two's-complement 8-bit integer, range `[-128, 127]`. Stored as MLIR signless `i8` (wire type kind 2, width 8; manifest dtype `"int8"`). The signed interpretation is declared by the operation: `reloc.quantize` produces s8, `reloc.dequantize` consumes s8. `si8`/`ui8` element types are rejected. |
| `fl32(e)` | The binary32 value nearest to the exact real `e`, ties to even (one rounding). |
| `rne(t)` | The integer nearest to `t`, ties to even. |
| Stage | One element type in a chain. Layout ops (`transpose`, `reshape`, `pad`, `plan_result`) preserve the stage type; a typed op changes it. |
| Logical shape | The `!sym.tensor` shape of the typed op's operand. Every typed op returns the same logical shape (rank and every dimension logically equal); only the element type changes. |

Policies and the operation/type pair each one is defined for:

| Policy | Operation | Type pair | Table |
| --- | --- | --- | --- |
| `ieee_rne` | `reloc.cast` | f32 → f16 | [§3.1](#31-reloccast-f32--f16-policy-ieee_rne) |
| `exact` | `reloc.cast` | f16 → f32 | [§3.2](#32-reloccast-f16--f32-policy-exact) |
| `symmetric_rne` | `reloc.quantize` | f32 → s8 | [§3.3](#33-relocquantize-f32--s8-policy-symmetric_rne) |
| `affine` | `reloc.dequantize` | s8 → f32 | [§3.4](#34-relocdequantize-s8--f32-policy-affine) |

Any other pair (`f32 → f32`, `f32 → f64`, `f32 → i8` as a cast, `f16 → s8`,
…) or any other policy/operation combination fails verification
(`test/dialect/reloc/typed_invalid_ops.mlir`). Extending the vocabulary means
adding a policy case, a table here, a verifier rule and a characterization case.

## 2. Floating-point environment

Every implementation that claims a policy must satisfy all of the following;
the reference formulas assume them.

- Arithmetic in binary32 with round-to-nearest-even; no extended-precision
  intermediates (`FLT_EVAL_METHOD == 0`). No formula in this document has an
  intermediate that is not itself a binary32 value or an exactly representable
  integer.
- Subnormals are honored on input and output: no flush-to-zero or
  denormals-are-zero mode. (libreloc's CUDA kernels are built without
  `--use_fast_math`; nvcc's default `--ftz=false`, `--prec-div=true` apply.)
- No fused multiply-add can arise: every formula is a single multiply, a
  single division, or an exact integer operation followed by one multiply.
- Floating-point exception flags are ignored; NaN inputs are never trapped.
- `max`/`min` follow IEEE `maxNum`/`minNum`: a quiet NaN operand yields the
  other operand. Implementations using x86 `MAXPS`/`MINPS` (which return the
  second operand when either is NaN) must order operands so §3.3's NaN rule
  holds (`MAXPS(v, lo)`, then `MINPS(v, hi)`), as `Quant.h` documents.

## 3. Operations

### 3.1 `reloc.cast` f32 → f16, policy `ieee_rne`

`y = convertFormat(x)` with IEEE 754 `roundTiesToEven`: the binary16 value
nearest to `x`, ties to even; magnitudes at or beyond the overflow threshold
round to infinity (no saturation); subnormal results are produced exactly.

| Input class | Output |
| --- | --- |
| `±0` | `±0` (sign preserved: `-0.0 → 0x8000`) |
| Normal, result in binary16 normal range (`2^-14 ≤ |y| ≤ 65504`) | Nearest binary16, ties to even. `1 + 2^-11 → 0x3C00` (tie, even), `1 + 3·2^-12 → 0x3C01` |
| Small (`|x| < 2^-14`) | Nearest multiple of `2^-24` (binary16 subnormal), ties to even: `2^-24 → 0x0001`, `2^-25 → 0x0000` (tie → even → zero), `1.5·2^-25 → 0x0001`, `2^-14 → 0x0400` |
| `65504 ≤ |x| < 65520` | `±65504` (`0x7BFF`/`0xFBFF`) |
| `|x| ≥ 65520` (the tie between 65504 and 2^16 rounds to even, which is infinity) | `±inf` (`0x7C00`/`0xFC00`); `1e5 → +inf` |
| `±inf` | `±inf` |
| NaN | A NaN. Payload and sign propagation are **outside conformance**: only "is a NaN" is promised. Observed: CPU scalar/F16C/AVX-512 and PyTorch CPU produce a quiet NaN with the truncated high payload bits (`0x7fc12345 → 0x7e09`, `0xffe00000 → 0xff00`); PyTorch CUDA produces the canonical `0x7fff` for every NaN, sign included. |

Conformance: bit-exact for every non-NaN input; NaN-ness for NaN inputs.
Characterization: `TypedSemantics.IeeeRneNarrowingWitnessVectors`,
`ConvertF32F16.ScalarSpecials`, `ConvertF32F16.SimdVariantsBitExactVsScalar`
(`libreloc/test/QuantTest.cpp`).

### 3.2 `reloc.cast` f16 → f32, policy `exact`

`y = x` exactly: every binary16 value is a binary32 value, so the result is
unique and no rounding occurs.

| Input | Output (bits) |
| --- | --- |
| `0x0001` (smallest subnormal, `2^-24`) | `0x33800000` |
| `0x0400` (smallest normal, `2^-14`) | `0x38800000` |
| `0x7BFF` (65504) | `0x477FE000` |
| `0x7C00` / `0xFC00` | `0x7F800000` / `0xFF800000` |
| `0x8000` | `0x80000000` |
| NaN | A NaN (payload outside conformance; the natural implementation shifts the payload left by 13 bits). |

Conformance: bit-exact for non-NaN inputs. Characterization:
`CudaTypedSemantics.ExactWideningWitnessBits`,
`CudaConvertF16F32.ExactWideningOfCpuConvertOutput` (`CudaKernelsTest.cpp`,
actual GPU). libreloc has no CPU widening kernel (R3 gap, §7).

### 3.3 `reloc.quantize` f32 → s8, policy `symmetric_rne`

Parameters: `scale` per tensor or per channel (§4); the zero point is `0`
(the only value this policy admits; a nonzero constant or a runtime zero
point is a verifier error). Steps, each with its own rounding:

1. `inv = fl32(1 / scale)` — the reciprocal is formed **once per parameter
   element**, in binary32, from the declared scale.
2. `t = fl32(x · inv)`.
3. `c = min(max(t, -128), 127)` — `max` first, then `min`, under §2's
   `maxNum`/`minNum` rule, so **NaN → -128**.
4. `q = rne(c)` — `c ∈ [-128, 127]`, so no integer overflow is possible.

`x / scale` is a different rounding and is **not** this policy: at
`scale = 0.3` the witness vector `[0.1, 0.3, 0.7, 1.1, 2.9, 10.1, 100.3, 1000.7]`
gives different bits for `x · fl32(1/0.3)` and `x / 0.3` in 3 of 8 cases
(`TypedSemantics.ReciprocalScaleIsFormedOnceInBinary32`; the pinned PyTorch
build shows the same 3 of 8 on CPU and CUDA). Clamp-then-round and
round-then-clamp agree for every finite `t` (the bounds are integers), so an
implementation may use either order for finite values, but NaN must map to
`-128`.

| Input (unit scale) | Output |
| --- | --- |
| Ties `[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]` | `[-2, -2, 0, 0, 2, 2]` |
| Limits `[-129, -128, -127, 126, 127, 128]` | `[-128, -128, -127, 126, 127, 127]` |
| `NaN`, `+inf`, `-inf` | `-128`, `127`, `-128` |
| `-0.0`, any `|x| < 0.5` (including subnormals) | `0` |
| `scale` in the subnormal range (`< 2^-126`) | Legal but degenerate: `inv = +inf`, so nonzero `x` saturates to `±127/-128` and `x = 0` gives `0 · inf = NaN → -128`. Deterministic; not rejected. |

Conformance: bit-exact for every input, NaN included. Characterization:
`TypedSemantics.SymmetricRneQuantizeWitnessVectors` (scalar and every SIMD
tier the host supports), `QuantizePack.ScalarSpecialValues`,
`QuantizePack.SimdVariantsBitExactVsScalar`,
`GatherQuantize.SimdVariantsBitExactVsScalar` (CPU);
`CudaTypedSemantics.SymmetricRneQuantizeWitnessVectors`,
`CudaQuantize.BitExactVsCpuScalar` (GPU).

### 3.4 `reloc.dequantize` s8 → f32, policy `affine`

Parameters: `scale` and `zero_point` per tensor or per channel (§4); zero
points may take any value in `[-128, 127]`.

1. `d = q - zero_point` — exact integer arithmetic, `d ∈ [-255, 255]`, hence
   exactly representable in binary32.
2. `y = fl32(d · scale)` — one rounding.

| Case | Output |
| --- | --- |
| `scale` a power of two | Exact (`|d| ≤ 255 < 2^24`). |
| `q ∈ [-128, -1, 0, 1, 127]`, `zero_point = 0`, `scale = 0.3` | bits `0xC219999A, 0xBE99999A, 0x00000000, 0x3E99999A, 0x42186667` |
| `q = -128`, `zero_point = 127`, `scale = 0.3` | `(-255) · 0.3 → 0xC2990000` (`-76.5`) |
| Non-finite inputs | None exist: the input domain is integer and scales are finite and positive, so every output is finite. |

Conformance: the formula has one rounding, so C4's tolerance against this
reference is **0 ulp** (bit-exact). Comparing against a differently rounded
formula (for example double-precision arithmetic) is outside the contract.
Dequantization is **not** an inverse of quantization: `dequantize(quantize(x))`
differs from `x` for every `x` that is not a multiple of `scale` inside the
range, and the pair is never cancelled (§5). Characterization:
`TypedSemantics.AffineDequantizeReferenceBits` (CPU reference arithmetic),
`CudaTypedSemantics.AffineDequantizeWitnessBits`,
`CudaDequant.ExactVsHostReference` (GPU, zero point 0).

## 4. Parameters

### 4.1 Scale denotes the step size

The IR parameter is the **scale** (the real value of one quantization step),
never the reciprocal. Programs and frameworks supply scales; `dequantize`
consumes the scale directly; `quantize` forms `inv = fl32(1/scale)` as its
first step (§3.3). The existing kernels take `invScales`
(`quantizePackF32S8`, `gatherQuantizeF32S8`, `quantizeF32S8`,
`pyreloc.prefold_s8`): the typed path (R3) must form `inv` from the declared
scale with exactly this rounding and must not accept a caller-supplied
reciprocal, so that one definition covers every implementation.

### 4.2 Domains

| Parameter | Element type | Domain | Checked |
| --- | --- | --- | --- |
| `scale` | `f32` | finite and strictly positive (`0`, negative, `NaN`, `±inf` rejected) | constants: verifier; runtime: bind time (C3) |
| `zero_point` | constants: any signless integer type; runtime: `i32` | integer in `[-128, 127]`; under `symmetric_rne` exactly `0` | constants: verifier; runtime: bind time (C3) |

The optional `zero_point` defaults to the per-tensor constant `0`.

### 4.3 Per-tensor and per-channel forms

| Form | Syntax | `scale` rank | `zero_point` rank | Parameter index |
| --- | --- | --- | --- | --- |
| Per tensor | no `axis` | 0 | 0 | none (one value) |
| Per channel | `axis a`, `0 ≤ a < rank(operand)` | 1, length `extent[a]` | 0 (broadcast to every channel) or 1, length `extent[a]` | the element's coordinate along axis `a` **of the typed op's operand** |

Rules:

- The channel axis is relative to the operand of the typed operation as
  written, before any later fold. It is a non-negative index; negative
  (Python-style) axes are rejected.
- Rank-1 parameters must have exactly `extent[a]` entries. The verifier
  rejects a provable mismatch (constant length versus constant extent, or a
  binding's declared extent that provably differs); an undecidable comparison
  (symbolic extent, or a binding declared with a different symbol) is
  accepted and becomes a bind-time guard (§4.4).
- No other broadcasting exists: a rank-1 parameter of length 1 against an
  extent greater than 1 is a length mismatch, and parameters never have rank
  greater than 1.
- Padded channels have no parameter: a pad on the channel axis before a
  quantize/dequantize has no defined per-channel meaning (§5).

Witness (`test/dialect/reloc/typed_ops.mlir`, `TypedSemantics.PerChannelAxisIsTheOperandAxisNotTheKernelChannel`):
shape `[2, 3, 4]`, `axis 1`, scales `[0.5, 1.0, 2.0]`, `x[i] = i - 11.5`. Slice
`[0, :, :]` quantizes to `[[-23, -21, -19, -17], [-8, -6, -6, -4], [-2, -1, -1, 0]]`
(the `-6, -6` pair is two ties rounding to even in opposite directions).
PyTorch's `quantize_per_channel(axis=1)` produces the same slice, confirming
the axis convention.

### 4.4 Inline constants and runtime bindings

| Declaration | Syntax | Identity | Element type and shape |
| --- | --- | --- | --- |
| Inline constant | `dense<...> : tensor<f32>` / `tensor<Nxf32>` / `tensor<Nxi32>` | the bits themselves; preserved exactly through print/parse (`0x3EAAAAAB` round-trips as `0.333333343`) | from the dense attribute |
| Runtime binding | `#reloc.binding<"name" : [extents], type>` | the **name**, function-wide, like a plan symbol | declared: `f32` for scales, `i32` for zero points; rank 0 or 1 with sym-expression extents |

- A binding name identifies one runtime tensor: the same name in several
  operations denotes the same bound value and must carry identical declared
  type and extents (C3 rejects a conflicting redeclaration across ops; the
  verifier rejects the within-op conflict of `scale` and `zero_point` sharing
  a name). Names are never addresses: no raw pointer or capture hint is part
  of the IR or the artifact.
- Static versus bind-time obligations:

| Obligation | Constant parameter | Runtime binding |
| --- | --- | --- |
| Kind, element type, rank, form (§4.3) | verifier | verifier |
| Values (finite positive scale, zero point range, `symmetric_rne` zero point `0`) | verifier | bind time (C3); `symmetric_rne` forbids runtime zero points outright |
| Rank-1 length equals `extent[a]` | verifier when both constant; bind time when the extent is symbolic | verifier when provably different; bind time otherwise |
| Presence, dtype and shape of the bound tensor | n/a | bind time (C3) |
| Any check by inspecting a capture-time hint or address | never | never |

Unknown runtime values are never validated by the verifier: symbolic extents
and declared bindings are preserved verbatim, and C3 receives the residual
guards above as its binding contract.

## 5. Order and composition (contract for C2)

- **Stage typing.** A chain is a sequence of stages; every layout op keeps
  the stage's element type (its verifier requires equal input/result element
  types), every typed op maps one stage type to the next under its table.
  Operation order is semantic: C2 may reorder or merge only where it proves
  bitwise equivalence under these tables.
- **Fill-value domains.** A `pad` fill belongs to the stage in which the pad
  occurs (its type must equal the element type: `f32` bits, `f16` bits, or an
  `s8` value in `[-128, 127]`). Consequently:
  - `pad(f32, v)` then `quantize` is **not** `quantize` then `pad(s8, w)`
    unless `w = quantize(v)` under the affected element's parameter, and a
    padded position on the channel axis has no parameter at all. C2 must
    keep the order or prove the fill equivalence; element counts agreeing is
    not a proof.
  - `pad(f32, v)` then `cast ieee_rne` equals `cast` then `pad(f16, ieee_rne(v))`
    exactly (the fill converts under the same table), and likewise for
    `exact` widening.
  - `pad(s8, w)` then `dequantize` equals `dequantize` then
    `pad(f32, affine(w))` only for per-tensor parameters or a pad off the
    channel axis; otherwise bail.
- **No cancellation.** `cast f32→f16` followed by `cast f16→f32` is lossy and
  not the identity; `quantize` followed by `dequantize` (or the reverse) is
  not the identity; two casts never compose into a "no-op". C2 must not fold
  any of these away (`typed_ops.mlir`, `@quantize_dequantize_is_not_cancelled`).
- **`no_copy`.** A plan is a pure view only if its layout is a view **and**
  its typed stage list is empty. Any typed stage requires data movement.
- **Direction.** A D2H computation compiles its requested forward order; a
  lossy value transform has no inverse, and the inverse-layout scatter API
  (`executeD2H*`) applies to layout-only plans only.
- **Channel axis under folds.** `transpose` permutes the channel axis index;
  a `reshape` that splits or merges the channel axis has no per-channel
  meaning without an index mapping and must bail; a `pad` on the channel axis
  before the typed op bails (§4.3). The kernels index the channel by the
  **outermost coalesced plan axis** (`gatherQuantizeF32S8`,
  `dequantRelocateS8F32`), so the mapping from the operand axis to that
  position is C2's to carry and R3's to establish
  (`TypedSemantics.PerChannelAxisIsTheOperandAxisNotTheKernelChannel` shows
  both the slice form and the axis-outermost plan form).
- **Fold pass.** Typed ops are chain members (C2): the layout folds into one
  `#reloc.plan`, every stage records its channel map over the logical result
  coordinates, fills fold through later per-tensor stages or the chain bails,
  and the result is a `reloc.typed_plan_result`
  ([reloc-typed-folding.md](reloc-typed-folding.md)).

## 6. Witness vectors

Downstream tests (C2–C4, R3) must include at least these inputs with these
outputs.

| Class | Input | Expected |
| --- | --- | --- |
| RNE ties, unit scale, zero point 0 | `[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]` | `symmetric_rne`: `[-2, -2, 0, 0, 2, 2]` |
| Signed-i8 limits | `[-129, -128, -127, 126, 127, 128]` | `symmetric_rne`: `[-128, -128, -127, 126, 127, 127]` |
| binary16 boundaries | `±0, 2^-24, 2^-25, 1.5·2^-25, 2^-14, 65504, 65519.99, 65520, 1e5, 1+2^-11, 1+3·2^-12` | `ieee_rne`: `0x0000/0x8000, 0x0001, 0x0000, 0x0001, 0x0400, 0x7BFF, 0x7BFF, 0x7C00, 0x7C00, 0x3C00, 0x3C01` |
| binary16 → binary32 | `0x0001, 0x0400, 0x7BFF, 0x7C00, 0xFC00, 0x8000` | `exact`: `0x33800000, 0x38800000, 0x477FE000, 0x7F800000, 0xFF800000, 0x80000000` |
| Specials | `NaN, +inf, -inf` | `symmetric_rne`: `-128, 127, -128`; `ieee_rne`: NaN, `0x7C00`, `0xFC00` |
| Reciprocal formation | `scale = 0.3`, `x ∈ [0.1, 0.3, 0.7, 1.1, 2.9, 10.1, 100.3, 1000.7]` | `x · fl32(1/0.3)` and `x / 0.3` differ in 3 of 8 bit patterns; the contract is the former |
| Dequantize bits | `q ∈ [-128, -1, 0, 1, 127]`, `scale = 0.3`, zero point 0 | `0xC219999A, 0xBE99999A, 0x00000000, 0x3E99999A, 0x42186667`; with zero point 127 and `q = -128`: `0xC2990000` |
| Parameters (static) | scale `0`, negative, NaN, `+inf`; zero point `128`, `-129`; nonzero or runtime zero point under `symmetric_rne` | verifier errors (`typed_invalid_ops.mlir`) |
| Parameters (bind time) | the same values arriving through a `#reloc.binding` | C3 rejects before execution |
| Channels | shape `[2, 3, 4]`, `axis 1`, scales `[0.5, 1, 2]`, `x[i] = i - 11.5` | slice 0: `[[-23, -21, -19, -17], [-8, -6, -6, -4], [-2, -1, -1, 0]]`; `axis 3`, `axis -1`, length 4 or 2 against extent 3: verifier errors |

## 7. Compatibility and characterization

Characterized on 2026-09-23: CPU on an AMD EPYC 7351 (AVX2 with F16C/FMA;
**no AVX-512**, so the AVX-512 tiers were not exercised here and rely on the
existing bit-exactness tests when run on an AVX-512 host); GPU on an NVIDIA
GeForce RTX 2080 Ti (sm_75, driver 595.71.05, CUDA 12.6.3); PyTorch 2.14.0
`+cpu` and `+cu126` on CPython 3.14.7 (the qualified baseline,
[torch-finalization/README.md](torch-finalization/README.md#selected-version-baseline)).
Skipped GPU tests never qualify a CUDA claim. The PyTorch rows come from
[typed-evidence/torch_witness.py](typed-evidence/torch_witness.py) with its
captured outputs [torch-witness-cpu.txt](typed-evidence/torch-witness-cpu.txt)
and [torch-witness-cuda.txt](typed-evidence/torch-witness-cuda.txt); the kernel
rows name their gtest cases.

| Implementation | Operation / policy | Status | Evidence and conditions |
| --- | --- | --- | --- |
| CPU `quantizePackF32S8` (scalar, AVX2, AVX-512) | `symmetric_rne`, per tensor or per channel with the channel as the outermost contiguous block | **Directly usable** | `TypedSemantics.SymmetricRneQuantizeWitnessVectors`, `QuantizePack.*`; consumes `inv` (R3 forms it, §4.1) |
| CPU `gatherQuantizeF32S8` (scalar, AVX-512, AVX-512Pf) | `symmetric_rne` fused with a layout plan | **Directly usable** when the channel axis is the outermost coalesced plan axis, rank ≥ 2, no pads, unit innermost dst stride | `GatherQuantize.*`, `TypedSemantics.PerChannelAxisIsTheOperandAxisNotTheKernelChannel`; other channel positions or pads: R3 gap |
| CUDA `quantizeF32S8` | `symmetric_rne`, channel = outermost block | **Directly usable** | `CudaTypedSemantics.SymmetricRneQuantizeWitnessVectors`, `CudaQuantize.BitExactVsCpuScalar` |
| CPU `convertF32F16` (scalar, F16C, AVX-512) | `ieee_rne` | **Directly usable** | `TypedSemantics.IeeeRneNarrowingWitnessVectors`, `ConvertF32F16.*`; NaN payload not promised |
| CUDA `convertF16F32` | `exact` | **Directly usable** | `CudaTypedSemantics.ExactWideningWitnessBits` |
| CPU f16 → f32 | `exact` | **Needs a new reference path (R3)**: no CPU widening kernel exists | — |
| CUDA `dequantS8F32`, `dequantRelocateS8F32` | `affine` with zero point `0` (`q · scale`) | **Directly usable** for zero point 0; **needs a reference/adapter path (R3)** for nonzero zero points and for channel axes other than the outermost | `CudaTypedSemantics.AffineDequantizeWitnessBits`, `CudaDequant.ExactVsHostReference`, `CudaDequantRelocate.MatchesHostOracle` |
| CPU s8 → f32 | `affine` | **Needs a new reference path (R3)**: no CPU dequantize kernel | reference arithmetic pinned by `TypedSemantics.AffineDequantizeReferenceBits` |
| Any kernel with `padRegions` | typed stage on a padded plan | **Unsupported** by the fused kernels (asserted); R3 must stage or bail | `Quant.h`, `CudaKernels.h` preconditions |
| int4 pack/unpack (`packS8S4`, `unpackS4S8`) | — | **Out of scope** (no typed operation; sub-byte storage has no contract) | [#132](https://github.com/JueonPark/sym/issues/132) quantization scope |
| PyTorch 2.14.0 `.to(torch.float16)` (CPU and CUDA) | `ieee_rne` | Matches every non-NaN witness bit for bit; NaN payload differs between CPU (truncated) and CUDA (canonical `0x7fff`, sign dropped) | outside conformance by §3.1 |
| PyTorch `.to(torch.float32)` from f16 | `exact` | Matches | — |
| PyTorch `torch.round` | `rne` | Matches (`[-2, -2, -0, 0, 2, 2]`) | — |
| PyTorch `quantize_per_tensor(..., dtype=torch.qint8, zero_point=0)` | `symmetric_rne`? | Ties and limits match; **NaN → 127** (kernel: `-128`), `+inf → 127`, `-inf → -128`. **Not a conformant implementation**; C4 uses the reference, not `torch.quantize_*` (also deprecated in 2.14 upstream) | difference explicitly excluded |
| PyTorch `quantize_per_tensor(zero_point ≠ 0)` | affine quantization (`clamp(rne(x·inv) + zp)`) | **Unsupported**: no C1 quantize policy has a zero point. Observed zp `127`: ties → `[125, 125, 127, 127, 127, 127]`, limits → `[-2, -1, 0, 127, 127, 127]`; a future policy needs its own name and NaN rule | — |
| PyTorch `quantize_per_channel(axis=1)` | channel convention | Matches §4.3 (witness slice) | — |
| PyTorch `.to(torch.int8)` | — | **Not a quantization**: truncation toward zero with platform-dependent overflow/NaN (CPU: `128 → -128`, `NaN/±inf → 0`; CUDA: `+inf → -1`, `3e9 → -1`) | excluded |
| PyTorch `dequantize` / `q.float() * scale` | `affine`, zero point 0 | Matches the reference bits on CPU and CUDA | — |

## 8. Handoff

**C2 (folding and canonicalization) consumed** the operation/policy schema
(§1, §3), the channel-axis meaning and its behavior under transpose, reshape
and pad (§4.3, §5) and the stage-typing and fill-value rules (§5); its
typed-plan representation, folding rules and rejection boundaries are
[reloc-typed-folding.md](reloc-typed-folding.md).

**C3 (versioned artifact and binding) receives** the parameter declaration
model (§4.4): constants by value, runtime parameters by name with declared
`f32`/`i32` type and rank-0/1 sym-expression extents; the bind-time
obligations table (values, symbolic lengths, presence/dtype/shape of bound
tensors, cross-op declaration consistency); and the fact that wire v0's type
table already encodes `f32`, `f16` and signless `i8` while typed stages,
parameters and bindings need a new wire version. Until then the exporter
answers `typed_unsupported` (`test/dialect/reloc/export_check.py`).

**C4 and R3 receive** the witness vectors (§6), the compatibility table (§7)
and these concrete kernel/reference gaps. R3 (issue [#147](https://github.com/JueonPark/sym/issues/147),
[runtime-dispatch.md](runtime-dispatch.md)) closed 1–3 and 5 with the scalar
reference (`reloc::typed::executeHost`: `inv` formed once from the declared
scale, CPU dequantize with any zero point, CPU widening, channel maps over
the result coordinates) and 6 for the reference path (pads are legal there;
the fused kernels stay excluded on padded plans); 4 remains an exclusion
(`no_cuda_kernel:nonzero_zero_point`) and 7 is not planned:

1. Form `inv = fl32(1/scale)` from the declared scale before calling the
   `invScales`-based kernels; never accept a caller reciprocal on the typed path.
2. CPU s8 → f32 dequantize reference (`affine`, any zero point).
3. CPU f16 → f32 widening reference (`exact`).
4. CUDA dequantize with a nonzero zero point.
5. Channel axis not outermost: slice loop, relayout, or fold-time axis motion.
6. Typed stages on plans with pad regions.
7. A PyTorch-style affine quantize (nonzero zero point, `NaN → qmax`) only as a
   new, separately named policy, if ever needed.

**Frontend.** C4 ([typed-relocation-support.md](typed-relocation-support.md))
imports the casts of §3.1/§3.2 and the `quantized_decomposed.dequantize_*`
operators (audited to be §3.4 bit for bit) into typed recipes and executes
them through R3; `quantized_decomposed.quantize_*` stays excluded
(`quantize_semantics_unproved`: double-rounded reciprocal, platform NaN
conversion), and `prepare_weights` stays gated by
`typed_artifacts_unavailable` ([torch-support.md](torch-support.md)). The
layout-only regression corpus, T2's same-dtype artifact checks and the v0
exporter behavior are unchanged by C1–C4.
