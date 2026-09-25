# Plan-driven typed dispatch (R3, issue [#147](https://github.com/JueonPark/sym/issues/147))

R3 executes C3's typed bound plans ([reloc-export.md](reloc-export.md),
[libreloc/README.md](../libreloc/README.md#typed-plans-c3-issue-143)) through
qualified paths and reports what it did. Three questions are kept apart:

1. **Capability** (`reloc::dispatch::queryCapability`, `pyreloc.query_capability`):
   which implementations are semantically equivalent to the reference and
   actually exist for this program, direction and device end. Pure; no cost
   model; the answer never changes the requested precision or drops a stage.
2. **Policy** (`prepareDispatch` / `selectImplementation`, `pyreloc.select_dispatch`):
   `original_cpu` forces the CPU reference pipeline, `auto` consults the
   central cost model ([CostModel.h](../libreloc/include/reloc/CostModel.h),
   issue [#111](https://github.com/JueonPark/sym/issues/111)) when a calibration is given and translates its advice to an
   eligible row or falls back with a recorded reason; an explicit row label
   is for tests and conformance evidence. Cost estimates rank; they never
   grant capability.
3. **Execution** (`executeDispatch`, `pyreloc.execute_dispatch`): exactly the
   selected row runs through R2's `CopyBackend` (host or CUDA), blocks until
   this request's work completed, and fills the report. Failures after the
   launch decision propagate; no second path is tried.

The Torch bridge is `reloc_torch.dispatch.prepare_typed_transfer(compiled,
source, device, *, parameters, policy="auto", calibration=None)` and
`execute_typed_transfer(request) -> DispatchResult(tensor, report)`;
parameters are CPU tensors bound **by declared name**, snapshotted at
preparation and re-checked by value at execution.

## Program model and the reference

A typed bound plan is one layout (an index map with fused pad fills in the
result dtype) plus ordered element-wise stages. Every valid source element
flows source scalar → stage 0 → … → stage S−1 → its destination cell; a
per-channel stage selects its parameter with the channel map evaluated over
the **logical result coordinates** (C2). Pads carry the original fill folded
to whatever boundary they are observed at. `reloc::typed::executeHost`
([TypedExecute.h](../libreloc/include/reloc/TypedExecute.h)) is the scalar
reference for the whole program and for any contiguous stage range
`[from, to)`: it reads the dense source layout at boundary `from`, writes the
dense padded result layout at boundary `to`, and refuses a cut where a pad
has not entered yet (`pads_not_settled`). The implemented C1 tables:

| Stage | Reference arithmetic | Reference function |
| --- | --- | --- |
| cast f32 → f16 `ieee_rne` | RNE, overflow → inf, subnormals kept | `quant::narrowF32F16` |
| cast f16 → f32 `exact` | exact | `quant::widenF16F32` |
| quantize f32 → s8 `symmetric_rne` | `inv = fl32(1/scale)` once per parameter element; `rne(clamp(x·inv, −128, 127))`, NaN → −128 | `quant::quantizeOneF32S8` |
| dequantize s8 → f32 `affine` | `fl32((q − zp) · scale)`, one rounding, any zero point in range | inline |

Anything else (other type pairs, other policies) is `unsupported_stage`
before any buffer is touched, for every row including the reference.

## Capability table

`S` is the stage count, `k` a stage boundary (0 = source side). "Wire" is the
tensor that crosses the link; its byte count is derived from the actual
partition (`typed::bytesAt`), never from a guessed ratio. Every row was
compared bit for bit against `executeHost` (`DispatchTest.cpp`,
`CudaDispatch.*` on an RTX 2080 Ti with CUDA 12.6.3; `test_typed_dispatch.py`
and `torch_frontend/test_dispatch.py -m gpu` against a NumPy oracle).

| Row | Direction | What runs where | Wire | Conditions | Method class |
| --- | --- | --- | --- | --- | --- |
| `cpu_reference` | H2D, D2H | CPU: layout + every stage; then the dense copy (H2D: result → device; D2H: device source → pinned staging first) | H2D: destination bytes; D2H: source bytes | every supported program, host or CUDA backend. **This is `original_cpu`.** | A (H2D) / B (D2H) |
| `cpu_stages_cuda_stages@k` | H2D (CUDA) | CPU: layout + stages `[0,k)` into the result layout with the fills folded to `k`; device: stages `[k,S)` element-wise | `bytesAt(k)` in the result layout | `padsSettledBy(k)`; every stage in `[k,S)` device-qualified (below) | A |
| `cuda_stages_then_cpu@k` | D2H (CUDA) | device: stages `[0,k)` element-wise in the **source** layout; CPU: layout + stages `[k,S)` + final fills | `bytesAt(k)` in the source layout | every stage in `[0,k)` device-qualified in the source layout | A |
| `cuda_dequant_relocate` | H2D (CUDA) | raw s8 source to the device; `dequantRelocateS8F32` (layout + stage 0 fused); stages `[1,S)` element-wise | source bytes | stage 0 = dequantize, zero point 0, per tensor or channel = coalesced outer axis (= result axis 0); no pads; rank ≥ 2; unit inner dst stride; later stages device-qualified | B |
| `cuda_relocate_f32` | H2D (CUDA) | raw f32 source to the device; `relocateF32` (layout); stages `[0,S)` element-wise | source bytes | f32 source; no pads; rank ≤ 8; every stage device-qualified | B |

**Device-qualified element-wise stages** (the existing kernels, each
bit-identical to the reference by its own tests):

| Stage | Kernel | Restriction |
| --- | --- | --- |
| cast f16 → f32 exact | `cuda::convertF16F32` | none |
| cast f32 → f16 ieee_rne | — | `no_cuda_kernel:cast_f32_f16` (no GPU narrowing kernel exists) |
| quantize symmetric_rne | `cuda::quantizeF32S8` | per tensor; per channel only when the channel map is the outermost axis of the buffer the kernel runs over (result axis 0 for H2D; for D2H, source axis 0 mapped one-to-one to a result axis) — `channel_not_outer_result_axis` / `channel_not_source_outer_axis` otherwise |
| dequantize affine | `cuda::dequantS8F32` | zero point 0 only (`no_cuda_kernel:nonzero_zero_point`); channel rule as above |

Exclusions are stable strings in `Capability.excluded` (`pads_not_settled`,
`stage j: <reason>`, `stage 0 is not a dequantize`, `layout has pads`,
`rank below two`, `rank above eight`, `source is not f32`,
`inner destination stride is not one`, `channel axis is not the coalesced
outer axis`, `no_cuda_device`). Without a CUDA backend only `cpu_reference`
is eligible, so CPU-only CI exercises the whole forced pipeline host-to-host.

Not implemented on purpose: int4 (no typed operation), a PyTorch-style
affine quantize (nonzero zero point, NaN → qmax), any inverse of a lossy
stage (D2H runs the **forward** program; the inverse-layout scatter API is
layout-only), device-resident parameter tensors (see below), and cuts where
a pad enters later than the cut.

## Policy and placement reasons

| Policy | Row | `placement_reason` |
| --- | --- | --- |
| `original_cpu` | `cpu_reference` | `forced`; when the program has no reference path the request fails with the program's code (`unsupported_stage`, `invalid_parameter`) and is **not** reported as forced |
| `auto`, one eligible row | `cpu_reference` | `only_qualified_path` |
| `auto`, no calibration | `cpu_reference` | `no_calibration` |
| `auto`, calibration lacks the keys for this pattern / wire ratio | `cpu_reference` | `model_missing_keys` |
| `auto`, decision A | cheapest-wire class-A row (`cpu_stages_cuda_stages@k` or `cpu_reference`) | `cost_model_prefers_a` |
| `auto`, decision B | cheapest-wire class-B row | `cost_model_prefers_b`; `advice_unavailable_path` (falls back to the class-A row) when no B row is eligible |
| `auto`, decision APrefold | the class-A row | `prefold_requires_prepared_artifact` (prefolding is T4's; the dispatcher never quantizes a weight to use it) |
| explicit label | that row | `explicit_selection`; `implementation_unavailable` when it is not eligible |

`auto` calls `costmodel::decide(model, classify(layout), sourceBytes, r,
threads)` with `r` = (cheapest class-A wire) / source bytes. Missing
calibration therefore changes placement only; it never disables the
reference nor authorizes another precision.

## Byte accounting

The report is scalar-only (no tensors, no pointers) and carries:

| Field | Meaning |
| --- | --- |
| `source_bytes` | the caller's dense source tensor |
| `wire_bytes` | the selected row's wire tensor (`bytesAt(wire_boundary, layout)`) |
| `destination_bytes` | the padded result (`== layout.total_bytes`) |
| `parameter_bytes` | every bound parameter snapshot (host) |
| `payload_bytes_transferred` | **observed** link traffic of the executed row: the wire tensor plus every device parameter upload the row needed (a representation footprint is not an observed total) |
| `device_temp_bytes` | device scratch the row allocates itself (intermediate boundaries, the raw source copy for the B rows) |
| `implementation`, `policy`, `placement_reason`, `method`, `wire_boundary`, `artifact_version` (= typed wire format 1), `executed` | selection and outcome |

Example (`quant_dequant_sym`, f32 → s8 → f32, N = 16, H2D): source 64,
`cpu_reference` wire 64, `cpu_stages_cuda_stages@1` wire 16, destination 64,
parameters 8; the D2H reference moves 64 bytes of source.

## Parameters and lifetimes (Task 4)

- Runtime parameters enter **by declared name** as `(dtype, extents, bytes)`
  snapshots; `bindTyped` copies and validates them (dtype, rank, channel
  length, byte size, finite positive scales, zero points in range). The
  bridge snapshots the CPU tensor values at `prepare_typed_transfer` and
  re-reads them at `execute_typed_transfer`: a changed value (in place or by
  replacement) is `stale typed transfer: parameter '<name>' changed after
  preflight`, a re-preparation re-validates and produces fresh results, and
  an invalid updated value fails preflight (`bind_error`).
- Device-resident parameter tensors are the stable exclusion
  `device_parameters_unavailable`: there is no ordered preparation path yet
  (stream-ordered copy into owned host validation storage, wait, retain).
  The device kernels receive their scales through the runtime's own pinned
  upload on the backend's stream 0, counted in `payload_bytes_transferred`.
- Prefold: `dispatch::prefoldSpecFor(plan)` / `pyreloc.typed_prefold_spec`
  answers whether a typed program is exactly the S8 quantize the existing
  prefolder implements (`s8_gather_quant` when source and destination strides
  differ, `s8_quant_pack` when they agree; per tensor or channel = the
  coalesced outer axis, which must be a bare result dimension; no pads;
  rank ≥ 2 after coalescing; packed destination) and returns the reciprocal
  scales formed from the declared scales. The binder coalesces an identity
  layout to one axis, so `s8_quant_pack` stays out of reach of a compiled
  identity artifact (T4's recorded handoff fact) until a channel-preserving
  plan exists. Everything else is
  `prefold_unavailable`, and an f32 weight is never quantized to reach the
  prefolder. `prefold::prefoldArtifact` now has an overload taking a
  `std::shared_ptr<CopyBackend>` that the artifact keeps alive until it has
  freed its staging (`PrefoldArtifact::ownsBackend`).
- **T4's freshness key** for a prepared typed artifact must cover: the
  program/artifact identity (plan bytes digest, wire version), the bound
  shape and channel mapping (symbol values), the weight identity, storage
  and version, every parameter identity **and value** (or version), dtype and
  device, and any execution-affecting configuration (policy, calibration
  identity, selected row). Mutation, replacement, `load_state_dict` or close
  require re-preparation or fallback. R3 introduces no second weight cache.

## Evidence

- `libreloc/test/TypedExecuteTest.cpp`: C1 witness bits for every stage,
  pad order, channel selection under a channel-moving layout and under
  equal-sized axes, resumption from an intermediate wire, rejections.
- `libreloc/test/DispatchTest.cpp`: capability rows per program/direction/
  device, host-to-host forced baseline in both directions with reports,
  policy translation with a synthetic calibration (B at small sizes, A at
  large, fallback when the advised class has no row), view/program
  rejections, backend failure propagation, prefold spec; `CudaDispatch.*`
  runs every eligible H2D and D2H row on the GPU against the reference.
- `libreloc/python/tests/test_typed_dispatch.py` (Torch-free, real exporter,
  NumPy oracle) and `torch_frontend/test_dispatch.py` (bridge; `gpu` rows).
