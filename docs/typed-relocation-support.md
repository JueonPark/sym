# Typed relocation support (C4, issue #144)

The typed relocation track (issues #132 and #133) is complete end to end:
C1 defines the value transforms ([reloc-typed-semantics.md](reloc-typed-semantics.md)),
C2 folds them with layouts ([reloc-typed-folding.md](reloc-typed-folding.md)),
C3 ships them as wire v1 artifacts ([reloc-export.md](reloc-export.md)), R3
executes them through qualified rows ([runtime-dispatch.md](runtime-dispatch.md)),
and this document is C4's contract: what the frontend imports, what every
combination is proved against, and how to reproduce the evidence.

Four support states are kept apart and each is gated on its own proof:

| State | Gate | Where |
| --- | --- | --- |
| **captured** | the FX importer recognizes the region and its operands | `reloc_torch.fx_import` (reason-coded exclusions otherwise) |
| **compiled** | the real exporter folds it (`--typed`, schema 2 / wire v1) | `reloc_torch.compiler`, `artifact._admit` |
| **bound** | symbols and named parameters pass C3's binder | `pyreloc.bind_typed` (reason `bind_error` on the frontend) |
| **executed** | an R3 row equivalent to the reference exists for this direction and device | `reloc_torch.dispatch` / `pyreloc.prepare_dispatch` (`original_cpu` always, `auto` picks among qualified rows) |

Accepting typed IR never turns execution on: a region that reaches
"compiled" but has no executable row, or whose runtime parameters fail
preflight, runs its **original PyTorch region** with the reason recorded in
the backend's diagnostics (`fallbacks`, `exclusions`).

## Reference and comparison rules

`libreloc/python/tests/typed_reference.py` is the independent oracle: C1's
tables in NumPy (binary32 throughout; NumPy's binary16 conversion is RNE with
subnormals and overflow to inf, `np.rint` is ties-to-even). Rules, fixed
before any kernel comparison:

- casts and quantization: **bit-exact**; for NaN inputs only NaN-ness (C1 §3.1);
- dequantization: **0 ulp** (one rounding in the formula; anything else is a
  different formula, not tolerance).

## Support matrix

Directions: H2D = host source to CUDA destination, D2H = CUDA source to host
destination; every typed program is a **forward** program in both directions
(a lossy stage has no inverse; the inverse-layout scatter API is layout-only).
Artifact: wire v1 / manifest schema 2 for every typed row. Evidence columns:
CPU = `test_typed_conformance.py` (Torch-free, real exporter, CI), CUDA =
`torch_frontend/test_typed_conformance_gpu.py` + `test_dispatch.py -m gpu` +
`DispatchTest.cpp`'s `CudaDispatch.*` on the RTX 2080 Ti (local; a skip is
not evidence), Torch = `torch_frontend/test_typed_import.py`.

| Operation (policy) | Source → destination | Parameters | Layout | Direction | Runtime rows | Frontend import | State | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cast `ieee_rne` | f32 → f16 | — | any (transpose/reshape/pad before or after) | H2D, D2H | `cpu_reference` only (no GPU narrowing kernel: `no_cuda_kernel:cast_f32_f16`) | `aten._to_copy(dtype=f16)` alone or with the transfer | executed | CPU `cast_transpose_f16`, `dequant_inline_zp_cast`; CUDA rows; Torch import + CPU op route + GPU `torch.compile` |
| cast `exact` | f16 → f32 | — | any | H2D | `cpu_reference`, `cpu_stages_cuda_stages@k` (`convertF16F32`), `cuda_relocate_f32`? no (f16 source) | `aten._to_copy(dtype=f32)` from f16 | executed | CPU `widen_reshape`; CUDA rows |
| cast `exact` | f16 → f32 | — | any | D2H | `cpu_reference`, `cuda_stages_then_cpu@1` | same | executed | CUDA rows |
| quantize `symmetric_rne` | f32 → s8 | inline or runtime scale, per tensor or per channel | any pad-free layout; pads only before a per-tensor stage or after the quantize | H2D | `cpu_reference`; `cpu_stages_cuda_stages@k` and `cuda_relocate_f32` when the channel is result axis 0 (`quantizeF32S8`) | **not imported**: `quantized_decomposed.quantize_*` is not C1 (`quantize_semantics_unproved`); explicit recipes only | executed (explicit recipes) | CPU `quantize_channel_transpose_sym`, `witness_channel`, `quantize_edges`, `reciprocal_formation`, both pad orders; CUDA rows; example |
| quantize `symmetric_rne` | f32 → s8 | as above | as above | D2H | `cpu_reference`; `cuda_stages_then_cpu@k` when the channel is source axis 0 mapped one-to-one | not imported | executed (explicit) | CUDA rows |
| dequantize `affine`, zero point 0 | s8 → f32 | inline or runtime scale; per tensor or per channel | any | H2D | `cpu_reference`; `cpu_stages_cuda_stages@0` (`dequantS8F32`) when channel = result axis 0; `cuda_dequant_relocate` for rank ≥ 2 pad-free layouts with channel = coalesced outer axis | `quantized_decomposed.dequantize_per_tensor(.default/.tensor)`, `dequantize_per_channel` with f32 scales, integer zero points, `quant_min/max = -128/127`, int8 input, f32 output | executed | CPU `dequant_scale_witness`, `dequant_channel_runtime_transpose` (zp per channel); CUDA rows; Torch import + CPU op route + GPU `torch.compile` |
| dequantize `affine`, zero point ≠ 0 | s8 → f32 | as above | any | H2D, D2H | `cpu_reference` only (`no_cuda_kernel:nonzero_zero_point`) | as above | executed (CPU reference) | CPU `dequant_zp127`, `dequant_inline_zp_cast`; CUDA `cpu_reference` |
| f32 → s8 → f32 | f32 → f32 | inline scales | any, padded destination allowed | H2D, D2H | `cpu_reference`, `cpu_stages_cuda_stages@1` (s8 wire = ¼ of the source), `cuda_relocate_f32` (no pad); with a trailing pad only `cpu_reference` | explicit recipes only | executed | CPU `quant_dequant_padded` (two bindings); R3 tests |
| dequantize → cast | s8 → f16 | inline | any | H2D, D2H | `cpu_reference` | explicit recipes only (a graph would need both audited ops) | executed | CPU `dequant_inline_zp_cast` |
| cast `ieee_rne` → cast `exact` | f32 → f32 (lossy) | — | any | H2D, D2H | `cpu_reference` | two `_to_copy` nodes, both kept (never cancelled) | executed | Torch import test |
| quantize import from `quantized_decomposed.quantize_per_tensor/_per_channel` | f32 → s8 | — | — | — | — | **excluded**, `quantize_semantics_unproved`: `1.0/scale` is formed in double (C1: fl32 once), NaN reaches `.to(int8)` (C1: −128) | captured, excluded | Torch import test |
| any other dtype pair (`bf16`, `.to(int8)`, `.to(int32)`, f64 scales) | — | — | — | — | — | **excluded**, `typed_transform_unavailable` / `parameter_dtype_unsupported` | excluded | Torch import test; C1 §7 |
| `dequantize_per_tensor` with `quant_min/max ≠ −128/127` or `out_dtype ≠ f32` | — | — | — | — | — | `unsupported_quantization_range` / `typed_transform_unavailable` | excluded | Torch import test |
| runtime parameters on a CUDA device | — | device tensors | — | — | — | captured; execution `device_parameters_unavailable` (fallback) | compiled, not executed | R3 tests |
| same-device cast without a transfer (`x.to(f16)`) | — | — | — | — | — | `same_device_transfer` (nothing to relocate); eager `.to(dtype)`: `typed_transform_unavailable` | excluded | Torch import + eager tests |
| int4, PyTorch-style affine quantize (nonzero zp, NaN → qmax), non-blocking typed transfers | — | — | — | — | — | out of scope | — | C1 §7 |

Rows for the CUDA columns are exactly what `pyreloc.query_capability(bound,
direction, "cuda")` lists for each corpus fixture (recorded per fixture in
its JSON and asserted by the conformance test), so the matrix cannot drift
from the runtime's own answer.

## Frontend import (Task 3)

Audit of Torch 2.14.0 raw forms (`libreloc/python/reloc_torch/compat.py`,
`fx_import.py`): `.to(torch.float16)` is `aten._to_copy(dtype=f16)`;
`.to("cuda", f16)` is one `_to_copy(dtype=f16, device=cuda)`; the
`quantized_decomposed` operators keep their scale/zero-point operands as
scalars (`dequantize_per_tensor.default`) or tensors (`.tensor`,
`dequantize_per_channel.default`, with the channel `axis`). Imported recipes:

- scalar operands become exact inline bits (`fl32(scale)`, the zero point as
  i32); tensor operands become runtime parameters **named by their graph
  node**, declared with the operand's channel extent, and travel as an
  explicit `Tensor[]` operand of `reloc_torch::typed_transfer` (schema
  `(Tensor src, Tensor[] parameters, str handle, SymInt[] symbols, SymInt[]
  out_shape, SymInt[] out_strides, Device device, ScalarType dtype) ->
  Tensor`), never read at import time;
- the destination dtype is explicit in the op, so the fake kernel's metadata
  is the recipe's descriptor and `verify_result` checks the real output
  against it;
- the original region keeps its exact operators and takes `(src,
  *symbols, *parameters)`; a guard miss, a runtime rejection (invalid scale,
  device parameters, no CUDA) or a missing row runs it once with the reason
  recorded; nothing is launched first;
- eager `.to(device, dtype)` stays `typed_transform_unavailable` (the
  identity recipe cannot carry a cast); the compile path handles casts.

## Remaining handoff

- **T3 / R2 for automatic replacement**: typed regions already flow through
  the guarded custom-op route and R2's storage/stream contract via
  `reloc_torch.dispatch`. Still outside the automatic route: device-resident
  parameter tensors (stable exclusion `device_parameters_unavailable`; R3
  Task 4's ordered preparation phase), non-blocking typed transfers, and
  quantize imports (no C1-equivalent PyTorch operator exists to import).
- **T4 / R4 for weights and lifetime**: `prepare_weights` keeps
  `typed_artifacts_unavailable` until T4 wires `pyreloc.typed_prefold_spec`
  and the dispatch bridge behind its freshness key
  ([runtime-dispatch.md](runtime-dispatch.md#parameters-and-lifetimes-task-4));
  R4 collects the combined evidence.
- Performance is descriptive only: the reports carry `source_bytes`,
  `wire_bytes`, `destination_bytes`, `parameter_bytes` and the observed
  `payload_bytes_transferred`; no speedup gate exists.

## Reproduction

```bash
export SYM_BUILD="$PWD/build/torch-cpu"            # any configured build with the exporter
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
cmake --build "$SYM_BUILD" --target check-sym          # typed_conformance.mlir, typed_export.mlir, goldens
ctest --test-dir "$SYM_BUILD" --output-on-failure       # TypedExecute.*, Dispatch.*, Prefold*
/tmp/sym-torch-cpu/bin/python -m pytest libreloc/python/tests -m 'not gpu' -q   # conformance + import gate
/tmp/sym-torch-cpu/bin/python libreloc/test/corpus/typed/generate_typed_corpus.py --check  # fixtures are fresh
/tmp/sym-torch-cpu/bin/python libreloc/python/examples/torch_typed_relocation.py          # host mode
```

CUDA rows (qualified cu126 environment, R2/R3-enabled CUDA build):

```bash
export SYM_BUILD="$PWD/build/torch-cuda"; export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"; export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
/tmp/sym-torch-cuda/bin/python -m pytest libreloc/python/tests/torch_frontend/test_typed_conformance_gpu.py \
    libreloc/python/tests/torch_frontend/test_typed_import.py libreloc/python/tests/torch_frontend/test_dispatch.py -m gpu -q
/tmp/sym-torch-cuda/bin/python libreloc/python/examples/torch_typed_relocation.py --cuda
```

The recorded run is [typed-evidence/c4-cuda-run.txt](typed-evidence/c4-cuda-run.txt)
(device, driver, CUDA and PyTorch versions, pass/skip counts). CPU CI runs
the Torch-free conformance suite against the real exporter, the typed
decoder/binder and R3's CPU path on every push (`.github/workflows/build.yml`,
"Run Tests"); the cp314 job runs the frontend import gate.
