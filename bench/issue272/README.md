# From a captured Torch operation to a specialized relocation

Issue [#272](https://github.com/JueonPark/sym/issues/272) asks which parts of
a relocation the compiler generates, which work is reused, and how the resulting
path compares with manual fusion and optimized Torch. The experiments cover
blocking inference transfers. The application supplies an ordinary Torch
function and a backend configuration.

## Findings

| Question | Measurement |
| --- | --- |
| How close does a captured plan come to manual fusion? | On a held-out rank-four cast/permutation, warm completed latency was 0.156 ms with the current native kernel, 1.044 ms with the prior kernel, 0.541 ms with true materialization, 0.217 ms with manual fusion, 0.204 ms with eager Torch and 0.273 ms with Inductor. All used the same wire and one transfer. The full automatic frontend took 0.467 ms. Eager Torch also had the lower total cost over 36 calls. [Matched comparison and records](CAPTURED_QUALITY.md) |
| What actually ran on untuned inputs? | Two additional chains passed 42 bitwise checks per run, including shape changes, parameter revisions, saved Torch fallbacks and held outputs. The separate untimed run observed the register cast and dequantization/cast loops. Restoring earlier parameter bytes reused the corresponding group. [Capture, dispatch and guard evidence](WITNESS.md) |
| When did reuse help? | Across 48 processes on the original E3 trace, program caching saved 40 preparations and 8.74% of repeated-request time on layout cast. Live dequantization had zero cache hits and group reuse cost 2.45%. All 2,880 outputs were exact. [Cold, total and warm results](RESULTS.md) |
| What does the caller supply, and what grows? | The automatic witnesses use ordinary Torch functions and a backend configuration. The archived manual comparator uses a 64-line C++ kernel family. The bounded specialization added 3,328 bytes of runtime executable text. Timed binding, compilation, cache inventories and all eight losses to the fastest manual/Torch control on the original E1 suite are recorded. [Archived controls and source counts](CONTROLS.md) |

## What is generated and what is reused

1. Dynamo calls `RelocBackend` with an FX graph. The importer checks operation,
   alias, ownership and metadata constraints. The exporter generates a guarded
   artifact containing the mapping, ordered value stages, parameter coordinates
   and permitted wire boundaries. The semantic cache can reuse it across
   concrete shapes. Dynamo may compile more than one enclosing graph.
2. At a call, the adapter binds shapes and snapshots the exact parameter bytes.
   Native capability checking determines the eligible dispatch rows. The
   automatic specialization key includes the artifact, bindings, tensor/view
   metadata, device and parameter revision. Admission and cache bounds apply
   before a native launch. Each live scale or zero point is checked again before
   use. Restoring the previous bytes can reuse an earlier binding.
3. A typed program contains its validated dispatch template. A group holds the
   structural schedule and rebinds current input and fresh output storage on
   each call. Indexed regions bind and check the current indices every time.
   Outstanding calls retain their resources across cache eviction. Host-observed
   completion retires the lease. A guard failure before launch may execute the
   saved Torch region. An error after launch propagates to the caller.
4. The compiler generates plan bytes. The runtime selects ahead-of-time kernel
   variants according to the bound layout and ordered stage sequence. Six
   AVX2/F16C variants implement eligible cast/dequantization runs, with scalar
   tails and buffered fallback. There is no per-shape executable code allocation.
   CPU loop specialization, prebuilt CUDA kernels and the explicit handwritten
   pooled DLRM producer have separate provenance. The opaque Torch operator
   carries the resulting contract into execution. No measured benefit is
   attributed to the operator itself.

[FX importer](../../libreloc/python/reloc_torch/fx_import.py) ·
[Prepared binding and guards](../../libreloc/python/reloc_torch/automatic.py) ·
[Group ownership](../../libreloc/python/reloc_torch/prepared_group.py) ·
[Typed kernel selection](../../libreloc/src/TypedExecute.cpp) ·
[Region legality](../../docs/relocation-regions.md)

## Reproduce

[Protocol](PROTOCOL.md) covers the paired controls and timing boundaries.
[Prior matched comparisons](CONTROLS.md) covers the original E1 and E3
traces, manual fusion, Torch/Inductor, code size, preparation and held-out
results on the RTX 2080 Ti. The new experiments use a Ryzen 7 7800X3D and
RTX 4070 Ti SUPER. Comparisons are within one machine and protocol.

For fresh GPU runs, use the repository's Python 3.14 / Torch 2.14.0+cu126
environment with pybind11, NumPy, CUDA, CMake, Ninja and GNU patch. Download the
[original E3 input archive](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04)
and the [#265 compiler bundle](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-265),
verify their SHA-256 inventories, and set the paths below. Run from the Sym
repository root in a checkout with history for the native #267 change.

```sh
python -m bench.issue272.build_controls --output "$BUILDS" \
  --compiler-bin "$ISSUE265/runtime/candidate/sym/tools" --cuda-arch 89
python -m bench.issue272.ablate run --artifact "$E3" --specialized-build "$BUILDS/on" \
  --generic-build "$BUILDS/off" --python "$(command -v python)" --rounds 3 \
  --affinity 0-7 --output "$RUN"
```

The build creates two Release libraries from the same sources, reversing only
the native change at `b319a0a` in the control. Both use identical Torch and
Python bindings, compiler and frontend files. The runner rotates the order
of the four arms and checks actual dispatch and payload at every step. The
60 requests in a process share state. Independent process pairs provide the
replication for the reported timing effects.
