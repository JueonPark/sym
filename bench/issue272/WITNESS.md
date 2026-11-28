# Capture and execution witness

The witness uses ordinary Torch functions with `torch.compile(fullgraph=True,
dynamic=True)` and `RelocBackend`. It runs two untuned chains under the same
admission policy used in #265, with a 64 KiB source threshold and one prior
occurrence per specialization. It compares the default device row policy and
a forced CPU row policy for the H2D relocation. The caller supplies only the
Torch function and backend configuration. [Witness source](witness.py) ·
[#265 policy](../issue265/POLICY.json)

## Reproduce

Run from the repository root after building the specialized runtime.
These commands show the Python 3.14.7
environment and build paths used in the recorded run with Torch 2.14.0+cu126
and an RTX 4070 Ti SUPER. Choose fresh output paths on subsequent runs.
[Setup instructions](README.md)

```sh
/tmp/sym272-venv/bin/python -m bench.issue272.witness \
  --build /tmp/sym272-build/on --output /tmp/sym272-witness.json

c++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra -Werror \
  -Ilibreloc/include -Ilibreloc/src bench/issue272/stage_probe.cpp \
  -ldl -o /tmp/sym272-stage-probe.so
env LD_PRELOAD=/tmp/sym272-stage-probe.so \
  /tmp/sym272-venv/bin/python -m bench.issue272.witness \
  --build /tmp/sym272-build/on --stage-probe /tmp/sym272-stage-probe.so \
  --output /tmp/sym272-witness-stage-probe.json
```

The frozen outputs contain the captured and rewritten graphs, serialized
plans and their hashes, bound symbols, eligible and excluded rows, per-request
completed dispatch reports, rebind counters, and compiler and runtime hashes.
[Default run](witness.json.gz) · [Untimed probe run](witness-stage-probe.json.gz)

The archived source hash for the interposer predates a formatting pass.
Recompiling the checked-in source with the command above reproduced the
recorded shared library byte for byte (SHA-256
`a5e3d16f63b79d6a8e422d70d5cadf1c22003e9add795f2c2b094562401389e4`).

## Observed execution

Each case and policy used one Dynamo callback and one compiler-generated
symbolic plan. Each run passed 42 bitwise and output-metadata checks against
eager Torch.

| Torch chain | Requests per policy | Group constructions / warm rebinds | Saved Torch calls | Device policy completed rows | CPU policy completed rows |
| --- | ---: | ---: | ---: | --- | --- |
| F32 → F16, rank-four permutation, F16 → F32 | 8 | 2 / 3 | 3 | `cuda_relocate_f32` ×5 | `cpu_reference`, `strided_runs` ×5 |
| Per-channel S8 → F32 dequantization, transpose, F32 → F16 → F32 | 13 | 4 / 4 | 5 | `cuda_dequant_relocate` ×6, `cpu_stages_cuda_stages@1` ×2 | `cpu_reference`, `tiled_transpose` ×8 |

The first use of each specialization and each below-threshold request called
the saved region once and submitted no native group. Prepared requests read
different source values and revisited earlier shapes. The scale and zero-point
tensors kept their storage addresses while values changed. A nonzero
zero-point revision excluded `cuda_dequant_relocate` and selected the split
row. A scale-only revision created another specialization. Restoring earlier
bytes reused the corresponding native group. All earlier outputs held distinct
storage and stayed bitwise exact through later calls and owner retirement.

The GNU Linux x86-64 ELF probe forwards the real `runStageRun` call and counts
successful returns. The CPU policy recorded 14,651 `register_cast_chain` runs
over 161,161 elements at stage range [0, 2), and 18,084
`register_dequant_cast_chain` runs over 569,920 elements at range [0, 3).
These counts cover the full inputs of the completed CPU-policy groups. The
device policy recorded zero successful register-stage runs. The runtime ISA
query reported AVX2/F16C/FMA eligibility. The probe observes execution at the
function-call boundary.

This evidence covers compiler-generated plans and automatic selection of
prebuilt native kernel variants. Performance comparisons use the separate
matched controls. The first invocation records diagnostic timing, and the
interposed invocation omits timings. [Qualification protocol](PROTOCOL.md)
