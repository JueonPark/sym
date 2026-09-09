# Torch transfer inventory (T1, issue #134)

T1 observes eager dispatch and inventories FX graphs. **No transfer replacement
is enabled.** A candidate remains excluded with `needs_compile_and_runtime_check`
until the [T2 importer](torch-finalization/t2-fx-import-and-guards.md) and
[T3 runtime replacement](torch-finalization/t3-custom-op-and-replacement.md)
prove that particular path. These are future execution tests, not current passes.

Observed on 2026-09-09: regular-GIL CPython 3.14.7,
`cpython-314-x86_64-linux-gnu`, PyTorch 2.14.0+cpu and 2.14.0+cu126
(CUDA 12.6), NumPy 2.5.3, pytest 9.1.1, pybind11 3.0.4. Each environment
uses its separately built cp314 extension. Exact resolved versions, extension
paths, records, exclusions and named scenario outcomes are in the compact
[CPU evidence](torch-evidence/cpu.json) and [CUDA evidence](torch-evidence/cuda.json).

The CUDA host is an NVIDIA GeForce RTX 4070 Ti SUPER, capability 8.9,
driver 595.79, NVCC 12.6.85. Its cu126 wheel lists sm_50, sm_60, sm_70,
sm_75, sm_80, sm_86, sm_90; allocation, kernel and CPU round-trip succeeded
on actual Ada hardware. A literal sm_89 cubin is not required for compatible
kernel execution. Turing qualification remains pending hardware.

| Python scenario | Raw Dynamo target | ATen observation | Direction / semantics | Candidate and current exclusion |
| --- | --- | --- | --- | --- |
| `.to("cuda")`, `.cuda()` | `to`, `cuda` | `aten._to_copy.default` | H2D | Blocking dense inference float32/float16/int8 only; candidate, `needs_compile_and_runtime_check` |
| `.cpu()` | `cpu` | `aten._to_copy.default` | D2H | Same restrictions; candidate, `needs_compile_and_runtime_check` |
| `copy_` | `copy_` | `aten.copy_.default` | H2D/D2H mutation | No; `mutation` |
| Frozen module parameters and buffers `.to()` | Not captured by this raw recipe | `aten._to_copy.default` | H2D, phase `module_to` | Candidate under same restrictions; `needs_compile_and_runtime_check` |
| `load_state_dict` | Not captured by this raw recipe | `aten.copy_.default` | H2D mutation into resident weights/buffers | No; `mutation` |
| Same-device `.to()` | `to` (when retained) | No eager dispatch for no-op | Identity/alias; no transfer | No; proven identity is `same_device_noop` |
| `.to(copy=True)` | Not captured by this raw recipe | `aten._to_copy.default` | Same-device allocation | No; `same_device_copy` |
| `.to(float16)` | `to` | `aten._to_copy.default` | Same-device cast | No; `typed_transform_unavailable` |
| `reshape`, `transpose` | `reshape`, `transpose` | `aten.view.default`, `aten.transpose.int` | Metadata-only views in tested recipe | No; `layout_only` |
| `contiguous` | `contiguous` | `aten.clone.default` | Materializes tested transposed view | No; `unsupported_operator` |
| Constant pad | `torch._C._nn.pad` | `aten.constant_pad_nd.default` | Padding recipe | No; compiler/runtime evidence absent |
| Symbolic size and division | `operator.floordiv` with shape input | `aten.sym_size.int`, `operator.floordiv` | Shape expressions | No; not a transfer |
| Intervening `add_` | `add_` | `aten.add_.Tensor` | Mutation; provenance cleared | No; `mutation` |
| Nonblocking `.to()` | `to` | `aten._to_copy.default` | H2D/D2H | No; `nonblocking_unavailable` |

Current tests: [raw Dynamo / symbolic ATen / real CUDA graph tests](../libreloc/python/tests/torch_frontend/test_graph_inventory.py),
[eager CPU/CUDA observations](../libreloc/python/tests/torch_frontend/test_inventory.py),
[pure eligibility contract](../libreloc/python/tests/torch_frontend/test_contract.py),
and [CLI accounting/failure tests](../libreloc/python/tests/torch_frontend/test_inventory_cli.py).
T2 must separately test normalization, symbolic guards, constant pad and multiple
users; T3 must prove stream, lifetime, alias and mutation correctness before any
execution row becomes supported.

Metadata snapshots contain shapes, strides, offset, dtype, device index, pinning,
layout, subclass status and capacity. Symbolic expressions are strings, never
forced to integers. Actual FX FakeTensor is accepted as metadata, without retaining
Torch objects in records. Absent tensor metadata is `metadata_unavailable`.
Graph inventory is read-only, never executes a graph, and uses the same pure
`classify` function as eager observations. Raw targets remain unchanged; a
boundary description is not authorization to rewrite them.

Eager `requires_grad` is captured at dispatch time below autograd, not from a
post-autograd output. A source requiring gradients excludes training. Nonzero
offsets, empty tensors, unsupported rank/dtype/device/layout, user subclasses,
nonblocking requests and mutation retain explicit exclusion reasons. Nearby layout
history is conservative and clears on mutation or unrelated arithmetic.

Reproduce from the repository root after installing
[requirements](../libreloc/python/requirements-torch-test.txt), selecting the CPU
index or cu126 index explicitly. Use current uv (qualified with 0.12.11) to install
Python 3.14.7; older catalogs may not contain this release. Configure each fresh
build with its exact `Python_EXECUTABLE`, `PYBIND11_FINDPYTHON=ON` and environment's
`pybind11_DIR`; CUDA additionally selects the CUDA 12.6 toolkit and
`RELOC_ENABLE_CUDA=ON`.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m 'not gpu' -q
"$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cpu --output /tmp/inventory-cpu.json
export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m gpu -q
PATH=/tmp/sym-cuda-toolkit-12.6.3/bin:$PATH "$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cuda --output /tmp/inventory-cuda.json
```

The CLI exits nonzero on scenario failure or unavailable requested CUDA. CPU
runs explicitly mark `.cuda()` as not requested. Every scenario reports event
and cross-device transfer counts, including no-ops. Optional observation also
works from a CMake build configured with pybind11 disabled; importing
`reloc_torch` remains Torch-free until an observation entry point is called.
The legacy runtime CI job keeps Torch absent, while the sibling CPU job installs
3.14.7 and builds its own ABI-specific artifacts.
