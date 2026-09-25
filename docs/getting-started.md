# Getting started with Sym

Sym describes tensor preparation as a **recipe**: a sequence of layout
operations and, when requested, value conversions. The compiler turns a
supported recipe into a **plan**. The runtime binds its symbolic dimensions
to actual sizes, checks constraints, and executes it on concrete buffers.

For example, a plan can reshape a vector of length `n` into `[n / 64, 64]`
and transpose it to `[64, n / 64]`. The same artifact can handle different
positive multiples of 64. A PyTorch backend can capture these recipes from
your code; the explicit compiler/runtime APIs also work without PyTorch.

## Build the compiler and CPU runtime

Run the commands below from the repository root. This first example needs
no GPU or PyTorch. You need Git, a C++17-capable compiler, CMake 3.20+, Ninja,
Python 3 for the example's file generation, and LLVM/MLIR **21.1.8**.

```bash
git clone https://github.com/JueonPark/sym.git
cd sym
```

If LLVM/MLIR is not installed, the repository helper downloads the pinned
revision, builds and tests MLIR, and builds Sym:

```bash
./build_tools/build_mlir.sh
export SYM_BUILD="$PWD/build/sym"
```

Building LLVM is the expensive part of this route. If you already have the
matching LLVM/MLIR build or installation, configure Sym directly instead.
Replace `/absolute/path/to/llvm` with the directory containing `lib/cmake`
and `bin/llvm-lit` (or set `LLVM_EXTERNAL_LIT` to your installed `lit`):

```bash
export SYM_LLVM=/absolute/path/to/llvm
export SYM_BUILD="$PWD/build/quickstart-cpu"
cmake -G Ninja -S . -B "$SYM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$SYM_LLVM/lib/cmake/mlir" \
  -DLLVM_DIR="$SYM_LLVM/lib/cmake/llvm" \
  -DLLVM_EXTERNAL_LIT="$SYM_LLVM/bin/llvm-lit" \
  -DRELOC_ENABLE_CUDA=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON
cmake --build "$SYM_BUILD" --target sym-opt sym-reloc-export reloc-run-artifact -j2
```

The direct build above omits Python bindings to keep the first example
independent of a Python environment. [PyTorch setup](#use-with-pytorch)
below builds those bindings separately. Although the repository build uses
LLVM/MLIR, the resulting runtime library and C++ artifact consumer do not
link either dependency.

## Compile and execute a plan on the CPU

The checked-in [split/transpose recipe](../libreloc/examples/recipes/split_transpose.mlir)
implements the vector example above. Export it into a fresh temporary
directory; the exporter refuses to overwrite existing output files.

```bash
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
export SYM_DEMO_DIR="$(mktemp -d)"
"$SYM_RELOC_EXPORT" libreloc/examples/recipes/split_transpose.mlir \
  --output "$SYM_DEMO_DIR/plan.bin" --manifest "$SYM_DEMO_DIR/manifest.json"

python3 - <<'PY'
import os
import struct
from pathlib import Path

root = Path(os.environ["SYM_DEMO_DIR"])
root.joinpath("input.bin").write_bytes(struct.pack("<128f", *range(128)))
PY

"$SYM_BUILD/libreloc/examples/reloc-run-artifact" "$SYM_DEMO_DIR/plan.bin" \
  --symbols s0=128 --input "$SYM_DEMO_DIR/input.bin" \
  --output "$SYM_DEMO_DIR/output.bin" --direction host

python3 - <<'PY'
import os
import struct
from pathlib import Path

root = Path(os.environ["SYM_DEMO_DIR"])
actual = struct.unpack("<128f", root.joinpath("output.bin").read_bytes())
expected = tuple(float(row * 64 + col) for col in range(64) for row in range(2))
assert actual == expected
print("Correct: 128 float32 values reshaped and transposed to [64, 2].")
PY
```

`plan.bin` contains the reusable layout program; `manifest.json` describes
its symbols, logical tensor shapes, constraints, and compiler identity.
`--symbols s0=128` supplies the concrete dimension. The C++ consumer prints
a JSON byte-count report, then the Python check verifies the result using
only the standard library. The input and output files contain raw float32
values in little-endian, row-major order.

To use another valid size, regenerate the input and change the symbol
binding; the plan itself stays the same. The consumer supports `host`,
`h2d` (CPU to GPU), and `d2h` (GPU to CPU), with the latter two requiring a
CUDA-enabled build. Its layout path consumes wire v0 artifacts; typed v1
recipes use the [typed runtime APIs](../libreloc/README.md#typed-plans-c3-issue-143).
See [the exporter contract](reloc-export.md) for writing your own recipes.

## Use with PyTorch

Follow the [environment installation](torch-integration.md#1-qualified-environments)
and [compiler/extension build](torch-integration.md#2-build-the-compiler-tools-and-the-cp314-extension)
instructions. They create separate CPU and CUDA environments and build
directories. The frontend checks for regular-GIL CPython **3.14.7** on
Linux x86_64 and PyTorch **2.14.0**, using `+cpu` or `+cu126` wheels.
The CUDA build also needs the CUDA **12.6** toolkit; a PyTorch wheel alone
does not supply the CUDA compiler.

After building the CUDA variant, select its interpreter, staged Python
packages, and compiler tools in the same shell:

```bash
export SYM_PYTHON=/tmp/sym-torch-cuda/bin/python
export SYM_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
"$SYM_PYTHON" -c 'import torch, pyreloc, reloc_torch; reloc_torch.check_version(); assert torch.cuda.is_available(); assert pyreloc.cuda_enabled'
```

Save the [README's PyTorch example](../README.md#quick-start) as a Python
file and run it with `"$SYM_PYTHON"`. It prepares tensors of three sizes,
returns contiguous CUDA outputs of shapes `[64, 2]`, `[64, 3]`, and
`[64, 4]`, and prints the backend's execution and fallback counters.

Or run the supplied examples, which compare their outputs against PyTorch:

```bash
"$SYM_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py \
  --direction h2d --sizes 128 192 256
"$SYM_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py \
  --direction d2h --sizes 128 192 256
"$SYM_PYTHON" libreloc/python/examples/torch_weight_loading.py --device cuda:0
```

`RelocBackend` handles supported regions inside `torch.compile`.
`eager_transfers` provides an explicit scope for eligible eager copies, and
`prepare_weights` manages reusable inference weight layouts. See
[activation and ownership](torch-integration.md#4-activation) before
integrating these into an application. Transfers complete before returning;
unsupported regions use the original PyTorch operations. Inspect
`backend.stats()` to distinguish runtime execution from fallback.

## Try casts, quantization, and dequantization

The [typed example](../libreloc/python/examples/torch_typed_relocation.py)
compiles explicit recipes, supplies scale/zero-point parameters, reuses
artifacts at different sizes, and checks results against an independent
reference. It reports which implementation ran and the source, wire, and
destination byte counts.

For a CPU run, use the CPU environment and binding build from the same
installation guide, then select them before running:

```bash
export SYM_PYTHON=/tmp/sym-torch-cpu/bin/python
export SYM_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
"$SYM_PYTHON" libreloc/python/examples/torch_typed_relocation.py
```

The default example executes on the CPU through `pyreloc` and NumPy,
without importing Torch. With the CUDA environment selected instead, add
`--cuda` to compare the forced `original_cpu` pipeline with `auto` dispatch.
Automatic dispatch only selects implemented paths with the same requested
semantics; it does not introduce quantization.

Explicit recipes support more operations than automatic PyTorch capture.
In particular, PyTorch quantize operators retain their original behavior
because their numerical semantics differ from Sym's quantization contract.
Device-resident quantization parameters and typed weight preparation remain
excluded. Consult the [typed support matrix](typed-relocation-support.md)
and [dispatch guide](runtime-dispatch.md) for the exact combinations.

## Troubleshooting and further reading

- **Cannot import `_pyreloc`:** use the interpreter used to build the
  extension, and put that build's `python` directory first on `PYTHONPATH`.
  The CPU command-line-only build above intentionally has no extension.
- **Unsupported Python/Torch version:** use the qualified versions and wheel
  variants above; `+cu130` and other builds are not accepted by the frontend.
- **No GPU execution:** check both `torch.cuda.is_available()` and
  `pyreloc.cuda_enabled`, then inspect backend exclusions/fallbacks.
- **Exporter refuses an output path:** use a new directory or new filenames.
- **Unexpected performance:** the integration is functionally validated;
  [published measurements](runtime-integration.md#5-evidence) show slower
  end-to-end transfers than PyTorch in the measured configuration.

For complete tests and recorded results, use the
[integration reproduction guide](runtime-integration.md#6-reproduction).
For the underlying compiler APIs, start with [symbolic shapes](symbolic-shapes.md).
