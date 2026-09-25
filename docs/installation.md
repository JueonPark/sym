# Installing Sym

Sym's wheel contains the compiler tools, native runtime, and Python APIs together.
An installed wheel finds its own compiler and runtime libraries; you do not need
`PYTHONPATH`, `SYM_OPT`, `SYM_RELOC_EXPORT`, or `LD_LIBRARY_PATH` for ordinary use.
Explicit compiler overrides remain available for development.

## Availability and supported environment

Binary release infrastructure is under development. **A qualified downloadable
release and runtime container images have not been published by this change.**
Use the source-wheel route below until a release includes
`installation-manifest.json`. The installer fails clearly if that asset is absent;
it does not compile LLVM or substitute another version.

The initial package target is glibc Linux x86_64, regular-GIL CPython **3.14.7**,
and PyTorch **2.14.0+cpu** or **2.14.0+cu126**. The variants share the same imports
and must be installed into separate environments. NumPy is a runtime dependency;
Torch is optional for the standalone runtime but required for the frontend and
complete installed demo. CUDA source builds require toolkit **12.6.3**; executing
an already built CUDA package requires a compatible NVIDIA driver and supported
GPU, not the development toolkit. Package support does not broaden the
[frontend's qualified environment](torch-integration.md#1-qualified-environments).

## Install a wheel from source today

This route builds Sym once against an existing LLVM/MLIR **21.1.8** installation.
You need Git, CMake 3.20+, Ninja, a C++17 compiler, and uv. If LLVM is absent, use
[the compiler build instructions](getting-started.md#build-the-compiler-and-cpu-runtime)
first. Building LLVM is still necessary on the build machine; it is not needed
by another compatible machine receiving the completed wheel.

Run from the checkout root, replacing the LLVM path:

```bash
uv python install 3.14.7
uv venv --python 3.14.7 build/install-cpu
export SYM_PYTHON="$PWD/build/install-cpu/bin/python"
uv pip sync --python "$SYM_PYTHON" --require-hashes --only-binary=:all: release/locks/cpu.txt
# Add tools only to this source-build environment.
uv pip install --python "$SYM_PYTHON" -r build_tools/release/requirements-build.txt
export MLIR_DIR=/absolute/path/to/llvm/lib/cmake/mlir
CMAKE_BUILD_PARALLEL_LEVEL=2 "$SYM_PYTHON" -m build --wheel --no-isolation
uv pip install --python "$SYM_PYTHON" --no-deps \
  dist/sym_reloc-0.1.0+cpu-cp314-cp314-linux_x86_64.whl
source build/install-cpu/bin/activate
sym-doctor --require-torch
sym-demo --device cpu
```

The wheel above has a local `linux_x86_64` tag. It is tested on its build system;
it is **not** a claim of manylinux portability. Release wheels require separate
[portability and clean-host qualification](releasing.md). Never rename an ordinary
Linux wheel to claim compatibility with older systems.

For the standalone runtime without Torch, install the wheel and its NumPy
dependency in a separate compatible interpreter. `import pyreloc` and
`import sym_reloc` do not import Torch. `sym-doctor` reports optional Torch absence;
`sym-demo` deliberately exercises the complete frontend and requires it.

## Install a published binary release

The following interface is for releases containing the installation manifest.
Download `install.sh` from the chosen
[Sym release](https://github.com/JueonPark/sym/releases), then run:

```bash
bash install.sh --cpu --version 0.1.0
# Or, on a qualified NVIDIA GPU host:
bash install.sh --cuda cu126 --version 0.1.0
```

`0.1.0` is the first proposed binary release, not a statement that it is already
available. Prerequisites are Bash, HTTPS-capable curl, tar, sha256sum, and glibc
Linux x86_64. The installer provisions uv, the exact Python interpreter, locked
Torch dependencies, and the matching Sym wheel. It validates checksums, runs
`sym-doctor` and the numerical demo, then activates the environment. A CUDA
request must pass actual device execution; it never silently installs CPU.

The default prefix is `${XDG_DATA_HOME:-$HOME/.local/share}/sym`. Use
`--prefix "/path/with spaces/sym"` to change it. The installer prints the exact
activation command. With the default prefix and no XDG override:

```bash
source "$HOME/.local/share/sym/current/bin/activate"
sym-doctor
sym-demo --device cpu
```

No system Python, existing project `.venv`, shell startup file, driver, or system
CUDA installation is changed. Omit `--version` to resolve the latest release
manifest once. For repeatable deployments, always select a version explicitly.

### Upgrades and recovery

Reinstalling the same validated release is a no-op. Select another release or
variant with `--upgrade`; the old environment remains under `PREFIX/envs/` and
`PREFIX/previous` records it. Failed validation leaves `PREFIX/current` unchanged
and writes a log under `PREFIX/logs/`. Concurrent installers cannot mutate the
same prefix together.

To roll back on Linux, first inspect `readlink PREFIX/previous`, then atomically
replace `current` with a symlink to that recorded environment and open a fresh
shell. Never rename or move an environment: its scripts contain absolute paths.
There is no automatic pruning or uninstall in this first implementation.

## Native C++ SDK and developer builds

CMake now supports installable native tools, headers and runtime:

```bash
cmake -G Ninja -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$MLIR_DIR" -DSYM_BUILD_TESTS=OFF -DSYM_BUILD_BENCHMARKS=OFF \
  -DSYM_BUILD_PYTHON=OFF
cmake --build build/sdk --target sym-opt sym-reloc-export reloc-run-artifact -j2
cmake --install build/sdk --prefix "$PWD/build/sdk-install"
```

Move the SDK as a unit. C++ applications configure with
`-DCMAKE_PREFIX_PATH=/path/to/sdk`, call `find_package(SymReloc CONFIG REQUIRED)`,
and link `SymReloc::runtime`. The SDK includes sample recipes under
`share/sym/recipes`. Its runtime and artifact consumer remain independent of
Torch/LLVM/MLIR. Compiler tools themselves include compiler code.

The [original getting-started guide](getting-started.md) remains the detailed
source-build and explicit recipe walkthrough. The
[PyTorch integration guide](torch-integration.md) documents developer CPU/CUDA
build environments and overrides. Keep those routes when modifying compiler or
runtime code; use a complete installed wheel for application use.
