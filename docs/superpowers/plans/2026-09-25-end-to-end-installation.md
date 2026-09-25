# End-to-end installation implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development`, according to the execution method
> chosen by the maintainer, to implement this plan task by task. Checkboxes
> describe future work; this document does not claim that installers or release
> packages already exist.

**Goal:** Install and verify the complete Sym compiler, runtime, Python frontend,
and matching PyTorch environment with one command, without requiring users to
build LLVM/MLIR or configure compiler/library search paths.

**Architecture:** Publish CPU and CUDA 12.6 binary distributions from the same
source revision. A small installer creates a private, versioned environment,
selects exact release artifacts and dependencies, runs installed-package checks,
and activates the environment only after validation. Complete runtime containers
consume these same artifacts; source builds remain available for development.

**Tech Stack:** CMake, scikit-build-core, pybind11 3.0.4, regular-GIL CPython
3.14.7, PyTorch 2.14.0 CPU/cu126, NumPy, uv, Linux ELF dependency auditing,
GitHub Actions/Releases, and GHCR. Compiler: LLVM/MLIR 21.1.8 at the repository
pin. CUDA builder: toolkit 12.6.3, with the existing `75;89` architecture targets.

**Spec:** The [installation contract below](#installation-contract) formalizes
the user's request and the accepted installation proposal. Preserve the
[current environment contract](../../torch-integration.md#1-qualified-environments),
[export contract](../../reloc-export.md), and
[runtime behavior](../../runtime-integration.md). Functional foundations are
[#131](https://github.com/JueonPark/sym/issues/131),
[#132](https://github.com/JueonPark/sym/issues/132), and
[#133](https://github.com/JueonPark/sym/issues/133); this work packages those
capabilities rather than expanding their numerical or frontend support.

## Implementation status — 2026-09-25

The first implementation provides source-built CPU wheels and native SDKs,
automatic compiler discovery, installed diagnostics/demos, exact dependency
locks, a release manifest validator, candidate construction, and the bootstrap
and transactional installer. Existing installation guides now point to the
[available source-wheel workflow](../../installation.md).

Local evidence: installed CPU examples, relocated SDK, non-GPU Python suite,
native CTest, and compiler lit tests. The whole release plan is **not complete**:
manylinux build/repair and clean-host qualification, CUDA runtime bundling and
actual GPU evidence, runtime containers, and release publication remain pending.
The [release guide](../../releasing.md) lists those gates. The detailed checklists
below remain the full release acceptance contract, including steps not yet met.

## Global constraints

- Keep the runtime and ordinary `pyreloc` imports independent of Torch and
  LLVM/MLIR. Shipping compiler tools beside the runtime does not change that
  linkage contract.
- Qualify Linux x86_64 first, with regular-GIL CPython **3.14.7** and PyTorch
  **2.14.0+cpu** or **2.14.0+cu126**. Do not relax `compat.py` to accommodate a
  convenient installer dependency. Free-threaded Python, other Python/Torch
  releases, macOS, Windows, ARM, and musl are outside this first release.
- Ordinary installation downloads binaries; it must not silently fall back to
  an LLVM, Sym, Torch, or extension source build.
- Use a private installation prefix. Do not modify system Python, an existing
  project `.venv`, shell startup files, drivers, or system CUDA installations.
- Preserve explicit compiler overrides and the existing build-tree workflow.
  No wire/manifest/portable-artifact schema change is needed for packaging.
- A requested CUDA installation either proves actual GPU execution or fails
  with a concrete diagnostic. A CPU fallback or skipped GPU check cannot certify it.
- Installation correctness, not a performance threshold, determines success.
- Build and stage release candidates before publication. Publishing a release
  is a separate authorized release operation, not an effect of opening a PR.

## Review focus

1. A wheel can import successfully while its exporter or shared libraries still
   refer to the build machine. Relocate artifacts and remove build/source paths
   before exercising every shipped executable (Tasks 2, 3, 6).
2. Torch or Sym can be the wrong CPU/CUDA variant despite matching public version
   numbers. Bind variant selection to the manifest, dependency lock, installed
   metadata, and actual execution (Tasks 1, 7, 8).
3. Renaming a finished virtual environment can invalidate absolute shebangs and
   interpreter links. Build it at its final versioned path and atomically change
   only the active symlink (Task 7).
4. A failed/repeated/concurrent install can damage an existing environment.
   Test interruption, locks, ownership, and activation failure (Task 7).
5. Checkout-relative examples can hide missing package contents. Validate from
   an unrelated working directory with no checkout, build tree, or exported
   search-path variables (Tasks 5, 6, 8, 9).

## Starting evidence

Audit baseline: `8e96a3f` (documentation link cleanup), with functional delivery
from [PR #158](https://github.com/JueonPark/sym/pull/158) and onboarding from
[PR #159](https://github.com/JueonPark/sym/pull/159).

| Current surface | Packaging consequence |
| --- | --- |
| Root `CMakeLists.txt` always discovers MLIR and adds benchmarks/tests | Release configuration needs explicit build options and install components |
| `libreloc/python/CMakeLists.txt` copies Python files into `build/python`; no wheel metadata/install rules | Add an installed layout and a wheel build without breaking the staged development layout |
| `CompilerClient.from_environment()` requires `SYM_RELOC_EXPORT` or `SYM_OPT` | Add bundled-tool discovery after explicit overrides |
| `compat.py` checks exact Python patch, ABI, Torch version and wheel suffix | The installer must provision that combination, including the regular-GIL interpreter |
| `torch_typed_relocation.py` imports an oracle under `libreloc/python/tests` and locates the repository by parent directories | Installed demonstrations must have packaged resources and independent expected results |
| `build_tools/Dockerfile` and `.github/workflows/docker.yml` produce an LLVM build image | Add distinct runtime images; preserve the existing CI image and tags |
| CUDA runtime links `CUDA::cudart`; CPU SIMD is selected at runtime | Audit CUDA runtime delivery and baseline CPU instructions separately from the driver |

## Installation contract

### User commands

These are target interfaces to implement, not currently available commands:

```bash
# From a checkout, or a downloaded installer from a published release:
./install.sh --cpu
./install.sh --cuda cu126

# Reproducible version and destination; 0.1.0 is the proposed first release.
./install.sh --cpu --version 0.1.0 --prefix "$HOME/.local/share/sym"

# Use the environment selected by the installer.
source "$HOME/.local/share/sym/current/bin/activate"
sym-doctor
sym-demo --device cpu
```

The installer accepts exactly one of `--cpu` or `--cuda cu126`; optional
`--version`, `--prefix`, and `--upgrade` control selection and activation. Default
prefix: `${XDG_DATA_HOME:-$HOME/.local/share}/sym`. Printed activation commands must
use the actual prefix, including spaces. `--version` selects an immutable release;
without it, resolve the latest published installation manifest once and then pin
every subsequent download to the version recorded there.

If the requested installation is already active, validate it and exit successfully
without reinstalling. Replacing a different active release/variant requires
`--upgrade`; preserve the previous environment. Never infer a GPU variant from
whatever Torch happens to be installed on the host.

Initial host prerequisites are Bash, HTTPS-capable curl, tar, SHA-256 tooling,
and glibc Linux x86_64. The bootstrap supplies uv and Python if needed; system
Python, compilers, CMake, NVCC, and LLVM are not end-user prerequisites. GPU use
requires a working compatible NVIDIA driver and supported hardware. Containers
add Docker and NVIDIA Container Toolkit as host prerequisites.

### Distribution and version choices

- Use distribution name `sym-reloc` for GitHub-hosted wheels. Public imports stay
  `pyreloc` and `reloc_torch`; a new Torch-free `sym_reloc` package owns tools,
  build information, diagnostics, and demos. PyPI name reservation/publication is
  deferred and is not needed for release-asset installation.
- Build one wheel per variant, with versions such as `0.1.0+cpu` and
  `0.1.0+cu126`, and regular `cp314-cp314` ABI tags. Both variants own the same
  modules and must not coexist in one environment. Select the exact wheel URL
  from the manifest, not an unconstrained package-index query.
- Build Linux binaries against `manylinux_2_28_x86_64`, then audit their actual
  ABI requirements. Test the resulting package on that baseline and Ubuntu
  22.04/24.04. Do not retag the current Ubuntu-built LLVM image as manylinux:
  dependencies compiled against a newer libc cannot be fixed by changing a tag.
- Keep Torch out of the wheel's mandatory dependencies so standalone runtime
  users retain a Torch-free installation. The end-to-end installer includes it
  through a variant-specific, hashed dependency lock. Include NumPy for demos;
  pin the installed baseline to 2.5.3 in these locks.
- Publish a native SDK archive per variant with compiler tools, the standalone
  runtime, public runtime headers, and `SymRelocConfig.cmake`. C++ consumers use
  `find_package(SymReloc CONFIG REQUIRED)` and `SymReloc::runtime`. Building their
  own C++ program requires a C++ compiler, but running the shipped example does not.
- Keep `sym-opt` and `sym-reloc-export` in both distributions. Include
  `reloc-run-artifact`, sample recipes, and dependency/license notices. Do not
  ship LLVM headers, LLVM test tools, or benchmark binaries in the user wheel.

### Installed wheel layout and discovery

```text
site-packages/
  pyreloc/                         # Python API and _pyreloc.cp314 extension
  reloc_torch/                     # optional Torch integration
  sym_reloc/
    __init__.py                    # no eager Torch import
    tools.py, doctor.py, demo.py
    _version.py, _build_info.json
    _native/bin/                   # sym-opt, sym-reloc-export, reloc-run-artifact
    _native/lib/                   # private native dependencies before repair
    examples/                      # packaged recipes and small demonstrations
```

The wheel repair step may adjust private library names/locations. Final loader
paths and `_build_info.json` must describe the repaired artifact, and every ELF
executable must be tested after repair. Add console entry points `sym-opt`,
`sym-reloc-export`, `reloc-run-artifact`, `sym-doctor`, and `sym-demo`. Native tool
wrappers execute an absolute packaged binary path with the original arguments
and exit status; they do not invoke a shell or recurse through their own PATH name.

Compiler resolution order: explicit `CompilerClient(path)`; otherwise
`SYM_RELOC_EXPORT`; otherwise sibling of `SYM_OPT`; otherwise the packaged native
exporter. An invalid explicit override is an error, not permission to ignore it.
Retain compiler identity in cache/artifact admission as today.

### Release manifest and installation state

Create `release/manifest.schema.json` with schema version 1. Required top-level
fields: `schema_version`, `release_version`, `source_revision`, `python_version`,
`python_abi`, `glibc_min`, `llvm_revision`, `installer_protocol`, and `variants`.
`variants.cpu` and `variants.cu126` each declare:

- full Sym package version, expected Torch version and CUDA runtime value;
- wheel, SDK, dependency-lock and installer-helper assets, each with exact
  filename, HTTPS URL, size and SHA-256;
- CUDA toolkit build version/architectures for cu126, or explicit CPU status;
- validation report assets and runtime container digest when qualified.

Generate hashes and resolved filenames from actual candidate artifacts. Do not
handwrite illustrative digest values into a production manifest. The bootstrap
must reject unknown schemas/protocols, missing variants, invalid hashes/sizes,
wrong platform/ABI, and asset/release version mismatches before activation.
Checksums detect corrupted or mismatched downloads; they are not an independent
authentication mechanism for a compromised release account.

Store each environment at its final absolute path under
`PREFIX/envs/VERSION-VARIANT-UNIQUE_ID/`. Keep managed Python/tool downloads in
prefix-owned directories and reusable download caches separately. A successful
installation writes `installation.json` with artifact digests, interpreter and
dependency versions, source revision, variant, and doctor/demo results, then
atomically replaces `PREFIX/current` with a symlink to that environment. Record
the previous target for manual rollback; never move the virtual environment.

## Delivery order

| Slice | Tasks | Reviewable result |
| --- | --- | --- |
| A — Installable product layout | 1–5 | Relocatable SDK/wheel, automatic discovery, installed diagnostics/demo |
| B — CPU end-to-end installation | 6–7 | Reproducible CPU candidate and one-command clean-host installation |
| C — CUDA qualification | 8 | cu126 candidate proving real transfers and installed typed execution |
| D — Containers and release handoff | 9–10 | Runtime images, staged/promoted releases, concise installation docs |

Tasks are sequential where their interfaces depend on earlier artifacts. CPU
delivery can be reviewed before CUDA hardware qualification, but the full proposal
is complete only after both variants and both runtime images meet acceptance.

## File map

Paths marked create are proposed. Test files live under `tests/packaging/` to
separate distribution checks from numerical/runtime tests.

| Action | Paths | Responsibility |
| --- | --- | --- |
| Create | `release/compatibility.toml`, `release/manifest.schema.json`, `release/locks/cpu.txt`, `release/locks/cu126.txt`, `release/bootstrap-tools.json` | Baseline, artifact schema, hashed runtime locks, pinned bootstrap tools |
| Create | `build_tools/release/requirements-build.txt`, `build_tools/release/prepare_version.py`, `build_tools/release/build_artifacts.py`, `build_tools/release/make_manifest.py`, `build_tools/release/audit_artifacts.py` | Locked build tooling, version generation, candidate construction and auditing |
| Modify/create | `CMakeLists.txt`, `libreloc/CMakeLists.txt`, `libreloc/python/CMakeLists.txt`, `sym/tools/CMakeLists.txt`, `libreloc/examples/CMakeLists.txt`; create `cmake/SymInstall.cmake`, `cmake/SymRelocConfig.cmake.in` | Build options, install components, private loader paths and C++ SDK exports |
| Create | `pyproject.toml`, `libreloc/python/sym_reloc/__init__.py`, `_version.py`, `tools.py`, `doctor.py`, `demo.py` in that package | Wheel definition and installed public commands |
| Create | `libreloc/python/sym_reloc/examples/__init__.py`, `layout.py`, `typed.py`, `weights.py`, `split_transpose.mlir` | Small standalone installed demonstrations |
| Modify | `libreloc/python/reloc_torch/compiler.py`, `libreloc/python/reloc_torch/backend.py`, `libreloc/python/pyreloc/__init__.py`, `libreloc/python/examples/torch_typed_relocation.py` | Discovery, descriptions, and shared example entry points where applicable |
| Create | `install.sh`, `build_tools/install_sym.py` | Portable bootstrap and standard-library installation state machine |
| Create | `build_tools/release/Dockerfile.manylinux`, `build_tools/release/Dockerfile.runtime`, `build_tools/release/qualify_install.py` | Binary builder, complete runtime images, clean-host qualification driver |
| Create | `.github/workflows/packages.yml`, `.github/workflows/release.yml` | Candidate artifacts, validation and explicit publication |
| Create | `tests/packaging/test_release_contract.py`, `test_install_tree.py`, `test_wheel.py`, `test_discovery.py`, `test_installed_commands.py`, `test_installer.py`, `test_containers.py`, `conftest.py` | Focused positive/negative packaging tests and isolated fixtures |
| Create | `tests/packaging/cpp_consumer/CMakeLists.txt`, `main.cpp` | C++ consumer outside the Sym build |
| Modify/create | `README.md`, `docs/getting-started.md`, `docs/torch-integration.md`, `docs/runtime-integration.md`, `libreloc/README.md`; create `docs/installation.md`, `docs/releasing.md` | User installation path and maintainer release procedure |

## Task 1: Freeze release identity, variants, and dependency inputs

**Interfaces:** `release/compatibility.toml` is the installation baseline, checked
against `reloc_torch.compat`. `make_manifest.py` consumes candidate assets and
emits a schema-1 manifest; its Python function
`validate_manifest(data: dict) -> dict` returns validated data or raises
`ValueError`. The installer helper will reuse this contract without importing
Sym or Torch. Keep the validator standard-library-only and package its code with
the release helper rather than expecting the repository on the target machine.

- [ ] Add contract tests for the exact Python, Torch suffixes, NumPy baseline,
  LLVM pin, variant exclusivity, and schema rejection. A fixture called
  `candidate_manifest` creates local dummy assets, computes their real hashes,
  and supplies a complete schema-1 manifest; it must not use fake success hashes.

  ```python
  def test_missing_cuda_variant_is_not_a_cpu_fallback(candidate_manifest):
      from build_tools.install_sym import select_variant
      del candidate_manifest["variants"]["cu126"]
      with pytest.raises(ValueError, match="variant"):
          select_variant(candidate_manifest, "cu126")
  ```

  Add `import pytest` in the test module. Define
  `select_variant(manifest: dict, variant: str) -> dict` in the installer helper
  as a pure validated lookup so this test requires neither network nor installation.
- [ ] Resolve runtime dependencies in fresh qualified environments. Write full
  transitive locks with hashes and exact wheel URLs, including the correct
  `torch==2.14.0+cpu` or `torch==2.14.0+cu126` selection. Source Torch from its
  explicit official variant index; generic dependencies retain their declared
  index. Do not use the development test requirements file as the runtime lock.
- [ ] Keep pybind11, pytest, build frontends, and audit tools in the separately
  pinned `requirements-build.txt`. Pin uv 0.12.11 initially, retrieve its official
  Linux x86_64 asset/checksum, and record those values in `bootstrap-tools.json`.
  Record an exact managed Python build available through that uv version, not
  merely a moving `3.14` request.
- [ ] Define `0.1.0` as the proposed first product version, check existing tags
  before assigning it, and derive `+cpu`/`+cu126` wheel versions automatically.
  Release tags, asset metadata, and source revision must agree. Distribution
  version and wire-format versions are separate fields.
- [ ] Run `python -m pytest tests/packaging/test_release_contract.py -q` first
  with the missing contract, then after implementation. Commit as
  `build: define installation variants and release manifest`.

## Task 2: Install a relocatable native SDK and Python payload

**Interfaces:** Add `SYM_BUILD_TESTS`, `SYM_BUILD_BENCHMARKS`,
`SYM_BUILD_EXAMPLES`, and `SYM_BUILD_PYTHON` options, defaulting to current
development behavior. With tests disabled, do not require `llvm_gtest`, lit,
FileCheck, or test corpus generation. `SYM_BUILD_PYTHON=ON` keeps today's
optional developer behavior; wheel builds additionally require the interpreter
and pybind11 and fail instead of silently omitting `_pyreloc`.

Use install components `Runtime`, `Tools`, `Python`, and `Development`.
`Development` includes public `reloc/` headers and the CMake target export, not
MLIR development headers. Add `SYM_SOURCE_REVISION` as an explicit compiler
build-identity override for release builds from archives; retain git discovery
as the developer fallback.

- [ ] Add a staged-install test that configures without tests/benchmarks, builds
  the requested tools/runtime, installs into a temporary prefix, renames that
  prefix, and executes the exporter and artifact consumer. Initially it should
  fail because no install rules exist.
- [ ] Implement build/install include interfaces and relative ELF loader paths.
  In the SDK use `bin/`, `lib/`, and `include/`; tools locate private libraries
  relative to their executable directory. In the wheel use the private layout
  above; `_pyreloc` must resolve its packaged runtime without `LD_LIBRARY_PATH`.
  Keep development build RPATH behavior working.
- [ ] Audit transitive dependencies of all three executables and the runtime.
  Prefer static LLVM/MLIR linkage for tools where practical; any remaining
  redistributable shared dependencies must be explicitly shipped/resolved.
  Do not copy the build machine's libc or NVIDIA driver into the distribution.
- [ ] Create the independent C++ fixture using only public headers and target:

  ```cmake
  cmake_minimum_required(VERSION 3.20)
  project(sym_sdk_consumer LANGUAGES CXX)
  find_package(SymReloc CONFIG REQUIRED)
  add_executable(consumer main.cpp)
  target_link_libraries(consumer PRIVATE SymReloc::runtime)
  ```

  `main.cpp` loads the exported split/transpose artifact, binds `s0=128`, executes
  through the host runtime, and checks all 128 output values against the nested
  row/column formula from the current quick start. Test both linking and running
  with the original source/build directories unavailable.
- [ ] Run `python -m pytest tests/packaging/test_install_tree.py -q`, existing
  compiler/runtime checks in the normal development build, and the native
  dependency checks on the installed files. Commit as
  `build: add relocatable runtime and compiler installation`.

## Task 3: Produce a wheel containing the complete Python/compiler path

**Interfaces:** `pyproject.toml` uses scikit-build-core with the Task 2 install
components. The distribution installs `pyreloc`, `reloc_torch`, and `sym_reloc`.
Generate `_version.py` and `_build_info.json` from the selected version, variant,
source revision and toolchain. Build information also lists native files and
supported wire versions `[0, 1]`; it is not a new artifact admission mechanism.

- [ ] Add wheel-content and fresh-venv tests asserting the extension, both Python
  APIs, native binaries, recipe resources, and license notices are present.
  Reject a wheel missing its exporter even if `import pyreloc` works.
- [ ] Implement package metadata with NumPy as a dependency and Torch absent
  from mandatory dependencies. Use a dynamic version provider reading generated
  `_version.py`; do not hand-edit two different project versions for the variants.
  Configure wheel builds with `SYM_BUILD_PYTHON=ON`, `SYM_BUILD_EXAMPLES=ON`,
  `SYM_BUILD_TESTS=OFF`, and `SYM_BUILD_BENCHMARKS=OFF`; require all three shipped
  tools explicitly so an optional developer setting cannot omit them.
- [ ] Register console commands with a focused layout such as:

  ```toml
  [project.scripts]
  sym-opt = "sym_reloc.tools:sym_opt_main"
  sym-reloc-export = "sym_reloc.tools:export_main"
  reloc-run-artifact = "sym_reloc.tools:run_artifact_main"
  sym-doctor = "sym_reloc.doctor:main"
  sym-demo = "sym_reloc.demo:main"
  ```

  Task 4 implements tool wrappers; Task 5 implements doctor/demo. The package
  skeleton must keep these imports lazy and must not claim those commands work
  until their tasks pass.
- [ ] Build with `python -m build --wheel`, install the resulting exact wheel
  into a clean interpreter environment, and run
  `python -I -c "import sys, pyreloc; assert 'torch' not in sys.modules"`.
  Verify wheel metadata rejects the wrong Python minor and the frontend still
  rejects unqualified patch versions/ABI variants through its existing check.
- [ ] Ensure package builds do not silently fetch/build LLVM. Source builds
  require a configured developer toolchain with a clear error; the installer
  later requests binary artifacts only.
- [ ] Run `python -m pytest tests/packaging/test_wheel.py -q`. Commit as
  `build: package Sym tools and Python APIs in variant wheels`.

## Task 4: Discover packaged compiler tools automatically

**Interfaces:** `sym_reloc.tools.native_tool(name: str) -> Path` returns a
validated packaged executable for an allowlisted name. Wrapper functions call
`os.execv` with that path and the original argument list. Extend
`CompilerClient.from_environment(environ=None)` with the resolution order in
the contract; retain the existing public method and error behavior for explicit
overrides. Stage `sym_reloc` in development builds as well, while tolerating the
absence of bundled binaries there when an explicit override is supplied.

- [ ] Add tests for packaged discovery, both override variables and their
  precedence, missing/not-executable overrides, executable paths containing
  spaces, and the no-bundle developer error. Define `installed_python` in
  `tests/packaging/conftest.py` as an isolated interpreter with the test wheel
  installed and no source-root path injection.

  ```python
  def test_packaged_discovery_needs_no_exports(installed_python, tmp_path):
      result = subprocess.run(
          [installed_python, "-I", "-c",
           "from reloc_torch import CompilerClient; "
           "c = CompilerClient.from_environment(); "
           "assert c.executable.is_file()"],
          cwd=tmp_path, env=clean_environment(), capture_output=True, text=True,
      )
      assert result.returncode == 0, result.stderr
  ```

  Import `subprocess`; define `clean_environment()` in the packaging test helper
  to preserve necessary OS variables while removing `PYTHONPATH`,
  `LD_LIBRARY_PATH`, `SYM_RELOC_EXPORT`, and `SYM_OPT`.
- [ ] Add packaged discovery only after the explicit paths have been considered.
  Fail clearly if no usable exporter exists. Do not choose an unrelated executable
  from PATH before the bundled matching compiler.
- [ ] Test console wrappers with a compiler exit 2 unsupported recipe, exit 1
  malformed recipe, spaces in file arguments, and an interrupted subprocess.
  Preserve argument boundaries, stderr, and exit codes.
- [ ] Run `test_discovery.py` and existing `test_compiler.py`, including strict
  portable artifact association and Torch-free imports. Commit as
  `feat: discover installed compiler tools without environment variables`.

## Task 5: Ship diagnostics and demonstrations that work after installation

**Interfaces:** `sym-doctor [--require-cuda] [--json]` reports installation/build
identity, compiler availability, wire support, Python/Torch compatibility, native
runtime availability, and requested device readiness. Exit 0 means all required
checks passed; exit 1 means a required check failed. Optional Torch absence is
reported as unavailable for standalone use; the end-to-end installer additionally
requires a compatible Torch result. Do not initialize Torch during ordinary
`sym_reloc`/`pyreloc` imports.

`sym-demo --device cpu|cuda [--json]` executes the installed recipe resources and
reports numerical results, actual runtime counters and exclusions. Exit 0 means
every required scenario executed and matched its reference. Both commands have
JSON schema version 1 and human-readable output by default. Required top-level
fields: `schema_version`, `status`, `variant`, `versions`, and `checks` (named
outcomes with detail); demo also includes per-scenario execution evidence.

- [ ] Add installed-command tests with all compiler/search-path environment
  variables removed, a working directory outside the repository, and no access
  to `libreloc/python/tests`. Initially reject the existing checkout-dependent
  typed example as evidence of an installed workflow.
- [ ] Implement `sym_reloc.examples.layout` using the packaged MLIR fixture:
  export one layout artifact, decode/bind/execute for 128/192/256 inputs, and
  compare with a direct reshape/transpose expression. Save and reload the
  portable artifact in another process as an additional scenario.
- [ ] Implement `sym_reloc.examples.typed` with explicit layout+f32→f16 and
  signed-int8 quantize/dequantize recipes and CPU-owned parameters. Use finite
  deterministic witnesses with analytical expected values, for example:

  ```python
  # C1 symmetric quantization at scale 0.5, with ties-to-even and saturation.
  source = np.array([-100.0, -0.75, -0.25, 0.25, 0.75, 100.0], np.float32)
  expected = np.array([-128, -2, 0, 0, 2, 127], np.int8)
  ```

  This small installation witness does not replace C4's full independent oracle.
  Keep full conformance in the source test suite. Refactor the source example
  into a wrapper/shared demonstration only where needed to remove path hacks;
  do not install the complete test tree as a workaround.
- [ ] Implement `sym_reloc.examples.weights` using a tiny module with one
  parameter and one buffer, `prepare_weights`, reuse, mutation and invalidation.
  CPU mode checks host preparation; CUDA mode checks real H2D delivery. Preserve
  the existing explicit exclusion for typed weight preparation.
- [ ] In CUDA mode run the README's `RelocBackend` input example and forward D2H
  witness on default and non-default streams. Require positive runtime execution
  counts and correct metadata/results. Check typed dispatch through `original_cpu`
  and a specifically qualified GPU-capable row; record actual implementation
  and byte counts. Auto without calibration may legitimately choose CPU, so it
  alone cannot prove CUDA transform kernel delivery.
- [ ] Test missing exporter, broken library loading, wrong Torch suffix, no GPU,
  CPU wheel with `--require-cuda`, and malformed build information. Diagnostics
  must tell users which installed component failed; never recommend rebuilding
  LLVM to fix an incomplete published wheel.
- [ ] Run `python -m pytest tests/packaging/test_installed_commands.py -q`, then
  the installed commands from an unrelated directory. Commit as
  `feat: add installed diagnostics and end-to-end demos`.

## Task 6: Build and qualify CPU release candidates without local toolchains

**Interfaces:** `build_artifacts.py --version VERSION --variant cpu --output DIR`
builds the wheel, native SDK, bootstrap/helper, and release metadata from one
revision. `audit_artifacts.py --artifacts DIR` validates dependencies/content.
`qualify_install.py --manifest PATH --variant cpu --output REPORT` drives clean
environments and emits machine-readable evidence. Artifact construction and
qualification are separate from uploading a public release.

- [ ] Create a pinned manylinux builder with the repository's LLVM revision,
  regular cp314 headers/interpreter, and locked packaging tools. Pin the builder
  image digest and record it. Cache LLVM by source pin, platform, compiler flags,
  and compiler version; do not reuse the current Ubuntu LLVM image for a lower
  glibc promise. Select a host compiler supported by both LLVM and the later
  CUDA 12.6 build; qualify GCC 12 for the shared build recipe instead of blindly
  using the manylinux image's newest compiler with NVCC.
- [ ] Build the CPU wheel and SDK without tests/benchmarks in the install payload.
  Run the ordinary source tests in a separate development configuration. Include
  licenses/notices for the actual shipped native dependencies and an inventory
  of versions/paths; record source and build identity without source-tree paths.
- [ ] Run `auditwheel show` and the required repair operation for the wheel.
  Audit every ELF in wheel and SDK, including standalone executables, using
  `readelf`/dependency traversal. Reject unresolved private dependencies, build
  RPATHs, unsupported libc/libstdc++ symbols, CPU artifacts linked to CUDA, or
  generic code accidentally compiled with `-march=native`. Do not use repair
  exclusions to conceal a missing LLVM or libreloc dependency.
- [ ] Unpack/install into fresh directories, make the original build tree and
  LLVM prefix unavailable, and exercise `sym-opt`, `sym-reloc-export`,
  `reloc-run-artifact`, `sym-doctor`, and `sym-demo --device cpu`. Check the
  repaired wheel, not the pre-repair copy. Confirm an unprivileged user can run it.
- [ ] Run the clean CPU installation on Ubuntu 22.04 and 24.04 and check binary
  compatibility at the declared glibc floor. Confirm no source build commands
  execute during installation. Validate from a directory containing spaces.
- [ ] Create the native SDK archive from the same build and validate the external
  C++ consumer using only the extracted prefix. Verify all transitive libraries
  resolve after moving the SDK; do not depend on Python or Torch for this path.
- [ ] Add `.github/workflows/packages.yml` to build/validate candidates and upload
  workflow artifacts with manifests/reports. Default permissions are read-only;
  PR builds cannot publish releases. A package check must fail if a requested
  binding/tool/demo is absent, not report a skip.
- [ ] Run the packaging test group and clean CPU qualification. Commit as
  `ci: build and validate complete CPU distribution artifacts`.

## Task 7: Implement the one-command environment installer

**Interfaces:** `install.sh` owns portable shell preflight and bootstrapping a
pinned uv/Python. It then invokes a checksum-verified standard-library Python
helper asset. Build that helper as a zipapp containing `build_tools/install_sym.py`,
the manifest validator from `build_tools/release/make_manifest.py`, package
initializers, and a generated `__main__.py`; it must not import the checkout or
an already installed Sym distribution. `build_artifacts.py` creates this asset.

Generate the standalone `install.sh` with the pinned bootstrap-tool metadata
embedded; it must not read `release/bootstrap-tools.json` from a checkout at
runtime. Bootstrap order is shell preflight → verified uv → managed Python →
manifest download → verified helper → full manifest validation → installation.
Before executing the helper, an embedded standard-library Python bootstrap
parses the manifest and checks its schema/protocol, selected release/variant,
helper HTTPS origin, filename, size and SHA-256. The helper then performs the
complete shared validation before downloading/installing product assets. Test
this bootstrap separately so helper verification has no circular dependency on
executing the helper itself.

`install_sym.py` exposes `select_variant`, `validate_platform`, `download_asset`,
`install_environment`, `validate_environment`, and `activate_environment` as
separate functions with explicit paths/data arguments. `main(argv)` implements
the public CLI. Use subprocess argument arrays rather than shell command strings.

- [ ] Write tests with a local fixture download provider and fake subprocess
  runner. In `conftest.py`, `installer` wraps the real state machine with those
  injected providers, allowing a forced validation failure without downloads:

  ```python
  def test_failed_upgrade_keeps_previous_environment(installer, tmp_path):
      prefix = tmp_path / "prefix with spaces"
      previous = installer.seed_managed_environment(prefix, variant="cpu")
      installer.fail_next_validation("typed demo mismatch")
      result = installer.run(prefix=prefix, variant="cu126", upgrade=True)
      assert result.returncode != 0
      assert (prefix / "current").resolve() == previous
      assert previous.joinpath("installation.json").is_file()
  ```

  The fixture creates a real prefix/current symlink and sentinel metadata.
  `fail_next_validation` injects a failing result at the validation boundary;
  production CLI must not expose a switch to skip validation.
- [ ] Shell preflight validates required tools, Linux/x86_64/glibc, writable
  prefix, and CLI conflicts. Download the pinned uv archive over HTTPS into a
  private temporary directory, verify its recorded checksum before extraction,
  and never invoke a downloaded shell script through a pipe. Check archive
  paths against traversal before extraction. Scope caches and managed Python
  installation to documented prefix-owned paths; do not change the host's
  uv configuration or default Python. Use invocation-specific temporary paths
  for bootstrap downloads and uv's own locking for shared interpreter storage;
  the later prefix lock also protects environment creation and activation.
- [ ] Resolve latest or explicit manifest once; validate schema, platform and
  selected variant; download exact assets with size/hash verification and
  bounded retry on transient failures. Fail on hash mismatch instead of retrying
  a different release/variant. Restrict executable payload origins to the
  configured official release/bootstrap hosts and their documented CDN redirects.
- [ ] Provision regular-GIL CPython 3.14.7 through pinned uv, then verify version,
  implementation and SOABI before creating the environment. Use a final absolute
  versioned path from the outset. Configure `UV_PYTHON_INSTALL_DIR` only for the
  installer subprocesses; preserve the managed interpreter needed by its venvs.
- [ ] Install the full variant lock with hash checking and binary-only resolution;
  install the verified Sym wheel by its local exact path with dependencies
  already satisfied. Do not offer a broad `--extra-index-url` that can choose a
  different Torch variant. Verify the installed Torch suffix/CUDA runtime,
  Sym variant and artifact revision agree with the manifest.
- [ ] Hold a prefix-local lock during environment creation/activation. Write a
  candidate ownership marker, run `sym-doctor` and `sym-demo`, write final
  `installation.json`, and only then atomically replace `current`. Remember the
  previous target. If validation fails, remove only files/environment directories
  created by this invocation, leave the active environment intact, and retain a
  readable diagnostic log. An interrupted candidate is never considered complete.
- [ ] Test same-version no-op, different version without `--upgrade`, successful
  upgrade, failed upgrade, two simultaneous installations, disk/write failures,
  interrupted download/install, symlinked or unrelated existing prefix entries,
  wrong ABI/architecture, missing variant, checksum mismatch, spaces, and failed
  post-install GPU validation. Existing `.venv`, system Python and shell profiles
  are sentinel files in these tests and must remain unchanged.
- [ ] Print the exact activation and direct interpreter commands. Document manual
  rollback by selecting the recorded previous environment; keep pruning old
  environments and a global uninstall command outside this initial implementation.
- [ ] Run `bash -n install.sh`, `python -m pytest tests/packaging/test_installer.py -q`,
  then the real CPU installer in the clean qualification environment. Commit as
  `feat: install and validate isolated Sym environments`.

## Task 8: Package and qualify the CUDA 12.6 variant

**Interfaces:** The `cu126` release uses the same installer and Python APIs, with
`RELOC_ENABLE_CUDA=ON`, Torch `2.14.0+cu126`, and the CUDA 12.6 native runtime.
`sym-doctor --require-cuda` and `sym-demo --device cuda` are mandatory before
activating a user-requested CUDA environment.

- [ ] Extend the binary builder with CUDA toolkit 12.6.3 and a supported pinned
  host compiler. Preserve `75;89` targets; do not infer broader GPU support from
  a successful build. Build/recheck LLVM and native binaries against the same
  declared libc floor used by the CPU distribution.
- [ ] Bundle the redistributable CUDA runtime dependencies required by libreloc
  privately with the CUDA wheel/SDK, and record their exact versions/notices.
  Audit the final dependency closure. The NVIDIA driver remains host-provided;
  do not bundle `libcuda` driver stubs or copy the host's driver into artifacts.
  A clean user environment must not need NVCC, CUDA headers, or a toolkit path.
- [ ] Test `_pyreloc` before importing Torch and Torch before `_pyreloc` in fresh
  processes. Both must load and interoperate without search-path exports; do not
  rely on Torch import side effects to find libreloc's CUDA dependencies. Check
  behavior with the separately installed Torch CUDA dependencies present.
- [ ] Use a GPU host with no installed development toolkit in the test container.
  Exercise generated v0/v1 plans, default/non-default stream H2D and forward D2H,
  immediate consumers, typed cast/quantize/dequantize, and repeated weight reuse
  and invalidation through the installed APIs. Test both the Python wheel and
  the SDK's artifact consumer.
- [ ] Run the existing GPU conformance suite against the installed packages in
  a separate qualification stage. The test checkout may be mounted read-only
  there, but must not supply imports, exporter binaries or shared libraries;
  record their resolved installed paths. Also retain the checkout-free smoke
  tests as an independent gate.
- [ ] Require actual GPU test execution; fail qualification if hardware/runtime
  absence skips required cases. Record GPU model/compute capability, driver,
  artifact hashes and all passed/failed/skipped counts. Test both Turing and Ada
  before advertising both as package-qualified; a missing hardware run is an
  explicitly unqualified row, not a fabricated result.
- [ ] Test CPU wheel with a CUDA request, cu130 Torch injected into a CUDA venv,
  absent/incompatible driver, unsupported device, and missing CUDA dependency.
  Refuse activation with diagnostics; never install CPU as an implicit fallback.
- [ ] Run clean cu126 qualification and packaging/GPU tests. Commit as
  `build: distribute and qualify the CUDA 12.6 installation`.

## Task 9: Build complete runtime containers from the same artifacts

**Interfaces:** Add CPU and cu126 images under
`ghcr.io/jueonpark/sym-runtime:VERSION-cpu` and `:VERSION-cu126`. These are
proposed runtime image names, separate from the existing LLVM builder at
`ghcr.io/jueonpark/sym`. Install into `/opt/sym`, put `/opt/sym/current/bin` on PATH,
and provide an unprivileged default user and writable work directory.

- [ ] Write image tests that run `sym-doctor` and `sym-demo` with no source or
  build directory mounted. The runtime image must contain the installed Sym tools,
  runtime, exact Python/Torch environment and resources; it must not build LLVM
  during startup or contain the full LLVM SDK just to make execution work.
- [ ] Build from digest-pinned base images and the exact validated release
  candidate assets/locks. A multi-stage build may prepare the environment, but
  copy it at the same absolute paths, including managed Python; moving only a
  virtual environment breaks its interpreter links. Test this explicitly.
- [ ] The CUDA image requires host GPU access through NVIDIA Container Toolkit.
  Image construction can happen without a GPU, but promotion requires a real
  GPU run. Do not weaken the public CUDA installer's mandatory validation just
  to make a Docker build succeed; use the internal construction stage followed
  by the separate mandatory image validation job.
- [ ] Make the documented commands work without a checkout or volume mount:

  ```bash
  docker run --rm ghcr.io/jueonpark/sym-runtime:0.1.0-cpu sym-demo --device cpu
  docker run --rm --gpus all ghcr.io/jueonpark/sym-runtime:0.1.0-cu126 sym-demo --device cuda
  ```

  Version `0.1.0` here denotes the candidate under test. Preserve ordinary
  command overrides so users can run Python or mount their own application.
- [ ] Run the demos with container networking disabled after the image is built:
  no compiler/package download may be hidden in the first execution. Test a
  non-root UID, read-only installed files, and writable temporary compilation
  space. CUDA without GPU access must report an actionable failure.
- [ ] Verify installed artifact hashes match the native release candidates and
  record final image digests. Preserve the existing build-image workflow/tags.
  Run `python -m pytest tests/packaging/test_containers.py -q` plus actual CPU/GPU
  container qualification. Commit as `build: add complete Sym runtime images`.

## Task 10: Gate publication and make installation the primary entry point

**Interfaces:** `.github/workflows/release.yml` consumes a selected source tag
and the verified candidate outputs from `packages.yml`. It stages a draft release,
checks a completeness matrix, and publishes only after the maintainer's release
approval. `make_manifest.py` emits the final downloadable manifest from the exact
validated assets, and the manifest points to immutable release URLs/digests.

- [ ] Define release readiness as CPU wheel+SDK, CUDA wheel+SDK, exact runtime
  locks, installer/bootstrap/helper, notices, both runtime images, and complete
  validation reports. Build infrastructure errors or missing GPU evidence keep
  the full release unpublished; they cannot be represented as successful checks.
  A CPU-only preview may be explicitly named and documented, but is not completion
  of this plan's two-variant release.
- [ ] Make release jobs consume the same candidate bytes that passed validation.
  Do not rebuild after testing and upload a different wheel under the same name.
  Verify asset hashes, image digests, package/source versions and expected files
  immediately before publishing. Never overwrite assets of an existing release
  with different bytes; issue a new version for corrections.
- [ ] Stage all assets and versioned images before updating any default channel.
  Account for the fact that a GitHub release and GHCR tags are not one atomic
  transaction. Update latest/default discovery only after every referenced asset
  is available, and test the anonymous download path used by the installer.
- [ ] Test the downloaded standalone installer from a directory with no checkout,
  with no system Python, LLVM, CMake or NVCC. Confirm bootstrap, environment
  creation, compiler discovery, import, artifact export, binding, execution and
  demo validation succeed. Repeat with explicit version selection against the
  staged candidate; after a prior product release exists, additionally install
  that older published version to prove selection remains reproducible.
- [ ] Replace README's build-first onboarding with CPU/CUDA installer commands
  and the short PyTorch example. `docs/installation.md` owns prerequisites,
  exact scope, activation, version selection, upgrades/rollback, containers,
  expected diagnostics and source-build escape hatch. Link from existing build
  guides rather than maintaining conflicting installation instructions.
- [ ] Document advanced direct-wheel installation into an already qualified
  environment and native SDK use. Explain that arbitrary `pip install sym` is
  not the product name or a supported variant-selection mechanism. Do not
  advertise PyPI commands before a corresponding distribution exists there.
- [ ] Write `docs/releasing.md` with candidate build/validation commands, lock
  refresh procedure, image/source pin updates, release promotion, failure
  recovery and the evidence checklist. Link every mentioned issue/PR to GitHub.
- [ ] Validate all published commands against the staged assets and check docs
  links/shell snippets. Commit as `docs: make verified binary installation the default`.

## Acceptance matrix

| Scenario | Required evidence |
| --- | --- |
| New Linux CPU user, no Python/compiler toolchain | Installer provisions the exact environment; doctor and CPU layout/typed/weight demos pass |
| New CUDA user, compatible driver, no CUDA toolkit | Installer selects cu126; real H2D/D2H, stream ordering, typed GPU execution and weight scenarios pass |
| Standalone runtime user | Wheel/native SDK loads and executes without importing or linking Torch/MLIR/LLVM into libreloc |
| Packaged compiler | Bundled tools execute after build/source/LLVM directories become unavailable |
| Existing project `.venv` | Sentinel contents unchanged; installation occurs under the requested private prefix |
| Repeated install | Same validated version/variant returns success without reinstalling |
| Failed upgrade or interrupted download | Previous active environment still runs; incomplete candidate never becomes current |
| Concurrent installers | At most one mutation/activation proceeds per prefix; the other reports the lock |
| Wrong ABI/variant/platform or unavailable GPU | Clear failure; no silent rebuild, version relaxation or CPU substitution |
| Installed examples | Run outside a checkout with search-path variables unset and package resources resolved locally |
| CPU/CUDA containers | Run complete demos without source mounts or network; CUDA qualification uses actual hardware |
| Release integrity | Downloaded assets match validated hashes, package identity, source revision and image digests |

The automated installer is complete only when these scenarios pass with release
artifacts. A successful source-tree build or import-only smoke test is insufficient.

## Risks and explicit boundaries

- **Binary size/build time:** compiler tools may carry substantial LLVM/MLIR code.
  Measure wheel/SDK/image/download sizes and cold-install time, strip unnecessary
  symbols in release artifacts, and retain debug symbols separately. Prefer one
  complete initial package over an unqualified split that reintroduces manual
  compiler installation; split distributions only if measured limits require it.
- **ABI portability:** audit against the claimed glibc floor before labeling the
  artifact. If a required dependency cannot meet it, resolve that build dependency
  or explicitly revise the installation contract and qualification matrix; do
  not weaken the audit or silently raise the requirement.
- **Bootstrap availability:** test that pinned uv actually provisions the exact
  regular-GIL Python build accepted by `compat.py`. If a pin changes, regenerate
  locks/metadata and rerun qualification; never select an adjacent Python patch
  because it happens to be downloadable.
- **CUDA host variability:** driver and GPU availability remain host responsibilities.
  Report tested hardware and exclusions; packaging does not broaden runtime
  semantics or make unsupported architectures qualified.
- **Scope:** no public PyPI release, universal Python matrix, offline installer,
  automatic driver installation, global environment replacement, performance
  optimization, or new typed/frontend operations in this first delivery. Native
  binary downloads and images can be cached, but full air-gapped installation is
  a separate feature with its own dependency-mirroring contract.

## Technical references

These sources inform the packaging mechanism; the choices and acceptance gates
above are Sym's proposed design, not claims that those tools automatically solve
the project-specific integration.

- [scikit-build-core CMake integration](https://scikit-build-core.readthedocs.io/en/latest/guide/cmakelists.html)
  describes wheel install destinations and modern Python discovery.
- [manylinux build environments](https://github.com/pypa/manylinux) document
  portable Linux build baselines.
- [auditwheel](https://github.com/pypa/auditwheel) documents native dependency
  inspection/repair and the limits of repairing binaries built against newer libc.
- [uv Python management](https://docs.astral.sh/uv/concepts/python-versions/)
  describes managed interpreters and their installation location.
- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
  describes explicit CPU/CUDA package indexes.
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  documents host GPU-container prerequisites.

## Plan review and execution handoff

This is a plan-only deliverable. Review the installation contract and the ten
task boundaries before implementation. Suggested first implementation PR: Tasks
1–2 (release contract and installable native layout), followed by separate reviews
for wheel/discovery, installed commands, CPU release/installer, CUDA, and final
container/publication work. No elapsed-time estimate substitutes for the explicit
CPU and hardware-dependent CUDA gates.
