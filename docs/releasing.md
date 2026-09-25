# Building and qualifying installation candidates

The installation work is staged according to the
[end-to-end implementation plan](superpowers/plans/2026-09-25-end-to-end-installation.md).
A successful local wheel is a development candidate, not a qualified public
release. This change adds the product layout, native SDK, installed commands,
locked dependencies, bootstrap, and transactional environment installer.

## Local candidate construction

Use a regular-GIL CPython 3.14.7 environment with
`build_tools/release/requirements-build.txt` installed, and the pinned LLVM/MLIR
build. From the repository root:

```bash
python -m build_tools.release.build_artifacts --version 0.1.0 --variant cpu \
  --mlir-dir /absolute/path/to/llvm/lib/cmake/mlir --output build/candidate-cpu
```

The output directory must be empty. The builder uses a private source snapshot,
generates variant metadata there, builds a wheel and native SDK, and adds the
standalone installer helper, bootstrap and exact runtime lock. `candidate.json`
records actual asset sizes/checksums and whether the source checkout was dirty.
Candidate metadata explicitly records `qualified: false`. It must not be renamed
to an installation manifest or treated as publication evidence.

The corresponding `--variant cu126` build requires CUDA 12.6.3. Its native
runtime dependencies must be audited/bundled and tested before distribution;
the local builder alone does not perform this CUDA dependency repair. It must
not be advertised as a working release until those gates pass.

## Dependency inputs

`release/compatibility.toml` records the exact qualified baseline.
`release/locks/*.in` selects the official Torch wheel URL and NumPy version;
`*.txt` contains resolved transitive versions and hashes. Refresh in the qualified
Linux cp314 interpreter, then review and test both variants:

```bash
uv pip compile release/locks/cpu.in --python 3.14.7 --generate-hashes \
  --no-emit-index-url -o release/locks/cpu.txt
uv pip compile release/locks/cu126.in --python 3.14.7 --generate-hashes \
  --no-emit-index-url -o release/locks/cu126.txt
```

Build tools are separately pinned. `release/bootstrap-tools.json` and the
standalone `install.sh` embed the same uv archive/version/hash; update them
together and verify the checksum against the official upstream release.
Do not change the Python/Torch baseline only to make dependency resolution pass.

## Required release gates still to complete

- Build against the declared manylinux 2.28 floor using a digest-pinned builder,
  audit and repair every wheel ELF and SDK dependency, and include dependency
  licenses. The current Ubuntu-based LLVM image cannot establish that floor.
- Install the repaired CPU artifact on clean Ubuntu 22.04/24.04 and at the libc
  floor, with source/build trees unavailable and no development toolchain.
- Bundle the redistributable CUDA runtime, qualify both Python import orders,
  and execute the full GPU conformance suite, transfers, streams, and typed GPU
  dispatch on actual supported hardware. A CPU fallback is not GPU evidence.
- Build complete CPU/CUDA runtime containers from the exact validated artifacts,
  record image digests, and run them without a checkout or network.
- Generate a schema-1 installation manifest with exact immutable release URLs,
  sizes and hashes for each variant's wheel, SDK, dependency lock and zipapp
  helper. Validate it using `python -m build_tools.release.make_manifest FILE`.
- Test the downloaded `install.sh` on a host with no system Python. Exercise
  upgrades, failure recovery and explicit prior-release selection.
- Stage the same tested bytes, verify anonymous asset downloads, and obtain
  release authorization before publishing or updating the default channel.

The implementation plan tracks these as Tasks 6 and 8–10. Local checks do not
replace the missing container/hardware evidence. No workflow in this change
publishes packages or marks these gates passed automatically.

## Local verification

Use the installed wheel interpreter and a built native configuration:

```bash
SYM_INSTALLED_PYTHON=/absolute/path/to/venv/bin/python \
SYM_PACKAGING_BUILD=/absolute/path/to/native-build \
python -m pytest tests/packaging -q
```

Unit checks cover manifest admission, executable override precedence, shell
argument boundaries, bootstrap CLI rejection, and transactional installer
failure behavior. The two environment variables enable integration checks:
relocate the SDK, link an external consumer, import the installed wheel outside
the checkout, and run installed diagnostics/numerical examples. Without these
variables those integration tests skip; that run is not package qualification.
