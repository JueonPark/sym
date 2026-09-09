# Task 1 report: version and record contract

## Delivered

- Added frozen, Torch-free `TensorMetadata`, `TransferRecord`, `GraphRecord`,
  and `Eligibility` plain-value records.
- Added pure eligibility classification with mutation-first categorization,
  CPU/CUDA direction checks, typed-cast and same-device-copy exclusions, and
  qualification of dense, contiguous, zero-offset, nonempty float32/float16/
  int8 blocking transfers observed as `aten._to_copy.default`.
- Added a lazy `check_version()` guard for exact CPython 3.14.7 final,
  regular-GIL `cpython-314-x86_64-linux-gnu`, and Torch 2.14.0 CPU or cu126
  wheels whose runtime metadata agrees with the wheel suffix.
- Added the exact Torch test requirements and focused contract tests.

## TDD evidence

RED 1: with no `reloc_torch` implementation, the contract command failed
during collection with `ModuleNotFoundError: No module named 'reloc_torch'`.

GREEN 1: after the minimal records/version/classifier implementation, the
focused command reported `25 passed`.

RED 2: new dense-root, nonempty-storage, destination, and CPU/CUDA endpoint
cases reported 5 expected failures because invalid metadata was still marked
candidate. GREEN 2 reported `31 passed` after the qualification checks.

RED 3: malformed device indices and aliased outputs reported 3 expected
failures. GREEN 3 reported `34 passed` after enforcing well-formed devices and
fresh outputs.

RED 4: Python prerelease/foreign-ABI cases reported 11 failures when the new
release-level input was absent. GREEN 4 and final verification reported
`36 passed`.

Final command:

```text
PYTHONPATH="$PWD/build/torch-cpu/python:$PWD/libreloc/python" \
  /tmp/sym-torch-cpu/bin/python -m pytest \
  libreloc/python/tests/torch_frontend/test_contract.py -q
.................................... [100%]
36 passed
```

Direct activation checks passed under both actual installed variants:
`2.14.0+cpu / None` and `2.14.0+cu126 / 12.6`. In each process,
`import reloc_torch` left `torch` absent from `sys.modules` until explicit
`check_version()` activation.

## Probe and environment handoff

The controller's fresh 2.14 CPU probes confirmed no dispatch event for no-op
`.to("cpu")`; forced copy and cast both use `aten._to_copy.default`; layout
operations use `aten.view.default`, `aten.transpose.int`, and
`aten.clone.default`. Symbolic `make_fx` additionally produced
`aten.sym_size.int` and `operator.floordiv`. These observations support the
allowlist used here; unknown operators remain excluded even with otherwise
valid metadata.

The qualification environment was CPython 3.14.7 regular GIL, Clang 22.1.3,
SOABI `cpython-314-x86_64-linux-gnu`, pybind11 3.0.4, NumPy 2.5.3, and pytest
9.1.1. The controller independently recorded fresh CPU CTest 10/10 and lit
27/27, and CUDA runtime checks 3/3. Turing runtime qualification remains
pending access to that hardware; this task does not claim it.
