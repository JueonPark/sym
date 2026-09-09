# Task 2 report: eager transfer observation

Implemented the explicitly activated `TransferObserver` and
`observe_transfers()` API. Importing `reloc_torch` remains Torch-free; resolving
the lazy observer export is the activation point and performs the pinned
version check.

The observer records immutable value-only metadata around exactly one
redispatch. Torch schema arguments determine mutable destinations, including
positional `copy_` options and keyword `out`. Layout provenance uses bounded
weak references with object-identity checks, is capped at 16 operations, stops
through unrelated arithmetic, and is cleared conservatively on mutation.
Metadata probing does not read tensor contents, call `.cpu()`/`.item()`, or
synchronize. Unsupported storage/stride metadata degrades to conservative
values so sparse tensors and subclasses retain their original behavior.

`TensorMetadata.requires_grad` is a dispatch-time snapshot. In Torch 2.14 the
mode runs below autograd: a `_to_copy` destination can report `False` inside
dispatch and receive its grad function afterward. The source flag reliably
excludes training transfers, and tests separately prove the returned result
keeps normal autograd behavior. Operator failures add frozen type/message
fields and re-raise the identical exception.
The pure classifier maps every such record to the stable exclusion reason
`operator_failed`; Task 3 can therefore retain failed scenarios without
mistaking them for runnable candidates.

## TDD evidence

Initial RED command:

```text
PYTHONPATH=$PWD/build/torch-cpu/python:$PWD/libreloc/python /tmp/sym-torch-cpu/bin/python -m pytest -q libreloc/python/tests/torch_frontend/test_inventory.py -m 'not gpu'
```

It failed during collection as intended:

```text
ImportError: cannot import name 'observe_transfers' from 'reloc_torch'
1 error in 1.93s
```

A second focused RED test passed a failed `aten.mm.default` record to
`classify` and got `unsupported_operator` instead of the required
`operator_failed` exclusion. The minimal classifier guard made this green.

The first GREEN attempt exercised the implementation and exposed the expected
stacked-mode behavior: 5 passed and 1 failed because the outer mode observes
the operation redispatched by the inner mode. The assertion was corrected to
pin 3 outer records and 1 inner record.

Final CPU command:

```text
PYTHONPATH=$PWD/build/torch-cpu/python:$PWD/libreloc/python /tmp/sym-torch-cpu/bin/python -m pytest -q libreloc/python/tests/torch_frontend
```

Result: `45 passed, 2 skipped, 1 warning in 2.48s`. The warning is Torch's
standard sparse-invariant warning from construction of the real sparse test
input.

Final actual-GPU command:

```text
PYTHONPATH=$PWD/build/torch-cuda/python:$PWD/libreloc/python /tmp/sym-torch-cuda/bin/python -m pytest -q libreloc/python/tests/torch_frontend
```

Result: `47 passed, 1 warning in 3.41s` on the qualified RTX 4070 Ti SUPER.
This covers pageable and pinned H2D, `.to(device)`, `.cuda()`, `.cpu()`, forced
same-device copy, D2H, cross-device `copy_`, parameter and buffer movement, and
`load_state_dict` mutation under the `weight_loading` phase.

Additional validation:

```text
git diff --check
PYTHONPATH=$PWD/build/torch-cpu/python:$PWD/libreloc/python /tmp/sym-torch-cpu/bin/python -m py_compile libreloc/python/reloc_torch/observer.py libreloc/python/reloc_torch/compat.py libreloc/python/reloc_torch/__init__.py libreloc/python/reloc_torch/records.py libreloc/python/tests/torch_frontend/test_inventory.py
```

Both completed with exit code 0.
