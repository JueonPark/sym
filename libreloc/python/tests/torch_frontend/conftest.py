import dataclasses
import os
from pathlib import Path

import pytest

from reloc_torch.records import TensorMetadata, TransferRecord


@pytest.fixture
def compiler():
    from reloc_torch import CompilerClient

    configured = os.environ.get("SYM_RELOC_EXPORT")
    if configured is None:
        sym_opt = os.environ.get("SYM_OPT")
        if sym_opt is None:
            pytest.fail("SYM_RELOC_EXPORT or SYM_OPT must select the R1 exporter")
        configured = str(Path(sym_opt).with_name("sym-reloc-export"))
    exporter = Path(configured)
    if not exporter.is_file():
        pytest.fail(f"configured R1 exporter is absent: {exporter}")
    return CompilerClient(exporter)


@pytest.fixture
def split_transpose_recipe():
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, FloorDiv, Symbol, dense_strides

    n = Symbol("s0")
    split = (FloorDiv(n, 64), Const(64))
    destination_shape = (Const(64), FloorDiv(n, 64))
    return Recipe(
        TensorSpec((n,), (Const(1),), Const(0), "float32"),
        (Reshape(split), Transpose((1, 0))),
        TensorSpec(
            destination_shape,
            dense_strides(destination_shape),
            Const(0),
            "float32",
        ),
        "h2d",
    )


@pytest.fixture
def cuda_device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def h2d_record():
    source = TensorMetadata(
        shape=(4, 6),
        strides=(6, 1),
        storage_offset=0,
        dtype="float32",
        device_type="cpu",
        device_index=None,
        requires_grad=False,
        layout="strided",
        pinned=False,
        is_subclass=False,
        storage_capacity_bytes=96,
    )
    destination = dataclasses.replace(
        source,
        device_type="cuda",
        device_index=0,
    )
    return TransferRecord(
        operator="aten._to_copy.default",
        phase="input",
        source=source,
        destination=destination,
        non_blocking=False,
        mutates=False,
        aliases_source=False,
        layout_history=(),
    )
