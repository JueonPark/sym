import dataclasses

import pytest

from reloc_torch.records import TensorMetadata, TransferRecord


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
