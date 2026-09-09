"""Torch-free immutable records shared by the optional frontend."""

from dataclasses import dataclass


Dimension = int | str


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[Dimension, ...]
    strides: tuple[Dimension, ...]
    storage_offset: Dimension
    dtype: str
    device_type: str
    device_index: int | None
    requires_grad: bool
    layout: str
    pinned: bool
    is_subclass: bool
    storage_capacity_bytes: int | None


@dataclass(frozen=True)
class TransferRecord:
    operator: str
    phase: str
    source: TensorMetadata
    destination: TensorMetadata
    non_blocking: bool
    mutates: bool
    aliases_source: bool
    layout_history: tuple[str, ...]
    failure_type: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class GraphRecord:
    node_name: str
    node_kind: str
    target: str
    input_nodes: tuple[str, ...]
    user_nodes: tuple[str, ...]
    tensor_metadata: TensorMetadata | None
    metadata_reason: str | None = None
    transfer: TransferRecord | None = None
    alias_semantics: str = "unknown"


@dataclass(frozen=True)
class Eligibility:
    category: str
    candidate: bool
    reason: str
