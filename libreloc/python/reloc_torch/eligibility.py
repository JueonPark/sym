"""Pure, reason-coded eligibility for observed transfer records."""

from .records import Eligibility, GraphRecord, TensorMetadata, TransferRecord


_TRANSFER_OPERATORS = frozenset({"aten._to_copy.default"})
_VIEW_OPERATORS = frozenset(
    {
        "aten.view.default",
        "aten.reshape.default",
        "aten.transpose.int",
        "aten.permute.default",
        "aten.as_strided.default",
    }
)
_SUPPORTED_DEVICES = frozenset({"cpu", "cuda"})
_SUPPORTED_DTYPES = frozenset({"float32", "float16", "int8"})


def _category(record: TransferRecord) -> str:
    if record.mutates:
        return "mutation"
    source = record.source
    destination = record.destination
    if (source.device_type, source.device_index) != (
        destination.device_type,
        destination.device_index,
    ):
        return "transfer"
    if source.dtype != destination.dtype:
        return "cast"
    if record.operator in _VIEW_OPERATORS:
        return "layout"
    if record.operator == "aten._to_copy.default":
        return "same_device_copy"
    return "other"


def _metadata_reason(metadata: TensorMetadata) -> str | None:
    if metadata.requires_grad:
        return "requires_grad"
    if metadata.device_type not in _SUPPORTED_DEVICES:
        return "unsupported_device"
    if (
        metadata.device_type == "cpu" and metadata.device_index is not None
    ) or (
        metadata.device_type == "cuda"
        and (metadata.device_index is None or metadata.device_index < 0)
    ):
        return "unsupported_device"
    if metadata.dtype not in _SUPPORTED_DTYPES:
        return "unsupported_dtype"
    if metadata.layout != "strided":
        return "unsupported_layout"
    if not metadata.shape:
        return "unsupported_rank"
    if any(dimension == 0 for dimension in metadata.shape):
        return "empty_tensor"
    if metadata.storage_capacity_bytes == 0:
        return "empty_tensor"
    if metadata.storage_offset != 0:
        return "storage_offset"
    if metadata.is_subclass:
        return "tensor_subclass"
    if not _is_dense_contiguous(metadata):
        return "unsupported_layout"
    return None


def _is_dense_contiguous(metadata: TensorMetadata) -> bool:
    if len(metadata.shape) != len(metadata.strides):
        return False
    expected_stride = 1
    for size, stride in zip(reversed(metadata.shape), reversed(metadata.strides)):
        if not isinstance(size, int) or not isinstance(stride, int) or size < 0:
            return False
        if size != 1 and stride != expected_stride:
            return False
        expected_stride *= size
    return True


def classify(record: TransferRecord) -> Eligibility:
    """Classify one observation without importing or consulting Torch."""
    if isinstance(record, GraphRecord):
        if record.transfer is None:
            return Eligibility("other", False, record.metadata_reason or "unsupported_operator")
        return classify(record.transfer)
    category = _category(record)
    if record.failure_type is not None:
        return Eligibility(category, False, "operator_failed")
    if category == "mutation":
        return Eligibility(category, False, "mutation")
    if category == "cast":
        return Eligibility(category, False, "typed_transform_unavailable")
    if category == "same_device_copy":
        return Eligibility(category, False, "same_device_copy")
    if category == "layout":
        return Eligibility(category, False, "layout_only")
    if category == "other":
        return Eligibility(category, False, "unsupported_operator")

    for metadata in (record.source, record.destination):
        reason = _metadata_reason(metadata)
        if reason is not None:
            return Eligibility(category, False, reason)
    if {record.source.device_type, record.destination.device_type} != {"cpu", "cuda"}:
        return Eligibility(category, False, "unsupported_device")
    if record.non_blocking:
        return Eligibility(category, False, "nonblocking_unavailable")
    if record.aliases_source:
        return Eligibility(category, False, "unsupported_layout")
    if record.source.dtype != record.destination.dtype:
        return Eligibility(category, False, "typed_transform_unavailable")
    if record.operator not in _TRANSFER_OPERATORS:
        return Eligibility(category, False, "unsupported_operator")
    return Eligibility(category, True, "needs_compile_and_runtime_check")
