"""R2 tensor, stream and lifetime adapter (issue #146).

``prepare_transfer`` validates a compiled layout recipe against a real source
tensor without allocating or launching: frontend metadata guards, exact symbol
binding, the standalone binder, the destination descriptor, storage identity
(allocation base, capacity and offset from the tensor's storage, never from
``data_ptr()``/``numel()``), CUDA device ownership through pointer attributes,
and the native span/plan-fit proof. ``execute_transfer`` allocates the fresh
dense destination on the caller's device, orders the runtime's private streams
after the caller's current CUDA stream, runs the blocking forward transfer, and
returns the destination once this request's work has completed.

Expected exclusions raise ``UnsupportedRecipe`` (T3 runs the original region
once). Failures after the launch decision are errors, never fallback signals.
Requests are single-use and recheck the source immediately before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyreloc

from . import compat
from .artifact import UnsupportedRecipe
from .runtime import (
    ConcreteDescriptor,
    bind_plan,
    bind_symbols,
    destination_descriptor,
    source_reason,
)


CAPABILITY_IDENTITY = f"blocking-v1/{'cuda' if pyreloc.cuda_enabled else 'host'}"

_DIRECTIONS = {"h2d": ("cpu", "cuda"), "d2h": ("cuda", "cpu")}


def _code(error):
    return str(error).split(":", 1)[0].strip() or "invalid_view"


def _storage_view(tensor, kind, device):
    """Describe `tensor`'s allocation and logical view for the native validator."""
    base, capacity, offset = compat.storage_span(tensor)
    return pyreloc.BufferView(
        base=base,
        capacity_bytes=capacity,
        offset_bytes=offset,
        extents=[int(d) for d in tensor.shape],
        strides=[int(s) for s in tensor.stride()],
        element_size=tensor.element_size(),
        kind=kind,
        device=device,
    )


@dataclass
class PreparedTransfer:
    """A validated invocation retained until completion; contains no original callable."""

    compiled: object
    source: object
    bindings: dict
    bound: object
    destination: ConcreteDescriptor
    direction: str
    device: object
    source_view: object
    source_span_bytes: int
    non_blocking: bool = False
    request: object = None
    consumed: bool = False
    _snapshot: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._snapshot = compat.storage_snapshot(self.source)

    def recheck(self):
        if compat.storage_snapshot(self.source) != self._snapshot:
            raise RuntimeError(
                "stale transfer request: source storage or metadata changed after preflight"
            )


def prepare_transfer(compiled, source, device, *, non_blocking=False):
    import torch

    if non_blocking:
        raise UnsupportedRecipe("nonblocking_unavailable", "blocking transfers only")
    device = torch.device(device)
    reason = source_reason(source)
    if reason is not None:
        raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
    direction = compiled.recipe.direction
    expected = _DIRECTIONS.get(direction)
    if expected is None or (source.device.type, device.type) != expected:
        raise UnsupportedRecipe(
            "direction_mismatch",
            f"{direction} recipe cannot move {source.device} -> {device}",
        )
    bindings = bind_symbols(compiled, source)
    bound = bind_plan(compiled, bindings)
    if direction == "h2d":
        if not pyreloc.cuda_enabled or not torch.cuda.is_available():
            raise UnsupportedRecipe("cuda_unavailable", "no CUDA-capable runtime")
        index = device.index if device.index is not None else torch.cuda.current_device()
        target = torch.device("cuda", index)
        view = _storage_view(source, "host", -1)
    else:
        if not pyreloc.cuda_enabled:
            raise UnsupportedRecipe("cuda_unavailable", "pyreloc built without CUDA")
        target = torch.device("cpu")
        index = source.device.index
        base, _, _ = compat.storage_span(source)
        try:
            owner = pyreloc.cuda_pointer_device(base)
        except pyreloc.TransferError as error:
            raise UnsupportedRecipe("device_mismatch", str(error)) from error
        if owner != index:
            raise UnsupportedRecipe(
                "device_mismatch",
                f"source storage belongs to cuda:{owner}, tensor declares cuda:{index}",
            )
        view = _storage_view(source, "cuda", index)
    destination = destination_descriptor(compiled, bindings, target)
    try:
        span = pyreloc.validate_transfer_source(bound, view, direction)
    except pyreloc.TransferError as error:
        raise UnsupportedRecipe(_code(error), str(error)) from error
    return PreparedTransfer(
        compiled, source, bindings, bound, destination, direction, target,
        view, span, non_blocking,
    )


def execute_transfer(request, *, n_buffers=4, n_streams=2, gather_threads=1, gather_pool=None):
    """Allocate the destination, order after the caller stream, run and complete."""
    import torch

    if request.consumed:
        raise RuntimeError("transfer request was already executed")
    request.recheck()
    request.consumed = True
    destination = request.destination
    out = torch.empty_strided(
        destination.shape, destination.strides,
        dtype=getattr(torch, destination.dtype), device=destination.device,
    )
    if request.direction == "h2d":
        dst_view = _storage_view(out, "cuda", out.device.index)
        cuda_device = out.device
    else:
        dst_view = _storage_view(out, "host", -1)
        cuda_device = request.source.device
    try:
        native = pyreloc.make_transfer(request.bound, request.source_view, dst_view, request.direction)
    except pyreloc.TransferError as error:
        raise RuntimeError(f"transfer request rejected at execution: {error}") from error
    request.request = native
    try:
        pyreloc.execute_transfer(
            native,
            caller_stream=compat.cuda_stream_handle(cuda_device),
            n_buffers=n_buffers,
            n_streams=n_streams,
            gather_threads=gather_threads,
            gather_pool=gather_pool,
        )
    except pyreloc.TransferError as error:
        raise RuntimeError(f"{request.direction} transfer failed: {error}") from error
    finally:
        # The source and destination owners stay referenced through the call
        # above; nothing is released before this request's work completed.
        del native
    return out


__all__ = ("CAPABILITY_IDENTITY", "PreparedTransfer", "execute_transfer", "prepare_transfer")
