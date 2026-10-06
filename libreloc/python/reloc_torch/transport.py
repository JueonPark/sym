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
from threading import Lock

import pyreloc

from . import compat
from .artifact import UnsupportedRecipe
from .runtime import (
    ConcreteDescriptor,
    bind_plan,
    bind_stacked_symbols,
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
    # torch.stack: every input in order (``source`` is ``stack_sources[0]``) and
    # their native views; empty for single-source requests.
    stack_sources: tuple = ()
    stack_views: tuple = ()
    staging: tuple = field(default=(), init=False)
    _snapshot: tuple = field(init=False, repr=False)
    _execution_lock: object = field(default_factory=Lock, init=False, repr=False, compare=False)

    def _inputs(self):
        return self.stack_sources or (self.source,)

    def __post_init__(self):
        self._snapshot = tuple(compat.storage_snapshot(tensor) for tensor in self._inputs())

    def recheck(self):
        if tuple(compat.storage_snapshot(tensor) for tensor in self._inputs()) != self._snapshot:
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


def prepare_stacked_transfer(compiled, sources, device, *, non_blocking=False):
    """prepare_transfer for a stacked recipe (Recipe.stack_inputs > 0): every
    host source is validated as one input of the logical [N, *S] source.
    Host-to-device only; allocates and launches nothing."""
    import torch

    if non_blocking:
        raise UnsupportedRecipe("nonblocking_unavailable", "blocking transfers only")
    sources = tuple(sources)
    if not sources or len(sources) != compiled.recipe.stack_inputs:
        raise UnsupportedRecipe(
            "plan_mismatch",
            f"stacked recipe expects {compiled.recipe.stack_inputs} inputs, got {len(sources)}",
        )
    device = torch.device(device)
    for source in sources:
        reason = source_reason(source)
        if reason is not None:
            raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
    if (compiled.recipe.direction != "h2d" or device.type != "cuda"
            or any(source.device.type != "cpu" for source in sources)):
        raise UnsupportedRecipe(
            "direction_mismatch", f"stacked h2d recipe cannot move {sources[0].device} -> {device}"
        )
    if not pyreloc.cuda_enabled or not torch.cuda.is_available():
        raise UnsupportedRecipe("cuda_unavailable", "no CUDA-capable runtime")
    bindings = bind_stacked_symbols(compiled, sources)
    bound = bind_plan(compiled, bindings)
    index = device.index if device.index is not None else torch.cuda.current_device()
    target = torch.device("cuda", index)
    views = tuple(_storage_view(source, "host", -1) for source in sources)
    destination = destination_descriptor(compiled, bindings, target)
    try:
        span = pyreloc.validate_stacked_sources(bound, list(views), "h2d")
    except pyreloc.TransferError as error:
        raise UnsupportedRecipe(_code(error), str(error)) from error
    return PreparedTransfer(
        compiled, sources[0], bindings, bound, destination, "h2d", target,
        views[0], span, non_blocking, stack_sources=sources, stack_views=views,
    )


def execute_transfer(request, *, n_buffers=4, n_streams=2, gather_threads=1,
                     gather_pool=None, resources=None, pinning="auto",
                     min_pinned_bytes=None):
    """Allocate a fresh output and complete; optionally reuse explicit resources.

    Omission/None uses a fresh native context for this call. In both cases the
    source and output are strongly owned through cleanup or quarantine.
    """
    from .resources import TransferResources, _transfer_configuration

    _transfer_configuration(resources, {"pinning": pinning, "min_pinned_bytes": min_pinned_bytes})

    if resources is not None and not isinstance(resources, TransferResources):
        raise TypeError("resources must be TransferResources or None")
    native_resources = None if resources is None else resources.native
    import torch

    # Guard preflight/output allocation too: those operations can release the
    # GIL before a native request exists. Distinct requests remain concurrent.
    if not request._execution_lock.acquire(blocking=False):
        raise RuntimeError("transfer request was already executed or is executing")
    try:
        if request.consumed:
            raise RuntimeError("transfer request was already executed")
        request.recheck()
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
            if request.stack_sources:
                native = pyreloc.make_stacked_transfer(
                    request.bound, list(request.stack_views), dst_view, request.direction
                )
            else:
                native = pyreloc.make_transfer(request.bound, request.source_view, dst_view, request.direction)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"transfer request rejected at execution: {error}") from error
        request.request = native
        # Consumed only once work can be launched: an allocation or validation
        # failure above leaves the request reusable, the native request itself
        # is single-use, and the source/destination owners stay referenced
        # through the blocking call.
        request.consumed = True
        try:
            pyreloc.execute_transfer(
                native,
                caller_stream=compat.cuda_stream_handle(cuda_device),
                n_buffers=n_buffers,
                n_streams=n_streams,
                gather_threads=gather_threads,
                gather_pool=gather_pool,
                resources=native_resources,
                owners=(request.stack_sources or request.source, out),
                pinning=pinning, min_pinned_bytes=min_pinned_bytes,
            )
        except pyreloc.TransferError as error:
            raise RuntimeError(f"{request.direction} transfer failed: {error}") from error
        finally:
            request.staging = tuple(getattr(native, "staging", ()))
        return out
    finally:
        request._execution_lock.release()


__all__ = (
    "CAPABILITY_IDENTITY", "PreparedTransfer", "execute_transfer", "prepare_stacked_transfer",
    "prepare_transfer",
)
