"""Explicit, blocking consumer-sized groups over fresh individual preparations.

A group completes all its members together. Split at consumer boundaries for
finer completion granularity; independent groups may use separate owners and
caller streams. No grouping decision crosses a model dependency implicitly.
"""
from contextlib import ExitStack
from dataclasses import dataclass, field
import operator
from threading import Lock
from types import MappingProxyType

import pyreloc

from . import compat
from .dispatch import PreparedTypedTransfer
from .transport import PreparedTransfer, _storage_view


@dataclass
class PreparedTransferGroup:
    requests: tuple
    device: object
    consumed: bool = field(default=False, init=False)
    native: object = field(default=None, init=False, repr=False)
    _execution_lock: object = field(default_factory=Lock, init=False, repr=False)

    @property
    def destinations(self):
        """One immutable logical descriptor per output, in request order."""
        return tuple(request.destination for request in self.requests)


@dataclass(frozen=True)
class GroupResult:
    tensors: tuple
    descriptors: tuple
    report: MappingProxyType


def prepare_transfer_group(requests):
    """Group 1..256 fresh layout/typed preparations on one CUDA device.

    Members may have different shapes, dtypes, parameters and directions.
    Preparation launches nothing. The group owns member sources/parameters;
    execution consumes each member exactly once and creates independent outputs.
    """
    import torch

    requests = tuple(requests)
    if not 1 <= len(requests) <= 256:
        raise ValueError("a group must contain between 1 and 256 requests")
    if len({id(request) for request in requests}) != len(requests):
        raise ValueError("a request cannot occur twice in a group")
    device = None
    for request in requests:
        if not isinstance(request, (PreparedTransfer, PreparedTypedTransfer)):
            raise TypeError("group members must be prepared layout or typed transfers")
        if request.consumed:
            raise RuntimeError("group member was already executed")
        if isinstance(request, PreparedTypedTransfer) and request.template is None:
            raise ValueError("typed group members require an immutable dispatch template")
        current = request.device if request.direction == "h2d" else request.source.device
        current = torch.device(current)
        if current.type != "cuda" or current.index is None:
            raise ValueError("group members need a canonical CUDA device")
        if device is not None and device != current:
            raise ValueError("group members must use the same CUDA device")
        device = current
    return PreparedTransferGroup(requests, device)


def execute_transfer_group(group, *, resources=None, gather_threads=8, gather_pool=None,
                           pinning="auto", min_pinned_bytes=None,
                           max_scratch_bytes=64 << 20):
    """Complete one group with bounded scratch and one current-stream ordering.

    Output/source allocations and cached immutable metadata are outside the
    scratch budget. No payload packing or output aliasing is introduced.
    Resources serialize groups on one owner; use separate owners for concurrent
    groups. Any submission failure consumes the whole group, never retries it.
    """
    import torch
    from .resources import TransferResources, _transfer_configuration

    if not isinstance(group, PreparedTransferGroup):
        raise TypeError("group must be a PreparedTransferGroup")
    max_scratch_bytes = operator.index(max_scratch_bytes)
    gather_threads = operator.index(gather_threads)
    if not 0 < max_scratch_bytes < 1 << 64:
        raise ValueError("max_scratch_bytes must be positive and fit size_t")
    if not 0 < gather_threads < 1 << 31:
        raise ValueError("gather_threads must be positive and fit a native int")
    _transfer_configuration(resources, {"n_buffers": 1, "n_streams": 1,
        "gather_threads": gather_threads, "gather_pool": gather_pool,
        "pinning": pinning, "min_pinned_bytes": min_pinned_bytes})
    if resources is not None and not isinstance(resources, TransferResources):
        raise TypeError("resources must be TransferResources or None")
    owner = None if resources is None else resources.native_typed
    with ExitStack() as locks:
        for request in (group, *group.requests):
            if not request._execution_lock.acquire(blocking=False):
                raise RuntimeError("group or member is already executing")
            locks.callback(request._execution_lock.release)
            if request.consumed:
                raise RuntimeError("group or member was already executed")
        # Finish all stale-storage/parameter checks before allocating outputs or
        # submitting any member. The native boundary validates every fresh view.
        for request in group.requests:
            request.recheck()
        descriptors = group.destinations
        outputs = tuple(torch.empty_strided(d.shape, d.strides,
                        dtype=getattr(torch, d.dtype), device=d.device) for d in descriptors)
        entries = []
        for request, output in zip(group.requests, outputs):
            view = _storage_view(output, "cuda" if output.is_cuda else "host",
                                 output.device.index if output.is_cuda else -1)
            plan = request.template if isinstance(request, PreparedTypedTransfer) else request.bound
            entries.append((plan, request.source_view, view, request.direction))
        native = pyreloc.prepare_dispatch_group(entries)
        group.native = native
        stream = compat.cuda_stream_handle(group.device)
        group.consumed = True
        for request in group.requests:
            request.consumed = True
        try:
            report = pyreloc.execute_dispatch_group(native, caller_stream=stream,
                gather_threads=gather_threads, gather_pool=gather_pool,
                pinning=pinning, min_pinned_bytes=min_pinned_bytes,
                max_scratch_bytes=max_scratch_bytes, resources=owner,
                owners=(tuple(r.source for r in group.requests), outputs))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"transfer group failed: {error}") from error
        return GroupResult(outputs, descriptors, MappingProxyType(report))
