"""Opt-in packed output lifetimes and explicit physical-capacity estimates.

The plan accounts for declared live allocations, not opaque allocator/library
state. Applications supply a measured physical budget and a workspace reserve;
an estimate is never a promise that an allocation cannot fail.
"""
from dataclasses import dataclass
import os
from threading import RLock

_MAX_BYTES = (1 << 63) - 1
_ALIGNMENT = 256


class MemoryBudgetError(BufferError):
    """A declared capacity cannot admit work; no transfer has been launched."""


def _integer(value, name, *, minimum=0, maximum=_MAX_BYTES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class TransferMemoryPlan:
    resident_groups: int
    resident_bytes: int
    slots: int
    contexts: int
    device_scratch_per_context: int
    output_bytes: int
    subgroup_members: int | None
    workspace_bytes: int
    retained_bytes: int
    safety_bytes: int
    available_bytes: int
    predicted_device_bytes: int
    headroom_bytes: int

    @property
    def native_scratch_bytes(self):
        """Host+device native quota, distinct from predicted device bytes."""
        return max(1, 2 * self.device_scratch_per_context) if self.contexts else 0


def plan_transfer_memory(group_output_bytes, group_scratch_bytes, *, available_bytes,
                         workspace_bytes, retained_bytes=0, max_slots=2,
                         max_resident_groups=0, safety_bytes=16 << 20,
                         subgroup_scratch_bytes=None):
    """Choose a finite prefix/depth plan using caller-declared physical space.

    Footprints must include alignment and all scratch for the qualified row.
    Prefer the largest resident prefix, then the deepest feasible window, then
    whole groups over single-member subgroups. One extra output group covers
    the consumer's previous iteration. Other held outputs belong in
    ``retained_bytes``; they are not silently assumed to have been released.
    """
    outputs, scratch = tuple(group_output_bytes), tuple(group_scratch_bytes)
    if len(outputs) != len(scratch) or len(outputs) > 4096:
        raise ValueError("one scratch estimate per group, at most 4096 groups")
    for value in outputs:
        _integer(value, "group output bytes", minimum=1)
    for value in scratch:
        _integer(value, "group scratch bytes")
    for name, value in (("available_bytes", available_bytes), ("workspace_bytes", workspace_bytes),
                        ("retained_bytes", retained_bytes), ("safety_bytes", safety_bytes)):
        _integer(value, name)
    _integer(max_slots, "max_slots", minimum=1, maximum=8)
    _integer(max_resident_groups, "max_resident_groups", maximum=len(outputs))
    small = None if subgroup_scratch_bytes is None else tuple(subgroup_scratch_bytes)
    if small is not None:
        if len(small) != len(outputs):
            raise ValueError("one subgroup scratch estimate per group")
        for value, full in zip(small, scratch):
            _integer(value, "subgroup scratch bytes", maximum=full)
    base = workspace_bytes + retained_bytes + safety_bytes
    for resident in range(max_resident_groups, -1, -1):
        resident_bytes = sum(outputs[:resident])
        depths = range(max_slots, 0, -1) if resident < len(outputs) else (0,)
        for slots in depths:
            variants = ((None, scratch),) if small is None else ((None, scratch), (1, small))
            for subgroup, requirements in variants:
                device_scratch = max(requirements[resident:], default=0)
                out_bytes = (slots + 1) * max(outputs[resident:], default=0) if slots else 0
                total = base + resident_bytes + slots * device_scratch + out_bytes
                if total <= available_bytes and 2 * device_scratch <= _MAX_BYTES:
                    return TransferMemoryPlan(resident, resident_bytes, slots, slots, device_scratch,
                                              out_bytes, subgroup if slots else None, workspace_bytes,
                                              retained_bytes, safety_bytes, available_bytes, total,
                                              available_bytes - total)
    raise MemoryBudgetError("no resident-prefix/prefetch plan fits the declared physical budget")


class PackedOutputArena:
    """Bound packed output storage until its final Tensor alias is released.

    Transfer completion never releases a block. A native storage owner waits
    until it alone retains the backing TensorImpl and storage, then waits for
    device completion before reuse. Original allocator ownership remains intact
    throughout; close only drops the owner's reference. ``max_bytes`` covers
    this arena's allocations only. Exported storage capacity cannot be resized.
    """

    def __init__(self, device, *, max_bytes, reuse=True):
        import torch
        from .fast import _load_native_views
        _integer(max_bytes, "max_bytes", minimum=1)
        if type(reuse) is not bool:
            raise TypeError("reuse must be a bool")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("packed outputs require CPU or CUDA")
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        bridge = _load_native_views()
        if bridge is None or not hasattr(bridge, "PackedStorage"):
            raise RuntimeError("packed output storage requires the matching native Torch bridge")
        self.device, self.max_bytes, self.reuse = device, max_bytes, reuse
        self._bridge, self._pid, self._lock = bridge, os.getpid(), RLock()
        self._blocks = []
        self._closed = False
        self._counts = dict(allocations=0, reuses=0, evictions=0, refusals=0,
                            retired_barriers=0, exported_bytes_at_close=0)

    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("packed output arena belongs to another process")

    def _layout(self, descriptors):
        import torch
        descriptors = tuple(descriptors)
        if not 1 <= len(descriptors) <= 256:
            raise ValueError("a packed group holds between 1 and 256 outputs")
        layout, cursor = [], 0
        for descriptor in descriptors:
            if torch.device(descriptor.device) != self.device:
                raise ValueError("all packed outputs must use the arena device")
            dtype = getattr(torch, descriptor.dtype, None)
            if not isinstance(dtype, torch.dtype) or dtype in (torch.qint8, torch.quint8, torch.qint32):
                raise ValueError("unsupported packed output dtype")
            shape = tuple(descriptor.shape)
            count, expected = 1, []
            for dimension in reversed(shape):
                _integer(dimension, "output dimension", minimum=1)
                expected.append(count)
                count *= dimension
                if count > _MAX_BYTES:
                    raise ValueError("packed output size overflows")
            if tuple(descriptor.strides) != tuple(reversed(expected)):
                raise ValueError("packed outputs must have canonical dense strides")
            itemsize = torch.empty((), dtype=dtype).element_size()
            nbytes = count * itemsize
            if nbytes > _MAX_BYTES - cursor - (_ALIGNMENT - 1):
                raise ValueError("packed output bytes overflow")
            layout.append((shape, tuple(descriptor.strides), dtype, cursor // itemsize))
            cursor += (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        return layout, cursor

    def _drop(self, block):
        block[1].close()
        self._counts["retired_barriers"] += block[1].barriers
        self._blocks.remove(block)
        self._counts["evictions"] += 1

    def allocate(self, descriptors):
        import torch
        self._check_process()
        with self._lock:
            if self._closed:
                raise RuntimeError("packed output arena is closed")
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("packed output arenas do not support CUDA graph capture")
            layout, required = self._layout(descriptors)
            if required > self.max_bytes:
                self._counts["refusals"] += 1
                raise MemoryBudgetError("packed group exceeds the output arena budget")
            free = sorted((b for b in self._blocks if b[1].expired()), key=lambda b: b[0])
            block = next((b for b in free if b[0] >= required), None) if self.reuse else None
            if block is None:
                for stale in free:
                    if self.reuse and sum(b[0] for b in self._blocks) + required <= self.max_bytes:
                        break
                    self._drop(stale)
                if sum(b[0] for b in self._blocks) + required > self.max_bytes:
                    self._counts["refusals"] += 1
                    raise MemoryBudgetError("caller-held output aliases exhaust the arena budget")
                backing = torch.empty(required, dtype=torch.uint8, device=self.device)
                block = (required, self._bridge.PackedStorage(backing, recycle=self.reuse))
                self._blocks.append(block)
                self._counts["allocations"] += 1
            else:
                self._counts["reuses"] += 1
            view = block[1].view()
            storage = view.untyped_storage()
            return [torch.empty(0, dtype=dtype, device=self.device).set_(storage, offset, shape, strides)
                    for shape, strides, dtype, offset in layout]

    def stats(self):
        self._check_process()
        with self._lock:
            allocated = sum(b[0] for b in self._blocks)
            live = sum(b[0] for b in self._blocks if not b[1].expired())
            if not self.reuse:
                allocated = live
            return dict(self._counts, allocated_bytes=allocated, live_bytes=live,
                        reusable_bytes=allocated-live, max_bytes=self.max_bytes,
                        quarantined_bytes=sum(b[0] for b in self._blocks if b[1].poisoned),
                        completion_barriers=self._counts["retired_barriers"] +
                        sum(b[1].barriers for b in self._blocks), closed=self._closed,
                        scope="logical storage capacity within Torch allocations; not physical or additive")

    def close(self):
        self._check_process()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failure = None
            for block in tuple(self._blocks):
                try:
                    if not block[1].expired():
                        self._counts["exported_bytes_at_close"] += block[0]
                    self._drop(block)
                except BaseException as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure

    def __enter__(self):
        self._check_process()
        if self._closed:
            raise RuntimeError("packed output arena is closed")
        return self

    def __exit__(self, *exc):
        self.close()
        return False
