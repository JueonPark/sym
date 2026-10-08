"""R3 typed dispatch bridge (issue #147).

``prepare_typed_transfer`` validates a compiled TYPED recipe against a real
source tensor and its runtime parameters without allocating or launching:
frontend metadata guards, exact symbol binding, parameter snapshots (CPU
values only; device-resident parameters are an explicit exclusion), the
standalone typed binder, the destination descriptor, storage identity, the
capability query and the policy decision. ``execute_typed_transfer``
allocates the fresh destination on the caller's device, rechecks the source
storage and the parameter values against their snapshots, orders the
runtime's private streams after the caller's current CUDA stream, runs
exactly the selected implementation and returns the tensor together with an
immutable scalar-only report.

Policies: ``"original_cpu"`` forces the CPU reference pipeline (layout and
every stage on the CPU, then the necessary copy); ``"auto"`` consults an
optional calibration and otherwise takes the reference with a recorded
reason. Neither can change the requested precision or skip a stage: only
implementations the runtime qualified as semantically equivalent are ever
selected (see docs/runtime-dispatch.md).

Expected exclusions raise ``UnsupportedRecipe``. Failures after the launch
decision are errors, never fallback signals. Requests are single-use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from threading import Lock
from types import MappingProxyType

import pyreloc

from . import compat
from .artifact import UnsupportedRecipe
from .runtime import (
    ConcreteDescriptor, bind_symbols, destination_descriptor, source_reason,
    _validated_preparation, _prepared_destination,
)
from .transport import _code, _storage_view

POLICIES = ("auto", "original_cpu")
_DIRECTIONS = {"h2d": ("cpu", "cuda"), "d2h": ("cuda", "cpu")}


def _parameter_snapshot(name, value, declared):
    """(dtype, extents, bytes) of one CPU parameter tensor; the runtime
    binder re-validates dtype, rank, extents, byte size and values."""
    if not compat.is_plain_tensor_or_parameter(value):
        raise UnsupportedRecipe("parameter_subclass", f"parameter {name!r} is not a plain tensor")
    if value.device.type != "cpu":
        raise UnsupportedRecipe(
            "device_parameters_unavailable",
            f"parameter {name!r} lives on {value.device}; only validated CPU parameter "
            "values are supported (device parameters have no qualified preparation path)",
        )
    import torch

    dtype = compat.dtype_name(value.dtype)
    data = value.detach().contiguous()
    if declared[0] == "int32" and value.dtype in (torch.int64, torch.int16, torch.int8):
        # Zero points arrive from graphs as whatever integer width PyTorch
        # gave them; the value is what matters and converts exactly, the
        # binder still range-checks it.
        if data.numel() and (int(data.min()) < -(2 ** 31) or int(data.max()) >= 2 ** 31):
            raise UnsupportedRecipe("bind_error", f"parameter {name!r} does not fit int32")
        data = data.to(torch.int32)
        dtype = "int32"
    if dtype != declared[0]:
        raise UnsupportedRecipe(
            "parameter_dtype", f"parameter {name!r} is {dtype}, declared {declared[0]}"
        )
    return (dtype, [int(d) for d in data.shape], data.numpy().tobytes())


@dataclass(frozen=True)
class DispatchResult:
    """The destination tensor and the immutable scalar-only report."""

    tensor: object
    report: MappingProxyType


@dataclass
class PreparedTypedTransfer:
    """A validated typed invocation retained until completion."""

    compiled: object
    source: object
    bindings: dict
    bound: object
    destination: ConcreteDescriptor
    direction: str
    device: object
    source_view: object
    policy: str
    calibration: object
    threads: int
    parameters: dict
    snapshots: dict
    capability: dict
    selected: dict
    program: object = None
    request: object = None
    template: object = field(default=None, kw_only=True)
    consumed: bool = False
    staging: tuple = field(default=(), init=False)
    _storage: tuple = field(init=False, repr=False)
    _execution_lock: object = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._storage = compat.storage_snapshot(self.source)

    def recheck(self):
        """Source storage and every parameter VALUE must be what was prepared;
        an unchanged tensor object is not proof."""
        if compat.storage_snapshot(self.source) != self._storage:
            raise RuntimeError(
                "stale typed transfer: source storage or metadata changed after preflight"
            )
        declared = self.compiled.parameter_extents(self.bindings)
        for name, value in self.parameters.items():
            try:
                current = _parameter_snapshot(name, value, declared[name])
            except UnsupportedRecipe as error:
                raise RuntimeError(f"stale typed transfer: {error}") from error
            if current != self.snapshots[name]:
                raise RuntimeError(
                    f"stale typed transfer: parameter {name!r} changed after preflight"
                )

    @property
    def report(self):
        """The selection as it stands before execution (no payload yet)."""
        return MappingProxyType(dict(self.selected))


@dataclass(frozen=True)
class _TypedTemplate:
    bound: object
    native: object
    capability: tuple
    selection: tuple

    def capability_dict(self):
        return {name: [dict(row) for row in rows] for name, rows in self.capability}


def prepare_typed_transfer(
    compiled, source, device, *, parameters=None, policy="auto", calibration=None,
    threads=8, implementation="",
):
    import torch

    if not getattr(compiled, "typed", False):
        raise UnsupportedRecipe("not_typed_recipe", "the compiled recipe has no value transform")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    device = torch.device(device)
    shared = _validated_preparation(compiled, source)
    if shared is None:
        reason = source_reason(source)
        if reason is not None:
            raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
    direction = compiled.recipe.direction
    expected = _DIRECTIONS.get(direction)
    if expected is None or (source.device.type, device.type) != expected:
        raise UnsupportedRecipe(
            "direction_mismatch", f"{direction} recipe cannot move {source.device} -> {device}"
        )
    bindings = shared[0] if shared is not None else bind_symbols(compiled, source)
    declared = compiled.parameter_extents(bindings)
    parameters = dict(parameters or {})
    missing = sorted(set(declared) - set(parameters))
    extra = sorted(set(parameters) - set(declared))
    if missing or extra:
        raise UnsupportedRecipe(
            "parameter_binding",
            f"declared parameters {sorted(declared)}; missing {missing}, unexpected {extra}",
        )
    snapshots = {name: _parameter_snapshot(name, parameters[name], declared[name]) for name in declared}
    # Normalize cache-key scalars before lookup so e.g. 1.0 cannot hit the
    # integer thread key and bypass native argument validation.
    threads = operator.index(threads)
    if not 1 <= threads <= (1 << 31) - 1:
        raise ValueError("threads must be between 1 and 2147483647")
    if not isinstance(implementation, str):
        raise TypeError("implementation must be a string")
    if calibration is not None and type(calibration) is not pyreloc.Calibration:
        raise TypeError("calibration must be pyreloc.Calibration or None")
    cuda_available = pyreloc.cuda_enabled and torch.cuda.is_available()
    index = (device.index if device.index is not None else
             torch.cuda.current_device() if cuda_available else None) if direction == "h2d" else source.device.index
    # Calibration is immutable in Python. Hold its object identity in the key
    # (not id(calibration), which can be reused after destruction).
    parameter_key = tuple((name, dtype, tuple(extents), data)
                          for name, (dtype, extents, data) in sorted(snapshots.items()))
    parameter_bytes = sum(len(data) for _, _, _, data in parameter_key)
    key = ("typed", tuple(sorted(bindings.items())), parameter_key, direction,
           source.device.type, source.device.index, device.type, index,
           policy, calibration, threads, implementation)
    cache = compiled._execution_templates
    template = cache.get(key, parameter_bytes)
    try:
        bound = template.bound if template is not None else pyreloc.bind_typed(
            compiled.decoded_plan, bindings, snapshots)
    except pyreloc.DecodeError as error:
        raise RuntimeError(f"compiled recipe holds an invalid typed plan: {error}") from error
    except pyreloc.BindError as error:
        raise UnsupportedRecipe("bind_error", str(error)) from error
    # Invalid parameter values still fail binding before CUDA availability,
    # including on hosts without a CUDA runtime. Cache hits use exact bytes.
    if not cuda_available:
        raise UnsupportedRecipe("cuda_unavailable", "no CUDA-capable runtime")
    if direction == "h2d":
        target = torch.device("cuda", index)
        view = _storage_view(source, "host", -1)
    else:
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
    destination = _prepared_destination(compiled, bindings, target, shared)
    if template is None:
        try:
            native = pyreloc.prepare_dispatch_template(
                bound, direction, "cuda", policy=policy, calibration=calibration,
                threads=threads, implementation=implementation,
            )
        except pyreloc.TransferError as error:
            raise UnsupportedRecipe(_code(error), str(error)) from error
        template = _TypedTemplate(bound, native, tuple(
            (name, tuple(tuple(row.items()) for row in rows))
            for name, rows in native.capability.items()), tuple(native.selection.items()))
        template = cache.put(key, template, parameter_bytes)
    return PreparedTypedTransfer(
        compiled, source, bindings, template.bound, destination, direction, target, view,
        policy, calibration, threads, parameters, snapshots,
        template.capability_dict(), dict(template.selection), program=template.native.program,
        template=template.native,
    )


def execute_typed_transfer(request, *, n_buffers=2, n_streams=2, gather_threads=None, gather_pool=None,
                           pinning="auto", min_pinned_bytes=None,
                           direct_dense_upload=True, pipeline=True, chunk_size=None, resources=None):
    """Complete a typed transfer, retaining scratch in ``resources`` when given.

    CPU-producing H2D rows use a bounded transform/copy ring when physical rows
    can be partitioned. ``chunk_size`` is a target in wire bytes (rows cannot be
    split); ``None`` uses the runtime heuristic. ``pipeline=False`` selects the
    whole-buffer control. ``n_buffers=1`` preserves the chunk schedule but waits
    for every copy before the next transform. Pinning and worker budgets remain
    explicit; unconfigured ``pinning="auto"`` stays conservative/pageable.
    """
    if not request._execution_lock.acquire(blocking=False):
        raise RuntimeError("typed transfer request is already executing")
    try:
        return _execute_typed_transfer(request, n_buffers=n_buffers, n_streams=n_streams,
            gather_threads=gather_threads, gather_pool=gather_pool, pinning=pinning,
            min_pinned_bytes=min_pinned_bytes, direct_dense_upload=direct_dense_upload,
            pipeline=pipeline, chunk_size=chunk_size,
            resources=resources)
    finally:
        request._execution_lock.release()


def _execute_typed_transfer(request, *, n_buffers, n_streams, gather_threads,
                            gather_pool, pinning, min_pinned_bytes,
                            direct_dense_upload, pipeline, chunk_size, resources):
    """Allocate the destination, recheck, order after the caller stream, run
    exactly the prepared implementation and complete."""
    import torch
    import operator
    from .resources import TransferResources, _transfer_configuration

    if not isinstance(pipeline, bool):
        raise TypeError("pipeline must be a bool")
    if chunk_size is not None:
        if isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer byte count")
        chunk_size = operator.index(chunk_size)
        if not 0 < chunk_size < 1 << 64:
            raise ValueError("chunk_size must be positive and fit size_t")

    _transfer_configuration(resources, {"n_buffers": n_buffers, "n_streams": n_streams,
        "gather_threads": request.threads if gather_threads is None else gather_threads,
        "gather_pool": gather_pool, "pinning": pinning, "min_pinned_bytes": min_pinned_bytes})
    if resources is not None and not isinstance(resources, TransferResources):
        raise TypeError("resources must be TransferResources or None")
    native_resources = None if resources is None else resources.native_typed

    if gather_threads is None:
        gather_threads = request.threads

    if request.consumed:
        raise RuntimeError("typed transfer request was already executed")
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
        if request.template is not None:
            native = pyreloc.prepare_dispatch_from_template(request.template, request.source_view, dst_view)
        else:
            native = pyreloc.prepare_dispatch(
                request.program, request.source_view, dst_view, request.direction,
                policy=request.policy, calibration=request.calibration, threads=request.threads,
                implementation=request.selected["implementation"],
            )
    except pyreloc.TransferError as error:
        raise RuntimeError(f"typed dispatch rejected at execution: {error}") from error
    request.request = native
    # Consumed only once work can be launched (see transport.execute_transfer).
    request.consumed = True
    try:
        report = pyreloc.execute_dispatch(
            native,
            caller_stream=compat.cuda_stream_handle(cuda_device),
            n_buffers=n_buffers,
            n_streams=n_streams,
            gather_threads=gather_threads,
            gather_pool=gather_pool,
            pinning=pinning, min_pinned_bytes=min_pinned_bytes,
            direct_dense_upload=direct_dense_upload,
            pipeline=pipeline, chunk_size=0 if chunk_size is None else chunk_size,
            resources=native_resources, owners=(request.source, out),
        )
    except pyreloc.TransferError as error:
        raise RuntimeError(f"{request.direction} typed dispatch failed: {error}") from error
    finally:
        request.staging = tuple(native.report.get("staging", ()))
    report = dict(report)
    report["placement_reason"] = request.selected["placement_reason"]
    report["policy"] = request.selected["policy"]
    return DispatchResult(out, MappingProxyType(report))


__all__ = (
    "POLICIES",
    "DispatchResult",
    "PreparedTypedTransfer",
    "execute_typed_transfer",
    "prepare_typed_transfer",
)
