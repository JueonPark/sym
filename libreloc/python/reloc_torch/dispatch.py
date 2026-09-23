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
from types import MappingProxyType

import pyreloc

from . import compat
from .artifact import UnsupportedRecipe
from .runtime import ConcreteDescriptor, bind_symbols, destination_descriptor, source_reason
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
    request: object = None
    consumed: bool = False
    _storage: tuple = field(init=False, repr=False)

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
    reason = source_reason(source)
    if reason is not None:
        raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
    direction = compiled.recipe.direction
    expected = _DIRECTIONS.get(direction)
    if expected is None or (source.device.type, device.type) != expected:
        raise UnsupportedRecipe(
            "direction_mismatch", f"{direction} recipe cannot move {source.device} -> {device}"
        )
    bindings = bind_symbols(compiled, source)
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
    try:
        plan = pyreloc.load_typed_plan(compiled.plan_bytes)
        bound = pyreloc.bind_typed(plan, bindings, snapshots)
    except pyreloc.DecodeError as error:
        raise RuntimeError(f"compiled recipe holds an invalid typed plan: {error}") from error
    except pyreloc.BindError as error:
        raise UnsupportedRecipe("bind_error", str(error)) from error
    # Parameters and the typed binding are checked before the device is: a
    # CPU-only host still reports every recipe/parameter problem precisely.
    if not pyreloc.cuda_enabled or not torch.cuda.is_available():
        raise UnsupportedRecipe("cuda_unavailable", "no CUDA-capable runtime")
    if direction == "h2d":
        index = device.index if device.index is not None else torch.cuda.current_device()
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
    destination = destination_descriptor(compiled, bindings, target)
    try:
        capability = pyreloc.query_capability(bound, direction, "cuda")
        selected = pyreloc.select_dispatch(
            bound, direction, "cuda", policy=policy, calibration=calibration,
            threads=threads, implementation=implementation,
        )
    except pyreloc.TransferError as error:
        raise UnsupportedRecipe(_code(error), str(error)) from error
    return PreparedTypedTransfer(
        compiled, source, bindings, bound, destination, direction, target, view,
        policy, calibration, threads, parameters, snapshots, capability, selected,
    )


def execute_typed_transfer(request, *, n_buffers=4, n_streams=2, gather_threads=1, gather_pool=None):
    """Allocate the destination, recheck, order after the caller stream, run
    exactly the prepared implementation and complete."""
    import torch

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
        native = pyreloc.prepare_dispatch(
            request.bound, request.source_view, dst_view, request.direction,
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
        )
    except pyreloc.TransferError as error:
        raise RuntimeError(f"{request.direction} typed dispatch failed: {error}") from error
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
