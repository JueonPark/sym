"""Scoped eager transfer replacement around T1's overload inventory.

Only real, blocking, dtype-preserving CPU<->CUDA ``aten._to_copy`` calls on
plain dense tensors are intercepted. Everything else redispatches to PyTorch
with a recorded reason. Interception is suspended while the common executor
runs so allocation, producer ordering, runtime-internal Torch calls, custom-op
execution and fallback are never re-intercepted; the dispatch mode itself is
thread-local, so concurrent threads outside the scope are untouched.

Activation qualifies the interpreter/Torch baseline and resolves the backend's
compiler (including that the exporter binary exists) and runtime bridge up
front, so a misconfiguration fails at the ``with`` statement rather than inside
a user's ``tensor.to()`` call. Compiler crashes (as opposed to explicit
``UnsupportedRecipe`` rejections) still propagate: they are errors, not
exclusions.

Dynamo does not trace a frame while a dispatch mode is active, so a
``torch.compile`` call first executed inside this scope runs eagerly and its
transfers are offloaded here, once each. Compile outside the scope for graph
replacement; a compiled function run inside the scope executes its custom op
with interception suspended, so nothing is offloaded twice.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

from . import compat
from .artifact import UnsupportedRecipe
from .eligibility import classify
from .records import TransferRecord
from .runtime import execute_or_fallback, interception_suspended, suspend_interception


@dataclass(frozen=True)
class Decision:
    reason: str | None
    direction: str | None = None
    device: object = None


def _dense_strides(shape):
    strides, running = [], 1
    for dim in reversed(shape):
        strides.append(running)
        running *= dim if isinstance(dim, int) else 1
    return tuple(reversed(strides))


def decide(func, args, kwargs):
    """Pure eligibility decision for one dispatched call; never executes it."""
    import torch

    if func is not torch.ops.aten._to_copy.default or not args:
        return Decision("unsupported_operator")
    src = args[0]
    if not compat.is_plain_tensor_or_parameter(src):
        return Decision("tensor_subclass")
    options = {name: value for name, value, _ in compat.bound_schema_arguments(func, args, kwargs)}
    if options.get("layout") not in (None, torch.strided):
        return Decision("unsupported_layout")
    if options.get("pin_memory"):
        return Decision("pinned_transfer_unavailable")
    if options.get("memory_format") not in (None, torch.preserve_format, torch.contiguous_format):
        return Decision("unsupported_memory_format")
    device = options.get("device")
    device = src.device if device is None else torch.device(device)
    dtype = options.get("dtype")
    dtype = src.dtype if dtype is None else dtype
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device = torch.device("cuda", index)
    if device == src.device:
        # Same-device casts and copies are the common ineligible case; decide
        # them before any storage query.
        return Decision("typed_transform_unavailable" if dtype != src.dtype else "same_device_copy")
    source = compat.tensor_metadata(src)
    if source.requires_grad and not torch.is_grad_enabled():
        # Gradient-requiring only while autograd could record; under no_grad a
        # parameter is an ordinary dense source (see runtime.source_reason).
        source = replace(source, requires_grad=False)
    destination = replace(
        source,
        strides=_dense_strides(source.shape),
        storage_offset=0,
        dtype=compat.dtype_name(dtype),
        device_type=device.type,
        device_index=index if device.type != "cpu" else None,
        pinned=False,
    )
    record = TransferRecord(
        operator="aten._to_copy.default",
        phase="eager",
        source=source,
        destination=destination,
        non_blocking=bool(options.get("non_blocking", False)),
        mutates=False,
        aliases_source=False,
        layout_history=(),
    )
    eligibility = classify(record)
    if not eligibility.candidate:
        return Decision(eligibility.reason)
    return Decision(None, "h2d" if src.device.type == "cpu" else "d2h", device)


_MODE_TYPE = None


def mode_type():
    """Version-pinned dispatch-mode subclass, built on first use so importing
    this module stays Torch-free."""
    global _MODE_TYPE
    if _MODE_TYPE is not None:
        return _MODE_TYPE

    class EagerTransferMode(compat.torch_dispatch_mode_type()):
        def __init__(self, backend):
            super().__init__()
            self.backend = backend

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            import torch

            kwargs = {} if kwargs is None else kwargs
            if func is not torch.ops.aten._to_copy.default or interception_suspended():
                return func(*args, **kwargs)
            decision = decide(func, args, kwargs)
            if decision.reason is not None:
                self.backend.diagnostics.record_redispatch(decision.reason)
                # Suspend so an enclosing scope neither re-records nor re-decides.
                with suspend_interception():
                    return func(*args, **kwargs)
            src = args[0]

            def original(source, *scalars):
                # The exact original operator and arguments are the fallback.
                return func(*args, **kwargs)

            try:
                entry = self.backend.eager_entry(src, decision.device, original, decision.direction)
            except UnsupportedRecipe as error:
                self.backend.diagnostics.record_exclusion(error.reason)
                with suspend_interception():
                    return func(*args, **kwargs)
            return execute_or_fallback(entry, src, None, decision.device)

    _MODE_TYPE = EagerTransferMode
    return _MODE_TYPE


@contextmanager
def eager_transfers(*, backend):
    """Intercept eligible eager transfers within the scope and route them to ``backend``."""
    if backend.closed:
        raise RuntimeError("RelocBackend is closed")
    compat.check_version()
    backend.compiler
    backend.runtime
    mode = mode_type()(backend)
    with mode:
        yield mode


__all__ = ("Decision", "decide", "eager_transfers", "mode_type")
