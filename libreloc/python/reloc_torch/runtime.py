"""Common guarded execution shared by the custom op and eager interception.

``execute_or_fallback`` is the single boundary between the frontend and the
runtime adapter. Expected rejections (``UnsupportedRecipe``) surface before a
destination is allocated or any work is launched, and run the saved original
region exactly once. Errors raised after the launch decision propagate with
recipe/direction context and never rerun the region. Both branches run with
eager interception suspended so runtime-internal Torch calls and fallbacks are
never re-intercepted.
"""

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass, field
import importlib
import threading
from typing import Protocol

from . import compat
from .artifact import UnsupportedRecipe
from .eligibility import _metadata_reason
from .symbolic import GuardError, expression


_local = threading.local()


class ExecutionError(RuntimeError):
    """A runtime failure after preflight; never a fallback signal."""

    def __init__(self, message, *, direction, handle):
        super().__init__(message)
        self.direction = direction
        self.handle = handle


@dataclass(frozen=True)
class ConcreteDescriptor:
    """Logical output metadata after symbol binding: shape, dense strides, dtype, device."""

    shape: tuple
    strides: tuple
    dtype: str
    device: object


def _metadata_snapshot(tensor):
    storage = tensor.untyped_storage()
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        str(tensor.dtype),
        str(tensor.device),
        storage.data_ptr(),
        storage.nbytes(),
    )


@dataclass(eq=False)
class PreparedCall:
    """Everything a validated invocation retains until completion.

    ``request`` is the runtime adapter's own validated request (R2's
    ``PreparedTransfer`` in production); ``bound`` is the standalone bound plan
    when the adapter binds through pyreloc. A prepared call is single-use and
    rechecks the source metadata immediately before execution.
    """

    compiled: object
    src: object
    bindings: dict
    bound: object
    destination: ConcreteDescriptor
    non_blocking: bool = False
    request: object = None
    _snapshot: tuple = field(init=False, repr=False, compare=False)
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._snapshot = _metadata_snapshot(self.src)

    def recheck(self):
        if _metadata_snapshot(self.src) != self._snapshot:
            raise RuntimeError(
                "stale prepared call: source metadata changed after preflight"
            )

    def consume(self):
        if self._consumed:
            raise RuntimeError("prepared call was already executed")
        self._consumed = True


class RuntimeAdapter(Protocol):
    """T3 / R2 bridge contract (shared contracts table)."""

    capability_identity: str

    def preflight(self, compiled, src, device, *, non_blocking=False) -> PreparedCall: ...

    def execute(self, call: PreparedCall): ...


class ExecutionEntry:
    """A compiled recipe, its saved original region, adapter and diagnostics."""

    def __init__(self, *, compiled, original, runtime, diagnostics, symbolic_bindings=(), extent_guards=()):
        self.compiled = compiled
        self.original = original
        self.runtime = runtime
        self.diagnostics = diagnostics
        self.symbolic_bindings = tuple(symbolic_bindings)
        self.extent_guards = tuple(extent_guards)
        self.handle = None
        self.closed = False
        self.fallback_calls = 0
        self._lock = threading.Lock()

    @property
    def direction(self):
        return self.compiled.recipe.direction

    def describe(self):
        return self.handle if self.handle is not None else "unregistered entry"

    def close(self):
        self.closed = True

    def fallback(self, src, *symbols):
        """Run the saved original region once with T2's ordered scalar arguments."""
        with self._lock:
            self.fallback_calls += 1
        return self.original(src, *self.scalar_arguments(symbols))

    def scalar_arguments(self, symbols):
        if not self.symbolic_bindings:
            return ()
        names = self.compiled.symbols
        if len(symbols) != len(names):
            raise RuntimeError(
                f"expected {len(names)} symbol values for {names}, got {len(symbols)}"
            )
        bindings = {name: int(value) for name, value in zip(names, symbols)}
        return tuple(
            expression(expr).evaluate(bindings) for _, expr in self.symbolic_bindings
        )


def interception_suspended():
    return getattr(_local, "depth", 0) > 0


@contextmanager
def suspend_interception():
    """Thread-local, reentrant suspension of eager transfer interception."""
    _local.depth = getattr(_local, "depth", 0) + 1
    try:
        yield
    finally:
        _local.depth -= 1


def source_reason(src):
    """Frontend metadata guard shared by every adapter; ``None`` when admitted.

    ``requires_grad`` excludes a source only while grad mode is enabled: that
    is when execution would be gradient-requiring. Under ``torch.no_grad()``
    parameters and buffers are ordinary dense inputs.
    """
    import torch

    if not compat.is_plain_tensor_or_parameter(src):
        return "tensor_subclass"
    metadata = compat.tensor_metadata(src)
    if metadata.requires_grad and not torch.is_grad_enabled():
        metadata = dataclasses.replace(metadata, requires_grad=False)
    return _metadata_reason(metadata)


def bind_symbols(compiled, src):
    """Exact name-to-value map for ``pyreloc.bind`` from real source metadata."""
    try:
        return compiled.bind_values(src)
    except GuardError as error:
        raise UnsupportedRecipe(error.reason, str(error)) from error


def load_plan(plan_bytes):
    """Decode the artifact's plan; decoding is cheap and keeps no hidden cache."""
    import pyreloc

    return pyreloc.load_plan(plan_bytes)


def bind_plan(compiled, bindings):
    """Bind through the standalone binder; expected rejections become a reason.

    Every call counts as one binder call on the diagnostics of the execution
    currently in preflight (see ``execute_or_fallback``), whatever the adapter
    decides afterwards.
    """
    import pyreloc

    diagnostics = getattr(_local, "diagnostics", None)
    if diagnostics is not None:
        diagnostics.increment("symbol_binds")
    try:
        return pyreloc.bind(load_plan(compiled.plan_bytes), bindings)
    except pyreloc.BindError as error:
        raise UnsupportedRecipe("bind_error", str(error)) from error


@contextmanager
def _counting_binds(diagnostics):
    previous = getattr(_local, "diagnostics", None)
    _local.diagnostics = diagnostics
    try:
        yield
    finally:
        _local.diagnostics = previous


def destination_descriptor(compiled, bindings, device):
    import torch

    def evaluate(value):
        try:
            return expression(value).evaluate(bindings, checked=True)
        except KeyError as error:
            raise UnsupportedRecipe("missing_symbol", str(error)) from error
        except GuardError as error:
            raise UnsupportedRecipe(error.reason, str(error)) from error

    logical = compiled.logical_destination
    shape = tuple(evaluate(dim) for dim in logical.shape)
    strides = tuple(evaluate(dim) for dim in logical.strides)
    if evaluate(logical.offset) != 0 or any(dim <= 0 for dim in shape):
        raise UnsupportedRecipe("destination_descriptor", "expected dense positive zero-offset output")
    return ConcreteDescriptor(shape, strides, logical.dtype, torch.device(device))


_dtype_name = compat.dtype_name


def verify_result(result, src, destination):
    """Enforce the functional promise on any result handed back to a caller.

    The result must be a fresh tensor (never aliasing the source) with the
    destination's shape, dtype, zero offset and device. Strides are compared
    only on extents larger than one: a size-one axis addresses no elements, so
    PyTorch's own fallback may legitimately report a different stride there.
    """
    import torch

    if not isinstance(result, torch.Tensor):
        raise RuntimeError("reloc_torch produced a non-tensor result")
    if compat.tensors_alias(result, src):
        raise RuntimeError("reloc_torch result must not alias its input")
    actual = (tuple(result.shape), tuple(result.stride()), _dtype_name(result.dtype), result.storage_offset())
    expected = (tuple(destination.shape), tuple(destination.strides), destination.dtype, 0)
    if actual[0] != expected[0] or actual[2] != expected[2] or actual[3] != 0:
        raise RuntimeError(f"reloc_torch output metadata mismatch: got {actual}, expected {expected}")
    for size, stride, promised in zip(actual[0], actual[1], expected[1]):
        if size != 1 and stride != promised:
            raise RuntimeError(f"reloc_torch output metadata mismatch: got {actual}, expected {expected}")
    device = destination.device
    if result.device.type != device.type or (
        device.index is not None and result.device.index != device.index
    ):
        raise RuntimeError(f"reloc_torch output device {result.device} does not match declared {device}")
    return result


def prepare_host_call(compiled, src, device, *, non_blocking=False):
    """Frontend preflight: metadata guards, exact binding, standalone bind, destination.

    Allocates nothing and launches nothing. Adapters add their own storage,
    device and capability checks around it.
    """
    import torch

    if non_blocking:
        raise UnsupportedRecipe("nonblocking_unavailable", "blocking transfers only")
    device = torch.device(device)
    reason = source_reason(src)
    if reason is not None:
        raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
    bindings = bind_symbols(compiled, src)
    bound = bind_plan(compiled, bindings)
    destination = destination_descriptor(compiled, bindings, device)
    return PreparedCall(compiled, src, bindings, bound, destination, non_blocking)


class TransportAdapter:
    """Production bridge to R2's ``reloc_torch.transport`` (issue #146).

    R2 owns storage validation, allocation, stream ordering and completion.
    While that module is absent every preflight is the expected exclusion
    ``runtime_unavailable``: the original region runs and nothing launches.
    """

    REQUEST_ATTRIBUTES = ("bindings", "destination")

    MODULE = f"{__package__}.transport"

    def __init__(self):
        self._module = None
        self.unavailable_reason = None
        try:
            module = importlib.import_module(self.MODULE)
        except ModuleNotFoundError as error:
            # Only the bridge module itself may be absent; a present module
            # that fails to import its own dependencies is a real error.
            if error.name != self.MODULE:
                raise
            self.unavailable_reason = (
                f"{self.MODULE} is not available ({error}); "
                "R2 (#146) has not been delivered"
            )
        else:
            self._module = module

    @property
    def available(self):
        return self._module is not None

    @property
    def capability_identity(self):
        if not self.available:
            return "reloc_torch.transport/unavailable"
        return f"reloc_torch.transport/{getattr(self._module, 'CAPABILITY_IDENTITY', '0')}"

    def preflight(self, compiled, src, device, *, non_blocking=False):
        import torch

        if not self.available:
            raise UnsupportedRecipe("runtime_unavailable", self.unavailable_reason)
        request = self._module.prepare_transfer(compiled, src, device, non_blocking=non_blocking)
        missing = [name for name in self.REQUEST_ATTRIBUTES if not hasattr(request, name)]
        if missing:
            raise RuntimeError(
                "R2 prepare_transfer result lacks the documented attributes "
                f"{missing}; reconcile TransportAdapter with reloc_torch.transport"
            )
        destination = request.destination
        if not isinstance(destination, ConcreteDescriptor):
            destination = ConcreteDescriptor(
                tuple(destination.shape), tuple(destination.strides),
                compat.dtype_name(destination.dtype), torch.device(destination.device),
            )
        return PreparedCall(
            compiled, src, dict(request.bindings), getattr(request, "bound", None),
            destination, non_blocking, request=request,
        )

    def execute(self, call):
        return self._module.execute_transfer(call.request)


def _derived_symbols(entry, src):
    values = []
    for name in entry.compiled.symbols:
        source = next((s for s in entry.compiled.symbol_sources if s.name == name), None)
        if source is None or source.axis >= src.dim():
            return None
        values.append(int(src.shape[source.axis]))
    return values


def _fallback(entry, src, symbols, reason, promised=None):
    if symbols is None:
        # Symbol values only matter for the original region's scalar
        # placeholders; eager identity entries have none.
        symbols = ()
        if entry.symbolic_bindings:
            symbols = _derived_symbols(entry, src)
            if symbols is None:
                raise RuntimeError(
                    "cannot evaluate the original region's scalar placeholders for a "
                    f"source of rank {src.dim()} (recipe symbols {entry.compiled.symbols})"
                )
    entry.diagnostics.record_fallback(reason)
    result = entry.fallback(src, *symbols)
    return result if promised is None else verify_result(result, src, promised)


def execute_or_fallback(entry, src, symbols, device, *, non_blocking=False, declared=None):
    """Preflight, then either dispatch once or run the original region once.

    ``symbols`` is the ordered list of concrete values supplied by the op (or
    ``None`` when the caller supplies none); the frontend guards bind the exact
    name-to-value map from real source metadata first and reconcile it with the
    supplied symbols, so every expected exclusion has a stable reason before the
    adapter is consulted. ``declared`` optionally carries the output metadata
    promised to the graph; a mismatch with the compiled descriptor is an error,
    never a fallback. Every result handed back, from the adapter or from the
    original region once the promised metadata is known, passes ``verify_result``.
    """
    import torch

    if entry.closed:
        raise RuntimeError("execution entry is closed")
    device = torch.device(device)
    with suspend_interception():
        if non_blocking:
            return _fallback(entry, src, symbols, "nonblocking_unavailable", declared)
        reason = source_reason(src)
        if reason is not None:
            return _fallback(entry, src, symbols, reason, declared)
        try:
            bindings = bind_symbols(entry.compiled, src)
            destination = destination_descriptor(entry.compiled, bindings, device)
            for guard in entry.extent_guards:
                if expression(guard).evaluate(bindings, checked=True) < 2:
                    raise UnsupportedRecipe("singleton_extent", f"{guard} binds below two")
        except UnsupportedRecipe as error:
            return _fallback(entry, src, symbols, error.reason, declared)
        except (KeyError, GuardError) as error:
            return _fallback(entry, src, symbols, getattr(error, "reason", "missing_symbol"), declared)
        promised = destination if declared is None else declared
        if symbols is not None:
            expected = [bindings[name] for name in entry.compiled.symbols]
            if [int(value) for value in symbols] != expected:
                return _fallback(entry, src, symbols, "symbol_mismatch", promised)
        if declared is not None and (
            tuple(declared.shape) != destination.shape or tuple(declared.strides) != destination.strides
        ):
            raise RuntimeError(
                f"declared output metadata {tuple(declared.shape)}/{tuple(declared.strides)} "
                f"does not match the compiled {entry.direction} descriptor "
                f"{destination.shape}/{destination.strides}"
            )
        try:
            with _counting_binds(entry.diagnostics):
                call = entry.runtime.preflight(entry.compiled, src, device, non_blocking=non_blocking)
        except UnsupportedRecipe as error:
            return _fallback(entry, src, symbols, error.reason, promised)
        if call.bindings != bindings or tuple(call.destination.shape) != destination.shape:
            raise RuntimeError("runtime adapter disagreed with the frontend binding")
        call.recheck()
        call.consume()
        entry.diagnostics.increment("runtime_executions")
        try:
            result = entry.runtime.execute(call)
        except Exception as error:
            raise ExecutionError(
                f"reloc_torch {entry.direction} execution failed for "
                f"{entry.describe()}: {error}",
                direction=entry.direction,
                handle=entry.handle,
            ) from error
        return verify_result(result, src, promised)


__all__ = (
    "ConcreteDescriptor",
    "ExecutionEntry",
    "ExecutionError",
    "PreparedCall",
    "RuntimeAdapter",
    "TransportAdapter",
    "bind_plan",
    "bind_symbols",
    "destination_descriptor",
    "execute_or_fallback",
    "interception_suspended",
    "load_plan",
    "prepare_host_call",
    "source_reason",
    "suspend_interception",
    "verify_result",
)
