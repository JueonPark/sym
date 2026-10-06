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
import os
import threading
from typing import Protocol

from . import compat
from .artifact import UnsupportedRecipe
from .eligibility import _metadata_reason
from .resources import AUTO, TransferResources, _transfer_configuration
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


_metadata_snapshot = compat.storage_snapshot


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
    # C4: the scalar-only R3 report the adapter attaches after a typed dispatch.
    report: object = None
    staging: tuple = ()
    # torch.stack: every stacked input in order (``src`` is ``sources[0]``);
    # empty for single-source calls.
    sources: tuple = ()
    # torch.stack: True when ``request`` itself rechecks every input
    # immediately before execution (the transport's stacked request); the
    # call then keeps no second snapshot of the same inputs.
    request_rechecks: bool = False
    _snapshot: tuple = field(init=False, repr=False, compare=False)
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.sources:
            self._snapshot = _metadata_snapshot(self.src)
        elif self.request_rechecks:
            self._snapshot = None
        else:
            self._snapshot = tuple(_metadata_snapshot(tensor) for tensor in self.sources)

    def recheck(self):
        if not self.sources:
            current = _metadata_snapshot(self.src)
        elif self.request_rechecks:
            return
        else:
            current = tuple(_metadata_snapshot(tensor) for tensor in self.sources)
        if current != self._snapshot:
            raise RuntimeError(
                "stale prepared call: source metadata changed after preflight"
            )

    def consume(self):
        if self._consumed:
            raise RuntimeError("prepared call was already executed")
        self._consumed = True


class RuntimeAdapter(Protocol):
    """T3 / R2 bridge contract (shared contracts table).

    ``preflight_stacked(compiled, sources, device, *, non_blocking=False)`` is
    an optional method for host ``torch.stack`` regions. It receives a
    stacked recipe (``Recipe.stack_inputs`` > 0) and every stack input in
    order, raises ``UnsupportedRecipe`` for expected rejections, and returns a
    ``PreparedCall`` with ``src`` = ``sources[0]``, ``sources`` = the inputs,
    the exact bindings of the logical [N, *S] source and the destination;
    ``execute`` then runs that call. An adapter without the method never gets
    a stacked region: the backend leaves it in PyTorch (exclusion
    ``runtime_unavailable``) and a stacked call made directly falls back.
    """

    capability_identity: str

    def preflight(self, compiled, src, device, *, non_blocking=False) -> PreparedCall: ...

    def execute(self, call: PreparedCall): ...


def stacked_preflight(runtime):
    """The adapter's optional ``preflight_stacked``; ``runtime_unavailable``
    when the adapter does not implement it (see ``RuntimeAdapter``)."""
    preflight = getattr(runtime, "preflight_stacked", None)
    if not callable(preflight):
        raise UnsupportedRecipe(
            "runtime_unavailable",
            f"runtime adapter {type(runtime).__name__} does not implement preflight_stacked",
        )
    return preflight


class ExecutionEntry:
    """A compiled recipe, its saved original region, adapter and diagnostics."""

    def __init__(self, *, compiled, original, runtime, diagnostics, symbolic_bindings=(), extent_guards=(), closed_event=None, min_stack_bytes=0):
        self.compiled = compiled
        self.original = original
        self.runtime = runtime
        self.diagnostics = diagnostics
        self.symbolic_bindings = tuple(symbolic_bindings)
        self.extent_guards = tuple(extent_guards)
        self.handle = None
        self._closed = False
        self._closed_event = closed_event
        self.fallback_calls = 0
        self.min_stack_bytes = int(min_stack_bytes)
        self._lock = threading.Lock()

    @property
    def direction(self):
        return self.compiled.recipe.direction

    def describe(self):
        return self.handle if self.handle is not None else "unregistered entry"

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed or (self._closed_event is not None and self._closed_event.is_set())

    @closed.setter
    def closed(self, value):
        self._closed = value

    def fallback(self, src, *symbols, parameters=()):
        """Run the saved original region once with T2's ordered scalar arguments
        and, for a typed region, its runtime parameter tensors (C4)."""
        with self._lock:
            self.fallback_calls += 1
        return self.original(src, *self.scalar_arguments(symbols), *parameters)

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

    @property
    def stack_inputs(self):
        return getattr(self.compiled.recipe, "stack_inputs", 0)

    def fallback_stacked(self, sources, *symbols):
        """Run the saved original region once over every stacked input."""
        with self._lock:
            self.fallback_calls += 1
        return self.original(*sources, *self.scalar_arguments(symbols))


def stack_below_threshold(count, elements, itemsize, min_stack_bytes):
    """The stack fusion size gate: a host ``torch.stack`` of ``count`` inputs
    of ``elements`` elements of ``itemsize`` bytes each fuses only from
    ``min_stack_bytes`` input bytes. The backend's static check and the
    per-call check both use it, so the two gates cannot drift."""
    return count * elements * itemsize < min_stack_bytes


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


def check_stacked_sources(sources):
    """``source_reason`` over every stacked input, paying for the full guard
    once; shared by the frontend and the transport.

    Input 0 runs ``source_reason``. A later input is admitted from a lean
    check against it that covers every property ``source_reason`` reads:
    the plain ``Tensor``/``Parameter`` type test, a strided layout, the
    requires_grad/grad-mode rule, and (from one ``compat.storage_snapshot``)
    the same shape, strides, storage offset, dtype and device as input 0
    and storage that is not empty. Any other input runs ``source_reason``
    itself, so a rejection carries exactly that guard's reason and no input
    it rejects is admitted.

    Returns ``(reason, snapshots, uniform)``: the reason of the first
    rejected input in order (``snapshots`` is then None) or None; one storage
    snapshot per input; and whether every input passed the lean check, so
    that all of them share input 0's dtype, shape, strides, offset and
    device.
    """
    import torch

    if not sources:
        return None, (), True
    reason = source_reason(sources[0])
    if reason is not None:
        return reason, None, False
    first = compat.storage_snapshot(sources[0])
    # storage_snapshot is (shape, strides, offset bytes, dtype, device, base,
    # capacity bytes); input 0 has a zero offset and non-empty storage.
    key = first[:5]
    plain = (torch.Tensor, torch.nn.Parameter)
    grad = torch.is_grad_enabled()
    snapshots = [first]
    uniform = True
    for src in sources[1:]:
        snapshot = None
        if type(src) in plain and src.layout is torch.strided and not (grad and src.requires_grad):
            try:
                snapshot = compat.storage_snapshot(src)
            except Exception:
                snapshot = None
            if snapshot is not None and (snapshot[:5] != key or not snapshot[6]):
                snapshot = None
        if snapshot is None:
            reason = source_reason(src)
            if reason is not None:
                return reason, None, False
            snapshot = compat.storage_snapshot(src)
            uniform = False
        snapshots.append(snapshot)
    return None, tuple(snapshots), uniform


def bind_symbols(compiled, src):
    """Exact name-to-value map for ``pyreloc.bind`` from real source metadata."""
    validated = getattr(_local, "validated_binding", None)
    if (validated is not None and validated[0] is compiled and validated[1] is src
            and validated[2] == compat.storage_snapshot(src)):
        return dict(validated[3])
    try:
        return compiled.bind_values(src)
    except GuardError as error:
        raise UnsupportedRecipe(error.reason, str(error)) from error


def bind_stacked_symbols(compiled, sources, uniform=False):
    """Exact name-to-value map of a stacked recipe's logical [N, *S] source,
    guarded across every input (count, dtype, shape, dense, zero offset).
    ``uniform`` is ``check_stacked_sources``' proof that every input shares
    input 0's metadata, so only input 0's descriptor is read."""
    try:
        return compiled.bind_stacked_values(tuple(sources), uniform)
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
        return pyreloc.bind(compiled.decoded_plan, bindings)
    except pyreloc.BindError as error:
        raise UnsupportedRecipe("bind_error", str(error)) from error


@contextmanager
def _validated_binding(compiled, src, bindings):
    previous = getattr(_local, "validated_binding", None)
    _local.validated_binding = (compiled, src, compat.storage_snapshot(src), dict(bindings))
    try:
        yield
    finally:
        _local.validated_binding = previous


@contextmanager
def _validated_stacked_binding(compiled, sources, snapshots, uniform, bindings):
    """The stacked twin of ``_validated_binding``: the frontend's checked
    inputs, their storage snapshots and exact binding, offered to the
    adapter's preflight of the same request (``validated_stacked_inputs``)."""
    previous = getattr(_local, "validated_stacked", None)
    _local.validated_stacked = (compiled, sources, snapshots, uniform, dict(bindings))
    try:
        yield
    finally:
        _local.validated_stacked = previous


def validated_stacked_inputs(compiled, sources):
    """``(snapshots, uniform, bindings)`` the frontend checked for this
    compiled recipe and these exact input objects in the current stacked
    preflight on this thread, or None. Like the single-source binding it is
    keyed on identity plus storage snapshots: an input that changed since the
    frontend's check gets every check again. The returned snapshots are
    therefore every input's current one."""
    validated = getattr(_local, "validated_stacked", None)
    if validated is None or validated[0] is not compiled:
        return None
    known = validated[1]
    if known is not sources and (len(known) != len(sources)
                                 or any(a is not b for a, b in zip(known, sources))):
        return None
    current = tuple(compat.storage_snapshot(src) for src in sources)
    if current != validated[2]:
        return None
    return current, validated[3], dict(validated[4])


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

    if hasattr(compiled, "_destination_metadata"):
        try:
            shape, strides, dtype = compiled._destination_metadata(tuple(sorted(bindings.items())))
        except KeyError as error:
            raise UnsupportedRecipe("missing_symbol", str(error)) from error
        except GuardError as error:
            raise UnsupportedRecipe(error.reason, str(error)) from error
        return ConcreteDescriptor(shape, strides, dtype, torch.device(device))
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


def prepare_stacked_host_call(compiled, sources, device, *, non_blocking=False):
    """prepare_host_call for a stacked recipe: every input passes the shared
    metadata guards (``check_stacked_sources``) before one logical binding,
    unless the frontend has just checked and bound this same request
    (``validated_stacked_inputs``). Allocates and launches nothing."""
    import torch

    if non_blocking:
        raise UnsupportedRecipe("nonblocking_unavailable", "blocking transfers only")
    sources = tuple(sources)
    validated = validated_stacked_inputs(compiled, sources)
    if validated is None:
        reason, _, uniform = check_stacked_sources(sources)
        if reason is not None:
            raise UnsupportedRecipe(reason, f"source tensor rejected: {reason}")
        bindings = bind_stacked_symbols(compiled, sources, uniform)
    else:
        bindings = validated[2]
    bound = bind_plan(compiled, bindings)
    destination = destination_descriptor(compiled, bindings, torch.device(device))
    return PreparedCall(compiled, sources[0], bindings, bound, destination, non_blocking, sources=sources)


class TransportAdapter:
    """Production bridge to R2's ``reloc_torch.transport`` (issue #146).

    R2 owns storage validation, allocation, stream ordering and completion.
    While that module is absent every preflight is the expected exclusion
    ``runtime_unavailable``: the original region runs and nothing launches.

    ``transfer_resources=AUTO`` lazily owns layout and typed resources; an explicit
    TransferResources is borrowed. None (the default) keeps per-call resources.
    Entry lifetimes never control this owner's lifetime; call close() to drain
    an owned cache. Typed scratch has its own explicit limits and one exclusive
    context; typed placement still uses the separate dispatch policy.
    """

    REQUEST_ATTRIBUTES = ("bindings", "destination")

    MODULE = f"{__package__}.transport"

    def __init__(self, *, transfer_resources=None, transfer_options=None):
        self._transfer_options = _transfer_configuration(transfer_resources, transfer_options)
        self._owns_resources = transfer_resources is AUTO
        self._resources = None if self._owns_resources else transfer_resources
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
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

    def _check_process(self):
        # Inherited Python locks are unsafe too, even before AUTO materializes.
        if os.getpid() != self._pid:
            raise RuntimeError("process_mismatch: transport adapter belongs to another process")

    def _require_open(self):
        self._check_process()
        if self._closed:
            raise RuntimeError("resources_closed: TransportAdapter is closed")

    def resource_stats(self):
        """Native counters, or None while disabled/unmaterialized; never allocate."""
        self._check_process()
        with self._lock:
            resources = self._resources
        return None if resources is None else resources.stats()

    def close(self):
        self._check_process()
        with self._lock:
            self._closed = True
            resources = self._resources if self._owns_resources else None
        # Native close can wait with the GIL released. No Python owner lock is
        # held, so in-flight work and stats can make progress while it drains.
        if resources is not None:
            resources.close()

    @property
    def available(self):
        return self._module is not None

    @property
    def capability_identity(self):
        if not self.available:
            return "reloc_torch.transport/unavailable"
        return f"reloc_torch.transport/{getattr(self._module, 'CAPABILITY_IDENTITY', '0')}"

    def preflight(self, compiled, src, device, *, non_blocking=False, parameters=None):
        self._require_open()
        import torch

        if not self.available:
            raise UnsupportedRecipe("runtime_unavailable", self.unavailable_reason)
        if getattr(compiled, "typed", False):
            # C4: typed recipes run through R3's dispatch bridge; parameters
            # are CPU tensors bound by declared name and snapshotted there.
            pool = self._transfer_options.get("gather_pool")
            threads = (pool.threads if pool is not None else
                       self._transfer_options.get("gather_threads", 8))
            if threads == 0:
                threads = os.cpu_count() or 1
            request = self._dispatch().prepare_typed_transfer(
                compiled, src, device, parameters=dict(parameters or {}), threads=threads,
            )
        else:
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

    def preflight_stacked(self, compiled, sources, device, *, non_blocking=False):
        """Stacked twin of preflight (torch.stack): one request over every input."""
        self._require_open()
        import torch

        if not self.available:
            raise UnsupportedRecipe("runtime_unavailable", self.unavailable_reason)
        if getattr(compiled, "typed", False):
            raise UnsupportedRecipe("typed_transform_unavailable", "stacked recipes are layout-only")
        sources = tuple(sources)
        request = self._module.prepare_stacked_transfer(compiled, sources, device, non_blocking=non_blocking)
        destination = request.destination
        if not isinstance(destination, ConcreteDescriptor):
            destination = ConcreteDescriptor(
                tuple(destination.shape), tuple(destination.strides),
                compat.dtype_name(destination.dtype), torch.device(destination.device),
            )
        # The transport's request rechecks every input right before execution.
        return PreparedCall(
            compiled, sources[0], dict(request.bindings), getattr(request, "bound", None),
            destination, non_blocking, request=request, sources=sources, request_rechecks=True,
        )

    def execute(self, call):
        self._check_process()
        typed = getattr(call.compiled, "typed", False)
        with self._lock:
            self._require_open()
            if self._owns_resources and self._resources is None:
                self._resources = TransferResources()
            resources = self._resources
        if typed:
            result = self._dispatch().execute_typed_transfer(call.request, resources=resources, **self._transfer_options)
            call.report = result.report
            call.staging = getattr(call.request, "staging", ())
            return result.tensor
        result = self._module.execute_transfer(call.request, resources=resources, **self._transfer_options)
        call.staging = getattr(call.request, "staging", ())
        return result

    def _dispatch(self):
        return importlib.import_module(f"{__package__}.dispatch")


def _derived_symbols(entry, src, axis_offset=0):
    values = []
    for name in entry.compiled.symbols:
        source = next((s for s in entry.compiled.symbol_sources if s.name == name), None)
        axis = None if source is None else source.axis - axis_offset
        if axis is None or not 0 <= axis < src.dim():
            return None
        values.append(int(src.shape[axis]))
    return values


def _bind_guarded(entry, device, bind, *args):
    """Bind (``bind(entry.compiled, *args)``), derive the destination and
    check extent guards: the steps that must all succeed, in order, before a
    symbol check is meaningful. Shared by both call shapes; the binder and
    its arguments are passed rather than wrapped, so the single-source hot
    path creates no closure. Raises ``UnsupportedRecipe`` with a normalized
    ``.reason`` for every expected rejection, including a bare
    ``KeyError``/``GuardError`` an extent guard's own expression evaluation
    can still raise directly (``bind`` and ``destination_descriptor``
    already normalize their own).
    """
    try:
        bindings = bind(entry.compiled, *args)
        destination = destination_descriptor(entry.compiled, bindings, device)
        for guard in entry.extent_guards:
            if expression(guard).evaluate(bindings, checked=True) < 2:
                raise UnsupportedRecipe("singleton_extent", f"{guard} binds below two")
    except UnsupportedRecipe:
        raise
    except (KeyError, GuardError) as error:
        raise UnsupportedRecipe(getattr(error, "reason", "missing_symbol"), str(error)) from error
    return bindings, destination


def _reconcile(entry, bindings, destination, symbols, declared, descriptor_label):
    """The supplied-symbol check (a fallback reason) and the declared/compiled
    metadata check (always an error, never a fallback): shared by both call
    shapes once the caller holds the exact binding and destination (and, for
    a stacked call, has already applied its size gate). Returns a fallback
    reason, or ``None`` when the caller should proceed to preflight.
    """
    if symbols is not None:
        expected = [bindings[name] for name in entry.compiled.symbols]
        if [int(value) for value in symbols] != expected:
            return "symbol_mismatch"
    if declared is not None and (
        tuple(declared.shape) != destination.shape or tuple(declared.strides) != destination.strides
    ):
        raise RuntimeError(
            f"declared output metadata {tuple(declared.shape)}/{tuple(declared.strides)} "
            f"does not match the compiled {descriptor_label} descriptor "
            f"{destination.shape}/{destination.strides}"
        )
    return None


_SINGLE_COUNTERS = ("runtime_executions",)
_STACKED_COUNTERS = ("runtime_executions", "stacked_executions")


def _launch(entry, call, bindings, destination, counters, label, verify, *subject):
    """Once preflight has returned a call: the adapter-agreement check,
    recheck/consume, the execution counters, the ``ExecutionError`` wrap
    around the actual dispatch (never replayed), and the diagnostics
    recorded after a launch that completed. ``label`` prefixes the direction
    in the error message (``""`` or ``"stacked "``). ``verify(result,
    *subject)`` applies the caller's own result and aliasing checks against
    the promised metadata and returns the value handed back to the caller.
    """
    if call.bindings != bindings or tuple(call.destination.shape) != destination.shape:
        raise RuntimeError("runtime adapter disagreed with the frontend binding")
    call.recheck()
    call.consume()
    for counter in counters:
        entry.diagnostics.increment(counter)
    try:
        result = entry.runtime.execute(call)
    except Exception as error:
        raise ExecutionError(
            f"reloc_torch {label}{entry.direction} execution failed for {entry.describe()}: {error}",
            direction=entry.direction,
            handle=entry.handle,
        ) from error
    if getattr(call, "report", None) is not None:
        entry.diagnostics.record_dispatch(call.report)
    entry.diagnostics.record_staging(getattr(call, "staging", ()))
    return verify(result, *subject)


def _fallback(entry, src, symbols, reason, promised=None, parameters=()):
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
    result = entry.fallback(src, *symbols, parameters=tuple(parameters))
    return result if promised is None else verify_result(result, src, promised)


def execute_or_fallback(entry, src, symbols, device, *, non_blocking=False, declared=None, parameters=()):
    """Preflight, then either dispatch once or run the original region once.

    ``symbols`` is the ordered list of concrete values supplied by the op (or
    ``None`` when the caller supplies none); the frontend guards bind the exact
    name-to-value map from real source metadata first and reconcile it with the
    supplied symbols, so every expected exclusion has a stable reason before the
    adapter is consulted. ``declared`` optionally carries the output metadata
    promised to the graph; a mismatch with the compiled descriptor is an error,
    never a fallback. ``parameters`` are the typed region's runtime parameter
    tensors in recipe declaration order (C4); they reach the adapter by
    declared name and the original region positionally. Every result handed
    back, from the adapter or from the original region once the promised
    metadata is known, passes ``verify_result``.
    """
    import torch

    if entry.closed:
        raise RuntimeError("execution entry is closed")
    device = torch.device(device)
    parameters = tuple(parameters)
    typed = getattr(entry.compiled, "typed", False)
    names = tuple(p.name for p in entry.compiled.parameters) if typed else ()
    if len(parameters) != len(names):
        raise RuntimeError(
            f"expected {len(names)} runtime parameters {names} for {entry.describe()}, got {len(parameters)}"
        )
    with suspend_interception():
        if non_blocking:
            return _fallback(entry, src, symbols, "nonblocking_unavailable", declared, parameters)
        reason = source_reason(src)
        if reason is not None:
            return _fallback(entry, src, symbols, reason, declared, parameters)
        try:
            bindings, destination = _bind_guarded(entry, device, bind_symbols, src)
        except UnsupportedRecipe as error:
            return _fallback(entry, src, symbols, error.reason, declared, parameters)
        promised = destination if declared is None else declared
        reason = _reconcile(entry, bindings, destination, symbols, declared, entry.direction)
        if reason is not None:
            return _fallback(entry, src, symbols, reason, promised, parameters)
        options = {"non_blocking": non_blocking}
        if typed:
            options["parameters"] = dict(zip(names, parameters))
        try:
            with _counting_binds(entry.diagnostics), _validated_binding(entry.compiled, src, bindings):
                call = entry.runtime.preflight(entry.compiled, src, device, **options)
        except UnsupportedRecipe as error:
            return _fallback(entry, src, symbols, error.reason, promised, parameters)
        return _launch(entry, call, bindings, destination, _SINGLE_COUNTERS, "", verify_result, src, promised)


def _verify_stacked(result, sources, promised):
    import torch

    if isinstance(result, torch.Tensor) and any(compat.tensors_alias(result, s) for s in sources[1:]):
        raise RuntimeError("reloc_torch result must not alias its input")
    return verify_result(result, sources[0], promised)


def _fallback_stacked(entry, sources, symbols, reason, promised=None):
    """_fallback for a stacked call: the original region takes every input."""
    if symbols is None:
        symbols = ()
        if entry.symbolic_bindings:
            # Logical axis k of the stacked source is input axis k - 1.
            symbols = _derived_symbols(entry, sources[0], axis_offset=1)
            if symbols is None:
                raise RuntimeError(
                    "cannot evaluate the original region's scalar placeholders for "
                    f"stacked inputs of rank {sources[0].dim()} (recipe symbols {entry.compiled.symbols})"
                )
    entry.diagnostics.record_fallback(reason)
    result = entry.fallback_stacked(sources, *symbols)
    return result if promised is None else _verify_stacked(result, sources, promised)


def execute_stacked_or_fallback(entry, sources, symbols, device, *, declared=None):
    """The stacked twin of execute_or_fallback (torch.stack fused into an H2D
    transfer). The size gate comes first and reads only the input count and
    input 0's size, so a call below it runs the original region before any
    per-input work. Above it every input passes the shared guards
    (``check_stacked_sources``: the full guard on input 0, a lean check
    against it on the others); equality of dtype, device and shape, the
    logical binding and the adapter preflight all run before any allocation
    or launch, and every expected rejection runs the original region exactly
    once. The adapter's preflight is offered the checked inputs, their
    storage snapshots and the binding, so it need not guard or bind again."""
    import torch

    if entry.closed:
        raise RuntimeError("execution entry is closed")
    device = torch.device(device)
    sources = tuple(sources)
    count = entry.stack_inputs
    if count < 1 or len(sources) != count:
        raise RuntimeError(
            f"expected {count} stacked sources for {entry.describe()}, got {len(sources)}"
        )
    with suspend_interception():
        first = sources[0]
        if stack_below_threshold(count, first.numel(), first.element_size(), entry.min_stack_bytes):
            return _fallback_stacked(entry, sources, symbols, "below_stack_threshold", declared)
        reason, snapshots, uniform = check_stacked_sources(sources)
        if reason is not None:
            return _fallback_stacked(entry, sources, symbols, reason, declared)
        if not uniform:
            # Uniform inputs share input 0's dtype, device and shape.
            for reason, differs in (
                ("stack_dtype_mismatch", lambda s: s.dtype != first.dtype),
                ("unsupported_device", lambda s: s.device != first.device),
                ("stack_shape_mismatch", lambda s: s.shape != first.shape),
            ):
                if any(differs(s) for s in sources):
                    return _fallback_stacked(entry, sources, symbols, reason, declared)
        try:
            bindings, destination = _bind_guarded(entry, device, bind_stacked_symbols, sources, uniform)
        except UnsupportedRecipe as error:
            return _fallback_stacked(entry, sources, symbols, error.reason, declared)
        promised = destination if declared is None else declared
        reason = _reconcile(entry, bindings, destination, symbols, declared, "stacked")
        if reason is not None:
            return _fallback_stacked(entry, sources, symbols, reason, promised)
        try:
            preflight = stacked_preflight(entry.runtime)
            with _counting_binds(entry.diagnostics), _validated_stacked_binding(
                entry.compiled, sources, snapshots, uniform, bindings,
            ):
                call = preflight(entry.compiled, sources, device)
        except UnsupportedRecipe as error:
            return _fallback_stacked(entry, sources, symbols, error.reason, promised)
        return _launch(
            entry, call, bindings, destination, _STACKED_COUNTERS, "stacked ", _verify_stacked, sources, promised,
        )


__all__ = (
    "ConcreteDescriptor",
    "ExecutionEntry",
    "ExecutionError",
    "PreparedCall",
    "RuntimeAdapter",
    "TransportAdapter",
    "bind_plan",
    "bind_stacked_symbols",
    "bind_symbols",
    "check_stacked_sources",
    "destination_descriptor",
    "execute_or_fallback",
    "execute_stacked_or_fallback",
    "interception_suspended",
    "load_plan",
    "prepare_host_call",
    "prepare_stacked_host_call",
    "source_reason",
    "stack_below_threshold",
    "stacked_preflight",
    "suspend_interception",
    "validated_stacked_inputs",
    "verify_result",
)
