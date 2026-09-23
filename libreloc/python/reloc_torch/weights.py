"""Explicit inference weight preparation with live-slot resolution (T4 Task 2).

``prepare_weights(module, recipes, *, backend, stable=False)`` returns a
``PreparedWeights`` owner. ``get(name, device=...)`` resolves the fully
qualified parameter/buffer slot on the live module at every call, relocates it
through the T3 backend's adapter and returns a fresh tensor on ``device``. It
never rewrites module slots, changes ``Parameter`` identity or alters
``requires_grad``.

``stable=False`` always reads current values and caches plans only.
``stable=True`` may retain the transformed host layout after validating
freshness conservatively: live object identity, descriptor fields, storage
identity, the mutation version when available, and a byte snapshot compared on
every reuse (a version counter alone misses ``.data`` or NumPy writes; bytes
also keep unchanged NaNs from counting as mutations). Stable data is rebuilt
whenever any of these differ, and a fresh destination is returned on every
``get`` so a caller's output mutation cannot poison the cache.

With grad mode enabled and a gradient-requiring source, the supplied recipe is
replayed through PyTorch so autograd is preserved; nothing is detached silently.
"""

from __future__ import annotations

import struct
import threading
import weakref

from . import compat
from .artifact import UnsupportedRecipe
from .recipe import Pad, Reshape, Transpose
from .runtime import ExecutionEntry, bind_plan, bind_symbols, execute_or_fallback, source_reason
from .symbolic import GuardError, expression, symbol_sources


def _fill_value(fill):
    if fill.dtype == "float32":
        return struct.unpack("<f", fill.bits.to_bytes(4, "little"))[0]
    if fill.dtype == "float16":
        return struct.unpack("<e", fill.bits.to_bytes(2, "little"))[0]
    return fill.bits - 256 if fill.bits >= 128 else fill.bits


def replay(recipe, tensor):
    """Apply a layout recipe with ordinary PyTorch operations (the original path)."""
    import torch

    bindings = {}
    for source in symbol_sources(recipe.source.shape):
        bindings[source.name] = int(tensor.shape[source.axis])
    result = tensor
    for operation in recipe.operations:
        if isinstance(operation, Transpose):
            result = result.permute(*operation.perm)
        elif isinstance(operation, Reshape):
            result = result.reshape([expression(dim).evaluate(bindings) for dim in operation.shape])
        elif isinstance(operation, Pad):
            rank = result.dim()
            widths = [0] * (2 * (rank - operation.axis))
            widths[2 * (rank - 1 - operation.axis)] = expression(operation.lo).evaluate(bindings)
            widths[2 * (rank - 1 - operation.axis) + 1] = expression(operation.hi).evaluate(bindings)
            result = torch.nn.functional.pad(result, widths, value=_fill_value(operation.fill))
        else:
            raise UnsupportedRecipe("unsupported_operator", f"cannot replay {operation!r}")
    return result.contiguous()


def _identity_recipe(descriptor, direction):
    from .backend import identity_recipe

    return identity_recipe(len(descriptor.shape), descriptor.dtype, direction)


def _byte_snapshot(tensor):
    """Owned copy of the tensor's bytes: the freshness witness must not alias the slot."""
    import torch

    return tensor.contiguous().view(torch.uint8).reshape(-1).clone()


class _StableState:
    __slots__ = ("fingerprint", "saved_bytes", "prepared")

    def __init__(self, fingerprint, saved_bytes, prepared):
        self.fingerprint = fingerprint
        self.saved_bytes = saved_bytes
        self.prepared = prepared


class PreparedWeights:
    def __init__(self, module, recipes, *, backend, stable=False):
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("prepare_weights expects a torch.nn.Module")
        compat.check_version()
        self._module = weakref.ref(module)
        self._recipes = dict(recipes)
        self._backend = backend
        self._stable = bool(stable)
        self._lock = threading.RLock()
        self._closed = False
        self._compiled = {}
        self._states = {}

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Release retained host data and reject later use; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._states.clear()
            self._compiled.clear()

    def invalidate(self, name=None):
        """Drop stable host data for one slot (or all); the next get rebuilds."""
        with self._lock:
            self._require_open()
            names = list(self._states) if name is None else [name]
            for key in names:
                if self._states.pop(key, None) is not None:
                    self._backend.diagnostics.increment("weight_invalidations")

    def _require_open(self):
        if self._closed:
            raise RuntimeError("PreparedWeights is closed")

    # ------------------------------------------------------------- resolution
    def _slot(self, name):
        module = self._module()
        if module is None:
            raise RuntimeError("the prepared module was garbage collected")
        try:
            return module.get_parameter(name)
        except AttributeError:
            pass
        try:
            return module.get_buffer(name)
        except AttributeError as error:
            raise RuntimeError(f"module slot {name!r} no longer exists") from error

    def _compile(self, name, recipe):
        compiled = self._compiled.get(name)
        if compiled is None:
            compiled = self._backend.compile_recipe(recipe)
            self._compiled[name] = compiled
        return compiled

    @staticmethod
    def _fingerprint(source):
        base, capacity, offset = compat.storage_span(source)
        try:
            version = int(source._version)
        except Exception:
            version = None
        return (
            id(source),
            tuple(source.shape),
            tuple(source.stride()),
            offset,
            str(source.dtype),
            str(source.device),
            base,
            capacity,
            version,
        )

    # -------------------------------------------------------------------- get
    def get(self, name, *, device):
        """Relocate the live slot `name` onto `device`; always a fresh tensor."""
        import torch

        with self._lock:
            self._require_open()
            if name not in self._recipes:
                raise KeyError(f"no recipe was supplied for slot {name!r}")
            recipe = self._recipes[name]
            device = torch.device(device)
            source = self._slot(name)
            if source.requires_grad and torch.is_grad_enabled():
                # Inference API used with autograd live: keep PyTorch semantics
                # and record it instead of detaching behind the caller's back.
                self._backend.diagnostics.record_fallback("requires_grad")
                return replay(recipe, source).to(device, copy=True)
            src = source.detach()
            try:
                compiled = self._compile(name, recipe)
            except UnsupportedRecipe as error:
                self._backend.diagnostics.record_exclusion(error.reason)
                return replay(recipe, src).to(device, copy=True)
            if not self._stable:
                return self._transfer(compiled, recipe, src, device, name)
            return self._stable_get(name, recipe, compiled, source, src, device)

    def _entry(self, compiled, original, label):
        entry = ExecutionEntry(
            compiled=compiled,
            original=original,
            runtime=self._backend.runtime,
            diagnostics=self._backend.diagnostics,
        )
        entry.handle = label
        return entry

    def _transfer(self, compiled, recipe, src, device, name):
        entry = self._entry(
            compiled,
            lambda tensor, *symbols: replay(recipe, tensor).to(device, copy=True),
            f"weight/{name}",
        )
        return execute_or_fallback(entry, src, None, device)

    def _stable_get(self, name, recipe, compiled, source, src, device):
        import torch

        state = self._states.get(name)
        fingerprint = self._fingerprint(source)
        # Data caching only for plain dense CPU sources the guards admit; other
        # sources (external storage, devices) take the current-value path.
        cacheable = src.device.type == "cpu" and source_reason(src) is None
        current = _byte_snapshot(src) if cacheable else None
        if state is not None:
            unchanged = (
                cacheable
                and state.fingerprint == fingerprint
                and torch.equal(current, state.saved_bytes)
            )
            if not unchanged:
                self._states.pop(name, None)
                self._backend.diagnostics.increment("weight_invalidations")
                state = None
        if not cacheable:
            return self._transfer(compiled, recipe, src, device, name)
        if state is None:
            prepared = self._materialize(compiled, recipe, src)
            if prepared is None:
                return replay(recipe, src).to(device, copy=True)
            # Publish only if the source did not change while preparing: the
            # snapshot taken above is owned, so this compares the live bytes
            # against a copy, not against themselves.
            if self._fingerprint(self._slot(name)) != fingerprint or not torch.equal(
                _byte_snapshot(src), current
            ):
                return replay(recipe, self._slot(name).detach()).to(device, copy=True)
            state = _StableState(fingerprint, current, prepared)
            self._states[name] = state
            self._backend.diagnostics.increment("weight_preparations")
        # The prepared host tensor already has the destination layout. A CPU
        # target needs a copy, not a transfer; a CUDA target moves it with an
        # identity artifact derived from that descriptor.
        if device.type != "cuda":
            return state.prepared.to(device, copy=True)
        identity = self._backend.compile_recipe(_identity_recipe(compiled.logical_destination, "h2d"))
        entry = self._entry(
            identity, lambda tensor, *symbols: tensor.to(device, copy=True), f"weight/{name}/prepared"
        )
        return execute_or_fallback(entry, state.prepared, None, device)

    def _materialize(self, compiled, recipe, src):
        """Owned host tensor in the destination layout via the CPU relocation executor."""
        import torch
        import pyreloc
        from pyreloc.torch_interop import as_ptr

        try:
            bindings = bind_symbols(compiled, src)
            bound = bind_plan(compiled, bindings)
        except UnsupportedRecipe as error:
            self._backend.diagnostics.record_fallback(error.reason)
            return None
        shape = tuple(expression(dim).evaluate(bindings) for dim in compiled.logical_destination.shape)
        prepared = torch.empty(shape, dtype=src.dtype)
        contiguous = src.contiguous()
        self._backend.diagnostics.increment("symbol_binds")
        pyreloc.relocate(bound, *as_ptr(contiguous), *as_ptr(prepared))
        return prepared


def prepare_weights(module, recipes, *, backend, stable=False):
    """Own explicit inference relocation of the named module parameters/buffers."""
    return PreparedWeights(module, recipes, backend=backend, stable=stable)


__all__ = ("PreparedWeights", "prepare_weights", "replay")
