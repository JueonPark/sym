"""Prepared transfer programs with minimal per-call work.

The ``torch.compile``/custom-op path validates every call from scratch: it
re-derives symbol bindings, snapshots parameters, rechecks storage and
rebuilds descriptors. That costs several hundred microseconds per transfer,
which dominates small KV-cache moves and adds up over weight groups. This
module binds a recipe to one source *shape signature* once (symbols,
parameters, native dispatch template and destination descriptor) and then
executes each call through a single native entry point.

What stays per call: the source tensor must match the prepared signature
(shape, strides, dtype, device type) and the native validator still proves
that the declared views fit the plan before anything launches. Parameters
(scales, zero points) are snapshotted when the program is prepared; a caller
that changes them prepares a new program. Values are never cached: every
call reads the live source and writes a fresh output.

``FastTransfers`` owns the native resources (bounded scratch, pinned staging
rings, worker pool, private streams) and exposes three program kinds:

* ``typed`` – a value-transforming recipe (cast/dequantize/...) at a fixed
  shape, executed blocking through the chunked CPU/DMA pipeline or a GPU row;
* ``layout`` – a layout-only recipe at a fixed shape, executed in groups
  (optionally relocating f32 tensors on the device);
* ``indexed`` – an ``IndexSelect`` recipe whose indices are bound per call
  straight from the index tensor's memory.

Groups (``submit``) enqueue several programs as one native submission with
one completion barrier and return a ``FastCompletion`` whose ``wait`` /
``wait_stream`` hand back the outputs. ``KVCacheSwap`` and ``FastWeights``
are thin conveniences over groups for the offload workloads.

With a calibration (``FastTransfers(calibration=...)`` or
``$RELOC_CALIBRATION``) every call prices the compiler's cost advice
(``reloc_torch.advice``) and takes the cheapest admissible path: the direct
kernel path (the producer kernel into a pinned wire, one copy) or the
dispatcher. Reports and ``stats()["advice"]`` record the reason.
"""
from __future__ import annotations

from collections import deque
import math
import os
from threading import RLock
from weakref import WeakSet

import pyreloc

from . import advice as _advice
from . import compat
from .artifact import UnsupportedRecipe
from .dispatch import _parameter_snapshot
from .resources import TransferResources
from .runtime import destination_descriptor
from .symbolic import GuardError
from .transport import _code, _storage_view

# libtorch-backed request builders (RELOC_ENABLE_TORCH_EXT), loaded lazily by
# FastTransfers after Torch itself is imported; None when the build lacks them.
_native_views = None
_native_views_checked = False


def _load_native_views():
    global _native_views, _native_views_checked
    if not _native_views_checked:
        _native_views_checked = True
        import torch  # noqa: F401 - the helper links against this interpreter's Torch
        try:
            from pyreloc import _pyreloc_torch
        except ImportError:  # pragma: no cover - depends on the build configuration
            _pyreloc_torch = None
        _native_views = _pyreloc_torch
    return _native_views

_DIRECTIONS = {"h2d": ("cpu", "cuda"), "d2h": ("cuda", "cpu")}


def _signature(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device.type)


def _check_plain(tensor, what):
    if not compat.is_plain_tensor_or_parameter(tensor) or tensor.requires_grad:
        raise ValueError(f"{what} must be a plain inference tensor")


class _Program:
    """Shared per-shape state: signature, destination descriptor, direction."""

    __slots__ = ("owner", "compiled", "bindings", "signature", "signature_dtype", "direction", "destination",
                 "_view_kind", "_view_device", "_out_dtype", "_out_device", "handle", "_registration",
                 "_op_args", "resolved", "__weakref__")

    def _init_common(self, owner, compiled, source, bindings):
        import torch
        self.owner, self.compiled, self.bindings = owner, compiled, bindings
        self.signature = _signature(source)
        self.signature_dtype = compat.dtype_name(source.dtype)
        self.direction = compiled.recipe.direction
        expected = _DIRECTIONS.get(self.direction)
        if expected is None or source.device.type != expected[0]:
            raise UnsupportedRecipe("direction_mismatch",
                                    f"{self.direction} recipe cannot start from {source.device}")
        if self.direction == "h2d":
            if source.device.type != "cpu":
                raise UnsupportedRecipe("direction_mismatch", "h2d source must be a CPU tensor")
            self._view_kind, self._view_device = "host", -1
            target = owner.device
        else:
            if source.device != owner.device:
                raise UnsupportedRecipe("device_mismatch", f"d2h source {source.device} is not {owner.device}")
            self._view_kind, self._view_device = "cuda", owner.device.index
            target = torch.device("cpu")
        self.destination = destination_descriptor(compiled, bindings, target)
        self._out_dtype = getattr(torch, self.destination.dtype)
        self._out_device = self.destination.device
        self.handle = self._registration = self._op_args = None
        # The compiler's cost advice under this shape (None without advice).
        advice = getattr(compiled, "advice", None)
        self.resolved = None if advice is None else _advice.resolve(advice, bindings, owner.gather_threads)

    @property
    def closed(self):
        return self.owner._closed

    def _register_op(self):
        # Registered at preparation (never inside a Dynamo trace): the op
        # arguments are the destination descriptor, constants to the tracer.
        from . import fast_ops
        self._registration = fast_ops.register(self)
        d = self.destination
        self._op_args = (self.handle, tuple(d.shape), tuple(d.strides), self._out_device, self._out_dtype)

    def check(self, source):
        if _signature(source) != self.signature:
            raise ValueError(f"source {_signature(source)} does not match the prepared program {self.signature}")

    def source_view(self, source):
        return _storage_view(source, self._view_kind, self._view_device)

    def allocate(self, *, pin=True):
        import torch
        d = self.destination
        if self._out_device.type == "cuda":
            return torch.empty_strided(d.shape, d.strides, dtype=self._out_dtype, device=self._out_device)
        return torch.empty_strided(d.shape, d.strides, dtype=self._out_dtype, pin_memory=pin)

    def output_view(self, out):
        if out.is_cuda:
            return _storage_view(out, "cuda", out.device.index)
        return _storage_view(out, "host", -1)


class TypedProgram(_Program):
    """A typed recipe bound to one shape and immutable parameter values."""

    __slots__ = ("bound", "template", "selection", "parameter_bytes", "_single")

    def __init__(self, owner, compiled, source, *, parameters=None, implementation="", policy="auto",
                 calibration=None):
        if not getattr(compiled, "typed", False) or compiled.indexed:
            raise UnsupportedRecipe("not_typed_recipe", "typed programs need a value transform")
        _check_plain(source, "source")
        try:
            bindings = compiled.bind_values(source)
        except GuardError as error:
            raise UnsupportedRecipe(error.reason, str(error)) from error
        self._init_common(owner, compiled, source, bindings)
        declared = compiled.parameter_extents(bindings)
        parameters = dict(parameters or {})
        if set(declared) != set(parameters):
            raise UnsupportedRecipe("parameter_binding",
                                    f"declared parameters {sorted(declared)}, got {sorted(parameters)}")
        snapshots = {name: _parameter_snapshot(name, parameters[name], declared[name]) for name in declared}
        self.parameter_bytes = sum(len(s[2]) for s in snapshots.values())
        try:
            self.bound = pyreloc.bind_typed(compiled.decoded_plan, bindings, snapshots)
            self.template = pyreloc.prepare_dispatch_template(
                self.bound, self.direction, "cuda", policy=policy, calibration=calibration,
                threads=owner.gather_threads, implementation=implementation)
        except pyreloc.BindError as error:
            raise UnsupportedRecipe("bind_error", str(error)) from error
        except pyreloc.TransferError as error:
            raise UnsupportedRecipe(_code(error), str(error)) from error
        self.selection = dict(self.template.selection)
        self._single = None
        if self.direction == "h2d":
            self._register_op()

    def _single_call(self):
        """The advice of a single blocking call: a typed program has no
        direct executor of its own (only weight groups do), so the direct
        path is not admissible here."""
        if self._single is None and self.resolved is not None:
            self._single = _advice.without_direct(self.resolved)
        return self._single

    def group_entry(self, source, out):
        return (self.template, self.source_view(source), self.output_view(out), self.direction)

    def tensor_entry(self, source, out):
        return (self.template, source, out, self.direction, self.signature_dtype)

    def op(self, source):
        """``execute(source)`` as a functional custom op (traceable by Dynamo, opaque to Inductor)."""
        from .fast_ops import TYPED_OP
        if self._op_args is None:
            raise RuntimeError("only H2D typed programs are available as ops")
        return TYPED_OP(source, *self._op_args)

    def execute(self, source, *, out=None, pipeline=True, gather_threads=None):
        """Blocking single transfer through the prepared row (chunked CPU/DMA pipeline for CPU rows)."""
        owner = self.owner
        owner._check_open()
        self.check(source)
        decision = owner._advise(self._single_call()) if pipeline else None
        if out is None:
            out = self.allocate()
        try:
            if _native_views is not None:
                native = _native_views.request_from_tensors(self.template, source, out, self.signature_dtype)
            else:
                native = pyreloc.prepare_dispatch_from_template(self.template, self.source_view(source),
                                                                self.output_view(out))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"typed dispatch rejected at execution: {error}") from error
        cuda_device = owner.device
        try:
            report = pyreloc.execute_dispatch(
                native, caller_stream=compat.cuda_stream_handle(cuda_device),
                n_buffers=owner.n_buffers, n_streams=1,
                gather_threads=owner.gather_threads if gather_threads is None else gather_threads,
                pinning=owner.pinning, direct_dense_upload=True, pipeline=pipeline,
                chunk_size=owner.chunk_size, resources=owner.native, owners=(source, out))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"{self.direction} typed dispatch failed: {error}") from error
        owner._record(_decided(report, decision))
        return out


class LayoutProgram(_Program):
    """A layout-only recipe bound to one shape; executed inside groups."""

    __slots__ = ("bound",)

    def __init__(self, owner, compiled, source):
        if getattr(compiled, "typed", False) or compiled.indexed:
            raise UnsupportedRecipe("not_layout_recipe", "layout programs take layout-only recipes")
        _check_plain(source, "source")
        try:
            bindings = compiled.bind_values(source)
        except GuardError as error:
            raise UnsupportedRecipe(error.reason, str(error)) from error
        self._init_common(owner, compiled, source, bindings)
        try:
            self.bound = compiled.bind_layout(bindings)
        except pyreloc.BindError as error:
            raise UnsupportedRecipe("bind_error", str(error)) from error
        if self.bound.no_copy:
            raise UnsupportedRecipe("no_copy_layout", "a no-copy layout has nothing to transfer")

    def group_entry(self, source, out):
        return (self.bound, self.source_view(source), self.output_view(out), self.direction)

    def tensor_entry(self, source, out):
        return (self.bound, source, out, self.direction, self.signature_dtype)

    def execute(self, source, *, out=None, device_layout=True):
        """Blocking single transfer as a one-item group."""
        completion = self.owner.submit([(self, source)], outputs=None if out is None else [out],
                                       device_layout=device_layout, gather_threads=1)
        return completion.wait()[0]


class _OneShot:
    """A per-call CUDA template (indexed or pooled producer) as a group member."""

    __slots__ = ("template", "direction", "signature_dtype", "destination", "_view")

    def __init__(self, template, source_view, destination, dtype):
        self.template, self.direction = template, "h2d"
        self.signature_dtype, self.destination, self._view = dtype, destination, source_view

    def check(self, source):
        pass  # the native validator checks the fresh program against the views

    def group_entry(self, source, out):
        return (self.template, self._view, _storage_view(out, "cuda", out.device.index), "h2d")

    def tensor_entry(self, source, out):
        return (self.template, source, out, "h2d", self.signature_dtype)


class IndexedProgram:
    """An IndexSelect (+ optional cast) H2D recipe; indices bind per call."""

    __slots__ = ("owner", "compiled", "_specs", "_resolved", "handle", "_registration", "_op_args", "__weakref__")

    def __init__(self, owner, compiled):
        import torch
        from .symbolic import Const
        if not compiled.indexed or compiled.recipe.direction != "h2d":
            raise UnsupportedRecipe("unsupported_index_select", "expected an H2D index_select recipe")
        self.owner, self.compiled = owner, compiled
        self._specs, self._resolved = {}, {}
        # The op needs the output metadata without binding: rows follow the
        # index length, the remaining extents are the recipe's constants.
        destination = compiled.recipe.destination
        columns = tuple(e.value if isinstance(e, Const) else None for e in destination.shape[1:])
        self.handle = self._registration = None
        from . import fast_ops
        self._registration = fast_ops.register(self)
        self._op_args = (None if None in columns else columns, self.handle, owner.device,
                         getattr(torch, destination.dtype))

    @property
    def closed(self):
        return self.owner._closed

    def op(self, source, index):
        """``execute(source, index)`` as a functional custom op; the output has ``len(index)`` rows."""
        from .fast_ops import INDEXED_OP
        columns, handle, device, dtype = self._op_args
        if columns is None:
            raise RuntimeError("index_select ops need constant trailing extents")
        shape = (index.shape[0],) + columns
        strides, stride = [], 1
        for extent in reversed(shape):
            strides.append(stride)
            stride = stride * extent
        return INDEXED_OP(source, index, handle, shape, tuple(reversed(strides)), device, dtype)

    def _prepare(self, source, index, out):
        """Bind the live indices natively; returns (program, source view, out)."""
        import torch
        owner = self.owner
        owner._check_open()
        _check_plain(source, "source")
        _check_plain(index, "index")
        if (source.device.type != "cpu" or not source.is_contiguous() or index.device.type != "cpu"
                or index.dim() != 1 or not index.is_contiguous()
                or index.dtype not in (torch.int32, torch.int64)):
            raise ValueError("index_select needs a dense CPU source and a contiguous CPU int32/int64 index")
        key = (_signature(source), index.shape[0], index.dtype)
        cached = self._specs.get(key)
        if cached is None:
            try:
                bindings = self.compiled.bind_values(source, index)
            except GuardError as error:
                raise UnsupportedRecipe(error.reason, str(error)) from error
            destination = destination_descriptor(self.compiled, bindings, owner.device)
            cached = self._specs[key] = (bindings, destination, compat.dtype_name(index.dtype),
                                        getattr(torch, destination.dtype))
        bindings, destination, index_dtype, out_dtype = cached
        if out is None:
            out = torch.empty_strided(destination.shape, destination.strides, dtype=out_dtype, device=owner.device)
        src_view = _storage_view(source, "host", -1)
        try:
            bound = pyreloc.bind_indexed_from_pointer(
                self.compiled.decoded_plan, bindings, index_dtype, [int(index.shape[0])],
                index.data_ptr(), index.numel() * index.element_size())
            program = pyreloc.prepare_index_select_program(bound, src_view)
        except pyreloc.BindError as error:
            raise UnsupportedRecipe("index_out_of_range" if "index_out_of_range" in str(error) else "bind_error",
                                    str(error)) from error
        except pyreloc.TransferError as error:
            raise RuntimeError(f"index_select dispatch rejected: {error}") from error
        return program, src_view, out

    def _cached(self, source, index):
        """(bindings, destination, index dtype, out dtype) for this source
        signature and index length, preparing (and validating) on a miss."""
        key = (_signature(source), index.shape[0], index.dtype)
        cached = self._specs.get(key)
        if cached is None:
            self._prepare(source, index, None)
            cached = self._specs[key]
        return key, cached

    def _decide(self, key, bindings):
        resolved = self._resolved.get(key)
        if resolved is None and self.compiled.advice is not None:
            resolved = self._resolved[key] = _advice.resolve(self.compiled.advice, bindings, self.owner.gather_threads)
        return self.owner._advise(resolved)

    def _direct(self, source, index, cached, decision, stream, gather_threads):
        """The row gather into a wire, then one copy on `stream`. Returns
        (out, event, report), or the reason the ring refused a wire."""
        owner = self.owner
        bindings, destination, _, out_dtype = cached
        # The gather kernel writes [rows, row elements]; any destination rank
        # is that matrix viewed to its shape.
        rows = destination.shape[0]
        wire, refusal = owner._ring.acquire((rows, math.prod(destination.shape) // max(rows, 1)), out_dtype)
        if wire is None:
            return refusal
        threads = owner.gather_threads if gather_threads is None else gather_threads
        try:
            _native_views.gather_rows(self.compiled.decoded_plan, bindings, source, index, wire, threads)
        except pyreloc.BindError as error:
            owner._ring.cancel(wire)
            raise UnsupportedRecipe("index_out_of_range" if "index_out_of_range" in str(error) else "bind_error",
                                    str(error)) from error
        except BaseException:
            owner._ring.cancel(wire)
            raise
        return owner._ring.ship(wire, destination.shape, destination.strides, stream, decision)

    def execute(self, source, index, *, out=None, gather_threads=None, pipeline=True):
        """Blocking fused row gather/cast/H2D; the index tensor is read in place.

        With advice the call may take the direct path (gather into a pinned
        wire, one copy ordered on the current stream): the source and the
        index are read before this returns, the output is ordered on the
        caller's stream. ``pipeline=False`` gathers the whole wire first and
        copies it once through the dispatcher (no chunk overlap, no advice);
        it exists for attribution measurements."""
        import torch
        owner = self.owner
        owner._check_open()
        _check_plain(source, "source")
        _check_plain(index, "index")
        key, cached = self._cached(source, index)
        decision = self._decide(key, cached[0]) if pipeline else None
        if decision is not None and decision.path == "direct" and out is None:
            direct = self._direct(source, index, cached, decision, None, gather_threads)
            if not isinstance(direct, str):
                owner._record(direct[2])
                return direct[0]
            decision = _advice.Decision("pipelined", direct, decision.predicted)
        if _native_views is not None and out is None:
            # One native call binds the live indices, prepares the program over
            # the source and selects the row; validation is the same as below.
            bindings, destination, _, out_dtype = cached
            out = torch.empty_strided(destination.shape, destination.strides, dtype=out_dtype, device=owner.device)
            try:
                native = _native_views.index_select_request(self.compiled.decoded_plan, bindings, source, index, out,
                                                            owner.gather_threads)
            except pyreloc.BindError as error:
                raise UnsupportedRecipe("index_out_of_range" if "index_out_of_range" in str(error) else "bind_error",
                                        str(error)) from error
            except pyreloc.TransferError as error:
                raise RuntimeError(f"index_select dispatch rejected: {error}") from error
            return owner._execute_request(native, source, out, (index,), gather_threads, pipeline, decision)
        program, src_view, out = self._prepare(source, index, out)
        return self.owner._execute_program(program, src_view, source, out, (index,), gather_threads, pipeline,
                                           decision)

    def submit(self, source, index, *, out=None, gather_threads=None):
        """Asynchronous fused gather; wait_stream() on the completion yields the output."""
        owner = self.owner
        owner._check_open()
        _check_plain(source, "source")
        _check_plain(index, "index")
        key, cached = self._cached(source, index)
        decision = self._decide(key, cached[0])
        if decision.path == "direct" and out is None:
            owner._retire()
            direct = self._direct(source, index, cached, decision, owner._stream, gather_threads)
            if not isinstance(direct, str):
                return _DirectCompletion(owner, [direct[0]], direct[1], direct[2])
            decision = _advice.Decision("pipelined", direct, decision.predicted)
        if _native_views is not None and out is None:
            import torch
            bindings, destination, _, out_dtype = cached
            with torch.cuda.stream(owner._stream):
                out = torch.empty_strided(destination.shape, destination.strides, dtype=out_dtype,
                                          device=owner.device)
            try:
                native = _native_views.index_select_group(self.compiled.decoded_plan, bindings, source, index, out,
                                                          owner.gather_threads)
            except pyreloc.BindError as error:
                raise UnsupportedRecipe("index_out_of_range" if "index_out_of_range" in str(error) else "bind_error",
                                        str(error)) from error
            except pyreloc.TransferError as error:
                raise RuntimeError(f"index_select dispatch rejected: {error}") from error
            return owner._submit_native(native, (source,), (out,), (index,), gather_threads, host=False,
                                        chunked_members=True, decision=decision)
        program, src_view, out = self._prepare(source, index, out)
        return self.owner._submit_program(program, src_view, source, out, (index,), gather_threads,
                                          compat.dtype_name(source.dtype), decision=decision)


class PooledProgram:
    """Fused EmbeddingBag(mode="sum") pooling over several tables, cast, H2D.

    ``tables`` are dense float32 ``[rows_t, dim]`` CPU tensors (they may differ
    in rows). Each call takes per-table ``indices``/``offsets`` (int64, the
    EmbeddingBag convention without include_last_offset) for ``bags`` bags and
    returns a fresh ``[bags, tables, dim]`` CUDA tensor equal to
    ``torch.stack([bag(t) for t]).transpose(0, 1).to(dtype)``. Sums are
    float32 in index order; the cast is IEEE round-to-nearest-even.
    """

    __slots__ = ("owner", "tables", "dim", "dtype", "_views", "_out_dtype", "handle", "_registration", "_resolved", "__weakref__")

    def __init__(self, owner, tables, *, dtype="float16"):
        import torch
        if dtype not in ("float16", "float32"):
            raise ValueError("pooled output dtype must be float16 or float32")
        tables = list(tables)
        if not tables:
            raise ValueError("at least one table is required")
        for table in tables:
            _check_plain(table, "table")
            if (table.device.type != "cpu" or table.dtype != torch.float32 or table.dim() != 2
                    or not table.is_contiguous() or table.shape[0] <= 0):
                raise ValueError("tables must be dense CPU float32 [rows, dim] tensors")
        dim = int(tables[0].shape[1])
        if any(int(t.shape[1]) != dim for t in tables):
            raise ValueError("all tables must share one embedding dimension")
        self.owner, self.tables, self.dim, self.dtype = owner, tables, dim, dtype
        self._views = [_storage_view(t, "host", -1) for t in tables]
        self._out_dtype = getattr(torch, dtype)
        self._resolved = {}
        self.handle = self._registration = None
        from . import fast_ops
        self._registration = fast_ops.register(self)

    @property
    def closed(self):
        return self.owner._closed

    def op(self, indices, offsets):
        """``submit(indices, offsets).wait_stream()[0]`` as a functional custom op."""
        from .fast_ops import POOLED_OP
        indices, offsets = list(indices), list(offsets)
        shape = (offsets[0].shape[0], len(self.tables), self.dim)
        return POOLED_OP(indices, offsets, self.handle, shape, self.owner.device, self._out_dtype)

    def _operands(self, indices, offsets, out):
        """Validated operands: (indices, offsets, bags, index pointers, offset
        pointers, out or None)."""
        import torch
        self.owner._check_open()
        indices, offsets = list(indices), list(offsets)
        if len(indices) != len(self.tables) or len(offsets) != len(self.tables):
            raise ValueError("one index and one offset tensor per table")
        bags = None
        operands_i, operands_o = [], []
        for index, offset in zip(indices, offsets):
            for value in (index, offset):
                _check_plain(value, "bag operand")
                if (value.device.type != "cpu" or value.dtype != torch.int64 or value.dim() != 1
                        or not value.is_contiguous()):
                    raise ValueError("indices and offsets must be contiguous CPU int64 vectors")
            if bags is None:
                bags = int(offset.shape[0])
            elif int(offset.shape[0]) != bags:
                raise ValueError("every table needs the same number of bags")
            operands_i.append((index.data_ptr(), int(index.shape[0])))
            operands_o.append((offset.data_ptr(), bags))
        if bags == 0:
            raise ValueError("at least one bag is required")
        if out is not None and (tuple(out.shape) != (bags, len(self.tables), self.dim)
                                or out.dtype != self._out_dtype or not out.is_contiguous()):
            raise ValueError("out must be a contiguous tensor of the pooled shape and dtype")
        return indices, offsets, bags, operands_i, operands_o

    def _prepare(self, indices, offsets, out):
        import torch
        owner = self.owner
        indices, offsets, bags, operands_i, operands_o = self._operands(indices, offsets, out)
        if out is None:
            out = torch.empty((bags, len(self.tables), self.dim), dtype=self._out_dtype, device=owner.device)
        try:
            program = pyreloc.prepare_pooled_program(self._views, operands_i, operands_o, bags, self.dim, self.dtype)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"pooled producer rejected: {error}") from error
        return program, out, tuple(self.tables[1:]) + tuple(indices) + tuple(offsets)

    def _decide(self, bags):
        resolved = self._resolved.get(bags)
        if resolved is None:
            resolved = self._resolved[bags] = _advice.resolve(
                _advice.pooled_advice(), {"bags": bags, "tables": len(self.tables), "dim": self.dim},
                self.owner.gather_threads)
        return self.owner._advise(resolved)

    def _direct(self, indices, offsets, bags, decision, stream, gather_threads):
        """The fused pooled producer into a wire, then one copy on `stream`.
        Returns (out, event, report), or the reason the ring refused a wire."""
        owner = self.owner
        wire, refusal = owner._ring.acquire((bags, len(self.tables), self.dim), self._out_dtype)
        if wire is None:
            return refusal
        threads = owner.gather_threads if gather_threads is None else gather_threads
        try:
            _native_views.pooled_bags(self.tables, indices, offsets, wire, threads)
        except pyreloc.TransferError as error:
            owner._ring.cancel(wire)
            raise RuntimeError(f"pooled producer failed: {error}") from error
        except BaseException:
            owner._ring.cancel(wire)
            raise
        return owner._ring.ship(wire, tuple(wire.shape), None, stream, decision)

    def execute(self, indices, offsets, *, out=None, gather_threads=None, pipeline=True):
        """Blocking pooled gather; returns the [bags, tables, dim] CUDA tensor.

        With advice the call may take the direct path (the producer into a
        pinned wire, one copy ordered on the current stream; the operands are
        read before this returns). ``pipeline=False`` sums every bag first and
        copies the wire once through the dispatcher (no chunk overlap, no
        advice); it exists for attribution measurements."""
        import torch
        indices, offsets, bags, _, _ = self._operands(indices, offsets, out)
        decision = self._decide(bags) if pipeline else None
        if decision is not None and decision.path == "direct" and out is None:
            direct = self._direct(indices, offsets, bags, decision, None, gather_threads)
            if not isinstance(direct, str):
                self.owner._record(direct[2])
                return direct[0]
            decision = _advice.Decision("pipelined", direct, decision.predicted)
        program, out, extra = self._prepare(indices, offsets, out)
        return self.owner._execute_program(program, self._views[0], self.tables[0], out, extra, gather_threads,
                                           pipeline, decision)

    def submit(self, indices, offsets, *, out=None, gather_threads=None):
        """Asynchronous pooled gather; wait_stream() on the completion yields the output."""
        owner = self.owner
        indices, offsets, bags, _, _ = self._operands(indices, offsets, out)
        decision = self._decide(bags)
        if decision.path == "direct" and out is None:
            owner._retire()
            direct = self._direct(indices, offsets, bags, decision, owner._stream, gather_threads)
            if not isinstance(direct, str):
                return _DirectCompletion(owner, [direct[0]], direct[1], direct[2])
            decision = _advice.Decision("pipelined", direct, decision.predicted)
        program, out, extra = self._prepare(indices, offsets, out)
        return owner._submit_program(program, self._views[0], self.tables[0], out, extra, gather_threads,
                                     "float32", decision=decision)


def _group_advice(program):
    """A group member's advice; only prepared weight snapshots have a direct
    group executor, so every other member prices without the direct path."""
    if isinstance(program, _WireEntry):
        return program.resolved
    if isinstance(program, TypedProgram):
        return program._single_call()
    return getattr(program, "resolved", None)


def _dense_strides(shape):
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _copy_out(wire, shape, strides, device, stream):
    """A fresh device tensor of `shape` (and `strides`) holding the dense
    `wire`, copied asynchronously on `stream` (None: the device's current
    stream, without a stream switch)."""
    import torch
    shape = tuple(shape)
    if stream is None:
        if strides is None or tuple(strides) == _dense_strides(shape):
            return wire.to(device, non_blocking=True).view(shape)
        return torch.empty_strided(shape, strides, dtype=wire.dtype, device=device).copy_(
            wire.view(shape), non_blocking=True)
    with torch.cuda.stream(stream):
        return _copy_out(wire, shape, strides, device, None)


def _decided(report, decision):
    """A dispatcher report with the advice decision that routed the call; the
    dispatcher's own row reason stays under row_reason."""
    if decision is None:
        return report
    return dict(report, placement_reason=decision.reason, predicted_ms=decision.predicted,
                row_reason=report.get("placement_reason"))


def _direct_report(decision, wire):
    nbytes = wire.nbytes
    return {"implementation": "direct", "placement_reason": decision.reason, "predicted_ms": decision.predicted,
            "wire_bytes": nbytes, "payload_bytes_transferred": nbytes}


class _DirectRing:
    """The direct path's wires, per (shape, dtype): pinned unless the owner
    stages pageable memory. A wire is reusable once the CUDA event recorded
    behind its copy (on the owner's device) completed; at most `slots` wires
    per shape and `max_bytes` in total, making room by dropping idle wires of
    other shapes least recently used first. A refusal is a per-call fallback
    to the dispatcher with its reason, never a wait or a failure. Each wire
    owns one event, re-recorded on every use (creating an event per call
    costs more than the copy of a small wire)."""

    def __init__(self, device, slots, max_bytes, pinned=True):
        self.device, self.slots, self.max_bytes, self.pinned = device, slots, max_bytes, pinned
        self.bytes = 0
        self._wires = {}    # (shape, dtype) -> [entry]
        self._entries = {}  # id(wire) -> entry = [wire, event, busy, recorded, last use, key, nbytes]
        self._clock = 0
        self._lock = RLock()

    @staticmethod
    def _idle(entry):
        return not entry[2] and (not entry[3] or entry[1].query())

    def acquire(self, shape, dtype):
        """(wire, None) for a free wire of this (tuple of ints) shape and
        dtype, or (None, reason): direct_ring_exhausted when every wire of the
        shape is in flight, direct_wire_over_budget when one wire exceeds the
        whole budget."""
        import torch
        key = (shape, dtype)
        with self._lock:
            self._clock += 1
            entries = self._wires.setdefault(key, [])
            for entry in entries:
                if self._idle(entry):
                    entry[2], entry[4] = True, self._clock
                    return entry[0], None
            nbytes = math.prod(shape) * dtype.itemsize
            if nbytes > self.max_bytes:
                return None, "direct_wire_over_budget"
            if len(entries) >= self.slots:
                return None, "direct_ring_exhausted"
            if self.bytes + nbytes > self.max_bytes:
                idle = sorted((e for e in self._entries.values() if e[5] != key and self._idle(e)),
                              key=lambda e: e[4])
                for entry in idle:
                    if self.bytes + nbytes <= self.max_bytes:
                        break
                    self._drop(entry)
                if self.bytes + nbytes > self.max_bytes:
                    return None, "direct_ring_exhausted"
            wire = torch.empty(shape, dtype=dtype, pin_memory=self.pinned)
            entry = [wire, torch.cuda.Event(), True, False, self._clock, key, nbytes]
            entries.append(entry)
            self._entries[id(wire)] = entry
            self.bytes += nbytes
            return wire, None

    def _drop(self, entry):
        self._wires[entry[5]].remove(entry)
        if not self._wires[entry[5]]:
            del self._wires[entry[5]]
        del self._entries[id(entry[0])]
        self.bytes -= entry[6]

    def ship(self, wire, shape, strides, stream, decision):
        """Copy `wire` into a fresh device tensor on `stream` (None: the
        owner device's current stream) and release it behind the copy;
        returns (out, event, report). The wire comes back on every exit."""
        try:
            out = _copy_out(wire, shape, strides, self.device, stream)
        except BaseException:
            self.cancel(wire)
            raise
        return out, self.release(wire, stream), _direct_report(decision, wire)

    def release(self, wire, stream):
        """The copy out of `wire` is enqueued on `stream` (None: the owner
        device's current stream, whichever device is current): the wire is
        busy until the returned event completes."""
        import torch
        entry = self._entries[id(wire)]
        entry[1].record(torch.cuda.current_stream(self.device) if stream is None else stream)
        with self._lock:
            entry[2], entry[3] = False, True
        return entry[1]

    def cancel(self, wire):
        """Nothing was enqueued from `wire` (the producer or the copy failed): free it."""
        with self._lock:
            self._entries[id(wire)][2] = False

    def close(self):
        """Forget every wire; the host allocator keeps a pinned block alive
        until copies recorded on it completed."""
        with self._lock:
            self._wires.clear()
            self._entries.clear()
            self.bytes = 0


class _DirectCompletion:
    """A direct-path submission: outputs ordered after one CUDA event.
    Interchangeable with FastCompletion for wait/wait_stream/done/report."""

    __slots__ = ("_owner", "_outputs", "_event", "_report", "_keep", "_done")

    def __init__(self, owner, outputs, event, report, keep=()):
        self._owner, self._outputs, self._event, self._report = owner, list(outputs), event, report
        self._keep, self._done = tuple(keep), False

    @property
    def report(self):
        return dict(self._report) if self._done else None

    def done(self):
        if not self._done and self._event.query():
            self._finish()
        return self._done

    def wait(self):
        """Host completion; the outputs may be used on the current stream."""
        if not self._done:
            import torch
            self._event.synchronize()
            self._finish()
            stream = torch.cuda.current_stream(self._owner.device)
            for tensor in self._outputs:
                tensor.record_stream(stream)
        return self._outputs

    def wait_stream(self, stream=None):
        """GPU-side dependency only; the completion stays registered with its
        owner until the copies finished."""
        import torch
        if stream is None:
            stream = torch.cuda.current_stream(self._owner.device)
        stream.wait_event(self._event)
        for tensor in self._outputs:
            tensor.record_stream(stream)
        if not self._done:
            self._owner._retain(self)
        return self._outputs

    def _finish(self):
        self._done = True
        if "items" in self._report:
            self._owner._record_group(self._report)
            self._owner._count_reason(self._report)
        else:
            self._owner._record(self._report)
        self._keep = ()


class FastCompletion:
    """One submitted group; outputs are handed out by wait()/wait_stream()."""

    __slots__ = ("_owner", "_native", "_outputs", "_group", "_host", "_done", "_decision")

    def __init__(self, owner, native, outputs, group, host, decision=None):
        self._owner, self._native, self._outputs, self._group, self._host = owner, native, outputs, group, host
        self._done, self._decision = False, decision

    @property
    def report(self):
        return _decided(dict(self._group.report), self._decision) if self._done else None

    def done(self):
        if self._done:
            return True
        if self._native.query():
            self._finish()
            return True
        return False

    def wait(self):
        """Host completion; CPU outputs are readable and sources may change afterwards."""
        if not self._done:
            self._native.wait()
            self._finish()
        self._record_outputs()
        return self._outputs

    def wait_stream(self, stream=None):
        """GPU-side dependency only (H2D groups): the current stream waits for the transfer.

        The completion stays registered with its owner until the native work
        finished, so dropping this object never turns into a host wait; the
        owner retires it on a later submission or at close().
        """
        import torch
        if self._host:
            if not self._done:
                self._native.wait()
                self._finish()
        else:
            if stream is None:
                stream = torch.cuda.current_stream(self._owner.device)
            if not self._done:
                self._native.wait_stream(stream.cuda_stream)
                self._owner._retain(self)
        self._record_outputs(stream)
        return self._outputs

    def _record_outputs(self, stream=None):
        # Transfer completion and consumer lifetime are independent, including
        # mixed groups that host-complete D2H but also return CUDA outputs.
        import torch
        for tensor in self._outputs:
            if tensor.is_cuda:
                if stream is None:
                    stream = torch.cuda.current_stream(self._owner.device)
                tensor.record_stream(stream)

    def _finish(self):
        self._done = True
        self._owner._record_group(self._group.report)
        if self._decision is not None:
            self._owner._count_reason(_decided({}, self._decision))
        self._native = None


class FastTransfers:
    """Owner of prepared programs and the native resources they execute on.

    ``contexts`` independent native contexts let that many groups be in flight
    at once (each owns a stream and a share of the scratch budget); one
    context serializes submissions. Use as a context manager or call close().
    """

    def __init__(self, device="cuda:0", *, compiler=None, gather_threads=8, pinning="pinned",
                 max_scratch_bytes=64 << 20, n_buffers=2, chunk_size=None, contexts=2, calibration=None,
                 advice=True, direct_wires=2, direct_bytes=None):
        import torch
        from .compiler import CompilerClient
        if not pyreloc.cuda_enabled or not torch.cuda.is_available():
            raise RuntimeError("fast transfers require a CUDA-capable runtime")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("fast transfers require a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        for name, value in (("gather_threads", gather_threads), ("contexts", contexts), ("n_buffers", n_buffers),
                            ("max_scratch_bytes", max_scratch_bytes), ("direct_wires", direct_wires),
                            ("direct_bytes", max_scratch_bytes if direct_bytes is None else direct_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if pinning not in ("pinned", "pageable"):
            raise ValueError("pinning must be pinned or pageable")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer or None")
        self.gather_threads, self.pinning, self.n_buffers = gather_threads, pinning, n_buffers
        self.chunk_size = 0 if chunk_size is None else chunk_size
        self.max_scratch_bytes = max_scratch_bytes
        self._compiler = compiler if compiler is not None else CompilerClient.from_environment()
        self._compiled = {}
        self._pid, self._lock = os.getpid(), RLock()
        self._closed = False
        self._registrations = WeakSet()
        self._resources = TransferResources(
            max_typed_retained_bytes=max_scratch_bytes * contexts, max_typed_live_bytes=max_scratch_bytes * contexts,
            max_typed_background_workers=max(1, gather_threads - 1) * contexts, max_typed_streams=contexts,
            max_typed_contexts=contexts, max_typed_contexts_per_device=contexts)
        self._stream = torch.cuda.Stream(device=self.device)
        self._totals = dict(calls=0, groups=0, logical_transfers=0, payload_bytes=0, dispatches={}, advice={})
        self.native_views = _load_native_views() is not None
        self._inflight = deque()
        self.max_inflight = 8
        # Cost advice: the machine calibration that prices the compiler's
        # advice (None: every call takes today's dispatcher path, reason
        # no_calibration), whether advice is used at all, memoized decisions
        # and the direct path's pinned wires.
        self.calibration = _advice.load_calibration(calibration)
        self.advice = bool(advice)
        self._decisions = {}
        # The direct path's wires have a budget of their own (default: the
        # scratch budget), separate from the dispatcher's scratch.
        self._ring = _DirectRing(self.device, direct_wires,
                                 max_scratch_bytes if direct_bytes is None else direct_bytes,
                                 pinned=pinning == "pinned")

    # -- lifecycle -------------------------------------------------------
    def _check_open(self):
        if os.getpid() != self._pid:
            raise RuntimeError("fast transfers belong to another process")
        if self._closed:
            raise RuntimeError("fast transfers are closed")

    @property
    def native(self):
        return self._resources.native_typed

    def _retain(self, completion):
        """Keep a consumed-but-pending completion alive; retire finished ones.

        Bounded: beyond max_inflight the oldest is host-waited, which also
        bounds the native scratch those groups still reserve.
        """
        while self._inflight and self._inflight[0].done():
            self._inflight.popleft()
        while len(self._inflight) >= self.max_inflight:
            self._inflight.popleft().wait()
        self._inflight.append(completion)

    def _retire(self, block=False):
        while self._inflight and (block or self._inflight[0].done()):
            self._inflight.popleft().wait()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for registration in tuple(self._registrations):
                registration.release()
            self._registrations.clear()
            try:
                self._retire(block=True)
            finally:
                self._ring.close()
                self._resources.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def stats(self):
        self._check_open()
        self._retire()  # finished direct completions record themselves first
        return dict(self._totals, dispatches=dict(self._totals["dispatches"]), advice=dict(self._totals["advice"]),
                    direct_wire_bytes=self._ring.bytes, resources=self._resources.stats()["typed"])

    def _count_reason(self, report):
        # Only decided calls carry predicted_ms; undecided attribution calls
        # (pipeline=False, explicit rows) are not advice outcomes.
        if "predicted_ms" in report:
            reason = report["placement_reason"]
            self._totals["advice"][reason] = self._totals["advice"].get(reason, 0) + 1

    def _record(self, report):
        self._totals["calls"] += 1
        self._totals["logical_transfers"] += 1
        self._totals["payload_bytes"] += int(report.get("payload_bytes_transferred", 0))
        label = report.get("implementation", "?")
        self._totals["dispatches"][label] = self._totals["dispatches"].get(label, 0) + 1
        self._count_reason(report)

    def _advise(self, resolved):
        """The path decision of one call with `resolved` advice, memoized per
        (calibration, advice flag, resolved advice)."""
        key = (id(self.calibration), self.advice, id(resolved))
        hit = self._decisions.get(key)
        if hit is not None and hit[0] is self.calibration and hit[1] is resolved:
            return hit[2]
        decision = _advice.select(self.calibration, resolved, direct_available=_native_views is not None,
                                  advice_enabled=self.advice)
        if len(self._decisions) >= 4096:
            self._decisions.clear()
        self._decisions[key] = (self.calibration, resolved, decision)
        return decision

    def _advise_group(self, members):
        """The decision of one group submission over the members' advice,
        memoized per (calibration, advice flag, member advice)."""
        key = (id(self.calibration), self.advice, "group") + tuple(id(m) for m in members)
        hit = self._decisions.get(key)
        if hit is not None and hit[0] is self.calibration and all(a is b for a, b in zip(hit[1], members)):
            return hit[2]
        decision = _advice.select_group(self.calibration, members, direct_available=_native_views is not None,
                                        advice_enabled=self.advice)
        if len(self._decisions) >= 4096:
            self._decisions.clear()
        self._decisions[key] = (self.calibration, tuple(members), decision)
        return decision

    def _submit_direct_weights(self, items, outputs, decision):
        """Every member is a prepared int8 [out, in] snapshot: copy the raw
        wire into a transient device buffer on the private stream and run the
        fused dequantize (f32, or f16 in the same kernel) into the output; one
        event completes the group. Bit-identical to the group row
        (cpu_stages_cuda_stages@0 runs the same kernels)."""
        import torch
        self._retire()
        outs, keep, items_report = [], [], []
        with torch.cuda.stream(self._stream):
            for position, (entry, source) in enumerate(items):
                out = entry.allocate() if outputs is None else outputs[position]
                q = torch.empty(tuple(source.shape), dtype=torch.int8, device=self.device)
                q.copy_(source, non_blocking=True)
                kernel = (_native_views.dequant_s8_f16 if out.dtype == torch.float16
                          else _native_views.dequant_s8_f32)
                kernel(q, entry.scale_device(), out, self._stream.cuda_stream)
                outs.append(out)
                keep += (q, source)
                items_report.append(dict(implementation="direct", payload_bytes_transferred=int(source.numel())))
            event = torch.cuda.Event()
            event.record(self._stream)
        payload = sum(item["payload_bytes_transferred"] for item in items_report)
        report = dict(implementation="direct", placement_reason=decision.reason, predicted_ms=decision.predicted,
                      wire_bytes=payload, payload_bytes_transferred=payload, logical_transfers=len(items),
                      items=items_report)
        return _DirectCompletion(self, outs, event, report, keep=keep)

    def _record_group(self, report):
        self._totals["groups"] += 1
        self._totals["logical_transfers"] += int(report["logical_transfers"])
        for item in report["items"]:
            self._totals["payload_bytes"] += int(item["payload_bytes_transferred"])
            label = item["implementation"]
            self._totals["dispatches"][label] = self._totals["dispatches"].get(label, 0) + 1

    # -- preparation -----------------------------------------------------
    def compile(self, recipe):
        """Compile once per recipe identity (the compiler client caches too)."""
        self._check_open()
        key = recipe.canonical_identity
        with self._lock:
            compiled = self._compiled.get(key)
            if compiled is None:
                compiled = self._compiled[key] = self._compiler.compile(recipe)
        return compiled

    def typed(self, recipe, source, **options):
        return TypedProgram(self, self.compile(recipe), source, **options)

    def layout(self, recipe, source):
        return LayoutProgram(self, self.compile(recipe), source)

    def indexed(self, recipe):
        return IndexedProgram(self, self.compile(recipe))

    def pooled(self, tables, *, dtype="float16"):
        return PooledProgram(self, tables, dtype=dtype)

    # -- one-shot producers (indexed / pooled programs) -------------------
    def _execute_request(self, native, source, out, extra, gather_threads, pipeline=True, decision=None):
        try:
            report = pyreloc.execute_dispatch(
                native, caller_stream=compat.cuda_stream_handle(self.device),
                n_buffers=self.n_buffers, n_streams=1,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, direct_dense_upload=True, pipeline=pipeline,
                chunk_size=self.chunk_size, resources=self.native, owners=((source,) + tuple(extra), out))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"producer dispatch failed: {error}") from error
        self._record(_decided(report, decision))
        return out

    def _execute_program(self, program, src_view, source, out, extra, gather_threads, pipeline=True,
                         decision=None):
        """Blocking single dispatch of a freshly prepared producer program."""
        try:
            native = pyreloc.prepare_dispatch(program, src_view, _storage_view(out, "cuda", out.device.index),
                                              "h2d", threads=self.gather_threads)
            report = pyreloc.execute_dispatch(
                native, caller_stream=compat.cuda_stream_handle(self.device),
                n_buffers=self.n_buffers, n_streams=1,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, direct_dense_upload=True, pipeline=pipeline,
                chunk_size=self.chunk_size, resources=self.native, owners=((source,) + tuple(extra), out))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"producer dispatch failed: {error}") from error
        self._record(_decided(report, decision))
        return out

    def _submit_program(self, program, src_view, source, out, extra, gather_threads, dtype, decision=None):
        try:
            template = pyreloc.prepare_dispatch_template_from_program(program, "h2d", threads=self.gather_threads)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"producer dispatch rejected: {error}") from error
        member = _OneShot(template, src_view, None, dtype)
        return self.submit([(member, source)], outputs=[out], gather_threads=gather_threads, extra_owners=extra,
                           chunked_members=True, decision=decision)

    def _submit_native(self, native, sources, outs, extra, gather_threads, *, host, chunked_members=False,
                       device_layout=False, decision=None):
        """Submit an already prepared native group (outputs allocated on the private stream)."""
        import torch
        self._retire()
        if host:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            completion = pyreloc.submit_dispatch_group(
                native, caller_stream=self._stream.cuda_stream,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, max_scratch_bytes=self.max_scratch_bytes,
                resources=self.native, owners=(tuple(sources), tuple(outs), tuple(extra)),
                device_layout=device_layout, n_buffers=self.n_buffers, chunked_members=chunked_members)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"transfer group failed: {error}") from error
        return FastCompletion(self, completion, list(outs), native, host, decision)

    # -- groups ----------------------------------------------------------
    def submit(self, items, *, outputs=None, device_layout=False, gather_threads=None, pin_outputs=True,
               extra_owners=(), chunked_members=False, decision=None):
        """Submit [(program, source), ...] as one group; returns a FastCompletion.

        H2D groups order after nothing (CPU sources are ready); D2H groups
        order after the caller's current stream. Outputs are allocated here
        unless ``outputs`` supplies them.
        """
        import torch
        self._check_open()
        self._retire()
        items = list(items)
        if not 1 <= len(items) <= 256:
            raise ValueError("a group holds between 1 and 256 transfers")
        if decision is None:
            decision = self._advise_group([_group_advice(program) for program, _ in items])
            if decision.path == "direct":
                return self._submit_direct_weights(items, outputs, decision)
        sources, outs, entries = [], [], []
        host = False
        native_views = _native_views is not None
        with torch.cuda.stream(self._stream):
            for position, (program, source) in enumerate(items):
                if not isinstance(program, (TypedProgram, LayoutProgram, _OneShot)):
                    raise TypeError("group members are typed or layout programs")
                program.check(source)
                out = program.allocate(pin=pin_outputs) if outputs is None else outputs[position]
                host |= program.direction == "d2h"
                sources.append(source)
                outs.append(out)
                if native_views:
                    entries.append(program.tensor_entry(source, out))
                else:
                    entries.append(program.group_entry(source, out))
        if host:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            native = (_native_views.group_from_tensors(entries) if native_views
                      else pyreloc.prepare_dispatch_group(entries))
            completion = pyreloc.submit_dispatch_group(
                native, caller_stream=self._stream.cuda_stream,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, max_scratch_bytes=self.max_scratch_bytes,
                resources=self.native, owners=(tuple(sources), tuple(outs), tuple(extra_owners)),
                device_layout=device_layout, n_buffers=self.n_buffers, chunked_members=chunked_members)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"transfer group failed: {error}") from error
        return FastCompletion(self, completion, outs, native, host, decision)


class FastPrefetch:
    """Keep up to ``slots`` groups in flight over an iterable of group items.

    Each iteration submits ahead and yields the next group's outputs with a
    current-stream dependency (H2D). Host-completing the oldest group frees
    its slot; outputs stay valid for the consumer through record_stream.
    """

    def __init__(self, owner, groups, *, slots=2, device_layout=False, gather_threads=None):
        self._owner, self._groups = owner, iter(groups)
        self._slots, self._device_layout, self._threads = slots, device_layout, gather_threads
        self._pending, self._consumed = deque(), deque()
        self._closed = False
        self._fill(limit=1)

    def _fill(self, limit=None, block=True):
        # In flight = submitted and not yet host-complete, whether or not the
        # consumer already enqueued its use. Retire finished ones cheaply;
        # block on the oldest only when every slot is taken and `block` allows
        # it. The window opens with one submission so that group's DMA starts
        # before the next is prepared; steady state keeps `slots` in flight.
        limit = self._slots if limit is None else min(limit, self._slots)
        while self._groups is not None and len(self._pending) < limit:
            while self._consumed and self._consumed[0].done():
                self._consumed.popleft()
            if len(self._pending) + len(self._consumed) >= self._slots:
                if not block or not self._consumed:
                    return
                self._consumed.popleft().wait()
                continue
            try:
                items = next(self._groups)
            except StopIteration:
                self._groups = None
                return
            self._pending.append(self._owner.submit(items, device_layout=self._device_layout,
                                                    gather_threads=self._threads))

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        # Any slot wait happens here, after the consumer enqueued the previous
        # group's work, never between handing out a group and its first use.
        self._fill(block=True)
        if not self._pending:
            self.close()
            raise StopIteration
        completion = self._pending.popleft()
        outputs = completion.wait_stream()
        self._consumed.append(completion)
        self._fill(block=False)
        return outputs

    def close(self):
        if self._closed:
            return
        self._closed = True
        for queue in (self._pending, self._consumed):
            while queue:
                queue.popleft().wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class KVCacheSwap:
    """Evict ``[H, S, Dh]`` GPU caches to ``[S, H, Dh]`` pinned CPU tensors and back.

    All tensors of one call form one group: one native submission, at most
    one kernel and one copy per tensor on the device, one completion.
    evict() returns after the host copies landed; restore() returns GPU
    tensors ordered on the current stream.
    """

    def __init__(self, owner, heads, head_dim, *, dtype="float32"):
        import torch
        from .recipe import Recipe, TensorSpec, Transpose
        from .symbolic import Const, Symbol, dense_strides
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("KV swap relocates float32, float16 or bfloat16 caches")
        if dtype != "float32" and head_dim % 2:
            raise ValueError("16-bit caches need an even head dimension")
        self.owner, self.heads, self.head_dim = owner, heads, head_dim
        # 16-bit caches move as float32 pairs: the layout kernels relocate
        # 4-byte elements, and a [H, S, Dh] half tensor is bit for bit a
        # [H, S, Dh/2] float32 tensor. The user sees the original dtype.
        self.dtype = getattr(torch, dtype)
        self._pair = dtype != "float32"
        inner = head_dim // 2 if self._pair else head_dim
        s = Symbol("s0")
        cache, stored = (Const(heads), s, Const(inner)), (s, Const(heads), Const(inner))

        def spec(shape):
            return TensorSpec(shape, dense_strides(shape), Const(0), "float32")
        self._evict_recipe = Recipe(spec(cache), (Transpose((1, 0, 2)),), spec(stored), "d2h")
        self._restore_recipe = Recipe(spec(stored), (Transpose((1, 0, 2)),), spec(cache), "h2d")
        self._programs = {}
        self._inflight = deque()

    def _program(self, recipe, tensor):
        key = (recipe is self._evict_recipe, tensor.shape)
        program = self._programs.get(key)
        if program is None:
            program = self._programs[key] = self.owner.layout(recipe, tensor)
        return program

    def _retire(self, block=False):
        while self._inflight and (block or self._inflight[0].done()):
            self._inflight.popleft().wait()

    def _pairs(self, tensor):
        import torch
        if not self._pair:
            return tensor
        if tensor.dtype != self.dtype:
            raise ValueError(f"cache dtype {tensor.dtype} is not the swap's {self.dtype}")
        return tensor.view(torch.float32)

    def _unpair(self, tensor):
        return tensor.view(self.dtype) if self._pair else tensor

    def evict(self, caches):
        """One packed pinned block holds every evicted cache; returns views of it."""
        import torch
        caches = [self._pairs(t) for t in caches]
        self._retire()
        programs = [self._program(self._evict_recipe, t) for t in caches]
        counts = [int(torch.Size(p.destination.shape).numel()) for p in programs]
        block = torch.empty(sum(counts), dtype=caches[0].dtype, pin_memory=True)
        outputs, offset = [], 0
        for program, count in zip(programs, counts):
            outputs.append(block[offset:offset + count].view(program.destination.shape))
            offset += count
        completion = self.owner.submit(list(zip(programs, caches)), outputs=outputs,
                                       device_layout=True, gather_threads=1)
        return [self._unpair(t) for t in completion.wait()]

    def restore(self, stored):
        """GPU caches ordered on the current stream; the host is not blocked."""
        stored = [self._pairs(t) for t in stored]
        self._retire()
        completion = self.owner.submit([(self._program(self._restore_recipe, t), t) for t in stored],
                                       device_layout=True, gather_threads=1)
        outputs = [self._unpair(t) for t in completion.wait_stream()]
        self._inflight.append(completion)
        return outputs

    def drain(self):
        self._retire(block=True)


class FastWeights:
    """Low-overhead grouped loads over PreparedWireWeights snapshots."""

    def __init__(self, owner, weights):
        self.owner, self.weights = owner, weights
        self._entries = {}

    def _items(self, names):
        items = []
        advice = getattr(self.weights, "advice", None)
        for name, (template, source, scale) in zip(names, self.weights.fast_entries(names)):
            entry = self._entries.get(name)
            if entry is None or entry[1] is not source:
                # A re-prepared snapshot is a new source object; rebuild its member.
                entry = self._entries[name] = (_WireEntry(self.owner, template, source, scale,
                                                          getattr(self.weights, "output_dtype", "float32"),
                                                          advice=advice),
                                               source)
            items.append(entry)
        return items

    def submit(self, names):
        return self.owner.submit(self._items(names), gather_threads=1)

    def load_many(self, names):
        return self.submit(names).wait()

    def prefetch(self, name_groups, *, slots=2):
        return FastPrefetch(self.owner, (self._items(names) for names in name_groups), slots=slots,
                            gather_threads=1)


class _WireEntry(TypedProgram):
    """A prepared INT8 [out,in] snapshot with its native template (no rebinding),
    its owned CPU scale and the dequantize recipe's advice at its shape."""

    __slots__ = ("scale", "_scale_device")

    def __init__(self, owner, template, source, scale, dtype="float32", *, advice=None):  # noqa: D401 - adapter
        import torch
        self.owner, self.compiled, self.bindings = owner, None, None
        self.signature = _signature(source)
        self.signature_dtype = "int8"
        self.direction = "h2d"
        self._view_kind, self._view_device = "host", -1
        self.template, self.bound, self.selection = template, None, dict(template.selection)
        self.parameter_bytes = 0
        from .runtime import ConcreteDescriptor
        self.destination = ConcreteDescriptor(tuple(source.shape), tuple(source.stride()), dtype, owner.device)
        self._out_dtype, self._out_device = getattr(torch, dtype), owner.device
        self.handle = self._registration = self._op_args = None
        self.scale, self._scale_device = scale, None
        # The packed source is [out, in]: the recipe's symbols, bound.
        self.resolved = None if advice is None else _advice.resolve(
            advice, {"out": int(source.shape[0]), "in": int(source.shape[1])}, owner.gather_threads)
        self._single = None
        self._register_op()

    def scale_device(self):
        """The scale on the owner's device, uploaded once per entry."""
        if self._scale_device is None:
            self._scale_device = self.scale.to(self.owner.device)
        return self._scale_device


__all__ = ("FastTransfers", "FastCompletion", "FastPrefetch", "TypedProgram", "LayoutProgram",
           "IndexedProgram", "PooledProgram", "KVCacheSwap", "FastWeights")
