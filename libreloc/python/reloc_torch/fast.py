"""Prepared transfer programs with minimal per-call work.

The ``torch.compile``/custom-op path validates every call from scratch: it
re-derives symbol bindings, snapshots parameters, rechecks storage and
rebuilds descriptors. That costs several hundred microseconds per transfer,
which dominates small KV-cache moves and adds up over weight groups. This
module binds a recipe to one source *shape signature* once (symbols,
parameters, native dispatch template and destination descriptor) and then
executes each call through a single native entry point.

What stays per call: live source/output metadata and storage must match the
prepared signature and fit the validated views. Bounded reusable group
slots retain programs and descriptor vectors between successful transfers. Parameters
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
"""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
import os
from threading import RLock
from weakref import WeakSet

import pyreloc

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
                 "_op_args", "__weakref__")

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

    __slots__ = ("bound", "template", "selection", "parameter_bytes")

    @classmethod
    def _from_template(cls, owner, compiled, source, bindings, bound, template, parameter_bytes):
        """Internal lowering of an already validated compiler specialization.

        The automatic adapter owns live operand guards and revision checks.
        It invokes the group directly, so no extra custom-op registration is
        needed. Native binding still checks every source/output view.
        """
        program = cls.__new__(cls)
        program._init_common(owner, compiled, source, bindings)
        program.bound, program.template = bound, template
        program.selection = dict(template.selection)
        program.parameter_bytes = parameter_bytes
        return program

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
        if self.direction == "h2d":
            self._register_op()

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
        owner._record(report)
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

    __slots__ = ("owner", "compiled", "_specs", "handle", "_registration", "_op_args", "__weakref__")

    def __init__(self, owner, compiled):
        import torch
        from .symbolic import Const
        if not compiled.indexed or compiled.recipe.direction != "h2d":
            raise UnsupportedRecipe("unsupported_index_select", "expected an H2D index_select recipe")
        self.owner, self.compiled = owner, compiled
        self._specs = {}
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

    def execute(self, source, index, *, out=None, gather_threads=None, pipeline=True):
        """Blocking fused row gather/cast/H2D; the index tensor is read in place.

        ``pipeline=False`` gathers the whole wire first and copies it once
        (no chunk overlap); it exists for attribution measurements."""
        owner = self.owner
        if _native_views is not None and out is None:
            # One native call binds the live indices, prepares the program over
            # the source and selects the row; validation is the same as below.
            owner._check_open()
            _check_plain(source, "source")
            _check_plain(index, "index")
            key = (_signature(source), index.shape[0], index.dtype)
            cached = self._specs.get(key)
            if cached is None:
                self._prepare(source, index, None)
                cached = self._specs[key]
            bindings, destination, _, out_dtype = cached
            import torch
            out = torch.empty_strided(destination.shape, destination.strides, dtype=out_dtype, device=owner.device)
            try:
                native = _native_views.index_select_request(self.compiled.decoded_plan, bindings, source, index, out,
                                                            owner.gather_threads)
            except pyreloc.BindError as error:
                raise UnsupportedRecipe("index_out_of_range" if "index_out_of_range" in str(error) else "bind_error",
                                        str(error)) from error
            except pyreloc.TransferError as error:
                raise RuntimeError(f"index_select dispatch rejected: {error}") from error
            return owner._execute_request(native, source, out, (index,), gather_threads, pipeline)
        program, src_view, out = self._prepare(source, index, out)
        return self.owner._execute_program(program, src_view, source, out, (index,), gather_threads, pipeline)

    def submit(self, source, index, *, out=None, gather_threads=None):
        """Asynchronous fused gather; wait_stream() on the completion yields the output."""
        owner = self.owner
        if _native_views is not None and out is None:
            owner._check_open()
            _check_plain(source, "source")
            _check_plain(index, "index")
            key = (_signature(source), index.shape[0], index.dtype)
            cached = self._specs.get(key)
            if cached is None:
                self._prepare(source, index, None)
                cached = self._specs[key]
            bindings, destination, _, out_dtype = cached
            import torch
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
                                        chunked_members=True)
        program, src_view, out = self._prepare(source, index, out)
        return self.owner._submit_program(program, src_view, source, out, (index,), gather_threads,
                                          compat.dtype_name(source.dtype))


class PooledProgram:
    """Fused EmbeddingBag(mode="sum") pooling over several tables, cast, H2D.

    ``tables`` are dense float32 ``[rows_t, dim]`` CPU tensors (they may differ
    in rows). Each call takes per-table ``indices``/``offsets`` (int64, the
    EmbeddingBag convention without include_last_offset) for ``bags`` bags and
    returns a fresh ``[bags, tables, dim]`` CUDA tensor equal to
    ``torch.stack([bag(t) for t]).transpose(0, 1).to(dtype)``. Sums are
    float32 in index order; the cast is IEEE round-to-nearest-even.
    """

    __slots__ = ("owner", "tables", "dim", "dtype", "_views", "_out_dtype", "handle", "_registration", "__weakref__")

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

    def _prepare(self, indices, offsets, out):
        import torch
        owner = self.owner
        owner._check_open()
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
        shape = (bags, len(self.tables), self.dim)
        if out is None:
            out = torch.empty(shape, dtype=self._out_dtype, device=owner.device)
        elif tuple(out.shape) != shape or out.dtype != self._out_dtype or not out.is_contiguous():
            raise ValueError("out must be a contiguous tensor of the pooled shape and dtype")
        try:
            program = pyreloc.prepare_pooled_program(self._views, operands_i, operands_o, bags, self.dim, self.dtype)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"pooled producer rejected: {error}") from error
        return program, out, tuple(self.tables[1:]) + tuple(indices) + tuple(offsets)

    def execute(self, indices, offsets, *, out=None, gather_threads=None, pipeline=True):
        """Blocking pooled gather; returns the [bags, tables, dim] CUDA tensor.

        ``pipeline=False`` sums every bag first and copies the wire once (no
        chunk overlap); it exists for attribution measurements."""
        program, out, extra = self._prepare(indices, offsets, out)
        return self.owner._execute_program(program, self._views[0], self.tables[0], out, extra, gather_threads,
                                           pipeline)

    def submit(self, indices, offsets, *, out=None, gather_threads=None):
        """Asynchronous pooled gather; wait_stream() on the completion yields the output."""
        program, out, extra = self._prepare(indices, offsets, out)
        return self.owner._submit_program(program, self._views[0], self.tables[0], out, extra, gather_threads,
                                          "float32")


class FastCompletion:
    """One submitted group; outputs are handed out by wait()/wait_stream()."""

    __slots__ = ("_owner", "_native", "_outputs", "_group", "_host", "_done",
                 "_prepared", "_snapshot", "_error")

    def __init__(self, owner, native, outputs, group, host):
        self._owner, self._native, self._outputs, self._group, self._host = owner, native, outputs, group, host
        self._done = False
        self._prepared = self._snapshot = self._error = None

    @property
    def report(self):
        self._owner._check_process()
        with self._owner._lock:
            if not self._done:
                return None
            if self._owner.report_mode == "native":
                return self._snapshot.report
            self._owner._eager_report_accesses += 1
            return deepcopy(self._snapshot)

    def done(self):
        self._owner._check_process()
        with self._owner._lock:
            self._raise_error()
            if not self._done:
                try:
                    if not self._native.query():
                        return False
                    self._finish()
                except BaseException as error:
                    self._fail(error)
                    raise
            return True

    def wait(self):
        """Host completion; CPU outputs are readable and sources may change afterwards."""
        self._owner._check_process()
        with self._owner._lock:
            self._wait_locked()
            self._record_outputs()
            return self._outputs

    def _wait_locked(self):
        self._raise_error()
        if not self._done:
            try:
                self._native.wait()
                self._finish()
            except BaseException as error:
                self._fail(error)
                raise

    def wait_stream(self, stream=None):
        """Order the consumer stream after H2D; host-complete groups with CPU outputs."""
        import torch
        self._owner._check_process()
        with self._owner._lock:
            self._raise_error()
            if self._host:
                self._wait_locked()
            else:
                if stream is None:
                    stream = torch.cuda.current_stream(self._owner.device)
                if not self._done:
                    try:
                        self._native.wait_stream(stream.cuda_stream)
                    except BaseException as error:
                        self._fail(error)
                        raise
                    self._owner._retain(self)
            self._record_outputs(stream)
            return self._outputs

    def _raise_error(self):
        if self._error is not None:
            raise self._error

    def _fail(self, error):
        self._error = error
        if self._prepared is not None:
            self._prepared._release(self, self._group, failed=True)
            self._prepared = None
        # The native completion/owner token retains unsafe buffers if needed.

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
        if self._owner.report_mode == "native":
            self._snapshot = self._owner._group_counters.capture(self._group)
        else:
            self._snapshot = self._group.report
            self._owner._record_group(self._snapshot)
        self._done = True
        self._native = None
        if self._prepared is not None:
            self._prepared._release(self, self._group)
            self._prepared = None
        self._group = None


class FastTransfers:
    """Owner of prepared programs and the native resources they execute on.

    ``contexts`` independent native contexts let that many groups be in flight
    at once (each owns a stream and a share of the scratch budget); one
    context serializes submissions. Use as a context manager or call close().

    ``prepared_groups`` enables the bounded implicit structural cache
    (``max_prepared_groups`` handles, at most ``contexts`` slots each).
    ``prepare_group`` creates an explicit handle. ``report_mode='native'``
    folds compact counters at completion and converts detailed reports only
    on access; ``'eager'`` and ``prepared_groups=False`` retain comparison
    paths. Current source values and fresh output allocation are unchanged.
    """

    def __init__(self, device="cuda:0", *, compiler=None, gather_threads=8, pinning="pinned",
                 max_scratch_bytes=64 << 20, n_buffers=2, chunk_size=None, contexts=2,
                 prepared_groups=True, report_mode="native", max_prepared_groups=64, resources=None,
                 output_arena=None):
        import torch
        if not pyreloc.cuda_enabled or not torch.cuda.is_available():
            raise RuntimeError("fast transfers require a CUDA-capable runtime")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("fast transfers require a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        _check_output_arena(output_arena, self.device)
        self.output_arena = output_arena
        for name, value in (("gather_threads", gather_threads), ("contexts", contexts), ("n_buffers", n_buffers),
                            ("max_scratch_bytes", max_scratch_bytes), ("max_prepared_groups", max_prepared_groups)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if pinning not in ("pinned", "pageable"):
            raise ValueError("pinning must be pinned or pageable")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer or None")
        if type(prepared_groups) is not bool:
            raise ValueError("prepared_groups must be a bool")
        if report_mode not in ("native", "eager"):
            raise ValueError("report_mode must be native or eager")
        if resources is not None:
            if not isinstance(resources, TransferResources):
                raise TypeError("resources must be a TransferResources owner or None")
            if resources.closed:
                raise RuntimeError("transfer resources are closed")
        self.prepared_groups, self.report_mode = prepared_groups, report_mode
        self.contexts, self.max_prepared_groups = contexts, max_prepared_groups
        self._group_counters = pyreloc.GroupCounters()
        self._eager_report_accesses = 0
        self._prepared_cache = OrderedDict()
        self._prepared_handles = WeakSet()
        self._wire_restores = WeakSet()
        self._prepared_counts = dict(handles_created=0, cache_hits=0, cache_misses=0, evictions=0,
                                     group_constructions=0, rebinds=0, view_descriptors=0)
        self.gather_threads, self.pinning, self.n_buffers = gather_threads, pinning, n_buffers
        self.chunk_size = 0 if chunk_size is None else chunk_size
        self.max_scratch_bytes = max_scratch_bytes
        self._compiler = compiler
        self._compiled = {}
        self._pid, self._lock = os.getpid(), RLock()
        self._closed = False
        self._registrations = WeakSet()
        self._owns_resources = resources is None
        self._resources = resources if resources is not None else TransferResources(
            max_typed_retained_bytes=max_scratch_bytes * contexts, max_typed_live_bytes=max_scratch_bytes * contexts,
            max_typed_background_workers=max(1, gather_threads - 1) * contexts, max_typed_streams=contexts,
            max_typed_contexts=contexts, max_typed_contexts_per_device=contexts)
        self._stream = torch.cuda.Stream(device=self.device)
        self._totals = dict(calls=0, groups=0, logical_transfers=0, payload_bytes=0, dispatches={})
        self.native_views = _load_native_views() is not None
        self._inflight = deque()
        self.max_inflight = 8

    # -- lifecycle -------------------------------------------------------
    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("fast transfers belong to another process")

    def _check_open(self):
        self._check_process()
        if self._closed:
            raise RuntimeError("fast transfers are closed")
        if self._resources.closed:
            raise RuntimeError("transfer resources are closed")

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
        self._check_process()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for registration in tuple(self._registrations):
                registration.release()
            self._registrations.clear()
            error = None
            try:
                for handle in (*tuple(self._wire_restores), *tuple(self._prepared_handles)):
                    try:
                        handle.close()
                    except BaseException as exc:
                        if error is None:
                            error = exc
                self._prepared_cache.clear()
                while self._inflight:
                    try:
                        self._inflight.popleft().wait()
                    except BaseException as exc:
                        if error is None:
                            error = exc
            finally:
                if self._owns_resources:
                    self._resources.close()
            if error is not None:
                raise error

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def stats(self):
        self._check_process()
        with self._lock:
            if self._closed:
                raise RuntimeError("fast transfers are closed")
            totals = dict(self._totals, dispatches=dict(self._totals["dispatches"]))
            totals["wire_restore"] = [owner.stats() for owner in self._wire_restores]
            native = self._group_counters.stats()
            for key in ("groups", "logical_transfers", "payload_bytes"):
                totals[key] += native[key]
            for key, value in native["dispatches"].items():
                totals["dispatches"][key] = totals["dispatches"].get(key, 0) + value
            prepared = dict(self._prepared_counts, cached_handles=len(self._prepared_cache), live_slots=0)
            for handle in self._prepared_handles:
                prepared["live_slots"] += handle._slots
                for key, value in handle._binder.stats().items():
                    prepared[key] += value
            totals["prepared_groups"] = prepared
            totals["report_materializations"] = (native["report_materializations"] + self._totals["groups"]
                                                  + self._eager_report_accesses)
            totals["resources"] = self._resources.stats()["typed"]
            totals["output_arena"] = None if self.output_arena is None else self.output_arena.stats()
            return totals

    def _allocate_outputs(self, programs, *, pin_outputs=True):
        """Caller holds the owner lock and selects the allocation stream."""
        if self.output_arena is not None:
            return self.output_arena.allocate([p.destination for p in programs])
        return [p.allocate(pin=pin_outputs) for p in programs]

    def _record(self, report):
        self._totals["calls"] += 1
        self._totals["logical_transfers"] += 1
        self._totals["payload_bytes"] += int(report.get("payload_bytes_transferred", 0))
        label = report.get("implementation", "?")
        self._totals["dispatches"][label] = self._totals["dispatches"].get(label, 0) + 1

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
                if self._compiler is None:
                    from .compiler import CompilerClient
                    self._compiler = CompilerClient.from_environment()
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
    def _execute_request(self, native, source, out, extra, gather_threads, pipeline=True):
        try:
            report = pyreloc.execute_dispatch(
                native, caller_stream=compat.cuda_stream_handle(self.device),
                n_buffers=self.n_buffers, n_streams=1,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, direct_dense_upload=True, pipeline=pipeline,
                chunk_size=self.chunk_size, resources=self.native, owners=((source,) + tuple(extra), out))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"producer dispatch failed: {error}") from error
        self._record(report)
        return out

    def _execute_program(self, program, src_view, source, out, extra, gather_threads, pipeline=True):
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
        self._record(report)
        return out

    def _submit_program(self, program, src_view, source, out, extra, gather_threads, dtype):
        try:
            template = pyreloc.prepare_dispatch_template_from_program(program, "h2d", threads=self.gather_threads)
        except pyreloc.TransferError as error:
            raise RuntimeError(f"producer dispatch rejected: {error}") from error
        member = _OneShot(template, src_view, None, dtype)
        return self.submit([(member, source)], outputs=[out], gather_threads=gather_threads, extra_owners=extra,
                           chunked_members=True)

    def _submit_native(self, native, sources, outs, extra, gather_threads, **options):
        self._check_open()
        with self._lock:
            self._check_open()
            self._retire()
            return self._launch_group(native, sources, outs, extra, gather_threads, **options)

    def _launch_group(self, native, sources, outs, extra, gather_threads, *, host, chunked_members=False,
                      device_layout=False):
        """Launch while holding the owner lock; outputs were allocated on its stream."""
        import torch
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
        return FastCompletion(self, completion, list(outs), native, host)

    # -- groups ----------------------------------------------------------
    def prepare_group(self, programs):
        """Prepare immutable members for repeated, checked tensor bindings."""
        from .prepared_group import PreparedGroup
        self._check_open()
        with self._lock:
            self._check_open()
            if _native_views is None:
                raise RuntimeError("prepared groups require the native Torch bridge")
            return PreparedGroup(self, programs, _native_views)

    def _invalidate_program(self, program):
        # Caller holds the owner lock. A snapshot lock must not be held here:
        # closing an evicted handle may wait for its in-flight transfer.
        for key in tuple(self._prepared_cache):
            if program in key:
                self._prepared_cache.pop(key).close()
                self._prepared_counts["evictions"] += 1

    def submit(self, items, *, outputs=None, device_layout=False, gather_threads=None, pin_outputs=True,
               extra_owners=(), chunked_members=False):
        """Submit [(program, source), ...], allocating fresh outputs unless supplied.

        The bounded structural cache reuses immutable programs, never tensor
        values. One-shot producers and builds without the Torch bridge use
        the original construction path.
        """
        self._check_open()
        with self._lock:
            self._check_open()
            self._retire()
            items = tuple(items)
            if not 1 <= len(items) <= 256:
                raise ValueError("a group holds between 1 and 256 transfers")
            programs = tuple(p for p, _ in items)
            options = dict(outputs=outputs, device_layout=device_layout, gather_threads=gather_threads,
                           pin_outputs=pin_outputs, extra_owners=extra_owners, chunked_members=chunked_members)
            if self.prepared_groups and _native_views is not None and all(
                    isinstance(p, (TypedProgram, LayoutProgram)) for p in programs):
                handle = self._prepared_cache.get(programs)
                if handle is None:
                    self._prepared_counts["cache_misses"] += 1
                    if len(self._prepared_cache) >= self.max_prepared_groups:
                        _, evicted = self._prepared_cache.popitem(last=False)
                        evicted.close()
                        self._prepared_counts["evictions"] += 1
                    handle = self._prepared_cache[programs] = self.prepare_group(programs)
                else:
                    self._prepared_counts["cache_hits"] += 1
                    self._prepared_cache.move_to_end(programs)
                return handle._submit_locked((source for _, source in items), **options)
            return self._submit_legacy(items, **options)

    def _submit_legacy(self, items, *, outputs, device_layout, gather_threads, pin_outputs,
                       extra_owners, chunked_members):
        import torch
        sources, outs, entries = [], [], []
        host = False
        native_views = _native_views is not None
        if outputs is not None:
            outputs = tuple(outputs)
            if len(outputs) != len(items):
                raise ValueError("one output per program is required")
        with torch.cuda.stream(self._stream):
            for program, source in items:
                if not isinstance(program, (TypedProgram, LayoutProgram, _OneShot)):
                    raise TypeError("group members are typed or layout programs")
                if not isinstance(program, _OneShot) and program.owner is not self:
                    raise ValueError("program belongs to a different owner")
                program.check(source)
            if outputs is None:
                outputs = self._allocate_outputs([p for p, _ in items], pin_outputs=pin_outputs)
            for position, (program, source) in enumerate(items):
                out = outputs[position]
                host |= program.direction == "d2h"
                sources.append(source)
                outs.append(out)
                entries.append(program.tensor_entry(source, out) if native_views else program.group_entry(source, out))
        try:
            native = (_native_views.group_from_tensors(entries) if native_views
                      else pyreloc.prepare_dispatch_group(entries))
        except pyreloc.TransferError as error:
            raise RuntimeError(f"transfer group failed: {error}") from error
        return self._launch_group(native, sources, outs, extra_owners, gather_threads, host=host,
                                  device_layout=device_layout, chunked_members=chunked_members)


class FastPrefetch:
    """Keep up to ``slots`` groups in flight over an iterable of group items.

    Each iteration submits ahead and yields the next group's outputs with a
    current-stream dependency (H2D). Host-completing the oldest group frees
    its slot; outputs stay valid for the consumer through record_stream.
    """

    def __init__(self, owner, groups, *, slots=2, device_layout=False, gather_threads=None):
        if type(slots) is not int or slots < 1:
            raise ValueError("prefetch slots must be a positive integer")
        self._owner, self._groups = owner, iter(groups)
        self._slots, self._device_layout, self._threads = slots, device_layout, gather_threads
        self._pending, self._consumed = deque(), deque()
        self._deferred, self._has_deferred = None, False
        self._closed = False
        self._fill(limit=1)

    def _fill(self, limit=None, block=True):
        from .memory import MemoryBudgetError
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
            if self._has_deferred:
                items = self._deferred
            else:
                try:
                    items = next(self._groups)
                except StopIteration:
                    self._groups = None
                    return
            try:
                completion = self._owner.submit(items, device_layout=self._device_layout,
                                                gather_threads=self._threads)
            except MemoryBudgetError:
                self._deferred, self._has_deferred = items, True
                # Only speculation can be deferred. Never wait indefinitely on
                # a caller-held alias or replay a possibly launched transfer.
                if not block or self._pending:
                    return
                raise
            self._deferred, self._has_deferred = None, False
            self._pending.append(completion)

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
        self._groups = self._deferred = None
        self._has_deferred = False
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
        self._wire_restore = None

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

    def restore_wire(self, stored):
        """Restore fresh dense [S,H,D] wire tensors for a qualified consumer.

        The caller explicitly selects this representation. Existing restore()
        keeps its [H,S,D] materialization. Both paths preserve the same values.
        """
        import torch
        from .consumer_layout import PackedWireRestore
        self.owner._check_open()
        if not isinstance(stored, (tuple, list)):
            raise TypeError("stored must be a list or tuple of cache tensors")
        if any(type(t) is not torch.Tensor or t.dtype != self.dtype or t.dim() != 3
               or tuple(t.shape[1:]) != (self.heads, self.head_dim) for t in stored):
            raise ValueError("wire tensors do not match the declared KV dtype/head layout")
        with self.owner._lock:
            self.owner._check_open()
            if self._wire_restore is None:
                self._wire_restore = PackedWireRestore(self.owner.device,
                    max_bytes=self.owner.contexts * self.owner.max_scratch_bytes,
                    max_inflight=self.owner.max_inflight)
                self.owner._wire_restores.add(self._wire_restore)
            return self._wire_restore.restore(stored)

    def drain(self):
        self._retire(block=True)
        if self._wire_restore is not None:
            self._wire_restore.drain()


def _check_output_arena(arena, device):
    if arena is None:
        return
    from .memory import PackedOutputArena
    if not isinstance(arena, PackedOutputArena):
        raise TypeError("output_arena must be a PackedOutputArena or None")
    if arena.device != device:
        raise ValueError("output arena and transfer owner must use the same device")
    if arena.stats()["closed"]:
        raise RuntimeError("output arena is closed")


class _WeightCompletion:
    """One logical weight group executed as bounded native subgroups."""
    def __init__(self, completions, outputs):
        self._completions, self._outputs = completions, outputs

    def done(self):
        return all([completion.done() for completion in self._completions])

    def wait(self):
        for completion in self._completions:
            completion.wait()
        return self._outputs

    def wait_stream(self, stream=None):
        for completion in self._completions:
            completion.wait_stream(stream)
        return self._outputs

    @property
    def report(self):
        reports = [completion.report for completion in self._completions]
        return None if any(report is None for report in reports) else dict(subgroups=reports)


class FastWeights:
    """Low-overhead grouped loads over PreparedWireWeights snapshots."""

    def __init__(self, owner, weights, *, output_arena=None, max_group_members=None, max_prefetch_slots=None):
        owner._check_open()
        _check_output_arena(output_arena, owner.device)
        for name, value, bound in (("max_group_members", max_group_members, 256),
                                    ("max_prefetch_slots", max_prefetch_slots, 8)):
            if value is not None and (type(value) is not int or not 1 <= value <= bound):
                raise ValueError(f"{name} must be an integer in [1, {bound}] or None")
        self.owner, self.weights = owner, weights
        self.output_arena = output_arena
        self.max_group_members, self.max_prefetch_slots = max_group_members, max_prefetch_slots
        self._memory_counts = dict(subgroup_validations=0, subgroups_submitted=0)
        self._entries = {}

    def _items(self, names):
        self.owner._check_open()
        with self.owner._lock:
            self.owner._check_open()
            names = tuple(names)
            items = []
            # fast_entries releases its snapshot lock before any eviction waits.
            for name, (template, source) in zip(names, self.weights.fast_entries(names)):
                entry = self._entries.get(name)
                if entry is None or entry[1] is not source:
                    if entry is not None:
                        self.owner._invalidate_program(entry[0])
                    entry = self._entries[name] = (_WireEntry(self.owner, template, source,
                                                              getattr(self.weights, 'output_dtype', 'float32')),
                                                   source)
                items.append(entry)
            return items

    def submit(self, names, *, outputs=None, device_layout=False, gather_threads=1):
        import torch
        if device_layout or gather_threads not in (None, 1):
            raise ValueError("prepared weights use their validated row with one gather thread")
        self.owner._check_open()
        with self.owner._lock:
            items = self._items(names)
            if not 1 <= len(items) <= 256:
                raise ValueError("a group holds between 1 and 256 transfers")
            split = self.max_group_members is not None and len(items) > self.max_group_members
            if outputs is not None:
                outputs = list(outputs)
                if len(outputs) != len(items):
                    raise ValueError("one output per weight is required")
            elif self.output_arena is not None or split:
                self.owner._retire()
                with torch.cuda.stream(self.owner._stream):
                    outputs = (self.output_arena.allocate([p.destination for p, _ in items])
                               if self.output_arena is not None else
                               self.owner._allocate_outputs([p for p, _ in items]))
            if not split:
                return self.owner.submit(items, outputs=outputs, gather_threads=1)
            # Cross-subgroup aliases and metadata must be rejected before ANY
            # member launches. Submissions still repeat their normal validators.
            try:
                if _native_views is not None:
                    _native_views.group_from_tensors([p.tensor_entry(source, out)
                                                      for (p, source), out in zip(items, outputs)])
                else:
                    pyreloc.prepare_dispatch_group([p.group_entry(source, out)
                                                    for (p, source), out in zip(items, outputs)])
            except pyreloc.TransferError as error:
                raise RuntimeError(f"weight group preflight failed: {error}") from error
            self._memory_counts["subgroup_validations"] += 1
            completions = []
            try:
                for offset in range(0, len(items), self.max_group_members):
                    stop = offset + self.max_group_members
                    completions.append(self.owner.submit(items[offset:stop], outputs=outputs[offset:stop],
                                                         gather_threads=1))
                    self._memory_counts["subgroups_submitted"] += 1
            except BaseException:
                for completion in completions:
                    try:
                        completion.wait()
                    except BaseException:
                        pass  # The native owner quarantines uncertain completion.
                raise
            return _WeightCompletion(completions, outputs)

    def load_many(self, names):
        return self.submit(names).wait()

    def prefetch(self, name_groups, *, slots=2):
        if type(slots) is not int or slots < 1:
            raise ValueError("prefetch slots must be a positive integer")
        depth = slots if self.max_prefetch_slots is None else min(slots, self.max_prefetch_slots)
        return FastPrefetch(self, name_groups, slots=depth, gather_threads=1)

    def stats(self):
        self.owner._check_process()
        with self.owner._lock:
            arena = self.output_arena if self.output_arena is not None else self.owner.output_arena
            return dict(self._memory_counts, max_group_members=self.max_group_members,
                        max_prefetch_slots=self.max_prefetch_slots,
                        output_arena=None if arena is None else arena.stats())


class _WireEntry(TypedProgram):
    """A prepared INT8 [out,in] snapshot with its native template (no rebinding)."""

    __slots__ = ()

    def __init__(self, owner, template, source, dtype="float32"):  # noqa: D401 - adapter
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
        self._register_op()


__all__ = ("FastTransfers", "FastCompletion", "FastPrefetch", "TypedProgram", "LayoutProgram",
           "IndexedProgram", "PooledProgram", "KVCacheSwap", "FastWeights")
