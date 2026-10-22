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
"""
from __future__ import annotations

from collections import deque
import os
from threading import RLock

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
                 "_view_kind", "_view_device", "_out_dtype", "_out_device")

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

    def group_entry(self, source, out):
        return (self.template, self.source_view(source), self.output_view(out), self.direction)

    def tensor_entry(self, source, out):
        return (self.template, source, out, self.direction, self.signature_dtype)

    def execute(self, source, *, out=None, pipeline=True, gather_threads=None):
        """Blocking single transfer through the prepared row (chunked CPU/DMA pipeline for CPU rows)."""
        owner = self.owner
        owner._check_open()
        if out is None:
            out = self.allocate()
        try:
            if _native_views is not None:
                native = _native_views.request_from_tensors(self.template, source, out, self.signature_dtype)
            else:
                self.check(source)
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

    __slots__ = ("owner", "compiled", "_specs")

    def __init__(self, owner, compiled):
        if not compiled.indexed or compiled.recipe.direction != "h2d":
            raise UnsupportedRecipe("unsupported_index_select", "expected an H2D index_select recipe")
        self.owner, self.compiled = owner, compiled
        self._specs = {}

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

    def execute(self, source, index, *, out=None, gather_threads=None):
        """Blocking fused row gather/cast/H2D; the index tensor is read in place."""
        program, src_view, out = self._prepare(source, index, out)
        return self.owner._execute_program(program, src_view, source, out, (index,), gather_threads)

    def submit(self, source, index, *, out=None, gather_threads=None):
        """Asynchronous fused gather; wait_stream() on the completion yields the output."""
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

    __slots__ = ("owner", "tables", "dim", "dtype", "_views", "_out_dtype")

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

    def execute(self, indices, offsets, *, out=None, gather_threads=None):
        """Blocking pooled gather; returns the [bags, tables, dim] CUDA tensor."""
        program, out, extra = self._prepare(indices, offsets, out)
        return self.owner._execute_program(program, self._views[0], self.tables[0], out, extra, gather_threads)

    def submit(self, indices, offsets, *, out=None, gather_threads=None):
        """Asynchronous pooled gather; wait_stream() on the completion yields the output."""
        program, out, extra = self._prepare(indices, offsets, out)
        return self.owner._submit_program(program, self._views[0], self.tables[0], out, extra, gather_threads,
                                          "float32")


class FastCompletion:
    """One submitted group; outputs are handed out by wait()/wait_stream()."""

    __slots__ = ("_owner", "_native", "_outputs", "_group", "_host", "_done")

    def __init__(self, owner, native, outputs, group, host):
        self._owner, self._native, self._outputs, self._group, self._host = owner, native, outputs, group, host
        self._done = False

    @property
    def report(self):
        return dict(self._group.report) if self._done else None

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
            if not self._host:
                import torch
                stream = torch.cuda.current_stream(self._owner.device)
                for tensor in self._outputs:
                    tensor.record_stream(stream)
        return self._outputs

    def wait_stream(self, stream=None):
        """GPU-side dependency only (H2D groups): the current stream waits for the transfer.

        The completion stays registered with its owner until the native work
        finished, so dropping this object never turns into a host wait; the
        owner retires it on a later submission or at close().
        """
        import torch
        if self._host:
            return self.wait()
        if stream is None:
            stream = torch.cuda.current_stream(self._owner.device)
        if not self._done:
            self._native.wait_stream(stream.cuda_stream)
            self._owner._retain(self)
        for tensor in self._outputs:
            tensor.record_stream(stream)
        return self._outputs

    def _finish(self):
        self._done = True
        self._owner._record_group(self._group.report)
        self._native = None


class FastTransfers:
    """Owner of prepared programs and the native resources they execute on.

    ``contexts`` independent native contexts let that many groups be in flight
    at once (each owns a stream and a share of the scratch budget); one
    context serializes submissions. Use as a context manager or call close().
    """

    def __init__(self, device="cuda:0", *, compiler=None, gather_threads=8, pinning="pinned",
                 max_scratch_bytes=64 << 20, n_buffers=2, chunk_size=None, contexts=2):
        import torch
        from .compiler import CompilerClient
        if not pyreloc.cuda_enabled or not torch.cuda.is_available():
            raise RuntimeError("fast transfers require a CUDA-capable runtime")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("fast transfers require a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        for name, value in (("gather_threads", gather_threads), ("contexts", contexts), ("n_buffers", n_buffers),
                            ("max_scratch_bytes", max_scratch_bytes)):
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
        self._resources = TransferResources(
            max_typed_retained_bytes=max_scratch_bytes * contexts, max_typed_live_bytes=max_scratch_bytes * contexts,
            max_typed_background_workers=max(1, gather_threads - 1) * contexts, max_typed_streams=contexts,
            max_typed_contexts=contexts, max_typed_contexts_per_device=contexts)
        self._stream = torch.cuda.Stream(device=self.device)
        self._totals = dict(calls=0, groups=0, logical_transfers=0, payload_bytes=0, dispatches={})
        self.native_views = _load_native_views() is not None
        self._inflight = deque()
        self.max_inflight = 8

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
            try:
                self._retire(block=True)
            finally:
                self._resources.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def stats(self):
        self._check_open()
        return dict(self._totals, dispatches=dict(self._totals["dispatches"]),
                    resources=self._resources.stats()["typed"])

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
    def _execute_program(self, program, src_view, source, out, extra, gather_threads):
        """Blocking single dispatch of a freshly prepared producer program."""
        try:
            native = pyreloc.prepare_dispatch(program, src_view, _storage_view(out, "cuda", out.device.index),
                                              "h2d", threads=self.gather_threads)
            report = pyreloc.execute_dispatch(
                native, caller_stream=compat.cuda_stream_handle(self.device),
                n_buffers=self.n_buffers, n_streams=1,
                gather_threads=self.gather_threads if gather_threads is None else gather_threads,
                pinning=self.pinning, direct_dense_upload=True, pipeline=True,
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

    # -- groups ----------------------------------------------------------
    def submit(self, items, *, outputs=None, device_layout=False, gather_threads=None, pin_outputs=True,
               extra_owners=(), chunked_members=False):
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
        sources, outs, entries = [], [], []
        host = False
        native_views = _native_views is not None
        with torch.cuda.stream(self._stream):
            for position, (program, source) in enumerate(items):
                if not isinstance(program, (TypedProgram, LayoutProgram, _OneShot)):
                    raise TypeError("group members are typed or layout programs")
                out = program.allocate(pin=pin_outputs) if outputs is None else outputs[position]
                host |= program.direction == "d2h"
                sources.append(source)
                outs.append(out)
                if native_views:
                    entries.append(program.tensor_entry(source, out))
                else:
                    program.check(source)
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
        return FastCompletion(self, completion, outs, native, host)


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
        from .recipe import Recipe, TensorSpec, Transpose
        from .symbolic import Const, Symbol, dense_strides
        if dtype != "float32":
            raise ValueError("KV swap relocates float32 caches")
        self.owner, self.heads, self.head_dim = owner, heads, head_dim
        s = Symbol("s0")
        cache, stored = (Const(heads), s, Const(head_dim)), (s, Const(heads), Const(head_dim))

        def spec(shape):
            return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
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

    def evict(self, caches):
        """One packed pinned block holds every evicted cache; returns views of it."""
        import torch
        caches = list(caches)
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
        return list(completion.wait())

    def restore(self, stored):
        """GPU caches ordered on the current stream; the host is not blocked."""
        stored = list(stored)
        self._retire()
        completion = self.owner.submit([(self._program(self._restore_recipe, t), t) for t in stored],
                                       device_layout=True, gather_threads=1)
        outputs = list(completion.wait_stream())
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
        for name, (template, source) in zip(names, self.weights.fast_entries(names)):
            entry = self._entries.get(name)
            if entry is None or entry[1] is not source:
                # A re-prepared snapshot is a new source object; rebuild its member.
                entry = self._entries[name] = (_WireEntry(self.owner, template, source), source)
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
    """A prepared INT8 [out,in] snapshot with its native template (no rebinding)."""

    __slots__ = ()

    def __init__(self, owner, template, source):  # noqa: D401 - adapter
        import torch
        self.owner, self.compiled, self.bindings = owner, None, None
        self.signature = _signature(source)
        self.signature_dtype = "int8"
        self.direction = "h2d"
        self._view_kind, self._view_device = "host", -1
        self.template, self.bound, self.selection = template, None, dict(template.selection)
        self.parameter_bytes = 0
        from .runtime import ConcreteDescriptor
        self.destination = ConcreteDescriptor(tuple(source.shape), tuple(source.stride()), "float32", owner.device)
        self._out_dtype, self._out_device = torch.float32, owner.device


__all__ = ("FastTransfers", "FastCompletion", "FastPrefetch", "TypedProgram", "LayoutProgram",
           "IndexedProgram", "PooledProgram", "KVCacheSwap", "FastWeights")
