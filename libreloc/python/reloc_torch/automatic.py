"""Compiler-derived prepared execution; immutable cache entries, live calls.

No capture recipes or workload names live here. Native capability is the
legality authority; the optional device-row preference is a heuristic.
"""
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import os
from threading import RLock
from time import perf_counter_ns

from .artifact import UnsupportedRecipe
from . import compat


@dataclass(frozen=True)
class PreparedExecutionPolicy:
    """Bound retained specializations and admit work by bytes/repeated use.

    ``min_reuses=1`` runs the first occurrence through the original region.
    Bounds include pending admissions and negative entries. Parameter bytes
    count copied immutable keys, not native scratch or outstanding calls.
    ``row_policy='cpu'`` is a reference control for qualification.
    """
    max_entries: int = 32
    max_parameter_bytes: int = 256 << 10
    min_bytes: int = 64 << 10
    min_reuses: int = 1
    row_policy: str = 'device'

    def __post_init__(self):
        for name in ('max_entries', 'max_parameter_bytes', 'min_bytes', 'min_reuses'):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f'{name} must be an integer')
            if value < (1 if name == 'max_entries' else 0):
                raise ValueError(f'{name} is below its minimum')
        if self.row_policy not in ('device', 'cpu'):
            raise ValueError('row_policy must be device or cpu')


def normalize_policy(value):
    if value is False:
        return None
    if value is True:
        return PreparedExecutionPolicy()
    if isinstance(value, PreparedExecutionPolicy):
        return value
    raise TypeError('prepared_execution must be bool or PreparedExecutionPolicy')


@dataclass
class _Specialization:
    """No live tensor references, including for admission/negative entries."""
    parameter_bytes: int
    seen: int = 0
    group: object = None
    program: object = None
    owner: object = None
    rejection: tuple = None
    ready: bool = False


@dataclass
class _Invocation:
    manager: object
    entry: _Specialization
    compiled: object
    source: object
    bindings: dict
    destination: object
    parameters: dict
    snapshots: dict
    declared: dict
    index: object = None
    index_snapshot: object = None
    indexed_program: object = None
    source_view: object = None
    consumed: bool = False
    source_snapshot: tuple = field(init=False)
    source_flags: tuple = field(init=False)

    def __post_init__(self):
        import torch
        self.source_snapshot = compat.storage_snapshot(self.source)
        self.source_flags = (self.source.requires_grad, torch.is_grad_enabled(), self.source.is_neg(), self.source.is_conj())

    def recheck(self):
        import torch
        from .dispatch import _parameter_snapshot
        if (compat.storage_snapshot(self.source) != self.source_snapshot or
            (self.source.requires_grad, torch.is_grad_enabled(), self.source.is_neg(), self.source.is_conj()) != self.source_flags):
            raise RuntimeError('stale prepared execution: source metadata changed')
        for name, value in self.parameters.items():
            try:
                current = _parameter_snapshot(name, value, self.declared[name])
            except UnsupportedRecipe as error:
                raise RuntimeError(f'stale prepared execution: {error}') from error
            if current != self.snapshots[name]:
                raise RuntimeError(f'stale prepared execution: parameter {name!r} changed')
        if self.index is not None:
            from .index_select import index_snapshot
            if index_snapshot(self.index) != self.index_snapshot:
                raise RuntimeError('stale prepared execution: index values or metadata changed')


class AutomaticPreparedExecution:
    def __init__(self, policy):
        self.policy = policy
        self._pid, self._lock = os.getpid(), RLock()
        self._closed = False
        self._cache, self._owners = OrderedDict(), {}
        self._parameter_bytes = 0
        self._counts = Counter(dict.fromkeys(('attempts', 'preparations', 'cache_hits', 'cache_misses',
            'evictions', 'executions', 'parameter_snapshots', 'parameter_snapshot_bytes',
            'index_bindings', 'preparation_ns', 'parameter_snapshot_ns'), 0))
        self._exclusions, self._bypasses = Counter(), Counter()
        self._final_native = None

    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError('process_mismatch: automatic preparation belongs to another process')

    def _require_open(self):
        self._check_process()
        if self._closed:
            raise RuntimeError('automatic prepared execution is closed')

    def _exclude(self, reason):
        self._exclusions[reason] += 1
        raise UnsupportedRecipe('prepared_' + reason, reason)

    def _owner(self, adapter, device, typed):
        from .fast import FastTransfers, _load_native_views
        if _load_native_views() is None:
            raise UnsupportedRecipe('prepared_native_bridge_unavailable', 'native Torch bridge required')
        key = (device.index, typed)
        if key not in self._owners:
            resources = adapter._prepared_resources()
            opts = adapter._transfer_options
            pinning = opts.get('pinning', 'auto')
            # Unconfigured auto is conservative/pageable in the legacy adapter.
            if pinning == 'auto':
                pinning = 'pageable'
            contexts = resources._configuration['max_typed_contexts_per_device']
            self._owners[key] = FastTransfers(device, resources=resources, contexts=max(1, contexts),
                gather_threads=opts.get('gather_threads', 8), pinning=pinning,
                n_buffers=opts.get('n_buffers', 2), max_prepared_groups=self.policy.max_entries)
        return self._owners[key]

    def prepare(self, adapter, compiled, source, device, *, non_blocking=False,
                parameters=None, index=None, implementation=''):
        """Return a call-owned request; None preserves an explicitly configured legacy path."""
        self._require_open()
        with self._lock:
            self._require_open()
            self._counts['attempts'] += 1
            opts = adapter._transfer_options
            bypass = None
            if adapter.placement is not None or implementation:
                bypass = 'placement'
            elif not adapter._owns_resources and adapter._resources is None:
                bypass = 'per_call_resources'
            elif (opts.get('gather_pool') is not None or opts.get('min_pinned_bytes') is not None
                  or opts.get('n_streams', 1) != 1 or opts.get('gather_threads', 8) == 0):
                bypass = 'transfer_options'
            elif compiled.recipe.direction != 'h2d':
                bypass = 'direction'
            if bypass is not None:
                self._bypasses[bypass] += 1
                return None
            started = perf_counter_ns()
            try:
                return self._prepare(adapter, compiled, source, device, non_blocking, parameters, index)
            except UnsupportedRecipe as error:
                # Admission exclusions count themselves; native/metadata errors
                # also remain visible without retaining exception tracebacks.
                if not error.reason.startswith('prepared_'):
                    self._exclusions[error.reason] += 1
                raise
            finally:
                self._counts['preparation_ns'] += perf_counter_ns() - started

    def _prepare(self, adapter, compiled, source, device, non_blocking, parameters, index):
        import torch
        import pyreloc
        from .dispatch import _parameter_snapshot
        from .fast import LayoutProgram, TypedProgram
        from .runtime import (_validated_preparation, source_reason, bind_symbols,
                              destination_descriptor, _local)
        from .symbolic import GuardError
        from .transport import _code, _storage_view
        if non_blocking:
            raise UnsupportedRecipe('nonblocking_unavailable', 'blocking transfers only')
        shared = _validated_preparation(compiled, source)
        if shared is None:
            reason = source_reason(source)
            if reason:
                raise UnsupportedRecipe(reason, reason)
        device = torch.device(device)
        if source.device.type != 'cpu' or device.type != 'cuda':
            raise UnsupportedRecipe('direction_mismatch', 'prepared execution requires H2D')
        if source.is_neg() or source.is_conj():
            self._exclude('view_flags')
        if compiled.indexed != (index is not None):
            raise UnsupportedRecipe('index_operand_required', 'index operand must match the compiler artifact')
        if not pyreloc.cuda_enabled or not torch.cuda.is_available():
            raise UnsupportedRecipe('cuda_unavailable', 'no CUDA-capable runtime')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        if shared is not None:
            bindings = shared[0]
        elif index is not None:
            from .index_select import bind_index_select
            bindings = bind_index_select(compiled, source, index)
        else:
            bindings = bind_symbols(compiled, source)
        destination = destination_descriptor(compiled, bindings, device)
        if source.numel() * source.element_size() < self.policy.min_bytes:
            self._exclude('small')
        parameters = dict(parameters or {})
        declared = compiled.parameter_extents(bindings) if compiled.typed and not compiled.indexed else {}
        if set(parameters) != set(declared):
            raise UnsupportedRecipe('parameter_binding', 'runtime operands differ from compiler declarations')
        started = perf_counter_ns()
        snapshots = {name: _parameter_snapshot(name, value, declared[name]) for name, value in parameters.items()}
        if snapshots:
            self._counts['parameter_snapshots'] += 1
            self._counts['parameter_snapshot_ns'] += perf_counter_ns() - started
        revision = tuple((name, dtype, tuple(extents), data) for name, (dtype, extents, data) in sorted(snapshots.items()))
        nbytes = sum(len(data) for _, _, _, data in revision)
        self._counts['parameter_snapshot_bytes'] += nbytes
        if nbytes > self.policy.max_parameter_bytes:
            self._exclude('parameter_budget')
        # Plan bytes include parameter coordinates and operations; metadata
        # excludes allocation identity and all live source/index values.
        key = (compiled.plan_bytes, tuple(sorted(bindings.items())), tuple(source.shape), tuple(source.stride()),
               compat.dtype_name(source.dtype), source.storage_offset(), device.index, revision,
               None if index is None else (tuple(index.shape), tuple(index.stride()), compat.dtype_name(index.dtype)))
        entry = self._cache.get(key)
        if entry is None:
            self._counts['cache_misses'] += 1
            while self._cache and (len(self._cache) >= self.policy.max_entries or
                                  self._parameter_bytes + nbytes > self.policy.max_parameter_bytes):
                _, evicted = self._cache.popitem(last=False)
                self._parameter_bytes -= evicted.parameter_bytes
                self._counts['evictions'] += 1
                # Drop cache ownership only. A preflighted invocation may hold
                # this entry, so closing the group here would invalidate it.
            entry = self._cache[key] = _Specialization(nbytes)
            self._parameter_bytes += nbytes
        else:
            self._cache.move_to_end(key)
            if entry.ready:
                self._counts['cache_hits'] += 1
        entry.seen += 1
        if entry.seen <= self.policy.min_reuses:
            self._exclude('first_use')
        if entry.rejection is not None:
            raise UnsupportedRecipe(*entry.rejection)
        if not entry.ready:
            try:
                owner = self._owner(adapter, device, compiled.typed or compiled.indexed)
                entry.owner = owner
                if compiled.indexed:
                    pass  # Shape entry only: native index program belongs to this invocation.
                elif compiled.typed:
                    bound = pyreloc.bind_typed(compiled.decoded_plan, bindings, snapshots)
                    capability = pyreloc.query_capability(bound, 'h2d', 'cuda')
                    rows = capability['eligible']
                    if self.policy.row_policy == 'cpu':
                        implementation = 'cpu_reference'
                    else:
                        # Prefer a legal device layout, then fewer CPU stages.
                        # Not a calibrated cost model; controls remain public.
                        chosen = min(rows, key=lambda row: (row['implementation'] == 'cpu_reference',
                            row['wire_result_layout'], row['wire_boundary'], row['wire_bytes'], row['device_temp_bytes']))
                        implementation = chosen['implementation']
                    template = pyreloc.prepare_dispatch_template(bound, 'h2d', 'cuda',
                        threads=owner.gather_threads, implementation=implementation)
                    entry.program = TypedProgram._from_template(owner, compiled, source, bindings,
                                                                 bound, template, nbytes)
                    entry.group = owner.prepare_group([entry.program])
                else:
                    entry.program = LayoutProgram(owner, compiled, source.detach() if source.requires_grad else source)
                    entry.group = owner.prepare_group([entry.program])
                entry.ready = True
                self._counts['preparations'] += 1
                diagnostics = getattr(_local, 'diagnostics', None)
                if diagnostics is not None:
                    diagnostics.increment('symbol_binds')
            except (pyreloc.BindError, pyreloc.TransferError, GuardError) as error:
                reason = 'bind_error' if isinstance(error, pyreloc.BindError) else _code(error)
                entry.rejection = (reason, str(error))
                raise UnsupportedRecipe(*entry.rejection) from error
            except UnsupportedRecipe as error:
                entry.rejection = (error.reason, str(error))
                raise
        invocation = _Invocation(self, entry, compiled, source, bindings, destination, parameters, snapshots, declared,
                                 index=index)
        if index is not None:
            from .index_select import index_snapshot
            invocation.index_snapshot = index_snapshot(index)
            invocation.source_view = _storage_view(source, 'host', -1)
            try:
                bound = pyreloc.bind_indexed(compiled.decoded_plan, bindings,
                    (compat.dtype_name(index.dtype), list(index.shape), invocation.index_snapshot[1]))
                invocation.indexed_program = pyreloc.prepare_index_select_program(bound, invocation.source_view)
                self._counts['index_bindings'] += 1
            except pyreloc.BindError as error:
                raise UnsupportedRecipe('index_out_of_range' if 'index_out_of_range' in str(error) else 'bind_error', str(error)) from error
            except pyreloc.TransferError as error:
                raise UnsupportedRecipe(_code(error), str(error)) from error
        return invocation

    def execute(self, invocation):
        import torch
        self._require_open()
        # Serialize close and per-manager execution; the existing resource
        # owner may still be used concurrently by independent adapters.
        with self._lock:
            self._require_open()
            if invocation.manager is not self:
                raise RuntimeError('prepared invocation belongs to another manager')
            if invocation.consumed:
                raise RuntimeError('prepared invocation was already executed')
            invocation.recheck()
            invocation.entry.owner._check_open()
            invocation.consumed = True
            if invocation.index is None:
                # Inference under no_grad accepts Parameter inputs. The native
                # binder requires a tensor without autograd history; detach
                # preserves this call's guarded storage and current values.
                source = invocation.source.detach() if invocation.source.requires_grad else invocation.source
                out = invocation.entry.group.submit([source], device_layout=True).wait()[0]
            else:
                d = invocation.destination
                out = torch.empty_strided(d.shape, d.strides, dtype=getattr(torch, d.dtype), device=d.device)
                out = invocation.entry.owner._execute_program(invocation.indexed_program, invocation.source_view,
                    invocation.source, out, (invocation.index,), None)
            self._counts['executions'] += 1
            return out

    def _native_stats(self):
        totals = dict(groups=0, logical_transfers=0, payload_bytes=0, report_materializations=0,
                      typed_executions=0, typed_payload_bytes=0, dispatches={}, typed_dispatches={}, prepared_groups={})
        for (_, typed), owner in self._owners.items():
            stats = owner.stats()
            for name in ('groups', 'logical_transfers', 'payload_bytes', 'report_materializations'):
                totals[name] += stats[name]
            # Indexed execution necessarily materializes one legacy report.
            totals['report_materializations'] += stats['calls']
            for name in ('dispatches', 'prepared_groups'):
                for key, value in stats[name].items():
                    totals[name][key] = totals[name].get(key, 0) + value
            if typed:
                totals['typed_executions'] += stats['logical_transfers']
                totals['typed_payload_bytes'] += stats['payload_bytes']
                for row, count in stats['dispatches'].items():
                    totals['typed_dispatches'][row] = totals['typed_dispatches'].get(row, 0) + count
        return totals

    def stats(self):
        self._check_process()
        with self._lock:
            return dict(self._counts, policy=asdict(self.policy), cache_entries=len(self._cache),
                retained_parameter_bytes=self._parameter_bytes, exclusions=dict(self._exclusions),
                legacy_bypasses=dict(self._bypasses), closed=self._closed,
                staging_reporting='aggregate_resource_counters; legacy per-call explanations excluded',
                native=deepcopy(self._final_native) if self._final_native is not None else self._native_stats())

    def close(self):
        self._check_process()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._parameter_bytes = 0
            self._final_native = self._native_stats()
            error = None
            for owner in self._owners.values():
                try:
                    owner.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
            self._owners.clear()
            if error is None:
                self._final_native['prepared_groups']['live_slots'] = 0
                self._final_native['prepared_groups']['cached_handles'] = 0
            if error is not None:
                raise error
