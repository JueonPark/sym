"""Explicit immutable INT8 wire snapshots for inference.

prepare() copies INT8 [in, out] into owned [out, in] storage and snapshots
FP32 scales. Later loads never consult the originals. Mutation, replacement,
load_state_dict and changed scales require another prepare(). Inputs must be
ready and stable for the duration of prepare(); no writable CPU alias escapes.
This deliberately differs from PreparedWeights' conservative live-slot cache.
"""
from dataclasses import dataclass
import os
from threading import RLock
from types import MappingProxyType
import weakref

from .asynchronous import _positive


class _Budget:
    def __init__(self, prepared, pinned, entries):
        self.lock = RLock()
        self.limit, self.pin_limit, self.entry_limit = prepared, pinned, entries
        self.entries = 0
        self.live = self.pinned = self.peak = self.pin_peak = 0

    def reserve(self, size, pinned):
        with self.lock:
            if (self.live + size > self.limit or self.pinned + pinned > self.pin_limit
                    or self.entries >= self.entry_limit):
                raise BufferError('prepared/pinned weight budget exceeded (including retired snapshots)')
            self.entries += 1
            self.live += size
            self.pinned += pinned
            self.peak = max(self.peak, self.live)
            self.pin_peak = max(self.pin_peak, self.pinned)

    def release(self, size, pinned, template=None):
        # The finalizer holds the immutable template for the source's lifetime,
        # including native completion/quarantine owners after invalidation.
        with self.lock:
            self.entries -= 1
            self.live -= size
            self.pinned -= pinned


@dataclass(frozen=True)
class _Entry:
    source: object
    template: object
    bindings: dict
    info: object


class PreparedWireWeights:
    """Bounded snapshots of symmetric INT8, per-output-channel FP32 weights.

    Each load returns fresh contiguous FP32 [out, in] CUDA tensors, with exact
    float32(q) * scale arithmetic. No quantization or FP32 pre-expansion occurs.
    prepare(name, q, scale) atomically replaces a revision, reserving both old
    and new storage until old transfers retire. invalidate() removes future
    lookup; submitted work retains its original revision. Use a context manager.

    max_prepared_bytes charges INT8 storage plus the native template's two
    FP32 scale copies and scalar zero point. Python/plan metadata and temporary
    binding copies are additional bounded overhead (max_entries and shape).
    max_pinned_bytes covers owned INT8 storage; the transfer queue has a separate
    max_scratch_bytes cap that includes its pinned staging. Torch's allocator
    may cache freed pages; these are live ownership limits, not process RSS caps.
    """

    def __init__(self, device='cuda:0', *, compiler=None, pin_memory=True,
                 max_prepared_bytes=1 << 30, max_pinned_bytes=1 << 30,
                 max_entries=4096, max_in_flight=2, max_output_bytes=128 << 20,
                 max_scratch_bytes=64 << 20, gather_threads=8):
        import torch
        from .compiler import CompilerClient
        from .recipe import BindingParam, Dequantize, Recipe, TensorSpec
        from .symbolic import Const, Symbol, dense_strides

        if type(pin_memory) is not bool:
            raise TypeError('pin_memory must be bool')
        self._max_entries = _positive(max_entries, 'max_entries')
        self._budget = _Budget(_positive(max_prepared_bytes, 'max_prepared_bytes'),
                               _positive(max_pinned_bytes, 'max_pinned_bytes'), self._max_entries)
        self._pin = pin_memory
        self._device = torch.device(device)
        if self._device.type != 'cuda':
            raise ValueError('prepared wire weights require a CUDA target')
        self._queue_options = dict(max_in_flight=_positive(max_in_flight, 'max_in_flight'),
            max_output_bytes=_positive(max_output_bytes, 'max_output_bytes'),
            max_scratch_bytes=_positive(max_scratch_bytes, 'max_scratch_bytes'),
            gather_threads=_positive(gather_threads, 'gather_threads'),
            pinning='pinned' if pin_memory else 'pageable')
        self._pid, self._lock = os.getpid(), RLock()
        self._entries, self._queue = {}, None
        self._closed, self._revision = False, 0
        rows, cols = Symbol('out'), Symbol('in')
        def spec(dtype):
            return TensorSpec((rows, cols), dense_strides((rows, cols)), Const(0), dtype)
        recipe = Recipe(spec('int8'), (Dequantize('float32',
            BindingParam('scale', 'float32', (rows,)), None, 0, 'affine'),),
            spec('float32'), 'h2d')
        self._compiled = (compiler or CompilerClient.from_environment()).compile(recipe)

    def _check(self):
        if self._pid != os.getpid():
            raise RuntimeError('prepared wire weights belong to another process')

    def _open(self):
        self._check()
        if self._closed:
            raise RuntimeError('prepared wire weights are closed')

    def prepare(self, name, q, scale):
        """Own a snapshot; return scalar-only metadata. Inputs may change on return.

        q must be a dense CPU INT8 matrix, scale a dense CPU FP32 [out]
        vector with finite positive values and no autograd requirement.
        Invalid input or a budget failure leaves the previous revision intact.
        """
        import torch
        import pyreloc
        from . import compat
        from .dispatch import _parameter_snapshot

        self._check()
        with self._lock:
            self._open()
            if not isinstance(name, str) or not name:
                raise ValueError('name must be a nonempty string')
            if name not in self._entries and len(self._entries) >= self._max_entries:
                raise BufferError('prepared weight entry budget exceeded')
            for value, dtype, rank in ((q, torch.int8, 2), (scale, torch.float32, 1)):
                if (not compat.is_plain_tensor_or_parameter(value) or value.device.type != 'cpu'
                        or value.dtype != dtype or value.ndim != rank or not value.is_contiguous()
                        or value.requires_grad or any(n <= 0 for n in value.shape)):
                    raise ValueError('weights require dense CPU int8[in,out] and float32[out] inference tensors')
            if scale.shape != (q.shape[1],):
                raise ValueError('scale must have one entry per output channel')
            bindings = {'out': q.shape[1], 'in': q.shape[0]}
            wire, parameter = q.numel(), scale.numel() * 4
            size, pinned = wire + 2 * parameter + 4, wire if self._pin else 0
            self._budget.reserve(size, pinned)
            finalizer = None
            try:
                snapshot = _parameter_snapshot('scale', scale, ('float32', [q.shape[1]]))
                bound = pyreloc.bind_typed(self._compiled.decoded_plan, bindings, {'scale': snapshot})
                template = pyreloc.prepare_dispatch_template(bound, 'h2d', 'cuda',
                    implementation='cpu_stages_cuda_stages@0',
                    threads=self._queue_options['gather_threads'])
                # One offline CPU transpose into final compact wire storage.
                # copy_ obeys the caller's Torch thread budget; it never converts.
                packed = torch.empty((q.shape[1], q.shape[0]), dtype=torch.int8,
                                     pin_memory=self._pin)
                finalizer = weakref.finalize(packed, self._budget.release, size, pinned, template)
                packed.copy_(q.t())
                revision = self._revision + 1
                info = MappingProxyType(dict(name=name, revision=revision,
                    shape=tuple(packed.shape), wire_dtype='int8', output_dtype='float32',
                    channel_axis=0, zero_point=0, wire_bytes=wire, parameter_bytes=parameter,
                    payload_bytes=wire + parameter, output_bytes=wire * 4,
                    prepared_bytes=size, pinned_bytes=pinned))
                self._entries[name] = _Entry(packed, template, bindings, info)
                self._revision = revision
                return info
            except BaseException:
                if finalizer is None:
                    self._budget.release(size, pinned)
                raise

    def describe(self, name):
        self._check()
        with self._lock:
            self._open()
            return self._entries[name].info

    def invalidate(self, name=None):
        """Remove future access; in-flight snapshots stay charged until retired."""
        self._check()
        with self._lock:
            self._open()
            if name is None:
                self._entries.clear()
            else:
                del self._entries[name]

    def _transfer_queue(self):
        if self._queue is None:
            import torch
            import pyreloc
            from .asynchronous import TransferQueue
            if not pyreloc.cuda_enabled or not torch.cuda.is_available():
                raise RuntimeError('prepared wire loads require a CUDA-capable runtime')
            self._queue = TransferQueue(self._device, **self._queue_options)
        return self._queue

    def _group(self, names):
        from .dispatch import PreparedTypedTransfer
        from .group import prepare_transfer_group
        from .runtime import ConcreteDescriptor
        from .transport import _storage_view

        self._check()
        with self._lock:
            self._open()
            if isinstance(names, str):
                raise TypeError('use a sequence of weight names')
            names = tuple(names)
            if not 1 <= len(names) <= 256:
                raise ValueError('a group must contain between 1 and 256 weights')
            entries = [self._entries[name] for name in names]
            device = self._transfer_queue().device
            requests = []
            for entry in entries:
                source, native = entry.source, entry.template
                destination = ConcreteDescriptor(tuple(source.shape), tuple(source.stride()), 'float32', device)
                # Strong ownership replaces value freshness checks ONLY here:
                # neither packed source nor bound parameters escape this owner.
                # Ordinary prepare_typed_transfer retains its full recheck.
                requests.append(PreparedTypedTransfer(self._compiled, source, entry.bindings,
                    None, destination, 'h2d', device, _storage_view(source, 'host', -1),
                    'auto', None, self._queue_options['gather_threads'], {}, {},
                    native.capability, native.selection, template=native))
            return prepare_transfer_group(requests)

    def submit(self, names):
        """Submit one named group; return a completion owning that revision."""
        self._check()
        with self._lock:
            return self._transfer_queue_checked().submit(self._group(names))

    def _transfer_queue_checked(self):
        self._open()
        return self._transfer_queue()

    def load_many(self, names):
        """Completed transfer into fresh outputs; return GroupResult with bytes."""
        with self.submit(names) as handle:
            return handle.wait()

    def load(self, name):
        return self.load_many((name,)).tensors[0]

    def prefetch(self, name_groups):
        """Context-managed lookahead over names; revisions resolve on submission."""
        self._check()
        with self._lock:
            queue = self._transfer_queue_checked()
            return queue.prefetch(self._group(names) for names in name_groups)

    def stats(self):
        self._check()
        with self._lock:
            with self._budget.lock:
                result = dict(entries=len(self._entries), live_snapshots=self._budget.entries,
                    revisions=self._revision, closed=self._closed,
                    prepared_bytes=self._budget.live, pinned_bytes=self._budget.pinned,
                    peak_prepared_bytes=self._budget.peak, peak_pinned_bytes=self._budget.pin_peak)
            result['queue'] = None if self._queue is None else self._queue.stats()
            return result

    def close(self):
        self._check()
        with self._lock:
            self._closed = True
            self._entries.clear()
            if self._queue is not None:
                self._queue.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
