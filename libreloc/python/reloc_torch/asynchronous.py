"""Bounded explicit transfer submission and consumer-stream ordering.

CPU preparation is synchronous; GPU work completes through an owned native
event. CPU sources must be ready and must not be mutated until wait()/done().
D2H tensors are exposed only after host completion. This API does not change
torch.compile's blocking tensor-return contract or its non_blocking fallback.
"""
from collections import deque
import operator
import os
from threading import RLock
from types import MappingProxyType

from .group import GroupResult, PreparedTransferGroup, _execute_transfer_group

# Only consumer-stream completion failures land here. The native runtime owns
# its separate producer/scratch quarantine. Unknown work must never see freed
# tensor storage, even if every public queue/handle is garbage collected.
_QUARANTINE = []


def _positive(value, name):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    value = operator.index(value)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _output_bytes(group):
    import torch
    return sum((0 if 0 in d.shape else 1 + sum((n-1)*s for n, s in zip(d.shape, d.strides)))
               * getattr(torch, d.dtype).itemsize for d in group.destinations)


class TransferCompletion:
    """One submitted group, retaining input/output owners until safe.

    wait_stream() returns GPU tensors with a consumer-stream dependency and
    records their allocator use. It does not authorize CPU source mutation.
    release() records the end of consumer use; close()/cancel() drain both the
    producer and those consumer events before releasing the queue's reservation.
    Cancellation cannot retract CUDA work already submitted.
    """

    def __init__(self, queue, group, native, tensors, descriptors, size):
        self._queue, self._group, self._native = queue, group, native
        self._tensors, self._descriptors, self._size = tensors, descriptors, size
        self._consumers, self._retired = {}, []
        self._released = self._closed = False

    def _check(self):
        self._queue._check_process()
        if self._closed:
            raise RuntimeError('transfer completion is closed')

    def done(self):
        """Poll and finalize completed GPU work, including any D2H CPU transform."""
        self._queue._check_process()
        with self._queue._lock:
            self._check()
            return self._native.query()

    def wait(self):
        """Host wait; returned CPU tensors and CPU source mutation are now safe."""
        self._queue._check_process()
        with self._queue._lock:
            self._check()
            self._native.wait()
            # Host completion makes values ready, but outputs were allocated
            # on another stream. Register ordinary current-stream consumption
            # too, so wait().tensors has the same safe lifetime as wait_stream().
            if not self._released and any(t.is_cuda for t in self._tensors):
                import torch
                self._record_consumer(torch.cuda.current_stream(self._queue.device))
            return GroupResult(self._tensors, self._descriptors,
                               MappingProxyType(dict(self._group.native.report)))

    def wait_stream(self, stream=None):
        """H2D-only GPU dependency; does not synchronize the host."""
        import torch
        self._queue._check_process()
        with self._queue._lock:
            self._check()
            if self._released:
                raise RuntimeError('consumer use was already released')
            stream = torch.cuda.current_stream(self._queue.device) if stream is None else stream
            self._queue._check_stream(stream)
            self._native.wait_stream(stream.cuda_stream)
            self._record_consumer(stream)
            return self._tensors

    def _record_consumer(self, stream):
        for tensor in self._tensors:
            if tensor.is_cuda:
                tensor.record_stream(stream)
        self._consumers[stream.cuda_stream] = stream

    def release(self):
        """Record consumer progress immediately after enqueueing its last use.

        The slot stays reserved until close(); subsequent layers on the same
        consumer stream do not extend this completion's retirement dependency.
        """
        import torch
        self._queue._check_process()
        with self._queue._lock:
            self._check()
            if not self._released:
                for stream in self._consumers.values():
                    event = torch.cuda.Event()
                    event.record(stream)
                    self._retired.append(event)
                self._released = True

    def close(self):
        self._queue._check_process()
        with self._queue._lock:
            if self._closed:
                return
            error = None
            try:
                self.release()
            except Exception as failure:
                error = failure
            try:
                self._native.wait()
                self._queue._record(dict(self._group.native.report))
            except Exception as failure:
                # Native failures drain or quarantine all producer owners.
                error = error or failure
            try:
                if self._released:
                    for event in self._retired:
                        event.synchronize()
                else:
                    for stream in self._consumers.values():
                        stream.synchronize()
            except Exception as failure:
                _QUARANTINE.append((self._tensors, self._group, self._native,
                                    tuple(self._consumers.values()), tuple(self._retired)))
                error = error or failure
                self._queue._closed = True
            self._closed = True
            self._native = self._group = None
            self._tensors, self._descriptors = (), ()
            self._consumers.clear()
            self._retired.clear()
            self._queue._release(self)
            if error is not None:
                raise error

    cancel = close

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TransferQueue:
    """One CUDA device/copy queue with bounded outstanding output reservations.

    CPU submissions to this queue serialize. Optional resources can be shared
    across queues: multiple typed contexts allow independent native producers,
    while admission completes an older producer when all slots are occupied.
    Closing a queue drains its handles and closes only resources it created.
    Already enqueued consumers run concurrently on their own streams. Fresh
    outputs never alias earlier outputs. A caller
    retaining tensors after close() owns that memory outside the queue's budget.
    Use an explicit context manager to drain, including on model exceptions.
    """

    def __init__(self, device='cuda:0', *, max_in_flight=2, max_output_bytes=128 << 20,
                 max_scratch_bytes=64 << 20, gather_threads=8, pinning='pinned',
                 resources=None):
        import torch
        from .resources import TransferResources
        if resources is not None:
            if not isinstance(resources, TransferResources):
                raise TypeError('resources must be TransferResources')
            resources.native_typed  # reject inherited use before any CUDA access
            if resources.closed:
                raise RuntimeError('transfer resources are closed')
        self.max_in_flight = _positive(max_in_flight, 'max_in_flight')
        self.max_output_bytes = _positive(max_output_bytes, 'max_output_bytes')
        self.max_scratch_bytes = _positive(max_scratch_bytes, 'max_scratch_bytes')
        self.gather_threads = _positive(gather_threads, 'gather_threads')
        if pinning not in ('pinned', 'pageable'):
            raise ValueError('pinning must be pinned or pageable')
        self.pinning = pinning
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('a transfer queue requires a CUDA device')
        self.device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        self._pid, self._lock = os.getpid(), RLock()
        self._closed = False
        self._held, self._bytes = set(), 0
        self._submitted = self._peak_bytes = self._peak_handles = 0
        self._totals = dict(groups=0, logical_transfers=0, source_bytes=0, wire_bytes=0,
                            destination_bytes=0, payload_bytes_transferred=0, dispatches={})
        # Allocate outputs on a dedicated stream. Ignoring the allocator's
        # stream would let a private DMA race with a recycled caller-stream block.
        self._stream = torch.cuda.Stream(device=self.device)
        self._owns_resources = resources is None
        self._resources = resources if resources is not None else TransferResources(max_typed_retained_bytes=max_scratch_bytes,
            max_typed_live_bytes=max_scratch_bytes, max_typed_background_workers=gather_threads - 1,
            max_typed_streams=1)

    def _check_process(self):
        if self._pid != os.getpid():
            raise RuntimeError('transfer queue belongs to another process')

    def _check_stream(self, stream):
        import torch
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
            raise ValueError('stream must belong to the transfer queue device')

    def can_submit(self, group):
        self._check_process()
        with self._lock:
            return (not self._closed and len(self._held) < self.max_in_flight and
                    self._bytes + _output_bytes(group) <= self.max_output_bytes)

    def submit(self, group, *, producer_stream=None):
        """Submit a fresh group; CPU input values must already be ready.

        CUDA sources default to ordering after the caller's current stream;
        CPU-only sources are independent of model compute. An explicit producer
        stream orders GPU accesses, not CPU reads of asynchronously produced RAM.
        """
        import torch
        self._check_process()
        if not isinstance(group, PreparedTransferGroup):
            raise TypeError('submit expects a PreparedTransferGroup')
        with self._lock:
            if self._closed:
                raise RuntimeError('transfer queue is closed')
            if group.device != self.device:
                raise ValueError('group device differs from the transfer queue')
            size = _output_bytes(group)
            if not self.can_submit(group):
                raise BufferError('transfer queue output/slot budget exceeded; close a prior completion')
            if producer_stream is None and any(r.direction == 'd2h' for r in group.requests):
                producer_stream = torch.cuda.current_stream(self.device)
            if producer_stream is not None:
                self._check_stream(producer_stream)
                self._stream.wait_stream(producer_stream)
            with torch.cuda.stream(self._stream):
                native, tensors, descriptors = _execute_transfer_group(group, submit=True,
                    resources=self._resources, gather_threads=self.gather_threads,
                    pinning=self.pinning, max_scratch_bytes=self.max_scratch_bytes)
            handle = TransferCompletion(self, group, native, tensors, descriptors, size)
            self._held.add(handle)
            self._bytes += size
            self._submitted += 1
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            self._peak_handles = max(self._peak_handles, len(self._held))
            return handle

    def _release(self, handle):
        self._held.remove(handle)
        self._bytes -= handle._size

    def _record(self, report):
        self._totals['groups'] += 1
        self._totals['logical_transfers'] += report['logical_transfers']
        for item in report['items']:
            for name in ('source_bytes', 'wire_bytes', 'destination_bytes', 'payload_bytes_transferred'):
                self._totals[name] += item[name]
            choices = self._totals['dispatches']
            label = item['implementation']
            choices[label] = choices.get(label, 0) + 1

    def stats(self):
        self._check_process()
        with self._lock:
            return dict(submitted=self._submitted, held=len(self._held), output_bytes=self._bytes,
                        peak_output_bytes=self._peak_bytes, peak_in_flight=self._peak_handles,
                        closed=self._closed, resources=self._resources.stats()['typed'],
                        owns_resources=self._owns_resources,
                        totals=self._totals | {'dispatches': dict(self._totals['dispatches'])})

    def prefetch(self, groups):
        return PrefetchWindow(self, groups)

    def close(self):
        self._check_process()
        with self._lock:
            self._closed = True
            error = None
            for handle in tuple(self._held):
                try:
                    handle.close()
                except Exception as failure:
                    error = error or failure
            try:
                if self._owns_resources:
                    self._resources.close()
            except Exception as failure:
                error = error or failure
            if error is not None:
                raise error

    def __enter__(self):
        self._check_process()
        if self._closed:
            raise RuntimeError('transfer queue is closed')
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class PrefetchWindow:
    """Submit the next consumer while the previously yielded consumer runs.

    Use as a context manager, including for early loop exit. Each iteration
    records the preceding consumer's end, retires the oldest slot if needed,
    and submits one next group. No item is prepared before its iterable yields
    it, so routing can explicitly determine the eligible expert sequence.
    """

    def __init__(self, queue, groups):
        self._queue, self._groups = queue, iter(groups)
        self._handles, self._last = deque(), None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._last is not None:
            self._last.release()
            self._last = None
        try:
            group = next(self._groups)
        except StopIteration:
            self.close()
            raise
        while not self._queue.can_submit(group) and self._handles:
            self._handles.popleft().close()
        handle = self._queue.submit(group)
        self._handles.append(handle)
        self._last = handle
        return handle.wait_stream()

    def close(self):
        if self._closed:
            return
        error = None
        for handle in self._handles:
            try:
                handle.close()
            except Exception as failure:
                error = error or failure
        self._handles.clear()
        self._last, self._closed = None, True
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
