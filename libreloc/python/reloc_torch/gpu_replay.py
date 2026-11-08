"""Bounded, explicit, pure GPU regions; transfers and routing stay outside capture.

All mutable state must be an argument. The callable must be functional and
capture safe. Prepare during a quiescent phase, then freeze before concurrent
serving. Outputs are copies, never leases on graph-private storage.
"""
from collections import Counter
from dataclasses import dataclass
import math
import os
import threading
import time

_CAPTURE_LOCK = threading.RLock()


@dataclass(frozen=True)
class GPUReplayPolicy:
    enabled: bool = True
    slots: int = 2
    max_buckets: int = 8
    max_bytes: int = 256 << 20

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError('enabled must be bool')
        for name, maximum in (('slots', 8), ('max_buckets', 128), ('max_bytes', (1 << 63) - 1)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1, {maximum}]')


@dataclass
class _Slot:
    inputs: list
    spec: object
    graph: object = None
    outputs: object = None
    done: object = None
    input_bytes: int = 0
    pool_bytes: int = 0

    @property
    def retained(self):
        return self.input_bytes + self.pool_bytes

    def close(self):
        if self.done is not None:
            self.done.synchronize()
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.outputs = self.done = None
        self.inputs.clear()
        self.input_bytes = self.pool_bytes = 0


def _pool_reserved(graph, device):
    import torch
    segments = torch.cuda.memory_snapshot()
    if not segments or any('segment_pool_id' not in segment for segment in segments):
        raise RuntimeError('graph pool accounting is unavailable')
    pool = tuple(graph.pool())
    return sum(segment['total_size'] for segment in segments
               if segment['device'] == device.index and tuple(segment['segment_pool_id']) == pool)


class GPUReplay:
    """Replay a pure inference callable with bounded exact-shape storage.

    ``prepare`` builds every slot for the example bucket; ``freeze`` forbids
    future capture. Callers must order incoming producers before their current
    stream, exactly as with ordinary Torch operations. A busy owner falls back
    to its original callable. ``invalidate`` retires buckets for explicit code
    or immutable revision changes; tensor values are refreshed on every call.

    ``max_bytes`` bounds retained input storage + private pool reservation, not
    transient capture, external inputs/outputs or physical device consumption.
    """
    def __init__(self, function, *, policy=None, device=None):
        import torch
        if not callable(function):
            raise TypeError('function must be callable')
        if policy is not None and not isinstance(policy, GPUReplayPolicy):
            raise TypeError('policy must be GPUReplayPolicy')
        device = torch.device('cuda' if device is None else device)
        if device.type != 'cuda':
            raise ValueError('GPU replay requires a CUDA device')
        self.device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        self.function, self.policy = function, policy or GPUReplayPolicy()
        self._pid, self._lock = os.getpid(), threading.RLock()
        self._buckets, self._counts, self._fallbacks = {}, Counter(), Counter()
        self._closed = self._poisoned = self._frozen = False
        self._costs = dict(warmup_ms=0., capture_ms=0., preparation_ms=0.)

    def _process(self):
        if os.getpid() != self._pid:
            raise RuntimeError('GPU replay belongs to another process')

    def _open(self):
        self._process()
        if self._closed or self._poisoned:
            raise RuntimeError('GPU replay is closed or poisoned')

    def _describe(self, args, kwargs):
        import torch
        from torch.utils._pytree import tree_flatten
        if torch.is_grad_enabled():
            raise RuntimeError('GPU replay requires torch.no_grad() or torch.inference_mode()')
        if torch.is_autocast_enabled('cuda'):
            return None, None, None, 'autocast'
        flat, spec = tree_flatten((args, kwargs))
        key, tensors = [], 0
        for value in flat:
            if isinstance(value, torch.Tensor):
                if type(value) not in (torch.Tensor, torch.nn.Parameter):
                    return None, None, None, 'tensor_subclass'
                if (value.device != self.device or value.layout != torch.strided
                        or not value.is_contiguous() or not value.numel()
                        or value.is_conj() or value.is_neg() or value.is_quantized):
                    return None, None, None, 'metadata'
                key.append(('tensor', tuple(value.shape), tuple(value.stride()), value.dtype))
                tensors += 1
            elif value is None or type(value) in (bool, int, str):
                key.append((type(value), value))
            elif type(value) is float and math.isfinite(value):
                key.append((float, value.hex()))
            else:
                return None, None, None, 'argument'
        if not tensors:
            return None, None, None, 'no_tensors'
        settings = (torch.is_inference_mode_enabled(), torch.are_deterministic_algorithms_enabled(),
                    torch.get_default_dtype(), str(torch.get_default_device()),
                    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic,
                    torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        return (spec, tuple(key), settings), flat, spec, None

    def _retained(self):
        return sum(slot.retained for bucket in self._buckets.values()
                   if isinstance(bucket, list) for slot in bucket)

    def _new_slot(self, flat, spec):
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten
        size = sum(x.numel() * x.element_size() for x in flat if isinstance(x, torch.Tensor))
        if size + self._retained() > self.policy.max_bytes:
            return None, 'memory_budget'
        slot = _Slot([], spec, input_bytes=size)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream), torch.inference_mode(False):
                slot.inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in flat]
            args, kwargs = tree_unflatten(slot.inputs, spec)
            versions = [x._version for x in slot.inputs if isinstance(x, torch.Tensor)]
            start = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    warm = self.function(*args, **kwargs)
            stream.synchronize()
            self._costs['warmup_ms'] += (time.perf_counter_ns() - start) / 1e6
            if versions != [x._version for x in slot.inputs if isinstance(x, torch.Tensor)]:
                slot.close()
                return None, 'input_mutation'
            outputs = tree_flatten(warm)[0]
            if not outputs or any(not isinstance(x, torch.Tensor) or x.device != self.device for x in outputs):
                slot.close()
                return None, 'output_metadata'
            del warm, outputs
            if self.policy.enabled:
                slot.graph = torch.cuda.CUDAGraph()
                start = time.perf_counter_ns()
                try:
                    with torch.cuda.graph(slot.graph, stream=stream):
                        slot.outputs = self.function(*args, **kwargs)
                except RuntimeError as error:
                    stream.synchronize()
                    slot.close()
                    if 'captur' not in str(error).lower():
                        raise
                    self._counts['capture_failures'] += 1
                    return None, 'capture_failed'
                finally:
                    self._costs['capture_ms'] += (time.perf_counter_ns() - start) / 1e6
                try:
                    slot.pool_bytes = _pool_reserved(slot.graph, self.device)
                except RuntimeError:
                    slot.close()
                    return None, 'pool_accounting'
                if slot.retained + self._retained() > self.policy.max_bytes:
                    slot.close()
                    return None, 'memory_budget'
                self._counts['captures'] += 1
            slot.done = torch.cuda.Event()
            slot.done.record(stream)
            slot.done.synchronize()
            return slot, None
        except BaseException:
            # Warmup/compile errors are not silently retried. Drain any work
            # before releasing its arguments, even when capture was not begun.
            try:
                stream.synchronize()
                slot.close()
            except BaseException:
                self._poisoned = True
            raise

    def _prepare(self, key, flat, spec):
        import torch
        if key in self._buckets:
            bucket = self._buckets[key]
            return bucket if isinstance(bucket, str) else None
        if self._frozen:
            return 'frozen_shape'
        if len(self._buckets) >= self.policy.max_buckets:
            return 'bucket_limit'
        start = time.perf_counter_ns()
        slots = self._buckets[key] = []
        try:
            with _CAPTURE_LOCK, torch.cuda.device(self.device):
                for _ in range(self.policy.slots):
                    slot, reason = self._new_slot(flat, spec)
                    if reason:
                        for previous in slots:
                            previous.close()
                        self._buckets[key] = reason
                        return reason
                    slots.append(slot)
        except BaseException:
            for slot in slots:
                slot.close()
            self._buckets.pop(key, None)
            raise
        finally:
            self._costs['preparation_ms'] += (time.perf_counter_ns() - start) / 1e6
        return None

    def prepare(self, *args, **kwargs):
        """Prepare all slots for this bucket while GPU activity is quiescent.

        Returns self. Inspect stats for admission or capture exclusions.
        """
        self._open()
        with self._lock:
            self._open()
            key, flat, spec, reason = self._describe(args, kwargs)
            if reason is None:
                reason = self._prepare(key, flat, spec)
            if reason:
                self._counts['prepare_exclusions'] += 1
            return self

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
        self._open()
        with self._lock:
            self._open()
            key, flat, spec, reason = self._describe(args, kwargs)
            if reason is None:
                reason = self._prepare(key, flat, spec)
            if reason is None:
                slot = next((s for s in self._buckets[key] if s.done.query()), None)
                if slot is None:
                    reason = 'busy_slots'
            if reason:
                self._fallbacks[reason] += 1
                # A fallback kernel still consumes temporary inputs on the
                # caller's stream. Register those uses before Python releases
                # its arguments, including metadata/device/budget exclusions.
                for value in tree_flatten((args, kwargs))[0]:
                    if type(value) in (torch.Tensor, torch.nn.Parameter) and value.is_cuda:
                        value.record_stream(torch.cuda.current_stream(value.device))
                return self.function(*args, **kwargs)
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                try:
                    for source, stable in zip(flat, slot.inputs):
                        if isinstance(source, torch.Tensor):
                            stable.copy_(source, non_blocking=True)
                            source.record_stream(stream)
                    if slot.graph is not None:
                        slot.graph.replay()
                        output = slot.outputs
                        self._counts['replays'] += 1
                    else:
                        positional, named = tree_unflatten(slot.inputs, spec)
                        output = self.function(*positional, **named)
                        self._counts['uncaptured_calls'] += 1
                    result = tree_map(lambda x: x.clone(), output)
                    slot.done.record(stream)
                    self._counts['calls'] += 1
                    return result
                except BaseException:
                    self._poisoned = True
                    # No fallback: the request may already have been launched.
                    slot.done.record(stream)
                    raise

    def freeze(self):
        self._open()
        with self._lock:
            self._frozen = True
        return self

    def invalidate(self):
        self._open()
        with self._lock:
            for bucket in self._buckets.values():
                if isinstance(bucket, list):
                    for slot in bucket:
                        slot.close()
            self._buckets.clear()
            self._frozen = False
            self._counts['invalidations'] += 1

    def stats(self):
        self._process()
        with self._lock:
            slots = [slot for bucket in self._buckets.values() if isinstance(bucket, list) for slot in bucket]
            return dict({name: self._counts[name] for name in ('calls', 'replays', 'uncaptured_calls',
                        'captures', 'capture_failures', 'invalidations', 'prepare_exclusions')},
                        **self._costs, fallbacks=dict(self._fallbacks), buckets=len(self._buckets),
                        exclusions=[b for b in self._buckets.values() if isinstance(b, str)],
                        slots=len(slots), input_bytes=sum(s.input_bytes for s in slots),
                        pool_reserved_bytes=sum(s.pool_bytes for s in slots),
                        retained_bytes=sum(s.retained for s in slots), max_bytes=self.policy.max_bytes,
                        frozen=self._frozen, closed=self._closed, poisoned=self._poisoned,
                        output_policy='copy; external outputs excluded from owner budget')

    def close(self):
        self._process()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for bucket in self._buckets.values():
                if isinstance(bucket, list):
                    for slot in bucket:
                        slot.close()
            self._buckets.clear()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
