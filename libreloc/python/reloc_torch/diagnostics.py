"""Compilation, binding, execution and fallback counters for the Torch frontend.

Counters are process-local, thread-safe integers. Snapshots are plain copies
that hold no tensor, graph or runtime references.
"""

from collections import Counter
import threading


COUNTERS = (
    "dynamo_compiles",
    "plan_compiles",
    "symbol_binds",
    "cache_hits",
    "runtime_executions",
    "typed_executions",
    "typed_payload_bytes",
    "weight_preparations",
    "weight_invalidations",
)


class Diagnostics:
    """Distinguishes Dynamo callbacks, compiler calls, binder calls, cache hits,
    actual dispatches, and reason-coded fallbacks/exclusions/redispatches."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = dict.fromkeys(COUNTERS, 0)
        self.fallbacks = Counter()
        self.exclusions = Counter()
        self.redispatches = Counter()
        self.dispatches = Counter()

    def increment(self, name, amount=1):
        with self._lock:
            if name not in self._counters:
                raise KeyError(name)
            self._counters[name] += amount

    def record_fallback(self, reason):
        with self._lock:
            self.fallbacks[str(reason)] += 1

    def record_exclusion(self, reason):
        with self._lock:
            self.exclusions[str(reason)] += 1

    def record_redispatch(self, reason):
        with self._lock:
            self.redispatches[str(reason)] += 1

    def record_dispatch(self, report):
        """Aggregate an R3 typed dispatch report: the implementation that ran
        and the bytes it moved (scalars only; the report holds no tensors)."""
        with self._lock:
            self._counters["typed_executions"] += 1
            self._counters["typed_payload_bytes"] += int(report.get("payload_bytes_transferred", 0))
            self.dispatches[str(report.get("implementation"))] += 1

    def snapshot(self):
        with self._lock:
            result = dict(self._counters)
            result["fallbacks"] = dict(self.fallbacks)
            result["exclusions"] = dict(self.exclusions)
            result["redispatches"] = dict(self.redispatches)
            result["dispatches"] = dict(self.dispatches)
        return result


__all__ = ("COUNTERS", "Diagnostics")
