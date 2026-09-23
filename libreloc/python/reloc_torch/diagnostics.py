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

    def snapshot(self):
        with self._lock:
            result = dict(self._counters)
            result["fallbacks"] = dict(self.fallbacks)
            result["exclusions"] = dict(self.exclusions)
            result["redispatches"] = dict(self.redispatches)
        return result


__all__ = ("COUNTERS", "Diagnostics")
