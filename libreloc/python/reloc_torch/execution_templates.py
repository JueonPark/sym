"""Bounded, per-artifact immutable metadata; never tensors or requests."""

from collections import OrderedDict
import os
from threading import Lock


class TemplateCache:
    """LRU bounded by entry count and copied parameter-key bytes.

    Native metadata has a fixed topology per owning artifact, with parameter
    copies proportional to the bounded keys. No allocation/stream/pointer or
    executable request belongs here. Factories run outside the lock; duplicate
    concurrent misses are harmless, and insertion returns the winning value.
    """

    def __init__(self, max_entries=32, max_parameter_bytes=256 << 10):
        self.max_entries = max_entries
        self.max_parameter_bytes = max_parameter_bytes
        self._reset()

    def _reset(self):
        self._lock = Lock()
        self._items = OrderedDict()
        self._bytes = self._hits = self._misses = self._evictions = self._bypasses = 0
        self._pid = os.getpid()

    def _process(self):
        # Metadata is portable, inherited Python locks are not. Never acquire
        # an inherited lock; start a new empty process-local cache instead.
        if self._pid != os.getpid():
            self._reset()

    def get(self, key, parameter_bytes=0):
        self._process()
        with self._lock:
            if self.max_entries == 0 or parameter_bytes > self.max_parameter_bytes:
                self._bypasses += 1
                return None
            item = self._items.get(key)
            if item is None:
                self._misses += 1
                return None
            self._hits += 1
            self._items.move_to_end(key)
            return item[0]

    def put(self, key, value, parameter_bytes=0):
        self._process()
        with self._lock:
            if self.max_entries == 0 or parameter_bytes > self.max_parameter_bytes:
                return value
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            while (len(self._items) >= self.max_entries or
                   self._bytes + parameter_bytes > self.max_parameter_bytes):
                _, (_, charge) = self._items.popitem(last=False)
                self._bytes -= charge
                self._evictions += 1
            self._items[key] = value, parameter_bytes
            self._bytes += parameter_bytes
            return value

    def clear(self):
        self._process()
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def info(self):
        self._process()
        with self._lock:
            return dict(entries=len(self._items), parameter_key_bytes=self._bytes,
                        max_entries=self.max_entries, max_parameter_key_bytes=self.max_parameter_bytes,
                        hits=self._hits, misses=self._misses, evictions=self._evictions, bypasses=self._bypasses)
