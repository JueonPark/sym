"""Bounded artifact cache and process-local execution-handle registry.

Artifact keys carry semantic identity only: canonical recipe (dynamic
dimensions stay symbolic, constant split factors stay constant), frontend and
compiler identity, wire version, dtype/layout family, transfer semantics and
runtime capability identity. Nothing is keyed by pointer or object id.

Execution handles are separate from artifact keys. A live graph owns its
registrations strongly, so cache eviction never invalidates it; handles are a
process-local naming scheme, not a portable artifact format.
"""

from collections import OrderedDict
from dataclasses import dataclass
import itertools
import secrets
import threading

from .artifact import UnsupportedRecipe


FRONTEND_IDENTITY = "reloc_torch/1"
WIRE_VERSION = 0
TYPED_WIRE_VERSION = 1
DEFAULT_CAPACITY = 128


@dataclass(frozen=True)
class ArtifactKey:
    recipe: tuple
    frontend: str
    compiler: str
    wire_version: int
    dtype: str
    layout_family: str
    direction: str
    blocking: bool
    value_transform: str
    runtime_capability: str


def artifact_key(
    recipe,
    *,
    compiler_identity,
    runtime_capability,
    frontend_identity=FRONTEND_IDENTITY,
    wire_version=None,
    blocking=True,
):
    # A typed recipe (C3) is a wire v1 artifact with value transforms; the
    # key says so, so a layout-only and a typed recipe can never share one.
    typed = recipe.typed
    if wire_version is None:
        wire_version = TYPED_WIRE_VERSION if typed else WIRE_VERSION
    return ArtifactKey(
        recipe.canonical_identity,
        str(frontend_identity),
        str(compiler_identity),
        int(wire_version),
        recipe.source.dtype,
        "dense",
        recipe.direction,
        bool(blocking),
        "typed" if typed else "layout_only",
        str(runtime_capability),
    )


class ArtifactCache:
    """LRU over compiled artifacts plus a bounded LRU of explicit rejections.

    Concurrent requests for one key compile once; the others wait and count as
    cache hits. Rejections are keyed identically, so a new compiler or runtime
    capability can never reuse a stale rejection. Unexpected compiler errors
    (crashes, missing exporter) are not cached: the failing caller sees the
    error and each waiter retries on its own.
    """

    def __init__(self, capacity=DEFAULT_CAPACITY):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._rejections = OrderedDict()
        self._pending = set()
        self._condition = threading.Condition(threading.Lock())

    def __len__(self):
        with self._condition:
            return len(self._entries)

    @property
    def rejections(self):
        with self._condition:
            return len(self._rejections)

    def _trim(self, table):
        while len(table) > self.capacity:
            table.popitem(last=False)

    def get_or_compile(self, key, compile, diagnostics):
        with self._condition:
            while True:
                if key in self._entries:
                    self._entries.move_to_end(key)
                    diagnostics.increment("cache_hits")
                    return self._entries[key]
                if key in self._rejections:
                    self._rejections.move_to_end(key)
                    diagnostics.increment("cache_hits")
                    reason, detail = self._rejections[key]
                    raise UnsupportedRecipe(reason, detail)
                if key not in self._pending:
                    self._pending.add(key)
                    break
                self._condition.wait()
        try:
            diagnostics.increment("plan_compiles")
            compiled = compile()
        except UnsupportedRecipe as error:
            with self._condition:
                self._rejections[key] = (error.reason, error.detail)
                self._trim(self._rejections)
                self._pending.discard(key)
                self._condition.notify_all()
            raise
        except BaseException:
            with self._condition:
                self._pending.discard(key)
                self._condition.notify_all()
            raise
        with self._condition:
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            self._trim(self._entries)
            self._pending.discard(key)
            self._condition.notify_all()
        return compiled


_HANDLE_COUNTER = itertools.count(1)


class Registration:
    """Strong ownership of one live handle; releasing is idempotent."""

    __slots__ = ("handle", "_registry", "__weakref__")

    def __init__(self, handle, registry):
        self.handle = handle
        self._registry = registry

    def release(self):
        registry, self._registry = self._registry, None
        if registry is not None:
            registry._release(self.handle)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class HandleRegistry:
    """Thread-safe process-local map from live handle to execution entry."""

    def __init__(self):
        self._entries = {}
        # Reentrant: a Registration finalizer may run inside a cyclic-GC pass
        # triggered while this thread already holds the lock in register().
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def register(self, entry):
        handle = f"reloc-{next(_HANDLE_COUNTER)}-{secrets.token_hex(4)}"
        with self._lock:
            self._entries[handle] = entry
        if entry.handle is None:
            entry.handle = handle
        return Registration(handle, self)

    def lookup(self, handle):
        with self._lock:
            entry = self._entries.get(handle)
        if entry is None or entry.closed:
            raise RuntimeError(
                f"unknown or closed reloc_torch execution handle {handle!r}"
            )
        return entry

    def _release(self, handle):
        with self._lock:
            self._entries.pop(handle, None)


REGISTRY = HandleRegistry()


def lookup_handle(handle):
    return REGISTRY.lookup(handle)


__all__ = (
    "DEFAULT_CAPACITY",
    "FRONTEND_IDENTITY",
    "REGISTRY",
    "TYPED_WIRE_VERSION",
    "WIRE_VERSION",
    "ArtifactCache",
    "ArtifactKey",
    "HandleRegistry",
    "Registration",
    "artifact_key",
    "lookup_handle",
)
