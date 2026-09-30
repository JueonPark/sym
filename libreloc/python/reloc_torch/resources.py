"""Explicit direct-transfer resource ownership (issue #169); Torch-free import."""

import os


class TransferResources:
    """Keep bounded staging, streams and workers across blocking transfers.

    Limits apply to this owner. Construction and stats do not initialize CUDA.
    Use a context manager or close() before shutdown. clear() retires idle
    resources and older active generations. Unknown completion retains resources
    and tensor owners for the process lifetime; close() reports that failure.
    """

    __slots__ = ("_native", "_pid", "__weakref__")

    def __init__(self, *, max_retained_bytes=256 << 20, max_contexts=4,
                 max_contexts_per_device=2, max_background_workers=64,
                 max_live_staging_bytes=None, acquire_timeout_ms=None):
        import pyreloc

        self._pid = os.getpid()
        self._native = pyreloc.TransferResourceCache(
            max_retained_bytes=max_retained_bytes, max_contexts=max_contexts,
            max_contexts_per_device=max_contexts_per_device,
            max_background_workers=max_background_workers,
            max_live_staging_bytes=max_live_staging_bytes,
            acquire_timeout_ms=acquire_timeout_ms,
        )

    @property
    def native(self):
        # Reject inherited use before a frontend can allocate output or touch
        # CUDA/pool state. Native PID/epoch checks independently protect bindings.
        if os.getpid() != self._pid:
            raise RuntimeError("process_mismatch: transfer resources belong to another process")
        return self._native

    @property
    def closed(self):
        return self.native.closed

    def stats(self):
        """Return scalar counters/gauges and per-device snapshots."""
        return self.native.stats()

    def clear(self):
        self.native.clear()

    def close(self):
        self.native.close()

    def __enter__(self):
        self.native.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __reduce_ex__(self, protocol):
        raise TypeError("transfer resources cannot be serialized")
