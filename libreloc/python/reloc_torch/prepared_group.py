"""Bounded reusable native groups; all state transitions use the owner's lock."""
from collections import deque
import os
from weakref import finalize, ref


def _account_binder(owner_ref, binder):
    owner = owner_ref()
    if owner is not None and owner._pid == os.getpid():
        with owner._lock:
            for key, value in binder.stats().items():
                owner._prepared_counts[key] += value


class PreparedGroup:
    """Immutable program revisions with up to ``owner.contexts`` reusable slots.

    Source values are read on every submission. Sources must stay unchanged
    until completion; returned outputs and reports survive subsequent reuse.
    Close drains active transfers, and does not invalidate returned outputs.
    Instances are created by :meth:`FastTransfers.prepare_group`.
    """

    def __init__(self, owner, programs, bridge):
        from .fast import LayoutProgram, TypedProgram
        self._owner = owner
        self._programs = tuple(programs)
        if not 1 <= len(self._programs) <= 256:
            raise ValueError("a group holds between 1 and 256 transfers")
        members = []
        for program in self._programs:
            if not isinstance(program, (TypedProgram, LayoutProgram)):
                raise TypeError("prepared groups require typed or layout programs")
            if program.owner is not owner:
                raise ValueError("program belongs to a different owner")
            d = program.destination
            members.append((program.template if isinstance(program, TypedProgram) else program.bound,
                            program.direction, program.signature_dtype, d.dtype,
                            program.signature[0], program.signature[1], program._view_device,
                            d.shape, d.strides, d.device.index if d.device.type == 'cuda' else -1))
        self._binder = bridge.PreparedTensorGroup(members)
        self._accounting = finalize(self, _account_binder, ref(owner), self._binder)
        self._accounting.atexit = False
        self._host = any(p.direction == 'd2h' for p in self._programs)
        self._free, self._active = [], deque()
        self._slots = 0
        self._closed = False
        owner._prepared_handles.add(self)
        owner._prepared_counts['handles_created'] += 1

    def submit(self, sources, *, outputs=None, device_layout=False, gather_threads=None,
               pin_outputs=True, extra_owners=(), chunked_members=False):
        """Bind live tensors and submit; supplied outputs must match the programs."""
        self._owner._check_open()
        with self._owner._lock:
            self._owner._check_open()
            self._owner._retire()
            return self._submit_locked(sources, outputs=outputs, device_layout=device_layout,
                                       gather_threads=gather_threads, pin_outputs=pin_outputs,
                                       extra_owners=extra_owners, chunked_members=chunked_members)

    def _submit_locked(self, sources, *, outputs=None, device_layout=False, gather_threads=None,
                       pin_outputs=True, extra_owners=(), chunked_members=False):
        import torch
        if self._closed:
            raise RuntimeError("prepared group is closed")
        sources = tuple(sources)
        if len(sources) != len(self._programs):
            raise ValueError("one source per prepared program is required")
        if outputs is not None:
            outputs = list(outputs)
            if len(outputs) != len(self._programs):
                raise ValueError("one output per prepared program is required")
        # Finalization removes the active entry and returns its healthy slot.
        while self._active and self._active[0].done():
            pass
        if not self._free and self._slots >= self._owner.contexts:
            self._active[0].wait()
        with torch.cuda.stream(self._owner._stream):
            if outputs is None:
                outputs = [p.allocate(pin=pin_outputs) for p in self._programs]
        native = self._free.pop() if self._free else None
        try:
            if native is None:
                native = self._binder.prepare(sources, outputs)
                self._slots += 1
            else:
                self._binder.rebind(native, sources, outputs)
        except BaseException as error:
            if native is not None:
                self._free.append(native)  # failed binding is atomic
            import pyreloc
            if isinstance(error, pyreloc.TransferError):
                raise RuntimeError(f"transfer group failed: {error}") from error
            raise
        try:
            completion = self._owner._launch_group(native, sources, outputs, extra_owners, gather_threads,
                                                   host=self._host, device_layout=device_layout,
                                                   chunked_members=chunked_members)
        except BaseException:
            # An execution failure can leave work quarantined in native resources.
            self._slots -= 1
            raise
        completion._prepared = self
        self._active.append(completion)
        return completion

    def _release(self, completion, native, *, failed=False):
        self._active.remove(completion)
        if failed or self._closed:
            self._slots -= 1
        else:
            self._free.append(native)

    def close(self):
        self._owner._check_process()
        with self._owner._lock:
            if self._closed:
                return
            self._closed = True
            error = None
            while self._active:
                try:
                    self._active[0].wait()
                except BaseException as exc:
                    if error is None:
                        error = exc
            self._free.clear()
            self._slots = 0
            self._accounting()
            self._binder = None
            self._programs = ()
            self._owner._prepared_handles.discard(self)
            if error is not None:
                raise error

    def __enter__(self):
        self._owner._check_open()
        if self._closed:
            raise RuntimeError("prepared group is closed")
        return self

    def __exit__(self, *exc):
        self.close()
        return False
