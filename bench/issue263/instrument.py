"""Diagnostic host phases and NVTX labels; off mode leaves callables untouched."""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import threading
import time


class Instrumentation:
    def __init__(self, mode='off', *, clock=time.perf_counter_ns):
        if mode not in ('off', 'counters', 'nvtx'):
            raise ValueError(f'unknown instrumentation mode: {mode}')
        self.mode, self.clock = mode, clock
        self.local = threading.local()
        self.records, self.undo = [], []
        self.lock = threading.Lock()
        self.nvtx = None
        if mode == 'nvtx':
            import torch
            self.nvtx = torch.cuda.nvtx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for owner, name, original, owned in reversed(self.undo):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.undo.clear()

    def _charge(self, state, now):
        state['exclusive_ns'][state['stack'][-1]] += now - state['last']
        state['last'] = now

    @contextmanager
    def request(self, label):
        if self.mode == 'off':
            yield None
            return
        if getattr(self.local, 'state', None) is not None:
            raise RuntimeError('nested request attribution is ambiguous')
        if self.nvtx:
            self.nvtx.range_push('request263/' + label)
        start = self.clock()
        state = dict(label=label, native_tid=threading.get_native_id(), start_ns=start,
                     last=start, stack=['other'], exclusive_ns=Counter(), calls=Counter(), groups=[], error=None)
        self.local.state = state
        try:
            yield state
        except BaseException as exc:
            state['error'] = type(exc).__name__
            raise
        finally:
            end = self.clock()
            self._charge(state, end)
            self.local.state = None
            if self.nvtx:
                self.nvtx.range_pop()
            row = {k: state[k] for k in ('label', 'native_tid', 'start_ns', 'groups', 'error')}
            row.update(elapsed_ns=end-start, exclusive_ns=dict(state['exclusive_ns']), calls=dict(state['calls']))
            with self.lock:
                self.records.append(row)

    @contextmanager
    def phase(self, category, *, operation=None, group=False):
        state = getattr(self.local, 'state', None)
        if state is None:
            yield
            return
        now = self.clock()
        self._charge(state, now)
        state['stack'].append(category)
        state['calls'][operation or category] += 1
        group_row = None
        if group:
            group_row = dict(id=len(state['groups']), operation=operation or category, start_ns=now)
            state['groups'].append(group_row)
            if self.nvtx:
                self.nvtx.range_push(f"group263/{state['label']}/g{group_row['id']}/{operation or category}")
        if self.nvtx:
            self.nvtx.range_push(f'phase263/{category}/{operation or category}')
        try:
            yield
        finally:
            if self.nvtx:
                self.nvtx.range_pop()
                if group:
                    self.nvtx.range_pop()
            end = self.clock()
            self._charge(state, end)
            state['stack'].pop()
            if group_row is not None:
                group_row['end_ns'] = end

    def wrap(self, owner, name, category, *, group=False):
        if self.mode == 'off':
            return
        original = getattr(owner, name)
        owned = name in vars(owner)
        @wraps(original)
        def wrapped(*args, **kwargs):
            if getattr(self.local, 'state', None) is None:
                return original(*args, **kwargs)
            with self.phase(category, operation=name, group=group):
                return original(*args, **kwargs)
        self.undo.append((owner, name, original, owned))
        setattr(owner, name, wrapped)

    def consumer(self, fn, name):
        if self.mode == 'off':
            return fn
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self.phase('consumer', operation=name):
                return fn(*args, **kwargs)
        return wrapped

    def install(self, controls):
        """Wrap runtime boundaries, never tensor operators captured by Dynamo."""
        if self.mode == 'off':
            return
        import pyreloc
        from reloc_torch import fast
        native = fast._load_native_views()
        for owner, names, category in (
            (fast._Program, ('check',), 'source_validation'),
            (fast._Program, ('allocate',), 'output_allocation'),
            (fast.TypedProgram, ('tensor_entry', 'group_entry'), 'entry_binding'),
            (fast.LayoutProgram, ('tensor_entry', 'group_entry'), 'entry_binding'),
            (fast.FastWeights, ('_items',), 'weight_binding'),
            (fast.FastCompletion, ('_finish',), 'completion_report'),
            (fast.FastTransfers, ('_record_group',), 'report_accounting'),
            (fast.FastCompletion, ('_record_outputs',), 'output_lifetime'),
            (fast.FastCompletion, ('wait', 'wait_stream'), 'completion_dependency'),
            (fast.FastTransfers, ('_retain', '_retire'), 'completion_retirement'),
            (fast.FastTransfers, ('_submit_native',), 'native_submission_wrapper'),
            (pyreloc, ('submit_dispatch_group',), 'native_submission'),
            (pyreloc, ('prepare_dispatch_group',), 'native_group_preparation'),
            (controls.TorchCompletion, ('__init__',), 'torch_submission'),
            (controls.TorchCompletion, ('wait', 'wait_stream', 'done'), 'completion_dependency'),
            (controls.TorchKV, ('_retire',), 'completion_retirement'),
            (controls.native, ('dequant', 'transpose_batch'), 'torch_conversion'),
            (fast.KVCacheSwap, ('evict', 'restore'), 'kv_python'),
        ):
            for name in names:
                self.wrap(owner, name, category)
        if native:
            for name in ('group_from_tensors',):
                self.wrap(native, name, 'native_group_preparation')
        for owner, name in ((fast.FastTransfers, 'submit'), (controls.TorchProducer, 'submit'),
                            (controls.TorchKV, 'evict'), (controls.TorchKV, 'restore')):
            self.wrap(owner, name, 'group_python', group=True)
