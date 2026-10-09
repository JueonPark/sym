"""Completed-call placement, separate from pinned/pageable allocation policy.

Profiles rank a closed set of semantically qualified implementations. They
contain scalar measurements, never code, tensors or permission to change a
recipe. Missing or incompatible evidence chooses the saved native region.
No device utilization polling, calibration, or speculative execution occurs
on the selection path. Applications supply load/overlap context explicitly.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading

from .recipe import Cast, Transpose

FORMAT = 'sym.completed-placement/1'
PATHS = ('native', 'torch_cpu', 'torch_gpu', 'sym_cpu', 'sym_gpu')


@lru_cache(maxsize=256)
def family(recipe):
    """Only parameter-free dense permutations and C1 casts are replay-qualified.

    Other recipes retain the existing runtime. In particular we neither
    approximate quantization nor move a pad across a value transformation.
    """
    ops = []
    for op in recipe.operations:
        if isinstance(op, Transpose):
            ops.append(('transpose', op.perm))
        elif isinstance(op, Cast):
            ops.append(('cast', op.dtype, op.policy))
        else:
            return None
    return json.dumps((recipe.direction, recipe.source.dtype,
        recipe.destination.dtype, len(recipe.source.shape), ops), separators=(',', ':'))


@lru_cache(maxsize=256)
def candidates(recipe):
    if family(recipe) is None:
        return ()
    rows = ['native', 'torch_cpu', 'torch_gpu', 'sym_cpu']
    if recipe.typed and len(recipe.source.shape) <= 8:
        rows.append('sym_gpu')
    return tuple(rows)


def gpu_implementation(recipe):
    stages = sum(isinstance(op, Cast) for op in recipe.operations)
    if recipe.direction == 'd2h':
        return f'cuda_stages_then_cpu@{stages}'
    return 'cuda_relocate_f32' if recipe.source.dtype == 'float32' else 'cpu_stages_cuda_stages@0'


def wire_bytes(recipe, shape, path):
    widths = {'float32': 4, 'float16': 2, 'int8': 1}
    if path == 'native':
        # The saved Torch graph is intentionally opaque to this policy. For
        # casts, qualify its wire accounting with the explicit CPU replay.
        # Native is therefore only eligible for dtype-preserving recipes.
        dtype = recipe.source.dtype
    elif path in ('torch_gpu', 'sym_gpu'):
        dtype = recipe.source.dtype if recipe.direction == 'h2d' else recipe.destination.dtype
    else:
        dtype = recipe.destination.dtype if recipe.direction == 'h2d' else recipe.source.dtype
    return math.prod(shape) * widths[dtype]


@lru_cache(maxsize=8)
def _hardware(device):
    """Cold-only discovery. Affinity and Torch thread settings are checked live."""
    import torch
    import pyreloc
    info = Path('/proc/cpuinfo').read_text()
    cpu = next((s.split(':', 1)[1].strip() for s in info.splitlines() if s.startswith('model name')), '')
    flags = next((s.split(':', 1)[1].strip() for s in info.splitlines() if s.startswith('flags')), '')
    props = torch.cuda.get_device_properties(device)
    devices = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,driver_version,pci.bus_id,pcie.link.gen.max,pcie.link.width.max',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    topology = next(line.split(', ') for line in devices.splitlines()
                    if line.split(', ')[0].removeprefix('GPU-') == str(props.uuid).removeprefix('GPU-'))
    pci = topology[2].lower()
    pci = pci[-12:]
    sysfs = Path('/sys/bus/pci/devices') / pci
    extension, = Path(pyreloc.__file__).parent.glob('_pyreloc*.so')
    library = Path(pyreloc.__file__).parents[2] / 'libreloc/libreloc_runtime.so'
    # Installed wheels may carry the shared library beside the extension.
    if not library.exists():
        libraries = list(Path(pyreloc.__file__).parents[1].rglob('libreloc_runtime.so*'))
        library = libraries[0] if libraries else extension
    return dict(cpu=cpu, cpu_flags=flags, torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=props.name, gpu_uuid=str(props.uuid), capability=[props.major, props.minor],
        driver=topology[1], pci_bus=pci, pcie_max=topology[3:],
        gpu_numa=(sysfs/'numa_node').read_text().strip(),
        gpu_local_cpus=(sysfs/'local_cpulist').read_text().strip(),
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        frontend_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in
            sorted(Path(__file__).parent.glob('*.py')))).hexdigest())


def hardware(device):
    import torch
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return dict(_hardware(str(device)), affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                thread_environment={k:os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})


def _context(value):
    value = dict(value)
    if set(value) != {'cpu_load', 'gpu_load', 'overlap'}:
        raise ValueError('placement context requires cpu_load, gpu_load, overlap')
    if any(value[k] not in ('unknown', 'idle', 'busy') for k in ('cpu_load', 'gpu_load')):
        raise ValueError('load must be unknown, idle or busy')
    if value['overlap'] not in ('none', 'cpu_h2d', 'consumer'):
        raise ValueError('overlap must be none, cpu_h2d or consumer')
    return value


class PlacementProfile:
    """Owned, versioned observations of the entire completed call in ms.

    Each observation contains family, shape, strides, path, wire_bytes,
    resource_state (cold/warm), context and completed_ms. No extrapolation
    beyond measured per-axis bounds is allowed. Interpolation uses the four
    closest shapes in log2 extent space, with inverse-square distance weights.
    """
    def __init__(self, *, hardware, execution, observations):
        self.hardware = json.loads(json.dumps(hardware))
        self.execution = json.loads(json.dumps(execution))
        self._observations = json.loads(json.dumps(observations))
        if len(self._observations) > 4096:
            raise ValueError('placement profile exceeds 4096 observations')
        self._index = {}
        for row in self._observations:
            shape = row['shape']
            if (not shape or len(shape) > 8 or any(type(n) is not int or n < 1 for n in shape)
                    or row['path'] not in PATHS or row['resource_state'] not in ('cold', 'warm')
                    or not math.isfinite(row['completed_ms']) or row['completed_ms'] <= 0
                    or type(row['wire_bytes']) is not int or row['wire_bytes'] <= 0):
                raise ValueError('invalid placement observation')
            strides, stride = [], 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= size
            if row['strides'] != strides:
                raise ValueError('placement observations require dense row-major strides')
            context = _context(row['context'])
            key = (row['family'], row['path'], row['resource_state'], json.dumps(context, sort_keys=True))
            self._index.setdefault(key, []).append(row)

    def to_dict(self):
        return json.loads(json.dumps(dict(format=FORMAT, hardware=self.hardware,
            execution=self.execution, observations=self._observations)))

    @classmethod
    def from_dict(cls, value):
        if value.get('format') != FORMAT:
            raise ValueError('incompatible completed-placement profile format')
        return cls(hardware=value['hardware'], execution=value['execution'], observations=value['observations'])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2)+'\n')

    def predict(self, identity, shape, path, state, context, wire):
        rows = self._index.get((identity, path, state, json.dumps(context, sort_keys=True)), ())
        rows = [r for r in rows if len(r['shape']) == len(shape)
                and r['wire_bytes'] * math.prod(shape) == wire * math.prod(r['shape'])]
        if not rows:
            return None
        if any(n < min(r['shape'][i] for r in rows) or n > max(r['shape'][i] for r in rows)
               for i, n in enumerate(shape)):
            return None
        points = sorted((sum(abs(math.log2(n/m)) for n,m in zip(shape,r['shape'])), r['completed_ms']) for r in rows)
        if points[0][0] == 0:
            exact = sorted(ms for distance, ms in points if distance == 0)
            return exact[len(exact)//2]
        if points[0][0] > 2:
            return None
        points = points[:4]
        weights = [1/(d*d) for d, _ in points]
        return sum(w*ms for w,(_,ms) in zip(weights,points))/sum(weights)


class PlacementPolicy:
    """Opt-in per-call selection; no calibration or source reads at selection.

    A callable context can supply scheduler load changes. Unknown/mismatched
    contexts conservatively keep native Torch. ``force(path)`` is a thread-local
    qualification/calibration override, still constrained by recipe capability.
    Observed successful paths are tracked in a bounded cache; a cleared native
    resource owner invalidates warm assumptions. Profiles describe blocking
    calls only and are not installed in asynchronous/grouped APIs.
    """
    def __init__(self, profile=None, *, context=None, capacity=128):
        if profile is not None and not isinstance(profile, PlacementProfile):
            raise TypeError('profile must be PlacementProfile or None')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        self.profile, self.capacity = profile, capacity
        self._context = context if context is not None else dict(cpu_load='unknown', gpu_load='unknown', overlap='none')
        if not callable(self._context):
            self._context = _context(self._context)
        self._pid = os.getpid()
        self._seen, self._counts = OrderedDict(), Counter()
        self._predictions = OrderedDict()
        self._cached_profile = profile
        self._execution = None
        self._execution_key = None
        self._lock, self._local = threading.Lock(), threading.local()
        self._last = None

    def _check(self):
        if os.getpid() != self._pid:
            raise RuntimeError('process_mismatch: placement policy belongs to another process')

    @contextmanager
    def force(self, path):
        self._check()
        if path not in PATHS:
            raise ValueError('unknown placement path')
        previous = getattr(self._local, 'forced', None)
        self._local.forced = path
        try:
            yield self
        finally:
            self._local.forced = previous

    def stats(self):
        self._check()
        with self._lock:
            return dict(choices=dict(self._counts), cached_paths=len(self._seen),
                        last=json.loads(json.dumps(self._last)))

    def choose(self, compiled, source, device, runtime):
        self._check()
        runtime._require_open()
        recipe = compiled.recipe
        identity = family(recipe)
        if identity is None:
            if getattr(self._local, 'forced', None) is not None:
                raise ValueError('forced placement is not qualified for this recipe')
            with self._lock:
                self._counts['legacy:unsupported_recipe'] += 1
            return None
        context = _context(self._context() if callable(self._context) else self._context)
        shape, strides = tuple(source.shape), tuple(source.stride())
        import torch
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        execution, epoch, retained = runtime.placement_configuration()
        with self._lock:
            if execution != self._execution:
                self._execution = execution
                self._execution_key = json.dumps(execution, sort_keys=True)
            execution_key = self._execution_key
        key = (identity, shape, strides, str(device), execution_key,
               (context['cpu_load'], context['gpu_load'], context['overlap']), epoch)
        eligible = candidates(recipe)
        import pyreloc
        if not pyreloc.cuda_enabled:
            eligible = tuple(p for p in eligible if not p.startswith('sym'))
        # For typed recipes, the controlled CPU replay provides native Torch
        # semantics with explicitly known wire precision; the opaque original
        # might perform a combined dtype/device copy using another wire dtype.
        if recipe.typed:
            eligible = tuple(p for p in eligible if p != 'native')
        fallback = 'torch_cpu' if recipe.typed else 'native'
        choice, reason, predictions, states = fallback, 'no_profile', {}, {}
        with self._lock:
            for path in eligible:
                states[path] = 'warm' if (key,path) in self._seen and (retained or not path.startswith('sym')) else 'cold'
        forced = getattr(self._local, 'forced', None)
        if forced is not None:
            if forced not in eligible:
                raise ValueError(f'placement {forced} is not qualified for this recipe')
            choice, reason = forced, 'forced'
        elif self.profile is not None:
            cuda_device = device if recipe.direction == 'h2d' else source.device
            try:
                current_hardware = hardware(cuda_device)
            except (OSError, RuntimeError, ValueError, StopIteration, subprocess.SubprocessError):
                current_hardware = None
            if current_hardware is None:
                reason = 'profile_hardware_unavailable'
            elif self.profile.hardware != current_hardware:
                reason = 'profile_hardware_mismatch'
            elif self.profile.execution != execution:
                reason = 'profile_execution_mismatch'
            else:
                prediction_key = (key, tuple(states.items()))
                with self._lock:
                    if self._cached_profile is not self.profile:
                        self._predictions.clear()
                        self._cached_profile = self.profile
                    predictions = self._predictions.get(prediction_key)
                    if predictions is not None:
                        self._predictions.move_to_end(prediction_key)
                if predictions is None:
                    predictions = {}
                    for path in eligible:
                        estimate = self.profile.predict(identity, shape, path, states[path], context,
                                                        wire_bytes(recipe, shape, path))
                        if estimate is not None:
                            predictions[path] = estimate
                    with self._lock:
                        self._predictions[prediction_key] = predictions
                        while len(self._predictions) > self.capacity:
                            self._predictions.popitem(last=False)
                if fallback not in predictions:
                    reason = 'profile_missing_native_coverage'
                else:
                    choice = min(predictions, key=predictions.get)
                    reason = 'lowest_completed_cost'
        record = dict(family=identity, shape=list(shape), strides=list(strides), path=choice,
            wire_bytes=wire_bytes(recipe, shape, choice), resource_state=states[choice],
            context=context, reason=reason, predicted_ms=predictions,
            unprofiled_paths=[p for p in eligible if p not in predictions])
        with self._lock:
            self._last = record
            self._counts[f'{choice}:{reason}'] += 1
        return choice, key, record

    def succeeded(self, decision, runtime):
        choice, key, _ = decision
        # AUTO may have created its owner during this call.
        _, epoch, _ = runtime.placement_configuration()
        key = (*key[:-1], epoch)
        with self._lock:
            self._seen[key,choice] = None
            self._seen.move_to_end((key,choice))
            while len(self._seen) > self.capacity:
                self._seen.popitem(last=False)


def replay(recipe, source, device, path):
    """Fresh, completed CPU- or GPU-transform Torch execution of the recipe."""
    import torch

    transform_device = torch.device('cpu') if path == 'torch_cpu' else (
        torch.device(device) if recipe.direction == 'h2d' else source.device)
    value = source.to(transform_device)
    for op in recipe.operations:
        if isinstance(op, Transpose):
            value = value.permute(op.perm)
        else:
            value = value.to(getattr(torch, op.dtype))
    value = value.contiguous().to(device)
    cuda_device = device if recipe.direction == 'h2d' else source.device
    torch.cuda.current_stream(cuda_device).synchronize()
    return value
