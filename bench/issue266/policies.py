"""Matched packed Torch/Sym producers; no changes to numerical workload code."""
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
import time

import torch
from reloc_torch.fast import FastPrefetch, FastTransfers, FastWeights
from reloc_torch.memory import PackedOutputArena, plan_transfer_memory
from reloc_torch.runtime import ConcreteDescriptor

MIB = 1 << 20


def descriptor(q):
    return ConcreteDescriptor(tuple(q.shape), tuple(q.stride()), 'float16', torch.device('cuda:0'))


def configuration(job):
    started = time.perf_counter_ns()
    kind = job['kind']
    if kind == 'e4':
        count, output, scratch, small = job['layers'], 24 * MIB, 16 * MIB, 8 * MIB
        workspace = 128 * MIB + job['layers'] * job['context'] * 4096 + job['context'] * 1024 * 16 + 32 * job['context'] ** 2
        available, safety, prefix = job['budget_mib'] * MIB, 32 * MIB, 8
    elif kind == 'e5':
        llm = job['case'] == 'llm_decode'
        count, output, scratch, small = (4, 24 * MIB, 16 * MIB, 8 * MIB) if llm else (16, 8 * MIB, 8 * MIB, 6 * MIB)
        # One 256 MiB aggregate transfer allowance; remaining physical capacity
        # covers shared fixture, compute/KV, held request outputs and libraries.
        available, workspace, safety, prefix = 256 * MIB // job['callers'], 0, 8 * MIB, 0
    else:
        count, output = 16, (8 * 64 * 2 if job['size'] == 'small' else 24 * MIB)
        scratch, small = (2 * MIB, MIB) if job['size'] == 'small' else (16 * MIB, 8 * MIB)
        available, workspace, safety, prefix = 2 * output + scratch + 8 * MIB, 0, 8 * MIB, 0
    plan = plan_transfer_memory([output] * count, [scratch] * count,
        available_bytes=available, workspace_bytes=workspace, safety_bytes=safety,
        max_resident_groups=prefix, subgroup_scratch_bytes=[small] * count)
    adaptive = job['policy'] == 'adaptive'
    slots = plan.slots if adaptive else 2
    native = plan.native_scratch_bytes if adaptive else (64 * MIB // job.get('callers', 1))
    return dict(policy=job['policy'], plan=asdict(plan), plan_applied=adaptive,
                planning_ms=(time.perf_counter_ns() - started) / 1e6,
                contexts=plan.contexts if adaptive else 2, slots=slots,
                native_scratch_per_context=native, subgroup_members=plan.subgroup_members if adaptive else None,
                output_capacity=plan.output_bytes if adaptive else (slots + 1) * output,
                resident_groups=plan.resident_groups if adaptive else
                    (job['layers'] if job.get('placement') == 'resident' else 8 if job.get('placement') == 'partial' else 0),
                packed=job['policy'] != 'fresh', recycle=job['policy'] in ('recycle', 'adaptive'))


class TorchCompletion:
    def __init__(self, outputs, sources, temporaries, event):
        self.outputs, self.sources, self.temporaries, self.event = outputs, sources, temporaries, event

    def done(self):
        ready = self.event.query()
        if ready: self.sources = self.temporaries = None
        return ready

    def wait(self):
        self.event.synchronize()
        self.sources = self.temporaries = None
        return self.outputs

    def wait_stream(self):
        stream = torch.cuda.current_stream()
        stream.wait_event(self.event)
        for out in self.outputs: out.record_stream(stream)
        return self.outputs


class PackedTorch:
    def __init__(self, images, config, controls):
        self.images, self.config, self.native = images, config, controls.native
        self.stream, self.pending = torch.cuda.Stream(), deque()
        self.arena = PackedOutputArena('cuda:0', max_bytes=config['output_capacity'], reuse=config['recycle'])
        self.groups = self.members = self.payload = self.subgroup_waits = 0

    def submit(self, names, *, outputs=None, **options):
        while self.pending and self.pending[0].done(): self.pending.popleft()
        names = tuple(names)
        sources = [self.images[n] for n in names]
        with torch.cuda.stream(self.stream):
            if outputs is None: outputs = self.arena.allocate([descriptor(q) for q, s in sources])
            if len(outputs) != len(sources): raise ValueError('output count')
            temporaries = []
            cap = self.config['subgroup_members'] or len(sources)
            for index, ((q, scale), out) in enumerate(zip(sources, outputs)):
                gpu_q, gpu_scale = q.to('cuda:0', non_blocking=True), scale.to('cuda:0', non_blocking=True)
                self.native.dequant(gpu_q, gpu_scale, out, self.stream.cuda_stream)
                temporaries.extend((gpu_q, gpu_scale))
                if cap < len(sources) and (index + 1) % cap == 0:
                    self.stream.synchronize()
                    temporaries.clear()
                    gpu_q = gpu_scale = None
                    self.subgroup_waits += 1
            event = torch.cuda.Event()
            event.record()
        completion = TorchCompletion(outputs, sources, temporaries, event)
        self.pending.append(completion)
        self.groups += 1
        self.members += len(names)
        self.payload += sum(q.numel() + 4 * s.numel() for q, s in sources)
        return completion

    def prefetch(self, groups, *, slots=2):
        return FastPrefetch(self, groups, slots=min(slots, self.config['slots']))

    def stats(self):
        return dict(groups=self.groups, members=self.members, payload_bytes=self.payload,
                    subgroup_waits=self.subgroup_waits, output_arena=self.arena.stats(), policy=self.config)

    def close(self):
        while self.pending: self.pending.popleft().wait()
        self.arena.close()


def install(module, inputs, config, observations):
    """Replace only factories for the opt-in policies; fresh controls are intact."""
    if not config['packed']: return

    class BudgetTransfers(FastTransfers):
        def __init__(self, *args, **kwargs):
            kwargs.update(contexts=config['contexts'], max_scratch_bytes=config['native_scratch_per_context'])
            super().__init__(*args, **kwargs)
            self.weight_arena = None
            self.loader = None

        def stats(self):
            result = super().stats()
            result['weight_policy'] = config
            result['weight_storage'] = self.loader.stats() if self.loader else None
            return result

        def close(self):
            super().close()
            if self.weight_arena is not None:
                self.weight_arena.close()
                observations.setdefault('retired_arenas', []).append(self.weight_arena.stats())

    def weights(owner, snapshots):
        owner.weight_arena = PackedOutputArena('cuda:0', max_bytes=config['output_capacity'], reuse=config['recycle'])
        owner.loader = FastWeights(owner, snapshots, output_arena=owner.weight_arena,
                                  max_group_members=config['subgroup_members'], max_prefetch_slots=config['slots'])
        return owner.loader

    def torch_producer(images, convert, dtype):
        return PackedTorch(images, config, inputs.controls)

    module.FastTransfers, module.FastWeights, module.TorchProducer = BudgetTransfers, weights, torch_producer


@contextmanager
def resident_preload(producer, groups, observations):
    """One contiguous resident allocation, serial per-matrix staging for both."""
    groups = list(groups)
    names = [n for g in groups for n in g]
    if not names:
        yield iter(())
        return
    is_sym = isinstance(producer, FastWeights)
    descriptions = ([p.destination for p, source in producer._items(names)] if is_sym else
                    [descriptor(producer.images[n][0]) for n in names])
    capacity = sum(((torch.empty((), dtype=torch.float16).element_size() *
                     __import__('math').prod(d.shape) + 255) // 256) * 256 for d in descriptions)
    arena = PackedOutputArena('cuda:0', max_bytes=capacity, reuse=False)
    outputs = arena.allocate(descriptions)
    arena.close()
    views = dict(zip(names, outputs))
    observations['resident_packed_bytes'] = capacity
    observations['resident_preload_matrices'] = 0
    def delivered():
        for group in groups:
            for name in group:
                completion = producer.submit([name], outputs=[views[name]])
                completion.wait()
                del completion
                torch.cuda.empty_cache()  # Preparation only, same opportunity for both.
                observations['resident_preload_matrices'] += 1
            yield [views[name] for name in group]
    yield delivered()
