"""Build/GPU audit plus sustained foreign CPU activity on the shared host."""
import os
from pathlib import Path
import time

from bench.issue227.run import audit as build_gpu_audit

_previous = None
_tick_rate = os.sysconf('SC_CLK_TCK')


def cpu_busy(samples, previous, allowed, *, now, tick_rate):
    own = set(allowed)
    for _ in range(10):
        own.update(pid for pid, value in samples.items() if value[0] in own)
    busy = []
    for pid, (parent, state, name, ticks, start) in sorted(samples.items()):
        age = now - start / tick_rate
        if pid in own or state.startswith(('T', 'Z')) or age < 5:
            continue
        rate = ticks / tick_rate / age
        if previous is not None:
            before, old = previous
            prior = old.get(pid)
            if prior is not None and prior[4] == start and now - before >= .5:
                rate = (ticks - prior[3]) / tick_rate / (now - before)
        if rate >= .5:
            busy.append((pid, 'CPU:' + name))
    return busy


def audit(allowed):
    global _previous
    busy = build_gpu_audit(allowed)
    samples = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            raw = (path / 'stat').read_text()
            end = raw.rindex(')')
            fields = raw[end + 2:].split()
            samples[int(path.name)] = (int(fields[1]), fields[0], raw[raw.index('(') + 1:end],
                int(fields[11]) + int(fields[12]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue  # A process may exit while /proc is being sampled.
    now = time.clock_gettime(time.CLOCK_BOOTTIME)
    seen = {pid for pid, _ in busy}
    busy += [(pid, name) for pid, name in cpu_busy(samples, _previous, allowed,
             now=now, tick_rate=_tick_rate) if pid not in seen]
    _previous = (now, samples)
    return busy
