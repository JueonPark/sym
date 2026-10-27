"""Foreign-work audit for the beat_torch measurements.

bench/issue227's audit flags build tools and foreign GPU compute. A shared
host also runs CPU-bound jobs it does not know (a CI runner's simulators
loaded this host to 18 during one calibration run), so this audit adds every
foreign process whose CPU share is at least 50%. `Watch` polls it from a
thread while an in-process measurement runs.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue227.run import audit

CPU_BUSY_PERCENT = 50.0


def foreign(allowed):
    """[(pid, name)] of foreign work: issue227's audit plus busy CPU processes
    outside `allowed`, this process and their descendants."""
    found = list(audit(set(allowed) | {os.getpid()}))
    rows = [line.split(None, 4) for line in subprocess.check_output(
        ['ps', '-eo', 'pid=,ppid=,pcpu=,stat=,comm='], text=True).splitlines()]
    own = set(allowed) | {os.getpid()}
    for _ in range(10):
        own.update(int(row[0]) for row in rows if int(row[1]) in own)
    found += [(int(pid), comm) for pid, ppid, pcpu, state, comm in rows
              if float(pcpu) >= CPU_BUSY_PERCENT and int(pid) not in own and not state.startswith(('T', 'Z'))]
    return sorted(set(found))


class Watch:
    """Poll `foreign` every `interval` seconds while the block runs; the
    observations are in `.observed` afterwards."""

    def __init__(self, allowed=(), interval=3.0):
        self.allowed, self.interval, self.observed = set(allowed), interval, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.observed += foreign(self.allowed)

    def __enter__(self):
        self.observed += foreign(self.allowed)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.observed = sorted(set(self.observed + foreign(self.allowed)))
        return False
