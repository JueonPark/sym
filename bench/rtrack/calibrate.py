#!/usr/bin/env python3
"""R0.3 (issue #76): per-session machine calibration, recorded once and
embedded into the CSV header by run_rtrack.py.

Every probe is best-effort: a missing tool records null plus a note,
never a crash. stdlib only.

  python3 bench/rtrack/calibrate.py --out calibration.json \
      [--load-bin ./bench-rtrack]

--load-bin points at the rtrack driver; when given, PCIe link status is
sampled while a short Method-B run keeps the bus busy (Gen4 cards
downtrain at idle, so idle lspci/nvidia-smi numbers understate the link).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

TRIAD_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* Schoolbook triad a[i] = b[i] + s*c[i] (NOT the licensed STREAM code).
   3 arrays x 8 bytes per element move per iteration; best of 5. The init
   loop is parallel too: first-touch page placement must match the triad
   threads or a multi-NUMA-node host measures remote-node bandwidth.
   Casts keep this valid C++ for the g++ fallback. */
int main(void) {
  const long n = 1L << 26; /* 3 x 512 MiB */
  double *a = (double *)malloc(n * 8), *b = (double *)malloc(n * 8),
         *c = (double *)malloc(n * 8);
  if (!a || !b || !c) return 1;
#pragma omp parallel for
  for (long i = 0; i < n; ++i) { b[i] = 1.5; c[i] = 2.5; a[i] = 0.0; }
  double best = 0;
  for (int rep = 0; rep < 5; ++rep) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#pragma omp parallel for
    for (long i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double gbps = 3.0 * 8.0 * n / sec / 1e9;
    if (gbps > best) best = gbps;
  }
  if (a[123] == 0.0) return 1; /* keep the loop alive */
  printf("%.2f\n", best);
  return 0;
}
"""


def run(cmd, timeout=60, **kw):
    """Run a command, return stdout or None (recording is best-effort)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, **kw)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def probe_cpu(notes):
    model = None
    cpuinfo = read_file("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    governors = sorted({
        g for g in (read_file(p) for p in glob.glob(
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"))
        if g
    })
    if not governors:
        notes.append("cpufreq governors unreadable")
    thp = read_file("/sys/kernel/mm/transparent_hugepage/enabled")
    return model, governors, thp


def probe_gpus(notes):
    fields = ("name,driver_version,pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max,"
              "clocks.sm,clocks.mem")
    out = run(["nvidia-smi", f"--query-gpu={fields}",
               "--format=csv,noheader"])
    if out is None:
        notes.append("nvidia-smi unavailable")
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        gpus.append({
            "name": parts[0], "driver": parts[1],
            "pcie_gen_current_idle": parts[2], "pcie_gen_max": parts[3],
            "pcie_width_current_idle": parts[4], "pcie_width_max": parts[5],
            "clocks_sm": parts[6], "clocks_mem": parts[7],
        })
    return gpus


def probe_pcie_under_load(load_bin, notes):
    if not load_bin:
        notes.append("pcie_under_load skipped: no --load-bin")
        return None
    # Small N so the driver reaches its copy loop quickly; sample the link
    # once per second for up to 25 s and keep the MAX gen/width observed --
    # a single fixed-delay sample can land during CUDA-context/fixture
    # setup while the bus is still idle and downtrained.
    proc = subprocess.Popen(
        [load_bin, "--transform", "T1", "--method", "b", "--n", "4096",
         "--chunk-mib", "64", "--warmup", "0", "--iters", "100000",
         "--no-verify", "--csv", os.devnull],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    best = None
    try:
        deadline = time.time() + 25
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1)
            out = run(["nvidia-smi",
                       "--query-gpu=pcie.link.gen.current,"
                       "pcie.link.width.current",
                       "--format=csv,noheader", "-i", "0"])
            if out is None:
                continue
            try:
                gen, width = [int(p.strip()) for p in out.split(",")[:2]]
            except ValueError:
                continue
            if best is None or (gen, width) > best:
                best = (gen, width)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if best is None:
        notes.append("pcie_under_load sample failed")
        return None
    return {"gen": str(best[0]), "width": str(best[1])}


def probe_triad(notes):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("g++")
    if cc is None:
        notes.append("triad skipped: no C compiler")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "triad.c")
        exe = os.path.join(tmp, "triad")
        with open(src, "w") as f:
            f.write(TRIAD_C)
        for flags in (["-O3", "-march=native", "-fopenmp"],
                      ["-O3", "-march=native"]):
            if run([cc, *flags, src, "-o", exe]) is not None:
                out = run([exe], timeout=300)
                if out is not None:
                    try:
                        return float(out)
                    except ValueError:
                        break
        notes.append("triad compile/run failed")
        return None


def probe_nvbandwidth(notes):
    if shutil.which("nvbandwidth") is None:
        notes.append("nvbandwidth not installed")
        return None
    result = {}
    for test, key in (("host_to_device_memcpy_ce", "h2d_gbps"),
                      ("device_to_host_memcpy_ce", "d2h_gbps")):
        out = run(["nvbandwidth", "-t", test], timeout=300)
        if out is None:
            notes.append(f"nvbandwidth {test} failed")
            continue
        for line in out.splitlines():
            if line.strip().startswith("SUM"):
                try:
                    result[key] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
    return result or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--load-bin", default=None,
                    help="bench-rtrack binary for the under-load PCIe sample")
    args = ap.parse_args()

    notes = []
    model, governors, thp = probe_cpu(notes)
    cal = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": os.uname().nodename,
        "kernel": " ".join(os.uname()),
        "cpu_model": model,
        "governors": governors,
        "transparent_hugepage": thp,
        "numactl_available": shutil.which("numactl") is not None,
        "cuda_toolkit": run(["nvcc", "--version"]) or
                        read_file("/usr/local/cuda/version.json"),
        "gpus": probe_gpus(notes),
        "pcie_under_load": probe_pcie_under_load(args.load_bin, notes),
        "triad_gbps": probe_triad(notes),
        "nvbandwidth": probe_nvbandwidth(notes),
        "notes": notes,
    }
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")
    print(f"calibration written to {args.out}", file=sys.stderr)
    if notes:
        print("notes: " + "; ".join(notes), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
