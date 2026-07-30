#!/usr/bin/env python3
"""R0.3 (issue #76): JSON config -> CSV rows.

  python3 bench/rtrack/run_rtrack.py --config bench/rtrack/configs/example.json

Loads the session calibration (running calibrate.py first when the file
is missing), writes '#'-prefixed metadata lines plus the driver's header
row, then invokes bench-rtrack once per (n, threads) point with the full
transform/method/chunk sweep, streaming its stdout rows into the CSV as
they are produced. Relative paths in the config resolve against the
CURRENT working directory (the session ritual runs everything from one
directory). A nonzero driver exit (e.g. a verify-gate failure) aborts the
run with the failing command echoed; an existing non-empty out_csv is
never overwritten. stdlib only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, list):
        out.append((prefix, "; ".join(str(v).replace("\n", " ")
                                      for v in value)))
    else:
        out.append((prefix, str(value).replace("\n", " ")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Absolute so Popen executes the file instead of searching PATH when
    # the config says just "bench-rtrack".
    bin_path = os.path.abspath(cfg["bin"])
    if not os.path.exists(bin_path):
        sys.exit(f"error: driver binary not found: {bin_path}")

    cal_path = cfg.get("calibration", "calibration.json")
    if not os.path.exists(cal_path):
        print(f"run_rtrack: {cal_path} missing, running calibrate.py",
              file=sys.stderr)
        calibrate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "calibrate.py")
        subprocess.run([sys.executable, calibrate, "--out", cal_path,
                        "--load-bin", bin_path], check=True)
    with open(cal_path) as f:
        calibration = json.load(f)

    numactl_prefix = []
    numactl_cfg = cfg.get("numactl", "")
    if numactl_cfg:
        if shutil.which("numactl") is None:
            print("run_rtrack: WARNING numactl configured but not installed;"
                  " running unpinned", file=sys.stderr)
        else:
            numactl_prefix = ["numactl", *numactl_cfg.split()]

    methods = cfg.get("methods", "both")
    transforms = cfg.get("transforms", "all")
    if isinstance(transforms, list):
        transforms = ",".join(transforms)
    chunk_mib = ",".join(str(c) for c in cfg.get("chunk_mib",
                                                 [4, 16, 64, 256]))

    out_csv = cfg.get("out_csv", "rtrack.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        sys.exit(f"error: {out_csv} already exists and is non-empty; "
                 "move it aside or point out_csv at a fresh file "
                 "(measured data is never silently overwritten)")
    with open(out_csv, "w") as csv:
        meta = []
        flatten("config", {k: v for k, v in cfg.items()
                           if k not in ("calibration",)}, meta)
        flatten("calibration", calibration, meta)
        for key, value in meta:
            csv.write(f"# {key}: {value}\n")
        csv.flush()

        first = True
        for n in cfg.get("n", [8192]):
            for threads in cfg.get("threads", [1]):
                cmd = numactl_prefix + [
                    bin_path,
                    "--transform", transforms,
                    "--method", methods,
                    "--n", str(n),
                    "--chunk-mib", chunk_mib,
                    "--threads", str(threads),
                    "--warmup", str(cfg.get("warmup", 5)),
                    "--iters", str(cfg.get("iters", 30)),
                    "--machine", cfg.get("machine", os.uname().nodename),
                    "--csv", "-",
                ]
                if "variant" in cfg:
                    cmd += ["--variant", str(cfg["variant"])]
                if first:
                    cmd.append("--csv-header")
                    first = False
                print(f"run_rtrack: {' '.join(cmd)}", file=sys.stderr)
                # Stream rows as the driver emits them so an interrupted
                # sweep keeps every completed measurement.
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        text=True)
                for line in proc.stdout:
                    csv.write(line)
                    csv.flush()
                proc.wait()
                if proc.returncode != 0:
                    sys.exit(f"error: driver exited {proc.returncode}: "
                             f"{' '.join(cmd)}")
    print(f"run_rtrack: wrote {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
