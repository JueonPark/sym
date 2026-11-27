"""Collect matched native transfer samples on one available CPU core (taskset)."""
import argparse
import csv
import io
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path, nargs="?")
args = parser.parse_args()
builds = {"baseline": args.baseline}
if args.candidate:
    builds["candidate"] = args.candidate

with args.output.open("x", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("block", "revision", "binary", "case", "iteration", "wall_ns", "fill_ns"))
    for block in range(6 if args.candidate else 3):
        for binary, case in (("transfer", "padded"), ("transfer", "plain"), ("phases", "padded")):
            order = list(builds) if block % 2 == 0 else list(reversed(builds))
            for revision in order:
                proc = subprocess.run([builds[revision] / binary, case], text=True,
                                      stdout=subprocess.PIPE, check=True)
                rows = list(csv.DictReader(io.StringIO(proc.stdout)))
                if [int(row["iteration"]) for row in rows] != list(range(50)):
                    raise ValueError("incomplete timing run")
                for row in rows:
                    writer.writerow((block, revision, binary, case, row["iteration"],
                                     row["wall_ns"], row["fill_ns"]))
                output.flush()
