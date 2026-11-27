"""Report per-process medians and the median across processes, in milliseconds."""
import csv
import statistics
import sys
from collections import defaultdict

samples = defaultdict(list)
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        metric = "fill_ns" if row["binary"] == "phases" else "wall_ns"
        samples[row["case"], metric, row["revision"], int(row["block"])].append(int(row[metric]))
medians = defaultdict(list)
for (case, metric, revision, block), values in sorted(samples.items()):
    assert len(values) == 50
    medians[case, metric, revision].append(statistics.median(values) / 1e6)
for key, values in medians.items():
    case, metric, revision = key
    print(case, metric.replace("_ns", "_ms"), revision, " ".join(f"{v:.4f}" for v in values), "median", f"{statistics.median(values):.4f}")
