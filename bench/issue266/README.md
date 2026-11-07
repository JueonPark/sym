# Budgeted output lifetimes and prefetch (#266)

This qualification covers packed output ownership, bounded reuse, physical
capacity estimates and prefetch/subgroup selection. Read the
[runtime contract](../../docs/budgeted-transfer-memory.md) and
[frozen protocol](PROTOCOL.md) before interpreting the results.

The numerical workloads and immutable source fixtures come from the
[original E1–E5 artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04).
The campaign reuses their E4/E5 loops and Torch kernel controls. A fresh-output
control retains existing native scratch reuse, so packing and lifetime reuse
are measured separately from that earlier optimization.

```sh
python -m pytest -q bench/issue266/test_analysis.py
python bench/issue266/run.py --artifact /path/to/sym_experiments_261104 \
  --build /path/to/qualified/runtime --output /path/to/new/campaign \
  --python /path/to/qualified/python
python3 bench/issue266/analyze.py /path/to/campaign
```

`run.py` executes the full360-job matrix serially. An explicit `--pilot-regex`
selects an excluded pilot; it cannot produce accepted analysis. Resuming a
campaign requires unchanged source, input, runtime and job hashes. Each job
uses a fresh process and a separate raw result, trace, execution log and audit.
The analyzer requires only Python's standard library and can run without CUDA
or Torch on a downloaded artifact.
