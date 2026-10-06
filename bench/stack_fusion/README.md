# Fused host `torch.stack` benchmark

Completed-call latency of `torch.stack(xs, dim).to("cuda")` on host inputs:
PyTorch eager and Inductor (as written, and rewritten as copy-then-GPU-stack,
plus a pinned host stack) against Sym's fused stacked transfer through the
direct adapter (FD) and `RelocBackend` (FB), and the single-source proxy (S1)
that bounds FD. `report.py` evaluates four gates:

- Gate 1: every output exact and the run complete.
- Gate 2: at >= 8 MiB, FD p50 <= 1.10 x S1 p50 in every configuration.
- Gate 3: at >= 32 MiB, FB p50 < min(B1, B3) p50 in every configuration.
- Gate 4: the smallest measured size from which gate 3's condition holds in
  every configuration from there up; it sets `DEFAULT_MIN_STACK_BYTES`.

```bash
export SYM_PYTHON=/path/to/cp314-cu126/bin/python
export PYTHONPATH=/path/to/cuda-build/python
export SYM_RELOC_EXPORT=/path/to/cuda-build/sym/tools/sym-reloc-export
bench/stack_fusion/run.sh --output bench/results/stack-fusion/latency.json
"$SYM_PYTHON" bench/stack_fusion/report.py bench/results/stack-fusion/latency.json \
  > bench/results/stack-fusion/README.md
```

Do not run it concurrently with other timing jobs or GPU test suites.
