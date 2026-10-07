# Fused host `torch.stack` benchmark

Completed-call latency of `torch.stack(xs, dim).to("cuda")` on host inputs:
PyTorch eager and Inductor (as written, and rewritten as copy-then-GPU-stack,
plus a pinned host stack) against Sym's fused stacked transfer through the
direct adapter (FD) and `RelocBackend` (FB with tuned transfer options, FBD
with `RelocBackend()`'s default ones), and the single-source proxy (S1) that
bounds FD. `stack_fusion.py`'s docstring has each method's exact call:

| Id | Method |
|---|---|
| B1 | eager as written, `torch.stack(xs, dim).to(dev)` |
| B2 | eager copy-then-stack, `torch.stack([x.to(dev) for x in xs], dim)` |
| B3 | Inductor as written, `torch.compile(B1)` |
| B4 | Inductor copy-then-stack, `torch.compile(B2)` |
| B5 | pinned host stack, `torch.stack(xs, dim, out=pinned).to(dev)` |
| S1 | Sym single-source transfer over a pre-stacked `[N, R, C]`, tuned options |
| FD | Sym fused through the direct adapter (`prepare_stacked_transfer` / `execute_transfer`), tuned options |
| FB | `torch.compile(B1, backend=RelocBackend(min_stack_bytes=0, transfer_options=<tuned options>))` |
| FBD | `torch.compile(B1, backend=RelocBackend(min_stack_bytes=0))`: default transfer options (`gather_threads` 1, pageable staging) |
| GB | `torch.compile(B1, backend=RelocBackend(min_stack_bytes=<one byte above the largest measured size>))`: every call falls back to PyTorch below the threshold, so GB times the below-threshold path, `torch.compile`'s own per-call cost included, at every size |

Measured configuration:

- The tuned transfer options, used by S1, FD and FB, are
  `{"pinning": "auto", "min_pinned_bytes": 8 << 20, "gather_threads": 8}`
  (`--min-pinned-bytes` and `--threads`). FBD and GB pass no
  `transfer_options`.
- `run.sh` pins the process to GPU 0's local cores (`STACK_FUSION_CPUS`,
  default `4-7,20-23`); the harness runs `--threads` Torch threads (default 8)
  and one interop thread.
- Sym's methods run on warm, retained transfer resources.

The results README's header prints the configuration recorded in the JSON.
After every timed method the harness checks that method's backend stats: FB
and FBD must execute every call through Sym with no fallback, and GB must
record `below_stack_threshold` for every call. Anything else fails the run
(gate 1).

`report.py` evaluates the gates:

- Gate 1: every output exact and the run complete.
- Gate 2: at >= 8 MiB, FD p50 <= 1.10 x S1 p50 in every configuration that
  stacks at dim 0 or 1 (the last dim is reported descriptively).
- Gate 3: at >= 32 MiB, FB (tuned options) p50 < min(B1, B3) p50 in every
  configuration.
- Gate 4, two rows by one rule: the smallest measured size from which the
  method beats min(B1, B3) in every configuration at every size from there
  up, or `none`. "4 (tuned options)" is FB's crossover, published for the
  tuned options; "4 (default options)" is FBD's and sets
  `DEFAULT_MIN_STACK_BYTES`.

`report.py` exits 1 when gate 1, 2 or 3 fails; gate 4 never changes the exit
status.

```bash
export SYM_PYTHON=/path/to/cp314-cu126/bin/python
export PYTHONPATH=/path/to/cuda-build/python
export SYM_RELOC_EXPORT=/path/to/cuda-build/sym/tools/sym-reloc-export
bench/stack_fusion/run.sh --output bench/results/stack-fusion/latency.json
"$SYM_PYTHON" bench/stack_fusion/report.py bench/results/stack-fusion/latency.json \
  > bench/results/stack-fusion/README.md
```

The results README is generated: record deviations (toolchain, gate history,
Inductor compile failures) in a `deviations.md` next to the JSON, with its own
`## Deviations` heading. `report.py` appends that file verbatim, so the
section survives regeneration.

Do not run it concurrently with other timing jobs or GPU test suites.
