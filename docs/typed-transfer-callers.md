# Retained resources in direct typed callers (#218)

LLM and MoE `WeightFetcher` calls now retain compatible typed scratch, streams,
and workers with the existing `TransferResources` API. Payloads, scales,
requests, caller streams, and output tensors remain fresh for each execution.
This change has no compiler or native runtime implementation changes.

The workload sources live in PR #162. This caller PR is stacked on that branch
and must run against current main's runtime, including #197–#199. The measured
runtime source is `45d64958ee30b4c3fa8c0733e0a992d450ac3757`. Historical numbers
in #191 describe the earlier optimization series; they are not an additional
gain from the caller adoption measured here.

## Audit and ownership

| Direct caller | Decision |
| --- | --- |
| `examples/workloads/common.py::WeightFetcher.fetch`, used by LLM/MoE | Retained by default, one lazy owner per canonical CUDA ordinal. `cuda` resolves using the current device on each call. `--weight-resources per-call` passes `None` for the control. |
| `examples/compiler_runtime_handoff.py::s_latency` | The repeated typed diagnostic reports both retained and per-call policies, first transfer separately from warmed medians, and resource counters before/after close. |
| `examples/torch_typed_relocation.py::run_cuda` | Keep explicit per-call conformance witnesses: two policies and small symbolic bindings, not a sustained weight-loading caller. |
| `sym_reloc/examples/typed.py::run` | Keep per-call installed-package correctness witnesses. No performance claim is derived from these tiny calls. |
| Direct dispatch unit/conformance/fault tests | Keep intentional policy controls and failure isolation; retained cases already opt in. |
| `reloc_torch/runtime.py::TransportAdapter` | Already supplies its frontend-owned/borrowed resources. No change. |

`RelocBackend()` manages frontend AUTO reuse. Direct
`execute_typed_transfer(request)` still means per-call resources unless the
caller supplies an owner. `WeightFetcher` adopts the existing API rather than
changing this default globally.

Use a context manager or `close()` for a fetcher. Closing is idempotent; later
fetches fail without recreating resources. Every device owner's close is
attempted even if execution/model work or another owner's close raises. Runtime
completion failures remain errors; quarantined resources retain the native
ownership contract. The helper is for sequential workload calls, not concurrent
submission. Per-device retention avoids context retirement on alternating
devices but does not introduce async prefetch or multi-GPU concurrency.

Each owner defaults to 64 MiB retained typed host+device scratch. The optional
`live_bytes` cap defaults to zero (unlimited); set it explicitly for a hard
scratch budget. Multiple devices multiply these limits. Tensor input/output
storage, Torch allocator memory, and layout-frontend owners are separate.
`weight_resources.before_close` and `after_close` report completed-call gauges
and cumulative counters. They are not peak live memory. Empty per-call snapshots
mean no retained-owner instrumentation, not zero resource allocations.

## Reproduce the matched policy comparison

Build current main with CUDA enabled, Release, and NVTX disabled. Run the
stacked caller sources with that staged runtime first on `PYTHONPATH`:

```sh
export SYM_BUILD=/path/to/current-main/build
export SYM_PYTHON=/path/to/qualified/python
export PYTHONPATH="$SYM_BUILD/python"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"

"$SYM_PYTHON" -m pytest -q libreloc/python/examples/workloads/tests
taskset -c 4-7,20-23 "$SYM_PYTHON" bench/typed_weight_resources.py \
  --build "$SYM_BUILD" --rounds 3 --output-dir /tmp/typed-weight-resources

"$SYM_PYTHON" libreloc/python/examples/workloads/run_workloads.py \
  --build "$SYM_BUILD" --python "$SYM_PYTHON" --only llm moe \
  --weight-resources per-call
```

Choose CPUs appropriate to the target GPU/NUMA topology. Do not run benchmark
processes concurrently with GPU tests or other timing work. Each policy/workload
round runs in a fresh process; process order is shuffled. Eight Torch threads
and one interop thread are used by default. The original model shapes,
computation, precisions, calibration, and correctness checks remain unchanged.
Torch precedes Sym within each process, as in PR #162; oracle checks touch data
outside the timer. GPU clocks are not locked.

The harness retains raw per-transfer samples, first calls, totals after omitting
the first call of each kind, full totals, policy resource snapshots, and source/
binary hashes. It verifies equal dispatch rows, bytes, workload outputs, and
correctness checks across policies. These are completed-transfer measurements,
not whole-model throughput. Final owner close is outside the transfer timer;
the ordinary process run includes cleanup before exit. Changing shapes after
the first call can still grow resources, so the remaining-call total is not a
fixed-shape steady-state microbenchmark.

## Qualification results

Results and validation are recorded below after the matched runs.
