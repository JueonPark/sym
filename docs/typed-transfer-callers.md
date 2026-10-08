# Retained resources in direct typed callers (#218)

LLM and MoE `WeightFetcher` calls now retain compatible typed scratch, streams,
and workers with the existing `TransferResources` API. Payloads, scales,
requests, caller streams, and output tensors remain fresh for each execution.
This change has no compiler or native runtime implementation changes.

The baseline WeightFetcher and its LLM/MoE callers were merged into main in
[PR #232](https://github.com/JueonPark/sym/pull/232). This optimization is
based directly on main and does not depend on #162. Its ancestry includes
the runtime changes #197–#199. The measured runtime source is
`45d64958ee30b4c3fa8c0733e0a992d450ac3757`, unchanged in both PRs. Historical
numbers in #191 describe the earlier optimization series; they are not an
additional gain from the caller adoption measured here.

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

Build this branch with CUDA enabled, Release, and NVTX disabled. Run the
caller sources with the staged runtime first on `PYTHONPATH`:

```sh
export SYM_BUILD=/path/to/this-branch/build
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

Measured 2026-10-08 on AMD EPYC 7351 / RTX 2080 Ti GPU 0, driver 595.71.05,
Torch 2.14.0+cu126 (CUDA 12.6), native CUDA toolkit 12.5.82, CPython 3.14.7.
Affinity was CPUs 4–7,20–23,
eight Torch threads and one interop thread. Caller/benchmark source was
`2957dd38309ecd4246cb6869a24d6679d23541ce`; runtime/compiler source was the main
revision above, rebuilt before testing. All 12 fresh-process runs passed all
3,978 check observations with identical dispatch rows, payload bytes, model
outputs and checks between policies. The weight rows use
`cuda_dequant_relocate`; CPU-reference ownership is qualified separately by
the caller tests.

Raw samples, first calls, round order, logs, hashes and policy snapshots are in
[the committed evidence](../bench/results/typed-transfer-callers-218/).
Milliseconds below are medians of three process totals; ranges are the minimum
and maximum process totals, not confidence intervals.

| Workload | Per-call, after first of each kind | Retained, after first of each kind | Reduction | Torch paired with retained |
| --- | ---: | ---: | ---: | ---: |
| LLM | 466.167 (465.507–468.482) | 360.394 (359.607–361.670) | 22.7% | 181.995 (180.433–182.442) |
| MoE | 199.691 (197.430–201.387) | 126.925 (126.397–126.991) | 36.4% | 75.496 (75.137–76.166) |

The LLM weight-only remaining total changes from 334.783 to 231.444 ms
(30.9%); KV calls use unchanged frontend ownership. Medians of individual
kinds need not sum to the median of the whole process. These improvements are
versus this caller's per-call control, not versus Torch. Torch remains faster
for both complete transfer workloads, and no model-throughput gain is claimed.

| Workload / transfer kind | Per-call first call | Retained first call |
| --- | ---: | ---: |
| LLM weights | 6.182 | 6.379 |
| LLM KV evict | 253.783 | 254.689 |
| LLM KV restore | 86.028 | 86.172 |
| MoE weights | 8.125 | 7.799 |

These are the first calls in initialized workload processes, after the Torch
pass. KV first calls include their graph compilation; the weight recipe is
compiled before its timer. They are not process-cold CUDA measurements. Small
first-call differences do not establish a startup improvement. Including every
transfer call, the median Sym totals are LLM 812.732 → 706.853 ms and MoE
207.828 → 134.608 ms. The paired retained Torch totals are 192.325 and 84.843 ms;
the MoE Torch all-call range is 84.481–104.458 ms, reflecting substantial
first-call variation.

| Retained direct owner, before close | LLM | MoE |
| --- | ---: | ---: |
| Requests / reuse hits | 256 / 255 | 170 / 169 |
| Native context creations | 1 | 1 |
| Device / host scratch allocations | 3 / 2 | 2 / 1 |
| Device / host retained bytes | 8,400,896 / 28,672 | 2,367,488 / 8,192 |
| Total retained scratch | 8,429,568 B (8.039 MiB) | 2,375,680 B (2.266 MiB) |
| Streams / background gather workers | 2 / 0 | 2 / 0 |

The measured GPU weight row needs no CPU gather workers; CPU-reference tests
separately assert that warmed worker counts remain stable. All retained runs
end with zero scratch bytes, streams, and workers after close and balanced
scratch allocation/free counts. The 64 MiB per-device retention limit remains
separate from the uncapped live setting (`live_bytes=0`). No peak-live-memory
claim follows from these completed-call gauges.

The original four-workload suite passed all 55 tests. Validation covers
changing payloads, scales, shapes, retained outputs, alternate
caller streams, canonical device aliases, two/four-GPU use, explicit per-call
execution, a hard live-budget failure, and cleanup when model/dispatch/owner
close raises. All 11 selected current-main GPU typed dispatch tests passed
(three non-GPU tests deselected), including real CUDA completion and
unknown-completion quarantine faults with the fault shim built.

The repeated typed handoff diagnostic passed with both resource policies.
Its CPU-reference retained row reports 19 requests, 18 reuse hits, one native
context, one host scratch allocation, two streams, and seven gather workers;
close reduces its retained bytes, streams and workers to zero. This is a single
descriptive diagnostic, not the fresh-process workload comparison above. Its
`source_dirty` flag reflects the pending documentation/evidence files; measured
caller code matches `2957dd3`. Exact commands and environment details are in
[environment.json](../bench/results/typed-transfer-callers-218/environment.json).
No C++/compiler runtime code is changed by this caller PR.

## Standalone WeightFetcher stack

After extracting the baseline into #232, the standalone baseline passed all
39 tests and the retained-resource branch passed all 51 tests, with no skips.
The four tests removed from the original suite concern the DLRM/GNN command
lines and their GPU examples, which remain in #162. All WeightFetcher,
LLM/MoE, resource lifetime, and runner coverage remains in the new stack.

The timing reports above retain their original source revisions and hashes;
they are not relabeled as measurements of a rewritten commit. Revalidation
checks that the measured LLM/MoE scripts, benchmark and handoff diagnostic
are byte-identical, that `common.py` differs only in its module docstring,
and that the runner's child environment is unchanged. Compiler/runtime
sources still match the measured main revision. Only the runner's model
selection, unrelated example tests, and documentation change during extraction.
See [restack-validation.json](../bench/results/typed-transfer-callers-218/restack-validation.json)
and the accompanying test logs for the current stack's checks.
