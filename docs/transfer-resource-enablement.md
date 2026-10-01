# Transfer resource default enablement (#173)

`RelocBackend()` now retains compatible layout-transfer resources across calls
through its owned runtime adapter. Construction remains lazy: the first real
layout execution creates the resource owner; compatible later calls reuse its
staging buffers, streams and gather workers. Every call still validates its
current source, allocates an independent output, establishes current-stream
ordering, and completes before returning.

This is the final implementation item for [#165](https://github.com/JueonPark/sym/issues/165).
R6 performance/overlap and R7 lifecycle qualification are merged and reviewed
below. The parent and #173 close when this enablement PR merges, not when the
local implementation or measurements finish. See the
[user guide](torch-integration.md#shared-resources-for-compiled-and-eager-calls)
for usage and ownership examples.

## Policy and scope

| Entry point or configuration | Resource policy |
| --- | --- |
| `RelocBackend()` with its own adapter | Omission selects lazy `AUTO`; explicit `AUTO` is equivalent. |
| `RelocBackend(transfer_resources=None)` | Per-call resources, preserving the comparison/opt-out path. |
| Explicit `TransferResources` passed to a backend | Borrowed owner; multiple backends/direct calls can share one budget. |
| Injected `runtime` | Omission or explicit `None` preserves that runtime's policy. Explicit `AUTO`, an owner, or non-None backend transfer options are rejected as ambiguous. |
| Standalone `TransportAdapter()` | Still defaults to per-call resources; configure reuse on the adapter explicitly. |
| Direct `execute_transfer` with omitted/None `resources` | Still uses per-call resources. |
| Typed transfers | Existing typed dispatch; no layout-resource cache is materialized. |

Compiled and eager calls share resources only when they share the backend or an
explicit owner. Leaving `eager_transfers` stops interception and leaves the
backend usable; close the backend when its calls are finished. Backend close
drains its owned adapter/cache, while borrowed adapters and owners remain the
caller's responsibility. Artifact eviction never closes the resource cache.
Inspection, compilation, fake execution and preflight do not create resources.

The tiled CPU transpose kernel, layout plans and H2D chunk schedule are unchanged.
CPU gather for the next chunk can overlap the previous chunk's PCIe copy, with
completion gating reuse of each staging slot. Forward D2H keeps its
single-download-then-gather schedule. Outputs and CUDA event objects are not
pooled. Typed caching, new dtype kernels, automatic CPU/GPU placement, D2H
pipelining and a nonblocking frontend remain outside this change.

The earlier native `CopyBackend::quiesce()` addition changes the C++ interface:
external backend implementations, runtime libraries and native consumers must
be rebuilt together. Serialized plan wire formats are unchanged. Resource
owners are process-local, cannot be pickled or reused after fork, and must be
closed before an external device reset.

## Gate review

**R6 passes the measured benefit and overlap gate.** The
[completed-transfer report](../bench/results/transfer-resource-reuse-171/README.md)
and its hashed raw artifacts use the actual compiled frontend with fresh
outputs and validation. On EPYC 7351 / RTX 2080 Ti, 4096×1024 FP32 H2D with eight
gather participants improves from 28.624 ms to 4.507 ms warmed p50, or 6.35×,
when switching from per-call resources to retention. CPU Inductor plus H2D is
5.503 ms; H2D plus GPU transpose is faster at 2.073 ms. CPU Inductor also wins
on the small and odd-sized cases. These are measured policy comparisons on one
machine, not a universal Torch speedup or a promise for the one-participant
default. At one participant the same reference case improves 29.948 → 6.789 ms
(4.41×). First-call costs and all sample distributions are reported separately.

Across 7,380 timed calls, all 78 warmed retained blocks add zero staging
allocations, stream creations and worker creations. The four-buffer trace has
CPU gather(n+1) overlapping actual GPU DMA(n) in all 15 adjacent transitions of
five requests, by 277–305 µs. The one-buffer control has zero overlapping
transitions with the same four 4 MiB chunks. The timeline proves preservation
of pipelining; most measured frontend improvement comes from avoiding setup.
Clocks were unlocked, and cache-allocation state required explicit benchmark
isolation. The report preserves that protocol, diagnostics and limitations.

**R7 passes the exercised lifecycle gate, with a recorded coverage limitation.**
The [qualification report](transfer-resource-qualification.md) covers byte-exact
changing-input/shape/layout execution, caller streams, concurrent callers,
multiple real GPUs, retained outputs under allocation churn, admission bounds,
clear/close/eviction races, failure ownership and process restrictions. All
supported exercised cases pass: 556 CUDA frontend tests, 384 CPU-only frontend
tests, 24 native-Python resource tests in each build, 4,000 extended-stress
transfers, Release/Debug native matrices, and 87 focused ASan+UBSan tests with
leak detection. Existing skips and exact configurations are enumerated there.

TSan builds but cannot start on this host (`unexpected memory mapping`, exit 66),
including with per-process ASLR disabled. This is unavailable coverage, not a
passing race check. #172 explicitly requires TSan when available; there is no
observed unresolved correctness failure in the exercised matrix. On that basis,
this review accepts enablement while retaining the limitation. It does not claim
absence of all races. [#71](https://github.com/JueonPark/sym/issues/71) continues
to own portable sanitizer CI. Fatal CUDA/driver faults, arbitrary allocator
teardown failures and multi-hour soak behavior were not tested. Supported
failure injection verifies conservative quarantine when completion is unknown;
those resources and tensor owners remain charged for process lifetime.

## Budget decision

Keep the implemented defaults rather than tuning them to a single benchmark:

| Setting | Default | Reason and tradeoff |
| --- | ---: | --- |
| `max_retained_bytes` | 256 MiB | Headroom for varying shapes/configurations; lazy capacity with idle eviction, not an upfront allocation. |
| `max_contexts` | 4 | Allows a small number of concurrent leases across configurations/devices. |
| `max_contexts_per_device` | 2 | Permits the two-caller case qualified on each device while bounding local contexts. |
| `max_background_workers` | 64 | Preserves explicit gather budgets within a per-owner limit; callers are excluded. |
| `max_live_staging_bytes` | `None` | Preserves large transfers that exceed the retained budget; applications can opt into a hard live-staging cap. |
| `acquire_timeout_ms` | `None` | Contention waits for capacity; applications can request bounded admission waiting. |

The R6 owner peaked at 32 MiB staging, two contexts, four streams and 14 owned
background workers. R7 additionally exercised four contexts across two GPUs
and tighter explicit byte/worker limits. These observations support the tested
workloads; they neither measure a universal optimum nor imply that every cache
uses its maximum. Requests still default to four buffers, two streams and one
gather participant, which creates no background workers. Retention does not
silently increase parallelism or reduce an explicit worker request.

The retained limit includes cacheable active and building contexts, not just
idle buffers. Oversize requests use ephemeral contexts subject to context,
worker and optional live-byte limits. There is no hard live-staging byte cap by
default, and multiple owners multiply limits. A shared explicit owner with
`max_live_staging_bytes` is appropriate for a process budget; input/output tensor
allocator memory and borrowed pools remain outside that budget. Impossible
requests fail admission; a timeout never cancels an already-started transfer.
The user guide documents `clear()`, `close()` and observable resource counters.

## Parent acceptance record

Each original #165 acceptance item has implementation and durable evidence:

| Parent acceptance item | Implementation and evidence |
| --- | --- |
| Compatible frontend calls stop allocating staging/streams/workers after initialization | R3–R5 (#168–#170); R6's 78 warmed blocks with zero creation deltas; R8 omitted-policy compiled/eager counter assertions. |
| Byte-exact repeated calls with changing contents/shapes, independent outputs, odd/multichunk and generic/padded layouts | R2/R4/R5 native/frontend matrices (#167, #169, #170); R7 compiled transpose/padded-transpose stress and R8 default-policy extension. |
| Alternating streams, concurrent callers/devices, exhaustion/failures, close/eviction; no premature staging reuse | R1–R5 completion, lease, budget, ownership and shutdown tests; R7 gated pending-copy/growth faults, concurrent two-GPU stress and unknown-completion quarantine checks. |
| Bounded staging/workers/contexts over a long changing-shape workload | R3 admission/accounting tests; R6 recorded counter peaks; R7 100-round stress with allocated-plus-reserved byte, retained-byte, per-device context, worker, event and close-balance assertions; R8 extends this to omission. Finite stress, not a multi-hour soak. |
| Completed real-frontend benchmarks with fresh outputs; first/warmed latency, configuration and sample distributions | R6 report, one/eight-participant raw JSON, environment/source hashes and committed reproduction harness. Timed intervals include CUDA completion. |
| CPU Inductor+H2D and H2D+GPU transpose comparison; one-buffer control | R6 allocation-policy-matched tables and same-chunk one/four-buffer runs; Nsight GPU activity correlation and portable timelines prove overlap separately from latency. |

Performance evidence is unchanged by this default-selection change: the R6
harness selects both policies explicitly. R7 adds tests and test-only fault
modes; R8 changes only Python default resolution and its regression coverage,
with no native execution changes. Historical reports retain the source
revisions and then-current rollout status at which they were produced.

## Default-policy validation

Tested source: `fe28fa85b05122818c7e8ec4e645fb7c5c6bfc32`, based on main after
merged #187 (`f0f3137998de5ca036c2bc01aa1f709655e4dd56`). Only documentation was
uncommitted during these runs. The same EPYC/RTX host, Torch 2.14.0+cu126,
CPython 3.14.7, CUDA 12.6 and Release builds described in R7 were used; both
Python build trees were rebuilt to copy the changed package.

| Check | Result | Evidence |
| --- | --- | --- |
| Full CUDA frontend | 575 passed, 3 existing skips | [log](qualification/transfer-resources-173/frontend-cuda.log) |
| CPU-only frontend with Python reference-count GIL assertions | 388 passed, 1 existing skip, 189 GPU cases deselected | [log](qualification/transfer-resources-173/frontend-cpu.log) |
| Extended CUDA qualification, 100 rounds | 8 passed, no hardware skips; 5,200 transfers plus 2 borrower-survival calls | [log](qualification/transfer-resources-173/gpu-stress.log) |

Coverage includes omitted/default, explicit AUTO, borrowed owner and explicit
None in both directions and all supported layout dtypes; shared compiled/eager
calls; injected runtime ownership; standalone adapter compatibility; lazy import,
fake execution, typed dispatch and clean shutdown. The extended run adds the
omitted policy to the earlier stream/shape/retained-output stress. Native
ASan/UBSan evidence is the unchanged R7 runtime qualification above.

Reproduce with the R7 build environment and commands:

```sh
python -m pytest -q -rs libreloc/python/tests/torch_frontend
# Use the CPU-only extension for this command:
python -m pytest -q -rs -m 'not gpu' libreloc/python/tests/torch_frontend
SYM_RESOURCE_STRESS_ROUNDS=100 python -m pytest -q -rs \
  libreloc/python/tests/torch_frontend/test_resource_qualification.py
```

The [environment record](qualification/transfer-resources-173/environment.json)
identifies source/runtime hashes and exact commands. Verify the logs and record
with `sha256sum -c SHA256SUMS` in `docs/qualification/transfer-resources-173/`.

## Delivered implementation items

| Item | PRs | State at R8 review |
| --- | --- | --- |
| R1 / #166, backend completion and event lifetimes | #175 | Merged (`7120a1d`) |
| R2 / #167, contexts and checked execution | #176, #177 | Merged (`25c7953`) |
| R3 / #168, bounded caching and leases | #179, #180 | Merged (`f3f8ef9`) |
| R4 / #169, explicit Python ownership | #182, #183 | Merged (`c72ce43`) |
| R5 / #170, compiled/eager sharing | #184 | Merged (`a3454b1`) |
| R6 / #171, measurements and overlap | #185, #186 | Merged (`23401a2`, `64a8387`) |
| R7 / #172, lifecycle qualification | #187 | Merged (`f0f3137`) |
| R8 / #173, default policy and documentation | This change | Ready after the validation above; parent closure waits for merge. |

The [original plan](transfer-resource-implementation-plan.md) and
[estimated diff](transfer-resource-reuse-diff.md) remain historical planning
references. Their estimates are not measured latency or remaining work.
