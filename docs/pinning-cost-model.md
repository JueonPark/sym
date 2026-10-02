# From an explicit pinning gate to a cost model

A staging buffer holds bytes while Sym transforms a tensor and transfers it
between CPU and GPU. Pinned host memory supports direct asynchronous DMA, but
allocating/registering it has a cost. Pageable memory has cheaper allocation;
the CUDA driver may make an internal pinned copy and may block the submitting
CPU thread. A retained buffer amortizes allocation across completed calls.
These mechanisms make pinning a latency decision, not a property of tensor size
alone.

## Implemented contract

Python accepts `pinning="auto" | "pinned" | "pageable"` and an optional
`min_pinned_bytes`. Unconfigured auto uses pageable staging. An explicit
threshold opts into a caller-qualified size gate for retainable staging:

```
auto_pins = threshold_is_configured && capacity_is_retainable
            && actual_wire_bytes >= min_pinned_bytes
```

This is the first-stage contract of #189. It does not predict first-call
profitability or infer future reuse. Native legacy `TransferOptions` still
starts in forced pinned mode. An already dense host input can upload directly;
this policy neither registers that input nor adds a staging copy to satisfy a
forced mode. Typed parameter uploads are assessed independently of the weights.
The layout ring uses total wire bytes for comparison, then reports its actual
allocated capacity and active slots separately.

A configured threshold is meaningful only within its measured device,
direction, layout, CPU/NUMA placement, thread count, chunk schedule, budgets and
reuse regime. The caller supplies that scope today; the runtime does not detect
hardware profile mismatches. Outside a qualified configuration, omit the
threshold. A future profile API should identify those dimensions explicitly.

Selection happens before allocation/acquisition. Memory kind is part of
compatibility; a request never changes the pinning of a live allocation.
Budgets and completion/ownership rules apply equally to both modes. Unknown
completion quarantines buffers and owners rather than freeing memory still
possibly in use. Per-request reports contain scalar decisions only: policy,
kind, wire and capacity bytes, threshold, reason, reuse and retention eligibility.

## Proposed adaptive model (not implemented)

Predict the latency of the **current completed request**, with either allocation
mode. Separate measured cache hits, growth and misses. Compare the same transfer
plan, outputs, CPU transformation and completion contract:

```
T_mode = frontend + acquisition_and_growth(mode, cache_state)
         + scheduled_CPU_and_DMA_critical_path(mode, plan)
         + completion_and_retirement(mode)
```

The critical path must respect dependencies and shared resources. For chunk
`i`, its gather must finish before its DMA can begin. The next CPU gather can
start after submission, subject to the availability of a staging slot; reuse
of that slot waits for the previous DMA. Each DMA waits for an available copy
engine and its producer. Pageable submission may add a driver staging copy or
host wait. Estimate this schedule, or calibrate its completed time directly.
Do not add the full CPU time to the full DMA time and then charge overlap again.
The typed CPU reference currently transforms the whole wire buffer before
copying, so it needs a different schedule from the chunked layout pipeline.
A direct typed upload has no Sym CPU transpose to overlap.

Model inputs should include:

- Actual wire bytes, staging capacity, active slots, chunk sizes and direction.
- Existing compatible idle capacity versus new allocation or growth; setup and
  retirement costs observed separately for pinned and pageable memory.
- CPU transform and host-copy costs for the selected operation, shape and
  strides, CPU ISA, worker count, affinity and NUMA placement.
- DMA bandwidth/latency and submission blocking for the device/PCIe topology,
  driver and mode; overlapping work and contention on the host and GPU.
- Aggregate busy and retained budgets, eviction costs, and live-byte limits.
- Prediction error, sample count, profile age and compatibility with the
  current environment.

An existing cache hit is known reuse. Future calls are not. Default to current
request latency; do not divide a new pin allocation by an assumed large reuse
count. A separately opted-in horizon policy could optimize a caller-declared
number of calls and report its predicted break-even call. It must state that
horizon and include allocation, eviction and final retirement in its total.

Select pinned only if its conservative predicted saving exceeds both an
absolute floor and a relative margin:

```
pageable_lower_bound - pinned_upper_bound
    > max(min_saving_ms, relative_margin * pageable_lower_bound)
```

Estimate bounds from repeated **independent round** results, not from treating
adjacent samples in one warmed process as independent. Insufficient evidence,
a profile mismatch or overlapping uncertainty intervals selects pageable with
an observable reason. Forced modes bypass performance selection while retaining
validation, ownership and resource budgets. Never silently run timing probes on
user tensors or delay a latency-critical call for calibration.

## Calibration and acceptance for the follow-up

Use offline completed-transfer sweeps for both directions and every supported
execution family. Separate a fresh owner, warmed capacity and per-call allocation;
include growth/eviction, parameter buffers, changing shapes and thread/NUMA
configurations. Compare forced modes and Torch with equal payload and completion
semantics. Keep first samples, raw round ordering, correctness results, source
and binary hashes, and exact machine configuration.

Fit on one subset and validate on held-out sizes/shapes and another supported
machine. Report wrong pinning choices and the resulting latency loss, including
cold-call regressions; do not accept a lower average hiding large tail losses.
Define maximum decision overhead and a tolerated regression bound before model
selection. A shadow mode can report a prediction without changing allocation.
Promote only profiles with sufficient held-out evidence; invalidation returns
them to the observable conservative fallback.

Profiler validation must correlate native CPU gather ranges with GPU memcpy
activity for the selected pinned layout path. Submission API durations alone
do not prove overlap. Keep a single-buffer control and characterize pageable
behavior without assuming it cannot overlap. This follows CUDA's documented
[API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/api-sync-behavior.html).

The adaptive implementation is tracked separately in [#208](https://github.com/JueonPark/sym/issues/208). Closing the initial #189
milestone requires the deterministic gate, diagnostics, correctness, measured
qualification and this extension design; it does not assert the model already
exists.
