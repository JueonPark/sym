# Region planner qualification, issue 271

The primary comparison is a finite measured oracle on one GPU. The region
reads immutable host FP32 groups, produces an FP16 concatenation, and runs a
specified pointwise consumer. Plans vary where to narrow, what to send,
whether to write directly into the consumer allocation, how much stays
resident, chunk count, lookahead depth and slot reuse. The executor owns
the same CPU, transfer stream and compute stream in all policy comparisons.
The output contract and group order are fixed. Full residency is a distinct
deployment option and includes preload and storage in lifecycle accounting.

## Information available at selection

Use only region facts, the executor's legal candidate catalog, its byte bounds,
the declared free capacity, the native per-transfer selector and independently
timed component samples. The two-stream probe calibrates contention. Preserve
the order in which the executor submits work, dependencies for slot retirement,
and the lifetimes of the fresh result and retained predecessor. Charge resources
held by the context. Reject uncovered or infeasible candidates. Refuse the
request when a qualified bounded fallback cannot fit.

Freeze the fit and control families before reading the exhaustive request
timings. The training shape serves to diagnose fit, never to select a winner by
table lookup. Test a different group shape, a repeated-consumer workload and a
different capacity. Held-out setup, completed service times, physical
observations, admission/OOM outcomes and arrival completions belong only to
scoring. Keep every candidate in the denominator and report coverage.

## Comparison and scoring

- Fixed controls use each offload representation with zero resident groups,
  one chunk, depth one and fresh storage. Include full residency as another
  fixed control. The current per-transfer cost model selects CPU or GPU
  transformation with the same remaining defaults.
- Placement-only varies representation and resident prefix. Schedule-only
  varies chunks and depth. Buffer-only varies slot reuse. The latter two
  keep the native single-transfer placement and zero residency. The joint
  search can vary all dimensions. All policies draw from one executable catalog.
- Warm cost is median completed request time. Lifecycle cost is setup, the
  first completed request, subsequent warm executions and cleanup over a fixed
  short horizon. Calibration and planning wall time are reported separately
  to show when they can amortize.
- In a separate paced experiment a serialized server receives requests at
  specified release times. A deadline applies to nearest-rank p95 sojourn
  from offered arrival through completion. Select an executable plan *and*
  offered load, then score that same pair. Report its completed throughput
  over the full trace and any SLO miss. This finite replay supports only its
  declared arrival trains.
- Exhaustively score every measured feasible choice on the same workload,
  budget and objective. Latency regret is selected/oracle minus one.
  Throughput regret is oracle/selected minus one. Report false admissions,
  SLO violations, refusals with a feasible oracle and cases with no feasible
  oracle separately. Report absolute percentage prediction error on comparable
  outcomes, including both favorable and regressing cells. A candidate timing
  or validation failure is never a faster fallback.
- Compare the training and held-out errors. If a learned resource model does
  not beat the controls or misses the deadline, narrow the claim to executable
  selection and quantify the miss. The results of earlier E2/E4/E5 campaigns
  retain their own hardware, physical-budget and concurrency contracts.

## Extending the executable catalog

A review of the first 49-plan sweep identified a missing lowering. Destination
forwarding folds narrowing into the write to the concatenation, while the
pointwise additions still launch separately. Extend the catalog with a fused
GPU narrowing-and-pointwise kernel at every offload/partial configuration.
Time the new kernel independently for each supported pass count. Model it as
one GPU submission after raw H2D, using the raw slot and writing directly to
the output. Include a fixed fused control. Repeat the **entire** randomized
sweep on the common executor. Preserve the existing prices, objectives,
capacity contract and control definitions. This change followed inspection
of the executor and the initial results. The expanded campaign uses the same
held-out tasks, so it is a repeat on known conditions.

A second executor audit established that the resident prefix and the first
offloaded DMA can overlap even with one slot. Apply the existing calibrated
link/compute rule to that overlap. The numerical input to the rule and the
objective stay fixed. Score the revised catalog and DAG on the new sweep.

Freeze one additional test before that rerun. `held_final` uses six groups
of 655,360 elements, one FP16 add, one retained predecessor, and 17 and
23.5 MiB budgets. It uses the same two offered periods, 40 requests and 3 ms
p95 deadline. Its shape and budgets have no pilot end-to-end timings. Measure
the five-call lifecycle directly using a new context for every successful arm.
