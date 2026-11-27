# Joint planning for a relocation region

This research prototype chooses a representation, transformation site, resident
prefix, chunk count, prefetch depth and scratch-buffer policy for one complete
producer–transfer–consumer region. The executable is a bounded PyTorch/CUDA
benchmark with a Triton fused arm. The region facts and the calibration are
inputs to a CPU planner. The exhaustive GPU campaign is its external oracle.
The scope is a single host, one submitter, one transfer stream and one compute
stream. [Region legality](../../docs/relocation-regions.md) · [Qualification protocol](PROTOCOL.md)

## What the measurements support

On the RTX 4070 Ti SUPER the final randomized sweep covered **585 plan/budget
combinations**. All 358 admitted configurations completed and produced the
expected bits. The other 227 hit the arena boundary. Predicted live peaks
matched every successful run. No driver OOM occurred in this experiment.

| Across nine workload/budget cells | Joint | Per-transfer | Placement-only |
|---|---:|---:|---:|
| Median warm regret vs measured oracle | 13.23% | 103.28% | 24.81% |
| Median five-call lifecycle regret | 20.66% | 105.05% | 19.63% |
| Worst warm regret | 313.21% | 1142.48% | 225.71% |

On the previously unmeasured six-group workload, the joint planner selected
the observed warm winner at both budgets. At 17 MiB it reused a single
compact offload slot. At 23.5 MiB it kept all six groups resident. The latter
choice also won the lifecycle comparison. In the 12 MiB training cell,
partial residency plus fusion cost 0.85 ms against a 0.21 ms resident oracle.
Full residency won warm latency wherever it fit, including two cells where
joint search priced another plan lower. A larger catalog improves the best
available choice but makes pricing mistakes consequential.

The selected rate met the 3 ms p95 deadline in **one of four** paced cells.
In the other three, the selected plan/load pair had a measured p95 of 8.32
to 35.33 ms. Every cell had a measured feasible alternative. The model uses
one steady service-time estimate per plan, and a train of arrivals can
accumulate delays that a median service time misses. Component-only
calibration gave median absolute warm prediction errors of 31.24% on the
training shape and 31.00–41.11% on the other workloads. These results support
bounded storage planning and demonstrate the cost of inaccurate latency
and tail predictions. The observed oracle can also favor timing noise across
arms in a single deployment trial. [All controls, errors and paced outcomes](RESULTS.md)

The archived application replay picked the measured winner in 16 of 24
primary decisions. Its worst loss was 28.11% at the physically imposed
512 MiB E4 budget. Removing its 32 MiB margin picked an arm that actually
OOMed in all three rounds. E2, E4 and E5 enter the same search using their
indivisible recorded endpoints and projected physical peaks/context limits.
Their source provides no stage timeline for a new hypothetical plan.
[Historical decisions, evidence and sensitivity](HISTORICAL.md)

## Cost to use the planner

The 7800X3D/4070 Ti SUPER run used PyTorch 2.14.0+cu126, Triton 3.8.0,
driver 617.42 and one Torch thread. CUDA initialization took 340 ms. Import,
cold compilation and fused-kernel qualification took 821 ms. Independent
component calibration took another 899 ms for 126 vectors of 11 samples.
Each of the four fused numerical checks used 253,958 boundary/edge inputs.
CUDA graph capture counted one kernel for each pass specialization.

One CPU replay took 0.21 ms to fit the component medians and 2.66–4.35 ms
to construct each 65-candidate catalog. Median search time was 4.84 ms for
warm, 4.35 ms for lifecycle and 10.38 ms for paced throughput. Across the
22 joint decisions the range was 1.60–14.61 ms. The JSON scorer reports fresh
fit, catalog and search times. A caller can build once and reuse a choice
while the region, calibration, objective and capacity contract remain valid.
Searching on every call can exceed the service time. Relative to the
matched per-transfer control, the observed warm savings would recover a
dedicated calibration in 478–5,473 requests. Including cold fused
qualification raises that range to 913–10,484, before CPU planning and
common initialization. [Amortization assumptions and per-cell estimates](RESULTS.md#warm-amortization-against-the-current-per-transfer-selector)

## The executable region

Pinned, immutable FP32 groups produce a concatenated FP16 result. The consumer
adds a request-dependent integer once or three times with an FP16 rounding
after each addition. Every plan preserves group order, a fresh result for the
declared caller lifetime, and the previous result in the held-output workloads.
There are four offload paths.

| Plan prefix | H2D representation | CUDA work for an offloaded chunk |
|---|---|---|
| `cpu` | Host-narrowed FP16 | Copy the compact slot into the result and run the additions |
| `gpu` | Raw FP32 | Narrow into a compact slot, copy into the result and run the additions |
| `forward` | Raw FP32 | Narrow directly into the result, then run the additions |
| `fused` | Raw FP32 | One kernel narrows, rounds each addition and writes the destination slice |

In a plan ID, `r` counts preloaded groups, `c` counts chunks per offloaded
group and `d` counts in-flight slots. `reuse` retains the scratch views and
any compact host buffers across requests. `fresh` acquires and retires them
per chunk. Resident groups always use the same device copy and FP16 consumer.
The catalog has 64 offload and partial configurations and one fully resident
configuration per workload. The fixed and restricted controls choose among
these same binaries and resource assignments. The per-transfer control calls
the native calibrated A/B selector using one group's byte count, then runs
its CPU/GPU choice in the shared executor. The separate profile-driven
blocking dispatcher and the unmerged #260 experimental policy have different
eligibility and timing contracts. [CostModel.h](../../libreloc/include/reloc/CostModel.h) ·
[Blocking dispatch](../../docs/completed-placement.md) · [#260 status and measurements](HISTORICAL.md#prediction-limits-and-prior-experimental-selector)

## Prediction, capacity and selection

1. Take the median independent component time at each calibrated size. Linearly
   interpolate within measured sizes for byte-dependent work. Pool fixed
   submission, synchronization, setup and release intercepts across sizes.
   Missing prices exclude the corresponding candidate.
2. Construct a DAG in submission order. Data edges join H2D to the consumer.
   FIFO edges serialize CPU, link and GPU work within each lane. Slot retirement
   prevents overwriting an in-flight source or destination. Compute may overlap
   H2D across slots or during a resident prefix. A simultaneous raw-copy/add
   probe gives a shared slowdown factor. For isolated times `a,b` and
   measured pair time `T`, use `1 + max(0,T-max(a,b))/min(a,b)`. Advance to
   the next completion at the service rates for the **currently active** lanes.
   Applying this one probe to other kernels and depths is a model assumption.
3. Count disjoint persistent resident, held-output and context bytes. Sweep
   temporary allocation-to-last-read intervals. Use the largest of the
   simultaneous live peak and the executor's setup and address-space floors.
   The executor physically allocates and touches the requested CUDA slab. It
   uses aligned sections and checks lease bounds and live overlap at acquisition.
   The same layout contract underlies planning and execution. The raw rows
   record slab bytes, high address, peak live bytes, pinned staging, allocator
   state and device free memory. The chosen budget covers the slab. CUDA
   context, loaded modules, driver memory and pinned host buffers are external.
   The original E4 campaign separately imposed device-wide pressure.
4. Enumerate qualified plans in name order. Reject byte floors above the
   budget. The contention-free critical path is a lower bound because slowdown
   factors are at least one. Skip strictly dominated lower bounds for latency
   objectives. Simulate remaining candidates, check their live peak, then
   rank by objective, peak bytes and plan ID. SLO search checks all its offered
   loads. The default bound is 256 candidates, with at most 1,024 events per
   candidate. Each simulation advances at least one completion per step. An
   oversized or ambiguous catalog, or a catalog with no feasible qualified
   choice, returns no plan. The CLI exits unsuccessfully on refusal. A caller
   can run an existing fallback only under its own legality and budget contract.

Warm prediction minimizes completed makespan. Lifecycle prediction minimizes
one setup/preload/close charge plus five completed makespans. The lifecycle
oracle times a newly created context through five requests and final output
release. Both run after process initialization and kernel preparation. For
SLO, simulate a serialized first-come-first-served executor on each offered
arrival sequence. Among loads whose nearest-rank p95 arrival-to-completion
time is at most 3 ms, maximize completions per second from first arrival to
last completion. The measured oracle uses the same finite traces and ranks
every feasible plan/load pair. A choice that misses its deadline gets an
invalid admission and no numerical regret. These 40-arrival trials establish
finite-trace outcomes. Longer serving runs, multiple concurrent executors
and tail prediction under external traffic are outside their coverage.

## Reproduction

All commands run from the repository root. CPU scoring and selection require
only Python 3.10 or later. The test suite uses pytest.

```sh
python -m pytest -q bench/issue271
python -m bench.issue271.analyze bench/issue271/measurements.json.gz
python -m bench.issue271.analyze bench/issue271/measurements.json.gz --json > /tmp/issue271-scores.json
python -m bench.issue271.historical --check
python -m bench.issue271.select_plan bench/issue271/measurements.json.gz \
  --workload held_final --budget-bytes 17825792 --objective warm
```

To replay a single decision, use a CUDA build of `pyreloc` on `PYTHONPATH`
and the measured CUDA-enabled PyTorch and Triton environment. The bundled
native per-transfer calibration targets the Ryzen 7 7800X3D and RTX 4070 Ti
SUPER. Generate a fresh component profile when the runtime or hardware
configuration changes. The offline command makes predictions for the host
represented by its input file.

```sh
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
python -m bench.issue271.measure --calibration-only --output /tmp/issue271-profile.json.gz
plan=$(python -m bench.issue271.select_plan /tmp/issue271-profile.json.gz \
  --workload held_final --budget-bytes 17825792 --objective warm --plan-only) &&
  python -m bench.issue271.measure --workload held_final --budget-bytes 17825792 \
    --plan-json "$plan" --output /tmp/issue271-execution.json.gz

python -m bench.issue271.measure --output /tmp/issue271-exhaustive.json.gz
python -m bench.issue271.analyze /tmp/issue271-exhaustive.json.gz
```

The archived gzip SHA-256 is
`7187feecd8defcfc7e58cf38a90dbc7f474bf3a9666b32f2907c3ca57493584c`.
The campaign file contains the environment, calibration and native A/B decisions,
candidate order, completed samples, capacity observations, output checks and
timestamped arrival/start/finish traces. Each measured cell is one context
trial for the warm and arrival measurements and a separate five-call lifecycle.
Twenty successive warm observations within that context do not estimate
between-deployment variability. [Fresh comparisons and prediction errors](RESULTS.md) ·
[Archived E2, E4 and E5 decisions](HISTORICAL.md)

## Relation to prior work

MAGIS coordinates DNN graph transformations and scheduling to reduce device
memory. TEMPI selects packing and communication paths for noncontiguous GPU
data in MPI. Sym's narrower question is whether calibrated costs for a legal
cross-boundary representation can choose the transform, consumer destination,
residency and scratch lifetime together under a completed-request objective.
The oracle evaluates the benefit and cost of those extra choices within one
finite executor. A performance or novelty claim relative to either system
would require a shared workload and compatible implementations. [MAGIS, ASPLOS 2024](https://github.com/pku-liang/MAGIS) ·
[TEMPI, HPDC 2021](https://carlpearson.net/pdf/20210624_pearson_hpdc.pdf)
