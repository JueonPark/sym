# Sym + Inductor: end-to-end qualification (#225)

## Protocol declared before acceptance measurements

Four paths: native Torch; Inductor; current Sym with eager FX compute; Sym with
Inductor compute. All start from identical data, precision, layouts and output
ownership. This is a new model-inclusive comparison, not a rerun of #217's
transfer-only totals. The older transfer totals cannot be divided by these
end-to-end latencies to claim a speedup.

| Regime | Timed work | Target / rationale |
| --- | --- | --- |
| DLRM, 512 / 2048 samples | 26 CPU EmbeddingBag lookups + stack, FP32→FP16 layout/H2D, bottom/top MLP and dot interaction | Large case: improve over eager-compute Sym by ≥10%; small case exposes fixed overhead |
| GraphSAGE, 256 / 1024 seeds | CPU feature gather + FP16 H2D, two GPU SAGE-mean layers | Large case: improve over eager-compute Sym by ≥10%; compare against fused Inductor host gather |
| GPT, 4 decode steps / 256-token prefill | 4 layers, INT8 weight prefetch, attention/MLP, logits, greedy token selection; decode includes growing KV eviction/restore | Prefill target ≥10% improvement over eager-compute Sym; decode may remain dispatch-bound |
| MoE, 1 / 512 tokens | 2 blocks, 8 experts, top-2 routing, selected weight prefetch, expert FFNs and combine | Dense target ≥10% improvement over eager-compute Sym; sparse case exposes queue/dispatch overhead |

The project-level target is ≥5% lower median completed latency than **both**
matched Torch controls in at least one useful regime, consistently in all three
rounds. Report misses, regressions and uncertainty; feature completion alone does
not establish this performance target. No sum of individual optimization gains.

DLRM and GraphSAGE model mathematics come from PR #162 at `cdc44e2`; their source
has not merged, so `models.py` contains the relevant benchmark definitions. GPT
and MoE use main's existing model weights and independently check the factored
compute against the original model forward. Model arithmetic is FP32, TF32 off.
Transfer results are byte-exact in the frontend tests; Inductor model results use
`rtol=2e-4, atol=2e-5`, with exact generated tokens / selected experts.

### Boundaries and fairness

- DLRM/GraphSAGE Sym paths capture transfer and GPU model compute in one graph.
  Torch controls get retained pinned FP16 staging and, for Inductor, a compiled
  host gather/layout/cast into that staging plus compiled GPU model compute.
  This explicit host boundary ensures the same FP16 wire representation.
  All paths include the CPU feature lookup; DLRM embedding-bag lookup/stack is
  common eager code. Graph sampling and local-index uploads, plus DLRM dense
  input upload, are common setup outside this inference unit.
- GPT/MoE all use an explicit immutable INT8 `[out,in]` snapshot with FP32 scales,
  and the same two-slot lookahead schedule: 128 MiB output and 64 MiB scratch
  budgets. Torch gets identical offline prepacking and GPU dequantization;
  Inductor compiles conversion **and** model tensor regions. Sym uses #224's
  prepared snapshots, #223's prefetch queue and #220's grouped submission.
  Snapshot and artifact/owner preparation costs are separate from warm reuse.
- Async queue iteration, MoE's dynamic active-expert discovery (`unique`, host
  list, `nonzero`), greedy token scalar reads and Python scheduling stay outside
  compute graphs and inside end-to-end timing. All compute regions use
  `fullgraph=True`; these deliberate boundaries are not implicit graph breaks.
  This does not claim a single captured graph for an entire offloaded model.
- Completed wall time includes allocations, transfer preparation/submission,
  prefetch startup/drain and consumer completion. Transfer diagnostics are
  separate serialized invocations, not subtractable phases or proof of overlap.
  GPT/MoE diagnostics cover all request weight groups but exclude KV/routing.
- First calls include Dynamo/AOT/Inductor and lazy runtime setup, with backend
  compile time and generated-kernel counts recorded separately. An empty cache
  per process does not make every later path's first call independently cold;
  rotated path order and compile counts expose shared-cache effects.
- Three independent processes, 5 warmups, 30 observations/path/case, path order
  rotated both between rounds and samples. Preserve every accepted raw sample
  in compact arrays. Record p50/p95, recompiles, graph breaks, resource counters,
  hardware, source and binary hashes. No tests, builds or unrelated GPU jobs
  during headline timing. Collect any profiler trace in a separate invocation.
- Shapes within each regime are known (including the four decode steps), so
  acceptance uses `dynamic=False` for all compiled paths, warming every shape
  before sampling. `--dynamic` separately exercises forced symbolic codegen;
  dynamic correctness/rebinding is also covered by the integration tests.

```sh
export PYTHONPATH=/tmp/sym-225-build/python
export SYM_RELOC_EXPORT=/tmp/sym-225-build/sym/tools/sym-reloc-export
export SYM_OPT=/tmp/sym-225-build/sym/tools/sym-opt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/sym225-round1 taskset -c 4-7,20-23 \
  /tmp/sym-210-venv/bin/python bench/issue225/compare.py --output /tmp/sym225-round1.json
```

## Results (2026-10-09)

Source and staged frontend: `fc3bd21097ca3edbeed9b7a557b7d8b43ca44c99`, based on
merged #224 (`9046522`). Release CUDA build, EPYC 7351, RTX 2080 Ti GPU 0,
driver 595.71.05, CUDA toolkit 12.5, Torch 2.14.0+cu126, CPython 3.14.7.
Affinity 4–7,20–23 (4 physical cores / 8 SMT threads), Torch/OpenMP threads 8,
interop 1, passive OpenMP waits, TF32 and CUDA graphs disabled. GPU clocks were
not locked. Process-tree-aware monitoring every 3 seconds observed **no foreign
build, packaging or GPU workloads** in any round. Each round took 109–112 seconds.

These are medians of the three independent round medians, in **milliseconds**;
lower is better. “Sym FX” is the existing eager-compute backend with current
transfer optimizations. “Sym + Inductor” is the new composition mode.

| Regime | Native Torch | Inductor | Sym FX | Sym + Inductor | Change vs Sym FX |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM, batch 512 | 3.633 | 3.837 | 4.709 | 4.410 | −6.4% |
| DLRM, batch 2048 | 7.855 | 7.965 | 9.101 | 8.779 | −3.5% |
| GraphSAGE, 256 seeds / 9,085 rows | 1.194 | 1.278 | 2.396 | 2.248 | −6.2% |
| GraphSAGE, 1024 seeds / 35,180 rows | 4.039 | 2.808 | 4.368 | 4.028 | −7.8% |
| GPT, four decode steps | 26.303 | 30.615 | 47.123 | 43.681 | −7.3% |
| GPT, 256-token prefill | 6.971 | 6.445 | 7.677 | 6.601 | −14.0% |
| MoE, one token | 3.206 | 4.434 | 4.776 | 4.559 | −4.6% |
| MoE, 512 tokens | 10.686 | 16.515 | 18.203 | 17.214 | −5.4% |

All eight aggregate medians improve over Sym FX, but this is **not a consistent
win in every round**: the small GraphSAGE case changes by −12.6%, −12.5%, then
**+4.6%**. Large GraphSAGE is essentially tied with native Torch in aggregate
(−0.3%) and loses to it in round 3 (+2.3%). GPT prefill beats native Torch in
each round (−6.6%, −0.8%, −5.6%) but trails Inductor (+0.4%, +7.2%, +2.8%).
Sym + Inductor trails Inductor in all eight aggregate results.

The ≥5% project-level win over **both** controls is **not achieved** in any
regime. Of the four ≥10% large-regime composition targets, only GPT prefill
passes in aggregate (14.0%; one round improves only 8.7%). DLRM 2048,
GraphSAGE 1024 and dense MoE miss their 10% targets. Completing the high-priority
implementation sequence therefore does not complete the performance goal.

Tail latency is also retained, not inferred from the median. Below are medians
of the three **per-round p95s**, in ms; the raw per-round values and all 6,120
timing observations are in [timings.csv](evidence/timings.csv).

| Regime | Native Torch p95 | Inductor p95 | Sym FX p95 | Sym + Inductor p95 |
| --- | ---: | ---: | ---: | ---: |
| DLRM 512 | 4.748 | 4.803 | 5.395 | 5.302 |
| DLRM 2048 | 8.131 | 8.619 | 10.240 | 9.431 |
| GraphSAGE 256 | 1.656 | 1.678 | 2.910 | 2.685 |
| GraphSAGE 1024 | 4.540 | 3.287 | 4.878 | 4.532 |
| GPT decode | 26.604 | 31.373 | 47.573 | 43.888 |
| GPT prefill | 7.148 | 6.646 | 8.113 | 7.066 |
| MoE sparse | 3.250 | 4.478 | 4.885 | 4.670 |
| MoE dense | 10.736 | 16.639 | 18.429 | 17.369 |

### Compilation and correctness

All measured Inductor paths generated kernels (2–49 per first model invocation).
Every compiled region used `fullgraph=True`; **zero implicit graph breaks, zero
warm compilation callbacks, and zero Sym runtime fallbacks** were observed.
Importer exclusions for unrecognized model math are not eager fallback: these
nodes reach Inductor, as the codegen tests and counters verify. Shape variants
compile during the first request; they are included below and then warmed.
The maximum model-output absolute difference was `2.504e-6`, below the declared
tolerance. Tokens and active-expert lists matched exactly.

| Regime | Inductor first call, seconds (range) | Sym + Inductor first call, seconds (range) | Initial graph callbacks: Inductor / Sym + Inductor |
| --- | ---: | ---: | ---: |
| DLRM 512 | 6.63–9.36 | 0.73–3.53 | 2 / 1 |
| DLRM 2048 | 1.58–2.77 | 0.73–1.96 | 2 / 1 |
| GraphSAGE 256 | 1.80–2.84 | 0.92–2.04 | 2 / 1 |
| GraphSAGE 1024 | 1.76–2.95 | 0.92–1.99 | 2 / 1 |
| GPT decode | 6.32–13.52 | 2.07–10.14 | 20 / 16 |
| GPT prefill | 0.68–2.90 | 0.41–2.86 | 7 / 3 |
| MoE sparse | 0.90–2.02 | 0.27–1.58 | 5 / 3 |
| MoE dense | 3.54–22.03 | 2.88–22.92 | 33 / 31 |

First-call order explains substantial shared-cache effects; these ranges do
**not** establish a cold-compile advantage. Native Torch first calls are
1.5–65.1 ms and Sym FX first calls 76.8–1747.3 ms. Raw first-call times and
backend-only compile times are in [diagnostics.jsonl](evidence/diagnostics.jsonl).
Model construction, checkpoint generation and common input setup precede these
first calls. Immutable snapshot preparation is separate: across the offload
cases it takes about 59–147 ms for the Torch controls and 62–153 ms for Sym
(medians of each path/case's three rounds). Owner/artifact setup is recorded too.
These costs are not amortized into warm timings or claimed free.

### Remaining costs

Completed transfer-only diagnostics below are independent invocations, in ms.
They **cannot be subtracted from end-to-end latency** to derive compute time:
the full model has different scheduling and overlap opportunities.

| Transfer scope | Native Torch | Inductor | Sym FX | Sym + Inductor |
| --- | ---: | ---: | ---: | ---: |
| DLRM 512 layout/cast/H2D | 0.264 | 0.323 | 0.794 | 0.807 |
| DLRM 2048 layout/cast/H2D | 1.615 | 1.584 | 2.149 | 1.968 |
| GraphSAGE 256 gather/cast/H2D | 0.732 | 0.592 | 1.545 | 1.571 |
| GraphSAGE 1024 gather/cast/H2D | 3.747 | 2.697 | 3.425 | 3.465 |
| GPT decode: request's weight groups | 22.306 | 19.196 | 20.302 | 20.195 |
| GPT decode: 48 KV transfers | 2.222 | 7.845 | 18.410 | 19.338 |
| GPT prefill: weight groups | 5.609 | 4.806 | 5.107 | 5.079 |
| MoE sparse: selected weight groups | 2.105 | 1.843 | 2.197 | 2.201 |
| MoE dense: selected weight groups | 8.189 | 7.140 | 7.264 | 7.225 |

1. **Small transfer dispatch remains expensive.** The decode KV diagnostic
   reproduces the model's shapes and transfer count, with eviction followed by
   restoration per step but no consumer compute. Sym takes roughly 8.7× native
   Torch here. Inductor wrapping of these small copy-only functions also costs
   more than eager Torch. This supports prioritizing size-sensitive native
   bypass / completed-cost selection in [#227](https://github.com/JueonPark/sym/issues/227).
2. **CPU gather/cast and its dispatch still trail the best Torch controls.**
   Inductor's host gather is faster in both GraphSAGE cases. Specialized CPU
   kernels in [#226](https://github.com/JueonPark/sym/issues/226) remain relevant;
   this measurement alone does not separate kernel time from frontend overhead.
3. **Compiling many small offload compute regions is not automatically faster.**
   Dense MoE's Inductor-only end-to-end path is 54.5% slower than eager Torch,
   despite faster isolated weight delivery. The fixed-shape benchmark produces
   33 initial graph variants for Inductor (31 for Sym + Inductor). This motivates
   investigating call granularity, shape specialization and queue/compute
   scheduling; attributing the loss to a particular kernel needs further profiling.

The prepared Sym snapshots retain 48.281 MiB (48 MiB pinned) for GPT and
64.375 MiB (64 MiB pinned) for MoE. Queue output peaks are 96 / 32 MiB;
native scratch peaks are 13.820 / 4.523 MiB. Both stay within the declared caps.
Scalar ownership/dispatch counters, exact binary/frontend hashes and per-round
configuration are retained in the evidence files. No new overlap claim is made
from wall-clock timings; #224's existing wire trace qualifies the unchanged
compact-upload native implementation. No verbose compiler logs or profiler
databases are checked in.

This comparison includes native Sym KV relocation for GPT decode, unlike
#224's weight-preparation benchmark, which deliberately used common Torch KV
copies. Thus the new decode total is not a regression attributable to #225
versus that older timing: the correctly matched control here is **Sym FX**.
Likewise, the PR #162 transfer-only totals use different units and cannot be
compared directly to these model-inclusive requests.

### Validation

- CUDA runtime/frontend suite: **1191 passed, 4 skipped**.
- Existing workload suite: **58 passed**.
- New composition tests on the CPU-only native extension: **4 passed**.
- CTest: **6/6**; compiler lit tests: **41/41**.
- All three benchmark rounds: correctness checks passed, no warm recompiles,
  no implicit graph breaks, no runtime fallback, no observed competing workloads.
