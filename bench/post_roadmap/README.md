# Post-roadmap workload remeasurement

**The performance goal remains unmet.** After merging #218–#230, none of the
three measured Sym configurations is at least 5% faster than **both** matched
Torch controls in all three rounds of any regime. GPT prefill comes closest;
the other regimes still have substantial gaps to the fastest Torch control.

## Completed model latency

Milliseconds, lower is better; each number is the median of three independent
round medians (30 samples per round). “Sym FX” uses eager compute; the other two
Sym columns use Inductor compute. Placement is an explicit opt-in configuration,
with the coverage limitations described below. All controls receive equivalent
prepacking and prefetch opportunities.

| Regime | Eager Torch | Inductor | Sym FX | Sym + Inductor | Sym + Inductor + placement |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM, batch 512 | 3.606 | 3.738 | 4.830 | 4.656 | 4.497 |
| DLRM, batch 2048 | 7.922 | 8.021 | 9.600 | 9.491 | 9.509 |
| GraphSAGE, 256 seeds | 1.270 | 1.307 | 2.447 | 2.230 | 2.182 |
| GraphSAGE, 1024 seeds | 4.098 | 2.993 | 5.500 | 5.126 | 5.223 |
| GPT, four decode steps | 26.321 | 30.179 | 46.927 | 43.008 | 43.158 |
| GPT, 256-token prefill | 7.001 | 6.430 | 7.671 | 6.540 | 6.454 |
| MoE, one token | 3.260 | 4.414 | 4.886 | 4.618 | 4.551 |
| MoE, 512 tokens | 10.789 | 16.589 | 18.345 | 17.318 | 17.203 |

For the regular Sym + Inductor path, the latency change relative to each control
is below. Positive numbers mean **slower**; negative numbers mean faster.

| Regime | vs eager Torch | vs Inductor | vs Sym FX |
| --- | ---: | ---: | ---: |
| DLRM 512 | +29.1% | +24.6% | −3.6% |
| DLRM 2048 | +19.8% | +18.3% | −1.1% |
| GraphSAGE 256 | +75.6% | +70.6% | −8.9% |
| GraphSAGE 1024 | +25.1% | +71.2% | −6.8% |
| GPT decode | +63.4% | +42.5% | −8.4% |
| GPT prefill | −6.6% | +1.7% | −14.7% |
| MoE sparse | +41.7% | +4.6% | −5.5% |
| MoE dense | +60.5% | +4.4% | −5.6% |

Inductor composition helps Sym in every aggregate median, especially prefill.
It does not close the gap to the fastest control. Prefill's regular Sym +
Inductor gain over eager Torch varies from 1.4% to 7.1% across rounds; its
placement arm is within −0.6% to +0.8% of Inductor. Placement makes no decisions
in prefill or MoE, so differences between those two Sym arms cannot be credited
to placement optimization.

Tail latency: median of the three **per-round p95s**, also in ms. This is not a
pooled p95; per-round medians, tails and every sample remain in the evidence.

| Regime | Eager Torch | Inductor | Sym FX | Sym + Inductor | + placement |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM 512 | 4.850 | 4.845 | 5.568 | 5.766 | 5.449 |
| DLRM 2048 | 8.203 | 8.278 | 9.866 | 10.168 | 9.937 |
| GraphSAGE 256 | 1.642 | 1.673 | 2.895 | 2.757 | 2.741 |
| GraphSAGE 1024 | 4.495 | 3.299 | 6.110 | 5.844 | 5.921 |
| GPT decode | 26.385 | 31.455 | 49.350 | 43.586 | 43.403 |
| GPT prefill | 7.048 | 6.567 | 7.851 | 6.657 | 6.595 |
| MoE sparse | 3.294 | 4.474 | 4.938 | 4.722 | 4.696 |
| MoE dense | 10.833 | 16.782 | 18.466 | 17.448 | 17.304 |

## Protocol (declared before measurement)

Rerun merged main `446bc9f95ea1a31809c73384639c74c06ddf9227` after #218–#230.
Reuse the eight model-inclusive regimes and matched controls from
[the #225 qualification](../issue225/README.md), without changing model math,
wire precision, shapes, prepacking or two-slot weight prefetch. The five paths
are eager Torch, Torch Inductor, Sym with eager FX compute, Sym with Inductor
compute, and Sym with Inductor plus opt-in calibrated placement.

- Three fresh workload processes; empty Inductor cache per process; five warmup
  requests and 30 completed samples per path/regime. Rotate path order across
  rounds and samples. Report median of round medians and median of round p95s;
  retain every observation and counter, including unfavorable results.
- The previous project-level criterion remains at least 5% lower model latency
  than **both** Torch controls in every round of a regime. Check each Sym path
  separately; do not pick a different path for each individual timing sample.
- Pin execution to GPU 0's local CPU affinity `4-7,20-23`, eight Torch/OpenMP
  threads, one interop thread, passive OpenMP waits, TF32 and CUDA graphs off.
  Release native build with NVTX off. Monitor foreign GPU/build work every
  three seconds; no builds, tests or profiling during accepted timing.
- Before each round, explicitly calibrate placement using #227's training
  shapes in a separate process against the current binaries and hardware.
  Save its full profile, held-out checks and wall time. No online calibration
  is hidden inside inference. The existing calibration covers matrix and KV
  transpose/cast families; unsupported recipes and missing family coverage
  retain their documented fallbacks. Report placement decisions.
- Fresh setup, immutable checkpoint snapshots and first calls stay separate
  from warm inference. First calls share process caches and **are not** an
  independent cold-start comparison. The #230 artifact disk cache stays off
  by default; its explicitly prepared serving API is qualified separately in
  [#230](../issue230/README.md). Warming the model requests here includes all
  shape-specific graph compilation and resource initialization.
- DLRM includes CPU embedding lookup; GraphSAGE includes CPU feature gather.
  Graph sampling, local-index upload, dense-input upload, checkpoint generation
  and model construction are common setup. GPT includes token reads and KV
  eviction/restore; MoE includes routing and dynamic expert discovery. Timing
  ends after CUDA completion. Separate weight/feature and KV transfer timings
  are independent diagnostics, not subtractable phases of model latency.
- Check every timed model output against eager Torch (`rtol=2e-4, atol=2e-5`),
  with exact tokens/expert selections and original-model offload oracles.
  Require zero warm compilation callbacks and implicit graph breaks.
- This is a single-GPU matched workload comparison, not a multi-GPU throughput
  study or proof that every optional API improves every workload. Historical
  #225 results are context, not a simultaneous baseline/candidate experiment.

Reproduce after building the current source with CUDA and Python enabled:

```sh
/tmp/sym-210-venv/bin/python bench/post_roadmap/run.py \
  --build /tmp/sym-post-roadmap-build --output /tmp/sym-post-roadmap-results
```

The output directory holds JSON observations, placement profiles, process logs
and compiler caches. Only compact measurement evidence belongs in the repo;
logs, caches and profiler databases remain outside it.

## Environment and validation

Measured on 2026-10-12: AMD EPYC 7351, RTX 2080 Ti GPU 0 (PCIe 3 x16,
NUMA node 1), driver 595.71.05, CPython 3.14.7, Torch 2.14.0+cu126.
Sym uses CUDA toolkit 12.5 and a fresh Release build of the revision above,
with explicit CUDA `-O3 -DNDEBUG`, native CUDA targets 75/89, and NVTX off.
GPU clocks were not locked. The eight allowed CPU threads are four physical
cores with SMT; this is not an eight-physical-core measurement.

Before timing, the fresh build passed 1,235 Python tests (four skips),
58 workload tests, 12 CTests and 41 compiler lit tests. The Python suite emitted
one existing fork-from-multithreaded-process deprecation warning. No source
changes to the compiler, runtime, frontend or model definitions accompany this
report.

All six monitored processes finished without observed foreign GPU/build work.
Each workload process took about 121 seconds; each offline calibration process
took about 50 seconds. All 3,600 timed model outputs passed, with maximum
absolute difference `2.5034e-6` and exact tokens/expert selections. There were
zero warm compilation callbacks, zero implicit graph breaks and zero backend
runtime fallbacks. Each Inductor path generated kernels (2–49 on the first
model invocation); planned placement choices/fallbacks are accounted separately.

## Preparation and first requests

The table gives the range of first-request times across rounds, in **seconds**,
after CUDA initialization and common model/input setup. They include graph
compilation and lazy execution resources. Paths share a process/compiler cache,
so these ranges **do not establish a cold-start advantage** for either backend.

| Regime | Inductor first request | Sym + Inductor first request |
| --- | ---: | ---: |
| DLRM 512 | 6.650–9.371 | 0.744–3.510 |
| DLRM 2048 | 1.590–2.774 | 0.743–1.956 |
| GraphSAGE 256 | 1.748–2.857 | 0.908–2.230 |
| GraphSAGE 1024 | 1.719–2.766 | 0.935–1.967 |
| GPT decode | 5.872–13.862 | 2.035–10.230 |
| GPT prefill | 0.514–2.922 | 0.411–2.790 |
| MoE sparse | 0.785–2.363 | 0.274–1.836 |
| MoE dense | 2.822–21.713 | 3.249–22.053 |

Immutable weight snapshot preparation precedes these first requests. Across
path/case medians, it costs 61.7–152.7 ms; Sym artifact/owner setup adds about
4.7–6.0 ms. These costs, the offline placement calibration, and common model
construction are not amortized into the warm inference numbers. Full first-call
times, compile counts and preparation costs for all five paths are retained.

Sym retains 48.281 MiB of prepared GPT weights (48 MiB pinned) and 64.375 MiB
for MoE (64 MiB pinned). Weight queues peak at two in-flight groups, 96 MiB
of GPT outputs / 32 MiB of MoE outputs; native scratch peaks at 13.820 / 4.523
MiB respectively. Resource counters show the bounded queues draining between
requests. These counters are not whole-process memory measurements.

## Transfer diagnostics and remaining gaps

These are independent completed invocations, in ms. They cannot be subtracted
from model latency to estimate compute time or overlapping phases.

| Transfer scope | Eager Torch | Inductor | Sym FX | Sym + Inductor | + placement |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM 512: layout/cast/H2D | 0.254 | 0.317 | 0.843 | 0.891 | 0.979 |
| DLRM 2048: layout/cast/H2D | 1.483 | 1.615 | 2.676 | 2.745 | 2.935 |
| GraphSAGE 256: gather/cast/H2D | 0.736 | 0.594 | 1.657 | 1.658 | 1.697 |
| GraphSAGE 1024: gather/cast/H2D | 3.758 | 2.710 | 4.706 | 4.774 | 4.738 |
| GPT decode: request weight groups | 22.340 | 19.175 | 20.270 | 20.227 | 20.333 |
| GPT decode: 48 KV transfers | 2.216 | 7.779 | 18.239 | 19.138 | 18.514 |
| GPT prefill: weight groups | 5.599 | 4.800 | 5.103 | 5.123 | 5.105 |
| MoE sparse: selected weight groups | 2.098 | 1.834 | 2.232 | 2.223 | 2.217 |
| MoE dense: selected weight groups | 8.146 | 7.156 | 7.353 | 7.348 | 7.326 |

- **Feature transfers remain expensive.** Sym + Inductor's isolated small
  DLRM transfer is 3.5× eager Torch's; small GraphSAGE is 2.8× Inductor's.
  The live transform/transfer path still needs improvement even after warming.
- **Decode's KV path remains a major gap.** Its 48 transfers take 19.138 ms
  with regular Sym + Inductor versus 2.216 ms eager and 7.779 ms Inductor.
  Placement reduces that diagnostic to 18.514 ms, without improving the
  aggregate model median. This points to work beyond merely choosing the
  transform location; it is not a new attribution of individual CPU phases.
- **Placement coverage is limited in these workloads.** DLRM selects
  `torch_cpu:profile_missing_native_coverage` because the existing profile lacks
  its rank-three cast family. GraphSAGE uses `legacy:unsupported_recipe`.
  GPT decode chooses profiled `torch_cpu` in rounds 1/3 and a mixture of
  `torch_cpu`/`torch_gpu` in round 2. Prefill/MoE do not enter the selector.
  This is a qualification of the existing #227 profile procedure, not an
  exhaustive search for the best workload-specific policy.
- **Startup improvements do not resolve the measured warm gap.** No compiler
  callbacks occurred during sampling. Prepared weight layouts and two-slot
  prefetch are active for every offload path, including both Torch controls.
  The raw-weight kernel optimized in #229 is a different path from these
  prepared wire weights; #228's multi-GPU capability and #230's explicit
  precompile/cache API have their own qualifications. Their completion does
  not imply additional speedups in this single-GPU, warmed integration.

## Historical comparison with #225

The earlier high-priority checkpoint used the same workload definitions and
machine. Below is regular Sym + Inductor latency; this is a **historical**
comparison, not a paired experiment isolating the later commits.

| Regime | #225, Oct 9 (ms) | Current, Oct 12 (ms) | Change |
| --- | ---: | ---: | ---: |
| DLRM 512 | 4.410 | 4.656 | +5.6% |
| DLRM 2048 | 8.779 | 9.491 | +8.1% |
| GraphSAGE 256 | 2.248 | 2.230 | −0.8% |
| GraphSAGE 1024 | 4.028 | 5.126 | +27.3% |
| GPT decode | 43.681 | 43.008 | −1.5% |
| GPT prefill | 6.601 | 6.540 | −0.9% |
| MoE sparse | 4.559 | 4.618 | +1.3% |
| MoE dense | 17.214 | 17.318 | +0.6% |

There is no broad model-level gain over that checkpoint. Large GraphSAGE is the
largest observed regression; its near-parity with eager Torch in #225 does not
reproduce here. A paired revision comparison would be needed to assign this
change to a specific implementation item.

## Evidence and reproduction

[summary.csv](evidence/summary.csv) contains aggregates, round-median ranges,
first-request ranges, errors and per-round ratios against both controls.
[manifest.json](evidence/manifest.json) records the measured main revision,
harness/dependency hashes, archive checksums and the empty qualifying-win list.
The nine compressed JSON files retain all three workload rounds, all three
calibration rounds and their exact profiles: 7,650 sampled model/transfer/KV
timings plus calibration observations, correctness, compiler and resource
counters, actual loaded-binary hashes and hardware metadata. Total evidence is
about 360 KiB. No process logs, compiler caches or profiler databases are added.

```sh
python3 bench/post_roadmap/summarize.py \
  /tmp/sym-post-roadmap-results bench/post_roadmap/evidence
python3 -c 'import gzip,json; r=json.load(gzip.open("bench/post_roadmap/evidence/workloads-r1.json.gz")); print(r["cases"]["llm_decode"]["paths"]["sym_inductor"]["samples_ms"])'
```
