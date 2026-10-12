# Remaining critical paths: pinning and targeted GPU tuning (#229)

## Outcome

Keep the dense 2D tiled dequantize/transpose: it reduces completed 4 MiB
raw-weight loads by **5.9–7.0%**, meeting the predeclared 5% target. It does
not close the Torch gap. Raw-weight GPT decode improves only 1.5%; sparse MoE's
1.6% median difference has overlapping round ranges. Already-prepacked models
do not execute this transpose and show no established end-to-end gain.

Keep pinning defaults unchanged. Warm retained pinning helps some large
transfers, while cold allocation, growth and retirement can reverse the result.
The explicit 8 MiB gate is not portable between execution families: typed
rings assess chunk bytes, whereas layout rings assess total wire bytes.
**The adaptive predictor is not implemented; #208 remains separate and open.**

Measured base `8342acc` versus candidate source `267115d`, on the machine and
protocol below. Later commits package evidence and clarify documentation.
Values below are milliseconds: **median of three round p50s [minimum–maximum
round p50]** unless labeled otherwise. Each warm round has 30 observations;
cold/lifecycle rounds have three. Ranges are observed process variation, not
confidence intervals. All 18 independent processes passed correctness and
isolation checks; model warm phases had zero compilation callbacks.

## Raw-weight kernel and completed calls

INT8 `[in,out]` plus FP32 scales still travels over PCIe; results remain
contiguous FP32 `[out,in]`. Both implementations perform the same FP32 multiply.
The original generic fused kernel remains the fallback and isolated control.
Only dense 2D transposes within CUDA's grid limits use the shared tile.

The completed-call table uses unconfigured auto and pageable source tensors.
Torch and Inductor perform the same upload, conversion and transpose with
fresh output allocation and final completion. These are transfer-only calls.

| Input shape | Sym before | Sym after | Reduction | Torch | Inductor |
|---|---:|---:|---:|---:|---:|
| 31×33 | 0.280 [0.279–0.282] | 0.280 [0.280–0.282] | −0.2% | 0.124 [0.123–0.124] | 0.171 [0.170–0.174] |
| 256×1024 | 0.364 [0.358–0.371] | 0.356 [0.354–0.356] | 2.0% | 0.184 [0.181–0.184] | 0.314 [0.310–0.324] |
| 1024² | 0.652 [0.650–0.656] | 0.627 [0.625–0.638] | 3.9% | 0.301 [0.298–0.307] | 0.439 [0.438–0.450] |
| 1024×4096 | 1.180 [1.178–1.192] | 1.110 [1.109–1.117] | 5.9% | 0.796 [0.796–0.797] | 0.834 [0.833–0.982] |
| 4096×1024 | 1.178 [1.172–1.179] | 1.095 [1.094–1.099] | 7.0% | 0.799 [0.794–0.804] | 0.843 [0.832–0.986] |
| 2049×1025 | 0.875 [0.873–0.878] | 0.839 [0.837–0.842] | 4.1% | 0.482 [0.480–0.485] | 0.603 [0.599–0.654] |

The 4 MiB cases' median round p95s improve from 1.290→1.208 and
1.279→1.215 ms. The tiny case is neutral; no material odd-shape regression
appears. Inductor's independent-process spread is preserved rather than
selecting its best or worst run.

With preallocated GPU buffers, alternating event-bracketed generic/tiled
measurements are 0.151 [0.150–0.151] / 0.076 [0.074–0.076] ms at 1024×4096,
and 0.149 [0.149–0.150] / 0.074 [0.073–0.074] ms at 4096×1024. Events include
launch gaps. Actual kernel activity from the separate trace is lower and is
reported next; event timing is not substituted for completed-call timing.

## Critical-path evidence

Separate NVTX/CUPTI traces contain five repeated warm calls per policy/shape
and five model calls per path. They repeat fixed inputs within each path;
headline samples rotate paths and mutate inputs. Instrumentation and cache
locality differ, so these are attribution evidence, not headline latencies.
Each cell is the trace median **before → after**.

| Transfer / policy | Host preparation¹ | CPU transform union² | Actual DMA | Actual Sym kernel | Native completion wait³ | Completed call |
|---|---:|---:|---:|---:|---:|---:|
| DLRM 2048 / pinned | 0.445→0.437 | 1.604→1.492 | 0.533→0.533 | 0→0 | 0.052→0.051 | 2.495→2.361 |
| GNN 35180 rows / pinned | 1.158→1.161 | 2.559→2.864 | 0.704→0.704 | 0→0 | 0.069→0.070 | 4.315→4.666 |
| KV restore 33 / pinned | 0.232→0.210 | 0.017→0.015 | 0.012→0.012 | 0→0 | 0.009→0.009 | 0.354→0.317 |
| KV evict 33 / pinned | 0.216→0.260 | — | 0.011→0.011 | 0→0 | 0.009→0.008 | 0.341→0.394 |
| Weight 1024×4096 / pageable | 0.347→0.344 | 0→0 | 0.329→0.329 | 0.121→0.052 | 0.199→0.129 | 1.003→0.926 |
| Weight 4096×1024 / pageable | 0.350→0.330 | 0→0 | 0.328→0.328 | 0.119→0.045 | 0.189→0.120 | 0.999→0.901 |

¹ Exclusive host time outside native execution and final explicit synchronization:
binding, validation, descriptors, output allocation and frontend preparation.
² Union of actual instrumented worker intervals, without summing parallel workers;
D2H scatter lacks a worker range and is **not measured as zero CPU work**.
³ Synchronization API time inside native execution, not an additional GPU phase.
**Do not add these columns:** CPU work, DMA, kernels and waits overlap; native
allocation/submission and other host intervals also remain in the call.
Full exclusive host partitions, CUDA API subsets and correlated intervals are
in the trace artifacts. For example, baseline DLRM/GNN CPU-transform/DMA overlap
is 0.373/0.503 ms; their transform and DMA durations are not additive.

The four model traces support the same prioritization. DLRM/GNN have no Sym GPU
transform kernel; their model-call CPU transform unions are 1.749/2.775 ms
versus 0.533/0.721 ms of DMA before this change. Prepared GPT/MoE Sym kernel
activity is effectively unchanged: 2.312→2.311 / 0.223→0.225 ms, with DMA
15.901→15.903 / 1.338→1.336 ms. GPT also makes 48 layout calls, 160 rechecks
and 416 storage snapshots per instrumented decode. Those request paths and
CPU transformations deserve attention before additional generic GPU tuning.
Host wrapper measurements include profiler overhead and do not promise the
same saving if a check is removed.

## Pinning: warm reuse versus allocation and retirement

All four modes get the same transform, payload, worker/chunk schedule and
resource limits. `auto` has no threshold; `gate8` explicitly configures 8 MiB.
The full sweep, including all sizes and per-request decisions, is preserved.

| Transfer / wire size | Pinned warm | Pageable warm | Auto warm | Gate8 warm |
|---|---:|---:|---:|---:|
| DLRM 4096 / 13 MiB | 4.170 [4.048–4.199] | 5.115 [5.100–5.181] | 5.256 [5.228–5.291] | 5.185 [5.169–5.343] |
| GNN 65539 / 16.001 MiB | 7.268 [7.198–7.388] | 7.995 [7.967–8.121] | 8.071 [8.047–8.680] | 8.100 [8.034–8.100] |
| KV restore 33 / 0.129 MiB | 0.230 [0.230–0.233] | 0.233 [0.233–0.239] | 0.234 [0.233–0.237] | 0.237 [0.236–0.240] |
| KV restore 2048 / 8 MiB | 2.364 [2.340–2.374] | 2.655 [2.614–2.800] | 2.657 [2.627–2.906] | 2.332 [2.152–2.335] |
| KV evict 2048 / 8 MiB | 1.797 [1.795–1.800] | 2.058 [2.049–2.062] | 2.076 [2.073–2.105] | 1.793 [1.779–1.800] |
| Raw weight / 4 MiB | 1.117 [1.116–1.120] | 1.113 [1.103–1.114] | 1.110 [1.109–1.117] | 1.113 [1.111–1.116] |

Warm pinned/pageable reductions are about 18.5% for the large DLRM transfer,
9.1% for GNN, and 11.0–12.7% for 8 MiB KV. Small KV savings are negligible in
absolute time. Typed DLRM/GNN ring slots remain pageable under `gate8`, reporting
`below_threshold` at approximately 1 MiB chunks even for these larger requests.
Layout KV at 8 MiB does select pinned. Direct dense weights borrow the source:
forcing staging pinning neither registers nor recopies it, so policies are a
neutral control there.

Cold means cleared native resources in an initialized process; clearing is
outside the timer. Ephemeral includes a new owner, completed call and close.

| Transfer | Pinned cold | Pageable cold | Pinned ephemeral | Pageable ephemeral |
|---|---:|---:|---:|---:|
| DLRM 4096 | 7.156 [7.035–7.264] | 5.915 [5.645–6.071] | 8.666 [8.395–8.729] | 5.821 [5.741–6.272] |
| GNN 65539 | 10.819 [10.767–10.917] | 9.722 [9.498–9.762] | 14.394 [14.275–15.202] | 9.017 [8.645–9.064] |
| KV restore 2048 | 11.697 [11.272–11.905] | 2.944 [2.916–3.073] | 14.694 [14.690–14.858] | 3.533 [3.518–4.282] |
| KV evict 2048 | 10.661 [10.518–10.714] | 2.502 [2.431–2.504] | 13.720 [13.439–13.732] | 2.943 [2.942–3.002] |

Growth sweeps retain owners through small→medium→largest→small, with immediate
reuse at each step, in three cycles per round. The following are the calls
entering the largest size and their next compatible calls:

| Growth target | Pinned growth | Pinned reuse | Pageable growth | Pageable reuse |
|---|---:|---:|---:|---:|
| DLRM 4096 | 4.377 [4.340–4.383] | 4.112 [4.072–4.201] | 5.240 [5.237–5.249] | 5.071 [4.924–5.181] |
| GNN 65539 | 9.409 [9.334–9.540] | 6.803 [6.785–6.847] | 8.243 [7.951–8.330] | 8.061 [7.970–8.204] |
| KV restore 2048 | 11.890 [11.761–12.158] | 2.374 [2.366–2.430] | 3.434 [3.193–4.789] | 2.713 [2.589–2.871] |
| KV evict 2048 | 9.971 [9.796–10.271] | 1.771 [1.761–1.799] | 2.015 [1.998–2.249] | 2.273 [2.176–2.446] |

Tensor growth is not necessarily proportional staging growth: DLRM/GNN switch
from whole-buffer execution to bounded rings and may retain older capacities.
The archive records counters before/after the first call and the reuse report.
Shrinking can also be expensive: pinned KV restore returning to 8 tokens takes
4.434 [4.387–4.666] ms, versus 0.226 [0.225–0.229] ms on immediate reuse; its
allocation counter increases despite the smaller tensor. Compatibility and
retirement history must be part of any future predictor.

Explicit prepared-source pinning is a different decision. At 1024×4096,
`PreparedWireWeights(pin_memory=True/False)` gives warm loads of
0.955 [0.950–0.955] / 1.107 [1.099–1.109] ms. Snapshot creation costs
5.072 [5.063–5.097] / 5.084 [5.059–5.108] ms and invalidation costs
0.062 [0.058–0.063] / 0.055 [0.054–0.056] ms. First loads after a snapshot
are 1.265 / 1.523 ms (median round p50s). This uses Torch's initialized caching
host allocator, not OS-cold page registration. The archive also includes
1/4/16-use totals with preparation and invalidation inside the timer. Current
offload models already use pinned prepared sources; staging-policy changes
cannot be credited with introducing that benefit.

## Model-inclusive comparison and priorities

These reuse #225's models, matched CPU participation, pinned Torch staging,
offline INT8 prepacking and two-slot prefetch. Sym uses Inductor for compute;
Torch is eager and the Inductor control compiles its matched compute regions.

| Model / unit | Sym pinned before | Sym pinned after | Torch | Inductor |
|---|---:|---:|---:|---:|
| DLRM / batch 2048 | 9.463 [9.435–9.663] | 9.440 [9.378–9.491] | 7.956 [7.854–8.011] | 8.033 [7.934–8.056] |
| GNN / 1024 seeds | 5.313 [5.162–5.342] | 5.264 [5.212–5.267] | 4.092 [4.071–4.095] | 3.012 [2.999–3.016] |
| GPT / four decoded tokens | 43.203 [43.100–43.436] | 43.128 [43.071–43.134] | 26.258 [26.251–26.328] | 30.502 [29.955–30.612] |
| MoE / one token | 4.623 [4.616–4.674] | 4.602 [4.597–4.608] | 3.238 [3.237–3.242] | 4.365 [4.346–4.370] |

| Model | Pinned | Pageable | Auto | Gate8 |
|---|---:|---:|---:|---:|
| DLRM | 9.440 [9.378–9.491] | 9.828 [9.823–9.946] | 9.914 [9.798–10.059] | 9.895 [9.748–9.951] |
| GNN | 5.264 [5.212–5.267] | 5.554 [5.544–5.573] | 5.564 [5.438–5.567] | 5.565 [5.453–5.596] |
| GPT | 43.128 [43.071–43.134] | 43.860 [43.826–43.887] | 43.801 [43.747–43.850] | 43.078 [43.018–43.079] |
| MoE | 4.602 [4.597–4.608] | 4.497 [4.491–4.502] | 4.495 [4.478–4.503] | 4.478 [4.477–4.495] |

Pinning helps DLRM/GNN model medians by roughly 4–5% versus forced pageable,
less than transfer-only gains. In GPT the policy affects KV; weights stay
explicitly pinned in every path. **MoE has no backend staging calls**, and its
prepared weights stay pinned in every path: its timing differences cannot be
attributed to changing staging memory kind. This negative control guards
against treating every small between-path delta as a policy effect.

Tail variation also remains. Candidate **median round p95 [range of round
p95s]** is:

| Model | Torch p95 | Inductor p95 | Sym pinned p95 |
|---|---:|---:|---:|
| DLRM | 10.576 [8.593–10.651] | 8.988 [8.035–13.826] | 10.189 [10.081–20.540] |
| GNN | 4.481 [4.465–4.777] | 3.294 [3.289–3.314] | 5.902 [5.817–5.920] |
| GPT | 26.377 [26.358–26.945] | 31.649 [30.989–32.841] | 43.700 [43.676–46.800] |
| MoE | 3.268 [3.253–3.268] | 4.414 [4.402–4.424] | 4.679 [4.669–4.710] |

The separate #223 raw-weight model experiment fixes eager model compute and
native Torch KV for **all** paths; its Inductor control compiles only weight
conversion. It changes live checkpoint values between samples and grants all
paths the same pinned checkpoint and two-slot prefetch. Do not compare these
absolute totals to the differently factored #225 model table above.

| Raw-weight model | Sym before | Sym after | Torch | Inductor weight conversion |
|---|---:|---:|---:|---:|
| GPT decode | 33.069 [32.845–33.091] | 32.580 [32.199–32.654] | 31.246 [31.187–32.165] | 26.421 [26.401–26.589] |
| Sparse MoE | 4.713 [4.681–4.727] | 4.639 [4.598–4.703] | 3.497 [3.448–3.497] | 3.622 [3.606–3.684] |

The resulting priorities are narrow: ship the verified raw-weight tile;
qualify warm pinning for each real lifetime rather than changing the default;
use the execution-family/cache evidence for #208's future profile design.
Deprioritize further generic GPU tuning for these four models: CPU transforms,
request preparation and remaining transfer time outweigh the demonstrated
kernel opportunity. This single-machine sweep does not satisfy #208's held-out
validation, additional-hardware, profile invalidation or conservative selection
requirements.

## Evidence and validation

[Raw samples and provenance](results/samples.json.gz),
[all per-round p50/p95 distributions](results/distributions.csv.gz), and
[SHA-256 manifest](results/manifest.json) accompany the report.
[Collection details](results/collection.json) record toolchain, validation and
the build-only timing helper's binary hash. The four
`*-trace.json.gz` files retain all scalar/host-phase samples and selected
request-relative CPU/CUDA/GPU intervals. Original profiler databases, logs and
compiler caches are omitted. The combined evidence is under 700 KiB compressed.

Validation: 6/6 CTest targets; 41/41 compiler tests; 1,214 Python tests passed
(4 skipped); 58 workload tests passed. New CUDA tests cover tile tails, skinny
shapes, all INT8 values, fresh scales/source contents, signed zero/subnormals,
offset pointers, a nonblocking stream and generic rank/stride fallback. All
transfer outputs are byte-checked; model outputs use #225's existing tolerance
with exact token/routes. The final NVTX-off build also passes the focused
dequantization tests. No adaptive policy or threshold default changed.

## Protocol fixed before acceptance

Base: merged main `8342acc` (#228). This qualification owns workload evidence;
the adaptive pinning predictor remains **unimplemented and tracked in #208**.
No new size threshold, online timing probes or adaptive policy are introduced.

Current-main diagnostic traces identify a targeted opportunity in the existing
raw INT8 `[in,out]` → FP32 `[out,in]` weight path: at 1024×4096, GPU dequantize /
transpose occupies about 0.121 ms of a 1.11 ms instrumented completed call.
The 1024² control spends 0.031 ms of 0.621 ms there. KV restore/evict has no
GPU transform kernel in this path. These instrumented values are selection
evidence, not headline latencies or additive CPU/GPU phase totals.

Evaluate a padded 32×32 shared-memory dequantize/transpose only for dense 2D
transposes, retaining the generic fused kernel as both fallback and control.
The target is at least 5% lower completed latency in the hot 4 MiB-weight
regimes without material tiny/odd-shape regressions. Report the kernel activity
and completed call separately. Keep INT8 plus identical FP32 scale payloads,
FP32 outputs, scale arithmetic, layout, allocation and completion semantics.
Prepared-wire weights already avoid this transpose; include them as an
unchanged control rather than attributing their behavior to the new kernel.

Pinning qualification covers CPU layout/cast H2D (DLRM), indexed gather/cast
H2D (GNN), KV layout transfers in both directions, and direct weight uploads.
Compare forced pinned/pageable, unconfigured auto, and an explicitly configured
8 MiB gate as a historical control, **not a proposed universal threshold**.
Measure cold staging calls with cleared owners, warm compatible reuse, growth and reuse after
growth, plus ephemeral ownership. Explicitly pinned prepared source snapshots
have separate preparation/retirement costs; staging policies do not pin a
borrowed dense source. Input tensors, parameters and output checks are fresh;
compilation, input construction and checks stay outside completed-call timing.

Compare native Torch and applicable Inductor controls with identical payload,
precision, layout, CPU participation and bounded temporary storage. Reuse
#225's model-inclusive DLRM, GraphSAGE, GPT decode and sparse MoE definitions
and matched prepacking/prefetch opportunities to qualify workload impact.
Model compute fusion uses the existing numerical tolerance; isolated transfers
are byte-exact. Report model and transfer-only measurements separately.
The model control is Sym with Inductor compute; both Torch controls receive
the same prepacking/prefetch opportunity. A separate raw-weight GPT decode/MoE
experiment reuses #223: all three paths use eager model compute and native Torch
KV transfers, and its
Inductor control compiles weight conversion only. Do not conflate the two.

Three independent process rounds, rotating path order, 30 warm samples and
three cold-owner observations per case; report per-round p50/p95 and round
ranges as uncertainty, not confidence intervals from correlated samples.
Cold-owner means initialized process/compiler/Torch allocator, not cold
process startup. Timing includes final stream completion; lifecycle tests also
include owner retirement. Main runs use Release with NVTX off; separate traces
correlate native CPU intervals, CUDA runtime calls, actual GPU memcpy/kernel
activity and completion. Never sum overlapping CPU/GPU durations. No foreign
builds, tests, packaging or GPU workloads during headline measurements.

The qualification machine uses GPU 0, EPYC 7351 CPUs `4-7,20-23`, eight CPU
participants, one Torch interop thread and passive OpenMP waits. The transfer
sweep and weight queues have 64 MiB native scratch limits. Model frontends reuse
#225's defaults (64 MiB typed retention, 256 MiB layout retention); their actual
staging footprints stay below 64 MiB. CUDA graphs and TF32 are disabled. Record exact source/binary
hashes, GPU/topology and policy/resource reports. Keep only compact raw samples,
scalar provenance and selected trace intervals; omit logs and compiler caches.

### Reproduction

Build the base and candidate with the same Release/CUDA flags and
`RELOC_ENABLE_NVTX=OFF`, `SYM_BUILD_TESTS=ON`. The build-only
`_cuda_kernel_test` extension supplies the generic fused control; it is not
installed. After correctness checks, with no other build/test/GPU jobs:

```sh
python bench/issue229/run.py --baseline /path/to/base-build \
  --candidate /path/to/candidate-build --python /path/to/torch-python \
  --output /tmp/issue229-rounds
```

For separate diagnostics, use NVTX-enabled runtimes and Nsight Systems
`--trace=cuda,nvtx --sample=none --cpuctxsw=none`, running `transfers.py` and
`models.py` with `--trace --samples 2`. Export each `.nsys-rep` to SQLite and
run `python bench/issue229/analyze.py trace.sqlite trace.json.gz`.
Use `python bench/issue229/compact.py /tmp/issue229-rounds /tmp/issue229-evidence
--traces /tmp/issue229-profiles` to retain raw samples (including #223's CSV
observations), distributions and selected trace witnesses.
The `phases` samples partition host wall time; correlated CUDA API durations
are subsets and actual GPU activities are separate, potentially overlapping
intervals. Event-bracketed isolated kernels include launch gaps; the trace also
reports actual kernel activity. Inputs, JIT, oracle checks and cold-owner
`clear()` are outside call timing. Ephemeral cycles include owner creation and
close. Prepared-source snapshot and invalidation costs are explicit; Torch's
host allocator may cache retired pages, so these are not OS-cold allocations.
