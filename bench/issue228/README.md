# Typed context concurrency — issue #228

The runtime adds bounded, compatible context leases to one typed resource owner.
Default execution still uses one context. See [ownership and budget semantics](../../docs/typed-contexts.md).

## Protocol

`concurrency.py` measures completed H2D transfers: fresh preparation, output
allocation, CPU transformation, copies and stream completion are timed. Batch
time also includes submission to persistent caller threads and joining every
result. Individual call latency includes resource admission. Compilation,
source allocation, input mutation and byte comparisons are outside samples.
These are transfer results, not model end-to-end timings.

The cast family transposes a 2048×1024 f32 tensor and narrows to f16; the small
control is 65×129. The chain preserves f32→f16→f32 rounding on 2,097,155 values.
Grouped rows transfer two separate cast tensors per GPU in one group. Inputs
change each sample; all results are compared byte for byte with eager CPU
Torch. Torch and Inductor controls transform on the CPU and copy the same
payload to the same GPUs. Inductor uses `emulate_precision_casts=True`. This
comparison does not select the fastest CPU/GPU placement for each framework.

`shared` uses one owner: baseline has one context, candidate has one slot per
caller (two on one GPU for the contention control). `independent` uses separate
owners in both builds, an already available concurrency control. `queue` uses
one asynchronous queue per GPU, each submitting a two-weight group and waiting
for completion. Baseline queues own separate resources; candidate queues borrow
one shared owner. This tests integration and completed throughput; it does not
measure transfer/compute prefetch overlap or multi-step model latency.

All Sym paths allow **64 MiB aggregate typed live/retained scratch**; separate
owners split it evenly. Queue output reservations sum to another 64 MiB.
Sources, fresh outputs, Torch allocator caches and queue ordering streams are
outside native scratch. Torch/Inductor allocate their own CPU intermediates
(no configurable native scratch owner); their logical temporary payload fits
within the same 64 MiB allowance, but their allocator retention is not capped.
The default parallel budget is two callers × four participants = eight CPU
threads, with six owned background workers. The sequential alternating-device
control permits the same aggregate budget but runs one four-participant call
at a time. Single-GPU controls use eight participants. The one-thread control
uses two callers without background workers; the explicitly oversubscribed
control permits sixteen participants on eight logical CPUs. Queue and owner
statistics expose actual retained resources and configured limits.

Affinity is explicit in every raw row. On the measured EPYC 7351, CPUs
`4-7,20-23` are four physical cores plus SMT on NUMA node 1. GPUs 0/1 share its
PCIe root. The parallel split is `4,5,20,21` / `6,7,22,23`. GPU 3 is on node 3;
its caller uses `12,13,28,29`. Each source uses a fresh anonymous mapping, filled by NumPy on its
submitting thread before workers are created. This avoids recycling malloc
arena pages placed by earlier configurations. Raw records include the source
address and its containing `/proc/self/numa_maps` entry; adjacent mappings can
merge, so an entry may cover more than one source. The remote-source control
first-touches both sources on node 1, then uses the same local worker placement
as the local-source control. This is first-touch placement under Linux's
existing policy, not an explicit page binding or migration guarantee. CPU
checks and source sign changes warm caches outside timing. GPU 2 is omitted
from placement tuning because its adjacent CPU NUMA node has no DRAM.

Each process has three initial calls, five warmups and 30 measured warm batches
per path. Before each initial Sym call, its owners are cleared; these are
cold-owner, warm-process/recipe samples. Torch initial calls have no equivalent
owner reset. `run.py` runs three independent process rounds, rotates path order
and alternates baseline/candidate order. Both builds are Release with native
NVTX disabled. Passive OpenMP waiting and one Inductor compile worker reduce
background contention. A process-tree-aware monitor rejects foreign builds,
packaging or GPU processes. Clock frequencies are recorded but not locked.

## Reproduce

The runner's affinity/configuration masks describe this four-GPU machine;
adjust `CONFIGS` and the runner mask for another topology before comparing.
Build matching baseline and candidate CUDA/Python runtimes separately.

```sh
python bench/issue228/run.py --baseline /path/to/baseline-build \
  --candidate /path/to/candidate-build --baseline-revision BASELINE_SHA \
  --output /tmp/sym-228-acceptance
python bench/issue228/compact.py --input /tmp/sym-228-acceptance \
  --output bench/issue228/evidence
```

Only compressed raw samples and scalar provenance are committed. Logs,
compiler caches and full profiler databases remain outside the repository.

## Results (2026-10-09)

Baseline is merged main `182179f`, using the frozen #226 library built at
`2d0103a` (executable runtime/compiler code is identical; intervening changes
are comments and documentation). Candidate runtime is `bd97293`, with the
first-touch benchmark correction in `25dbf57`. Hardware: EPYC 7351, four RTX
2080 Ti GPUs, driver 595.71.05, CUDA toolkit 12.5, Torch 2.14.0+cu126 and
CPython 3.14.7. All six accepted processes passed byte comparisons and observed
no foreign build/packaging or GPU processes. The earlier allocator-based pilot
is excluded. [Raw samples](evidence/samples.csv.gz) contain 31,878 timings;
[provenance](evidence/runs.json.gz) contains binary/frontend/exporter hashes,
loaded libraries, topology, source page-placement observations, distributions
and resource snapshots.

Cells below are **medians of three round statistics**, in milliseconds;
`p50/p95` does not mean a pooled percentile. Batch latency includes completion
of every listed GPU transfer. `Separate` is the candidate with separate owners;
Torch/Inductor values are CPU-transform controls from the candidate rounds.

| Configuration | Old shared p50 | New shared p50/p95 | Shared speedup | Separate p50 | Torch p50 | Inductor p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, GPU 0 | 0.284 | 0.311/0.325 | **0.91×** | 0.309 | 0.137 | 0.227 |
| Cast, GPU 0 | 2.394 | 2.389/2.447 | 1.00× | 2.388 | 3.974 | 2.037 |
| Chain, GPU 0, whole | 2.182 | 2.189/2.276 | 1.00× | 2.177 | 2.139 | 2.133 |
| Cast, alternating 0→1 | 21.371 | 6.686/6.785 | 3.20× | 4.790 | 8.132 | 4.824 |
| Cast, concurrent 0+1 | 21.709 | 4.296/4.370 | 5.05× | 4.291 | 5.060 | 4.755 |
| Cast, 0+3, remote source | 23.129 | 4.182/4.235 | 5.53× | 4.191 | 4.525 | 5.079 |
| Cast, 0+3, local sources | 22.810 | 4.122/4.175 | 5.53× | 4.121 | 4.516 | 4.620 |
| Cast, two callers on GPU 0 | 22.745 | 4.451/4.495 | 5.11× | 4.411 | 5.121 | 4.710 |
| Cast, 0+1, 16 participants | 23.415 | 4.354/4.617 | 5.38× | 4.386 | 5.090 | 3.844 |
| Cast, 0+1, two participants | 26.797 | 6.314/6.370 | 4.24× | 6.281 | 5.116 | 8.361 |
| Chain, 0+1, whole | 34.548 | 3.780/3.836 | 9.14× | 3.781 | 4.058 | 4.161 |
| Chain, 0+1, 32 KiB ring | 17.682 | 5.076/5.227 | 3.48× | 5.070 | 4.054 | 4.170 |
| Chain, 0+1, 1 MiB ring | 13.055 | 2.967/3.002 | 4.40× | 3.026 | 4.057 | 4.164 |
| Chain, 0+1, 4 MiB ring | 37.424 | 3.469/3.587 | 10.79× | 3.529 | 4.059 | 4.164 |
| Two-weight groups, 0+1 | 44.845 | 7.175/7.283 | 6.25× | 7.184 | 9.515 | 8.735 |

The large gains over the old shared owner combine warm resource retention and
concurrent execution. They are not proportional PCIe bandwidth gains. The new
owner matches separate owners on the parallel cases, bringing their existing
concurrency under one aggregate budget. GPU 0+1 cast batch medians are
4.328/4.270/4.296 ms in rounds 1/2/3. Its completed wire throughput is
**1.953 GB/s**, with individual-call p50/p95 of 3.801/3.905 ms. The local 0+3
case reaches 2.035 GB/s and 3.523/3.644 ms call latency. The 1 MiB chain reaches
**5.656 GB/s** and 2.482/2.672 ms call latency; grouped casts reach 2.338 GB/s
and 6.615/6.827 ms. Every case's call distributions remain in the raw evidence.

The default single-context small-call control regresses about **9%** in batch
latency (27 µs); individual-call p50 increases 0.218→0.234 ms. Native admission
adds work, but this measurement does not isolate its cost from host scheduling.
Single-GPU large cast and chain results are essentially unchanged. Sequential
alternation is also variable: shared round medians are 6.686/7.116/4.799 ms,
versus 4.637/7.106/4.790 ms for separate owners. Both retain contexts, so there
is no established latency advantage over separate owners in that case. CPU
scheduling/locality is a possible contributor, not a demonstrated cause.
Inductor still wins the single-GPU cast and several controls; these results do
not establish a general or model-level advantage over Torch.

### Qualified scheduling choices

For the tested concurrent cast, use four participants per GPU on disjoint
CPU masks (eight aggregate). Reducing to one participant per GPU increases
batch latency to 6.314 ms. Oversubscribing to sixteen participants gives
4.354 ms, versus 4.296 ms with eight, and worsens p95 from 4.370 to 4.617 ms.
There is no reason to increase the default worker budget for this family.
Two callers sharing GPU 0 take 4.451 ms; the hardware copy path remains a
contention point even though CPU transformations have separate contexts.

For the tested chain, the existing 1 MiB ring is 1.27× faster than whole-buffer
execution. The 32 KiB ring loses to repeated scheduling/copy overhead; 4 MiB
chunks expose less overlap here. Keep chunk selection explicit: these cases
do not justify a universal change for other layouts or devices.

Local first-touch on GPU 3 improves this comparison only slightly
(4.182→4.122 ms median, about 1.5%); the difference is small relative to
between-round variation. The page observations confirm node-1 placement in
the remote control and node-3 pages for the local GPU-3 source. Prefer local
placement without claiming a general NUMA speedup. Comparing 0+1 with 0+3 also
changes PCIe roots and memory controllers, so it does not isolate either one.

### Resource reuse, cold costs and queues

Across the 60 measured calls in each alternating-device round, the old owner
creates 60 contexts and records zero hits; the new owner creates zero contexts
and records 60 hits after warmup. Parallel 0+1 cast records the same zero/60
new-owner counts, versus 57–60 creations in the baseline. New peak leases are
one for sequential alternation and two for parallel calls. Two cast contexts
retain **9 MiB**, six background workers and two runtime streams, compared
with the baseline's one 4.5 MiB context, three workers and one stream. Both
have the same 64 MiB scratch allowance and eight-participant application
budget. The new ring retains 5 MiB aggregate; grouped casts retain 18 MiB.
All measured live peaks remain below 64 MiB. Source/output/allocator storage
is separately owned as described above.

Cold-owner costs remain material: new single-GPU cast p50 is 8.312 ms,
parallel 0+1 cast 14.037 ms, 1 MiB chain 13.101 ms and grouped casts 26.768 ms.
These are allocation/context costs in an already warm process, not compilation.
The async two-queue grouped path changes 7.551→7.459 ms, with candidate p95
7.519 ms: broadly equivalent to the existing separate-owner queues. Its new
benefit is shared bounded ownership; no model prefetch speedup is claimed.

### CPU and GPU timeline evidence

A separate Nsight Systems 2024.2.3 run loads the same runtime code compiled
with NVTX enabled. Its timings are excluded from the headline results.
The [compact trace](evidence/trace.json.gz) retains profiler/runtime hashes,
loaded-library provenance, all 30 measured batch summaries and one detailed
interval witness per configuration. `trace.py` intersects actual CUPTI H2D
intervals on two devices and unions real native CPU transformation intervals;
CUDA API waits and worker barriers do not count as CPU work.

| Shared-owner case | Batches with simultaneous H2D | Median H2D overlap | Batches with CPU/DMA overlap | Median CPU/DMA overlap |
| --- | ---: | ---: | ---: | ---: |
| Cast, 0+1 | 10/10 | 0.318 ms | 1/10 | 0.000 ms |
| Cast, local 0+3 | 3/10 | 0.000 ms | 10/10 | 0.264 ms |
| Chain, 1 MiB ring, 0+1 | 10/10 | 0.507 ms | 10/10 | 0.921 ms |

This proves real overlap while showing that its form depends on the schedule.
The whole-buffer 0+1 casts usually finish both CPU transformations before their
copies overlap. Local 0+3 often overlaps one GPU's DMA with the other call's
CPU work; simultaneous DMA is intermittent. The chunked chain overlaps both.
Each chain batch contains 18 H2D copies, including each tensor's short tail.
There are no GPU transform kernels in these explicitly CPU-placed recipes.

To reproduce tracing, use an NVTX-enabled build and run:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/sym228-trace python bench/issue228/concurrency.py \
  --variant candidate --configs pair01 pair03_local chain_ring1m \
  --paths shared --samples 10 --trace --output /tmp/sym228-trace-run.json
nsys export --type=sqlite --output=/tmp/sym228-trace.sqlite /tmp/sym228-trace.nsys-rep
python bench/issue228/trace.py /tmp/sym228-trace.sqlite /tmp/sym228-trace.json.gz \
  --runtime /path/to/nvtx-build/libreloc/libreloc_runtime.so \
  --run /tmp/sym228-trace-run.json
```

Use the same Python/exporter paths, passive OpenMP settings and CPU masks as
the runner. The recorded run loaded `/tmp/sym-228-trace-lib/libreloc_runtime.so`;
its explicit hash identifies the instrumented library. The metadata's normal
build-library hash identifies the uninstrumented headline build.

Validation: 6/6 CTest targets, 41/41 compiler tests, 1,214 Python tests passed
(4 skipped), and 58 workload tests passed. New tests cover FIFO admission,
limits/timeouts, LRU/affinity compatibility, concurrent blocking and async
execution, current-stream ordering, changing values/parameters, shared queue
ownership, clear/close and one quarantined context beside a healthy pending
sibling. The one-context default remains in place; additional contexts and
scheduling settings are explicit caller choices.
