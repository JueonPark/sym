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
its caller uses `12,13,28,29`. NumPy allocates and first-touches each source on
its submitting thread before workers are created. The remote-source control
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
