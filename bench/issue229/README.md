# Remaining critical paths: pinning and targeted GPU tuning (#229)

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
Measure new-owner cold calls, warm compatible reuse, growth and reuse after
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
experiment reuses #223: all three paths use eager model compute, and its
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
participants, one Torch interop thread, passive OpenMP waits, and 64 MiB native
scratch limits. CUDA graphs and TF32 are disabled. Record exact source/binary
hashes, GPU/topology and policy/resource reports. Keep only compact raw samples,
scalar provenance and selected trace intervals; omit logs and compiler caches.

### Reproduction

Build the base and candidate with the same Release/CUDA flags and
`RELOC_ENABLE_NVTX=OFF`, `SYM_BUILD_TESTS=ON`. The build-only
`_cuda_kernel_test` extension supplies the generic fused control; it is not
installed. After correctness checks, with no other build/test/GPU jobs:

```sh
python bench/issue229/run.py --baseline /path/to/base-build \
  --candidate /path/to/candidate-build --output /tmp/issue229-rounds
```

For separate diagnostics, use NVTX-enabled runtimes and Nsight Systems
`--trace=cuda,nvtx --sample=none --cpuctxsw=none`, running `transfers.py` and
`models.py` with `--trace --samples 2`. Export each `.nsys-rep` to SQLite and
run `python bench/issue229/analyze.py trace.sqlite trace.json.gz`.
The `phases` samples partition host wall time; correlated CUDA API durations
are subsets and actual GPU activities are separate, potentially overlapping
intervals. Event-bracketed isolated kernels include launch gaps; the trace also
reports actual kernel activity. Inputs, JIT, oracle checks and cold-owner
`clear()` are outside call timing. Ephemeral cycles include owner creation and
close. Prepared-source snapshot and invalidation costs are explicit; Torch's
host allocator may cache retired pages, so these are not OS-cold allocations.
