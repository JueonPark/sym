# bench/rtrack — R0.3 pipeline & measurement harness (issue #76)

The R-track's Method-A vs Method-B measurement rig over the R0.1 CPU quant
kernels (#74/#77) and R0.2 GPU kernels (#75/#78).

**Isolated metric** (issue #76, verbatim): end-to-end (transform+transfer)
latency for one tensor, source in host DRAM (cold pinned staging),
destination = final layout in GPU global memory.

- **Method A**: per-chunk CPU transform (gatherChunk /
  `gather_quantize_f32_s8` / `quantize_pack_f32_s8` / `convert_f32_f16`,
  parallelized over a `GatherPool`) into double-buffered pinned staging
  (2 × chunk, event-gated reuse) → `cudaMemcpyAsync` of r·S bytes into the
  final layout. R2's dequant/unpack receive kernels slot in as new
  `GpuStage` values when the r-sweep lands.
- **Method B**: per-chunk pageable→pinned staging copy (same thread budget
  as A's transform) → DMA of S fp32 bytes → R0.2 GPU transform kernels
  (`relocate_f32`, `quantize_f32_s8`, plus a bench-local f32→f16 kernel for
  T5, which issue #75's set does not include).

Both methods produce the identical final artifact (dtype_out in the plan's
dst layout); every (workload, method, chunk) config is verified bit-exact
against a scalar CPU reference before it is timed.

## Workloads

| ID | transform | dtype_out | r | A's host read |
|----|-----------|-----------|---|---------------|
| T1 | `transpose` | f32 | 1.0 | strided (runs of 1 elem) |
| T1b | `blocked_transpose` (the sym#63 anchor) | f32 | 1.0 | strided (runs of N elems) |
| T2 | `transpose_quant` | s8 | 0.25 | strided, fused gather+quant |
| T3 | `quant` | s8 | 0.25 | contiguous |
| T4 | `nchw_nhwc_quant` (B,C,H,W = N/64,64,64,N/64) | s8 | 0.25 | strided, fused |
| T5 | `convert_f16` | f16 | 0.5 | contiguous |

Caveats to read T-rows correctly:

- **Chunk semantics differ per method**: Method A's chunk request budgets
  staged OUTPUT bytes (the staging buffer holds transformed rows), Method
  B's budgets INPUT bytes. For r < 1 an "A at 4 MiB" chunk touches 4/r MiB
  of source. Compare best-C *within* a method; cross-method chunk-size
  comparisons are confounded with r. Chunk requests that clamp to the same
  1-chunk plan are measured once and skipped thereafter (stderr notes it).
- **Quant scale granularity**: the channel is the plan's coalesced outer
  axis (the `gather_quantize_f32_s8` contract), so T4's scales are per
  batch image, not per C channel.
- **Method B kernel path**: `relocate_f32` takes its SMEM-tiled path only
  for the exact rank-2 transpose shape (T1); T1b and T4 run the naive
  fallback (bit-identical output, different bandwidth).
- **Stage attribution overhead**: per-chunk H2D event pairs live inside
  the timed pipeline (~2 event records per chunk); at the 4 MiB end of the
  sweep this adds a fixed few-µs-per-chunk cost to both methods.

Plans are **hand-authored** in `plans.h` and verified against independent
index-math oracles + bijectivity in `RtrackTest.cpp`. Do NOT switch them to
decoding `bench/reference_plan.h`: the frozen golden blob on main still
carries the non-injective pre-fix strides from issue #63 (the fix lives on
`fix-reference-plan-transpose`), and a decode-then-execute self-check
cannot catch that class of bug.

## Build

CMake (needs `RELOC_ENABLE_CUDA=ON`): target `bench-rtrack`; the CPU-side
tests build as `bench-rtrack-test` and run in CI under ctest.

Standalone, without the MLIR tree (libreloc is MLIR-free) — the per-TU ISA
flags matter, or the quant kernels silently run scalar-only:

```sh
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX2.cpp \
  -mavx2 -mfma -mf16c -o QuantAVX2.o
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX512.cpp \
  -mavx512f -mavx512bw -o QuantAVX512.o
nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -Ibench \
  bench/rtrack/rtrack_bench.cu libreloc/src/*.cpp libreloc/quant/Quant.cpp \
  QuantAVX2.o QuantAVX512.o \
  libreloc/cuda/CudaBackend.cu libreloc/cuda/CudaKernels.cu \
  -o bench-rtrack -Xcompiler -pthread
```

(`-arch=sm_89` on the Ada box.)

## Session ritual (environment controls, issue #76)

1. `sudo cpupower frequency-set -g performance` (or the sysfs equivalent).
2. `sudo nvidia-smi -pm 1` (persistence mode; resets at reboot). Without
   it links/clocks ramp per process and cold measurements read a fraction
   of the real bandwidth. Fix GPU clocks where the driver allows:
   `sudo nvidia-smi -lgc <sm_clock>` (`-rgc` to reset afterwards).
3. On multi-NUMA boxes, pin the process to the GPU's affinity cores
   (`nvidia-smi topo -m`; `taskset -c ...`) — see
   docs/m0-2080ti-bringup.md for why unpinned runs are bimodal on the
   EPYC box.
4. Record calibration ONCE per session — this also samples the PCIe link
   **under load** (Gen3/Gen4 cards downtrain to gen1 at idle; idle numbers
   understate the link):

   ```sh
   python3 bench/rtrack/calibrate.py --out calibration.json \
       --load-bin ./bench-rtrack
   ```

5. Run the sweep pinned to one NUMA node when `numactl` is available
   (`"numactl": "--membind=0 --cpunodebind=0"` in the config; the runner
   warns and continues unpinned when it is not installed):

   ```sh
   python3 bench/rtrack/run_rtrack.py --config bench/rtrack/configs/example.json
   ```

   The output CSV carries the calibration + config as `#` comment lines;
   kernel/driver/CUDA versions therefore ride in every file's header.
   Relative paths in the config resolve against the current working
   directory; the runner refuses to overwrite an existing non-empty
   `out_csv`, and the driver suppresses `--csv-header` into a non-empty
   file (no mid-file header rows).

6. Figure 1:

   ```sh
   python3 bench/rtrack/figure1.py --csv gen3.csv gen4.csv --out figure1.png
   ```

   Best chunk is chosen **per method** (they legitimately differ); bars
   from any `unstable` row (IQR/median > 5%) are hatched.

## Protocol

5 warmup + 30 timed iterations per config; the CSV reports wall median /
min / p95, `iqr_over_median_pct`, and `unstable` (> 5%). Stage columns
(all medians): `gpu_pipeline_ms` (CUDA events, start recorded before the
first stage, stop after the last enqueued op, then stream sync),
`cpu_stage_ms` (fenced `steady_clock` sums of the per-chunk CPU sections),
`h2d_ms` (summed per-chunk DMA event pairs), `gpu_kernel_ms` (Method B's
transform kernels). `effective_input_GBps` = S / wall median.

## R0 exit criteria status

- Kernel correctness: R0.1/R0.2 test suites + this harness's per-config
  bit-exact gates; the quant round-trip max-abs-err bound
  (|x − q·scale| ≤ scale/2) is asserted in `RtrackTest.cpp`.
- Regression anchor: "reproduces sym#63 numbers on Gen4 (gather ~14, H2D
  ~24, Method B flat) within ±10%" — run T1b/T3 on the 7800X3D + 4070 Ti
  SUPER box and compare `cpu_stage_ms`-derived gather GB/s, `h2d_ms`-derived
  DMA GB/s, and `gpu_kernel_ms` flatness. **Caveat:** the sym#63 gather
  figure was measured with the pre-fix reference plan (issue #63); after
  the fix the blocked-transpose gather pattern changed (runs of N elements
  instead of 1–2 KiB), so re-baseline the anchor on the first post-fix
  Gen4 run rather than treating 14 GB/s as gospel.
