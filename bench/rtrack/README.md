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
  final layout. On the r-sweep rows (issue #83) the wire payload is
  compressed and the DMA target is *not* the final artifact, so a real
  in-stream GPU receive kernel runs per chunk to decompress it back to f32
  in the dst layout — `convertF16F32`, `dequantS8F32`, or
  `unpackS4S8`+`dequantS8F32` depending on the wire dtype — timed
  separately as `gpu_recv_ms` (a new `RecvStage`, orthogonal to Method B's
  `GpuStage`).
- **Method B** (`b`, staged): per-chunk pageable→pinned staging copy (same
  thread budget as A's transform) → DMA of S fp32 bytes → R0.2 GPU transform
  kernels (`relocate_f32`, `quantize_f32_s8`, plus a bench-local f32→f16
  kernel for T5, which issue #75's set does not include).
- **Method B_fair** (`b_fair`, issue #95): identical to Method B but the
  source is resident in pinned memory, so the per-chunk staging memcpy is
  gone and the DMA reads the source directly (`host_stage_ms` = 0). This is
  the admissible baseline — a competent pure-relocation implementation would
  not carry a pageable→pinned copy. `b` is kept as the comparison, not
  replaced. Select all three with `--method all`.

Both methods produce the identical final artifact (dtype_out in the plan's
dst layout); every (workload, method, chunk) config is verified bit-exact
against a scalar CPU reference before it is timed (see the r-sweep
section below for how that reference differs by method on a compressed
wire).

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

## R2 r-sweep workloads

Issue #83 fixes the final artifact at **f32** and sweeps only the wire
dtype — the payload Method A actually stages onto the PCIe link — across
the four T1b/T2/T3/T4 families at r ∈ {1.0, 0.5, 0.25, 0.125} (wire
f32/f16/s8/s4). 16 rows total, selected with `--transform rsweep` (the
R1 matrix rows above are `--transform matrix`; `--transform all` is a
backward-compatible alias for `matrix`, NOT the union of both sets — it
selects the same variant!="rsweep" rows as `matrix`, so existing configs
that say `"all"` keep meaning the R1 matrix after this lands. Run the
rsweep rows explicitly via `--transform rsweep` or a comma-separated list
of `*R###` IDs). CSV rows carry `variant=rsweep` vs `variant=matrix` so
the two sweeps
never mix in `gates.py`/`figure_rstar.py`.

| ID | family (`transform`) | wire | A's CPU stage | A's recv stage | runs B |
|----|----|----|----|----|----|
| T1bR100  | `blocked_transpose` | f32 | GatherF32     | none            | yes |
| T1bR050  | `blocked_transpose` | f16 | GatherF16     | ConvertF16F32   | no  |
| T1bR025  | `blocked_transpose` | s8  | GatherQuant   | DequantS8       | no  |
| T1bR0125 | `blocked_transpose` | s4  | GatherQuantS4 | UnpackDequantS4 | no  |
| T3R100   | `quant`              | f32 | CopyF32       | none            | yes |
| T3R050   | `quant`              | f16 | ConvertF16    | ConvertF16F32   | no  |
| T3R025   | `quant`              | s8  | QuantPack     | DequantS8       | no  |
| T3R0125  | `quant`              | s4  | QuantPackS4   | UnpackDequantS4 | no  |
| T2R100   | `transpose_quant`    | f32 | GatherF32     | none            | yes |
| T2R050   | `transpose_quant`    | f16 | GatherF16     | ConvertF16F32   | no  |
| T2R025   | `transpose_quant`    | s8  | GatherQuant   | DequantS8       | no  |
| T2R0125  | `transpose_quant`    | s4  | GatherQuantS4 | UnpackDequantS4 | no  |
| T4R100   | `nchw_nhwc_quant`    | f32 | GatherF32     | none            | yes |
| T4R050   | `nchw_nhwc_quant`    | f16 | GatherF16     | ConvertF16F32   | no  |
| T4R025   | `nchw_nhwc_quant`    | s8  | GatherQuant   | DequantS8       | no  |
| T4R0125  | `nchw_nhwc_quant`    | s4  | GatherQuantS4 | UnpackDequantS4 | no  |

- **Only the R100 row of each family runs Method B.** B's GPU transform
  kernel reads the full fp32 tensor regardless of what Method A stages on
  the wire, so it is r-independent by construction; one measurement per
  family covers all four r values (`methodB=false` on R050/R025/R0125).
- **s4 scale**: `invScale = 7/maxAbs` per channel (vs `127/maxAbs` for
  s8) — a packed nibble is a signed 4-bit value, range [-8, 7], so 7 is
  the largest magnitude representable both signs need to stay symmetric.
- **Two-pass CPU stages are deliberate, measured costs, not an assumed
  fusion.** `GatherF16` (the r=0.5 strided rows) is a gather-to-f32 pass
  followed by a convert-to-f16 pass; `GatherQuantS4`/`QuantPackS4` (the
  r=0.125 rows) are a gather-or-quant pass followed by a nibble-pack pass.
  A fused single-pass kernel is a noted future optimization, not something
  the harness silently credits Method A with.
- **Verification differs by method on compressed wires.** Method B (R100
  rows only) verifies against the exact CPU reference of the f32 artifact.
  Method A verifies against a CPU **quantize→dequantize roundtrip**
  reference instead: the same per-channel float scale and the same RNE
  quantize run on both the reference and the GPU receive path, so a
  compressed-wire row is lossy by construction but bit-exact by contract
  against that roundtrip, not against the untouched f32 values.
- **`--variant avx2`/`avx512pf` fail fast on `*S4` rows.** `packS8S4`
  (`libreloc/quant/Quant.cpp`) only has scalar and AVX-512 tiers — no
  AVX2, and `avx512pf` isn't accepted for this kernel either — so
  `GatherQuantS4`/`QuantPackS4` workloads reject those two variants before
  any buffers are allocated (`rtrack_bench.cu`'s pre-run variant check,
  the same fail-fast convention as R0.1's `quant_bw`).

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

7. R2 r-sweep figure (issue #83): A/B speedup vs r per family, plus
   measured and model-predicted critical r* (the r where A overtakes B):

   ```sh
   python3 bench/rtrack/figure_rstar.py --csv rsweep.csv \
       [--rooflines r2_rooflines/*.json] [--h2d GBPS] [--n N] \
       [--threads T] --out figure_rstar.png --json rstar.json
   ```

   Reads `variant=rsweep` rows only, at the largest `N`/`threads` present
   per family unless overridden. `r*_measured` is the linear-in-log2(r)
   interpolated 1.0 crossing of `speedup(r) = median_ms(B at r=1.0) /
   median_ms(A at r)`; `None` when the curve never crosses in [0.125,
   1.0]. The model composes stage rooflines (`--rooflines`, effective
   *input* GB/s) with a pipelined (`min(BW_cpu(r), H2D/r)`) and a serial
   (`1/BW_cpu + r/H2D`) variant; `--json` feeds `gates.py --rstar` for
   R2-G5. `--b-method b|b_fair` (issue #95) selects which Method-B
   baseline anchors the speedup curve (default `b`, the staged R2
   figure); the choice is recorded in the JSON as `b_method`.

   Note: the committed `bench/results/r2_rstar_gen4.json` was built from the
   stabler-preference-merged CSVs (per the R2 report's rerun rule); re-running
   `figure_rstar.py` on the two rsweep CSVs (raw + rerun) reproduces the same
   r* values with only ~0.005-level speedup differences at the reran points.

8. Gates:

   ```sh
   python3 bench/rtrack/gates.py --csv run.csv           # R1/EXP-1 (default)
   python3 bench/rtrack/gates.py --exp r2 --csv run.csv \
       [--rstar rstar.json]                              # R2/EXP-2 gates
   python3 bench/rtrack/gates.py --exp v1 --csv run.csv  # V1 admissibility bar
   ```

   `--exp v1` (issue #95) checks whether each Method-B baseline reaches
   ≥ 0.90 × the box's measured pinned H2D on the r=1.0 workloads; `b`
   (staged) is expected to fail the bar and `b_fair` to pass it. Bars are
   fixed in `gates.py` before the data is read. For the multi-GPU
   pre-fold path (`bench-multigpu-reloc`, issue #98), `--reuse
   1,2,4,16` and `--streaming` sweep the folded artifact's per-load
   amortization against repeated vs. cold single-use loads, and
   `python3 bench/rtrack/exp4v_gate.py --json <run.json ...>` evaluates
   its V4-G1..G3 gates (speedup bar, DMA-leg admissibility, counter-case
   + rule-match) over the resulting JSON.

## Protocol

5 warmup + 30 timed iterations per config; the CSV reports wall median /
min / p95, `iqr_over_median_pct`, and `unstable` (> 5%). Stage columns
(all medians): `gpu_pipeline_ms` (CUDA events, start recorded before the
first stage, stop after the last enqueued op, then stream sync),
`cpu_stage_ms` (fenced `steady_clock` sums of the per-chunk CPU sections),
`h2d_ms` (summed per-chunk DMA event pairs), `gpu_kernel_ms` (Method B's
transform kernels; for `b_pipelined` this column is instead the per-chunk
SUM of `Pipeline::kernBeg`/`kernEnd` spans — it therefore includes
launch-granularity overhead and, for the rank-2 transposePlan family, the
naive-fallback kernel cost noted in `launchBPipeChunkKernel`'s comment,
whereas `b`/`b_fair` report the single monolithic `kBeg`→`kEnd` span).
`effective_input_GBps` = S / wall median.

R2 (issue #83) adds four columns: `gpu_recv_ms` (median, summed per-chunk
in-stream receive-kernel event time — Method A's r-sweep decompress
stage; 0 on rows with `RecvStage::None`, i.e. all matrix rows and all
`wire=f32` rsweep rows), `verified` (1/0 — whether the `--no-verify` gate
ran for this config; a row is only ever written after a passing verify,
so `verified=0` just means the run passed `--no-verify` and skipped the
check), `variant` (`matrix` | `rsweep` — absent in pre-R2 CSVs, and both
`gates.py`/`figure_rstar.py` treat a missing column as `matrix`), and
`wire` (the workload's on-wire dtype — `f32` | `f16` | `s8` | `s4` — set
from the `Workload` table regardless of method or variant, so e.g. a
matrix T2 row also reports `wire=s8`). Column order in the CSV
(`bench/rtrack/csv.h::csvHeaderLine`) is: ...,
`gpu_kernel_ms,gpu_recv_ms,verified,variant,wire,h2d_occupancy`.

BP1 (issue #114) adds `h2d_occupancy`: median over iterations of
sum(per-chunk h2d event time) / (evStart→evStop span); ~1.0 = the
pipeline is DMA-saturated (kernels fully hidden), lower = exposed kernel
or gaps. Populated for every method (issue #114).

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

## R2 gates (issue #83)

```sh
python3 bench/rtrack/gates.py --csv r2_gen4_matrix_nsweep.csv --exp r2 \
    [--rstar rstar.json]
```

Bars are fixed in `gates.py`'s docstring *before* the Gen4 data is read
(these are falsification gates — the issue predicts the losses, not the
wins): R2-G1 Gen4 H2D floor in [20, 26] GB/s; R2-G2 `quant` (T3) A ≥
1.50x B; R2-G3 `transpose_quant`/`nchw_nhwc_quant` (T2/T4) A/B < 0.95;
R2-G4 `blocked_transpose` (T1b) A/B in [0.40, 0.80] (barred on T1b, not
plain T1, per sym#63's blocked-gather anchor); R2-G5 per-family measured
r* within 2x of the `figure_rstar.py --json` prediction. `load_rows()`
filters to `variant=matrix` unconditionally, so R2-G1..G4 read the
output of `configs/r2_gen4_matrix_nsweep.json` /
`r2_gen4_matrix_tsweep.json` (`"transforms": "matrix"`);
`configs/r2_gen4_rsweep.json` (`"transforms": "rsweep"`, same machine and
N sweep as the nsweep config) instead feeds `figure_rstar.py --json
rstar.json`, which is then passed back into `gates.py --rstar` for
R2-G5. All three configs target the 7800X3D + 4070 Ti SUPER (Gen4) box,
no `numactl` (single-NUMA).

### CM1 Gen4 recv-kernel run (issue #109 runbook)

The Gen4 box has no `copy_f32` ceiling measurement and no usable committed
recv-kernel data (`gpu_recv_ms` in the pipeline CSVs is chunked/overlapped
event time and collapses on several cells), so all three `recv.m.*` keys
are omitted from `calibration/7800x3d-4070tis.cal` until this run lands.
On the 7800X3D/4070 Ti SUPER box:

1. Build `bench-hiding-ratio` (CUDA cmake tree, or the standalone recipe in
   `docs/m0-2080ti-bringup.md` with `-arch=sm_89`).
2. `bench-hiding-ratio --json bench/results/cm1_recv_kernel_bw_7800x3d_4070tis.json`
   — this also produces the first real Gen4 `copy_f32` ceiling.
3. Commit the JSON, then regenerate:
   `python3 bench/rtrack/make_calibration.py --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal`
   (the three `recv.m.*` keys appear automatically; every pre-existing
   line must stay byte-identical) and commit the `.cal`.
4. `python3 bench/rtrack/v3_gate.py` — CALIBRATION-REGEN must PASS.

Gen4 run notes (as executed, 2026-08-05): the box runs CUDA 13.2, which
removed `cudaDeviceProp::memoryClockRate` (`hiding_ratio.cu` reads the
memory clock via `cudaDeviceGetAttribute` instead), and the standalone
nvcc recipe additionally needs `libreloc/quant/Quant.cpp` plus
`QuantAVX2.cpp` built with `-mavx2 -mfma -mf16c` (its per-file cmake
flags). On this WSL2 box the WDDM driver repositions P-states in the
inter-kernel verify gaps: unlocked, the memory clock oscillates
10251<->5001 MHz mid-run and the recv-kernel medians (and the
copy_f32/kernel m ratios with them) swing run-to-run by up to 2x, at
temperature and power nowhere near their limits. Lock the clocks first
from an elevated Windows-side shell (`nvidia-smi -lgc 2610` and
`-lmc 10251`; `-rgc`/`-rmc` to undo) — the committed artifact was
measured with 100% locked-clock residency and
`iqr_over_median_pct < 2.5` on every N=16384 cell.

Deliberately out of scope here: re-baselining the existing Gen4 `hbm.*`
proxy keys onto the new run's ceiling. That would change committed keys
("byte-identical except the new keys" would fail) and re-open V3's
as-measured verdicts — a separate decision for CM5's re-run, recorded
here so it isn't lost. `pipeline.chunks_per_buffer` is likewise deferred
to CM5 for the same frozen-verdict reason: the C++ `Overlapped` fill/drain
term is implemented and unit-tested but dormant in both committed
calibrations, since emitting the key would activate it on the frozen V3
prediction cells and silently re-score `bench/results/v3_prediction_report.json`
without pre-registration.

Also note for CM5: `recv.m.convert_f16_f32`/`recv.m.dequant_s8_f32` derive from
L2-warm in-pipeline `gpu_recv_ms`, while `recv.m.unpack_dequant_s4` derives from
an isolated cold run — the two bases differ by ~20% on dequant. CM5's `m_eff`
composition must first decide on a single measurement basis (warm in-pipeline
vs. cold isolated) before consuming the `recv.m.*` keys.

## V3 cost-model tools (issue #97)

- **`make_calibration.py`** — assembles a `calibration/<machine>.cal` flat
  file (the `reloc::costmodel::CostModel` parser's input format) from
  committed `bench/results` artifacts only; every emitted `key value` line
  carries a trailing provenance comment naming its source file, and
  regeneration is deterministic (`git diff --exit-code calibration/`
  after regenerating is expected to be clean). `--machine
  {epyc7351-2080ti,7800x3d-4070tis} --out PATH`.
- **`v3_gate.py`** — the pre-registered V3 prediction gate: bars
  (`MISCLASS_BAR=0.15`, `RSTAR_ABS_BAR=0.15`, `REGRET_P90_BAR=0.20`) are
  fixed in the script's source before any prediction data exists, per this
  track's standing discipline. `--report bench/results/v3_prediction_report.json`
  evaluates winner-misclassification rate, `|r*_pred − r*_meas|` on
  families where both sides have a crossing, and p90 regret vs. the
  measured oracle over the committed R-track cells; see
  `docs/v3-costmodel.md` for the verdicts and a documented gap in how
  `mismatch_one_sided` r* rows are excluded from the bar.
- **`--plan-wire PATH`** (on `bench-rtrack`) — decodes a corpus wire-format
  plan blob (`reloc::decodePlan`) and `bind()`s it against `--n`'s `N`
  instead of using a hand-authored `plans.h` plan; only valid together
  with `--transform TW` (bijective requirement, checked both directions).
  Closes the compiler→measurement gap: a plan produced by the real fold
  pipeline (`generate_corpus.py`'s MLIR chain → `sym-opt --reloc-fold` →
  `encodePlan`) is measured with the same A/B/b_fair harness as every
  hand-authored workload. See `docs/v3-costmodel.md` §6 for the acceptance
  comparison against the hand-authored `T1b` row and the L=64-decomposition
  sensitivity finding.
