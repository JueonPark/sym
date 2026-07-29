# V2 — Single-host isolation: link generation vs host ISA (issue #96)

**Status: COMPLETE.** The host-ISA half (sections 1–4) was measured on
the Gen4 box 2026-07-29; the gen3 half (section 5a) was executed the same
day on `rebel-gpu1` (bare-metal EPYC 7351 / RTX 2080 Ti — an AVX2-only
host, so the gen3×avx512 cell is recorded impossible, the case the
runbook anticipated). Pre-registered expectations and reading rules were
posted to issue #96 before any data was taken. Final attribution: §6.

**Plan change (recorded).** Issue #96's link task asked for a BIOS
gen3/x8 downgrade of the Gen4 box (one host, one variable). The owner
decided to fill the gen3 axis on a 2080 Ti server instead. Consequence,
stated up front: the link axis is then a **cross-box** comparison (link +
host generation + environment change together), not a single-host link
isolation. What keeps the 2×2 informative is the ISA-dispatch axis
measured *within* each box (one host, one variable on that axis) plus the
per-ISA rooflines, which — given the ISA-null result below — pin the
cross-box difference on the host generation rather than the vector ISA.
If the 2080 Ti box BIOS allows a gen4→gen3-style slot control in reverse
(or an x8 re-seat), a true single-host link cell can still be added later.

## 1. Design (host-ISA half, Gen4 box)

Session 2026-07-29 on `JueonAtHome` (7800X3D / 4070 Ti SUPER, **WSL2
caveat as in R2**), fresh calibration
(`v2_isa_calibration_7800x3d_4070tis.json`: gen4 x16 under load, triad
37.11 GB/s). Two arms, same box, same link, same session, `--method a`:
`--variant avx512` vs `--variant avx2`; one shared `b_fair` anchor run (B
is variant-independent, so a shared anchor removes B session-variance from
the ISA contrast). Configs `bench/rtrack/configs/v2_isa_*.json`; data
`bench/results/v2_isa_*_7800x3d_4070tis.csv` (matrix arms: T1, T1b, T3,
T5 over N=2048…16384; rsweep arms: blocked+quant eligible rows at
N=16384); per-ISA stage rooflines `bench/results/v2_rooflines/*_7800x3d.json`
(T=8 and T=1). Every config bit-exact verified, 0 failures.

**Code-level variant map** (`Quant.cpp::kernelHasVariant`) — this
constrains what an ISA flip can test, and was stated before running:

| kernel | avx512 | avx2 | consequence |
|---|---|---|---|
| `gather_f32`, `contig_read`, `CopyF32` | — | — | no SIMD paths: identical code in both arms (the E1 null) |
| `quantize_pack` (T3), `convert_f32_f16` (T5, \*R050) | ✓ | ✓ | the clean ISA cells |
| `gather_quantize` (T2/T4, strided \*R025), `pack_s8_s4` (\*R0125) | ✓ | none | excluded from the avx2 arm; EPYC's `Auto` ran these **Scalar** |

`Auto` ≡ AVX512 for every variant kernel on this host (`resolveFor`), so
the V1 session (#104) doubles as an avx512-arm cross-check.

## 2. Results against the pre-registered expectations

**E1 — gather rows identical across arms: CONFIRMED.** Matrix A-side
avx2/avx512 effective-GB/s ratios (best chunk): transpose 0.65–1.07,
blocked 0.89–1.04 — every cell off unity traces to a row flagged
`unstable` (IQR > 5%, the known small-N WSL2 pattern); stable-row cells
sit at 0.98–1.03. Kernel-level: `gather_f32` avx2/avx512 = 1.00 (T=8 and
T=1). The T1b gather wall has no vector-ISA component *in this codebase*
— R1's "AVX2-only host" wording is a misattribution as to ISA; the
correct variable is the host memory system (see E2).

**E2 — T3 `quantize_pack`: ISA is NOT the driver (attribution flips).**
Pre-registered rule: this box's avx2 ≥ 0.9× its avx512 → not ISA-driven.
Measured: 1.01 / 0.99 (identity/blocked rooflines, T=8); matrix quant A
cells 0.98–1.11. Even at **T=1** (compute-bound regime) avx2/avx512 =
**1.08** — Zen4 executes AVX-512 double-pumped on 256-bit datapaths, so
the wider ISA buys ~nothing here at any thread count. The EPYC-AVX2 23.2
→ Zen4 38.4 GB/s gap is **host generation (memory system/core), not
vector ISA**.

**E3 — T5 `convert_f32_f16`: same verdict.** avx2/avx512 = 0.97–0.99 at
T=8, **1.17 at T=1** (avx2 faster). Not ISA-driven.

**E4 — r\* barely moves with ISA: CONFIRMED.** Shared-anchor r\* fits
(`v2_isa_rstar_{avx512,avx2}.json`):

| family | avx512 r\* | avx2 r\* | note |
|---|---|---|---|
| quant | 0.597 | 0.579 | 3% apart; speedup curves within 4% at every shared r |
| blocked_transpose | 0.396 | < 0.5 (unbracketable) | avx2 arm has no r=0.25 point (`gather_quantize` has no AVX2 path); curves agree at r=0.5 (0.760 vs 0.718) and r=1.0 (0.625 vs 0.644) |

(avx512-arm blocked r\* 0.396 also cross-checks V1's 0.374 from the same
box a day earlier — session variance on a flat crossing.)

## 3. Attribution statement (issue #96 acceptance)

- **Supported**: R1/R3's *substantive* claim that a stronger transform
  host flips the CPU-bound outcomes — the 7800X3D really is 1.7–8×
  faster per kernel than the EPYC.
- **Refuted as stated**: the attribution of that difference to **AVX-512
  vs AVX2**. For every kernel with both paths, the two ISAs are equal on
  this host (0.97–1.03 at T=8; avx2 even faster at T=1). For the gather
  kernels the codebase has no ISA paths at all, so "AVX2-only host
  hitting a gather wall" was never a vector-ISA statement. Reports should
  say **host generation (memory system / core)**, not "AVX-512".
- **Untestable by dispatch**: `gather_quantize` / `pack_s8_s4` exist only
  as AVX512 (and Scalar); no AVX2 comparison is possible anywhere.

## 4. Instability disclosure

Same WSL2 pattern as R2/V1: flagged rows concentrate at N=2048 and in
method-a rsweep rows at N=16384. The four family-level r\* fits carry
`unstable=true` from contributing rows. Because both arms are equally
affected and the conclusion rests on *contrasts* (which are 0.97–1.03
everywhere), no stabler-preference rerun was spent in this draft pass;
if any final verdict cell becomes borderline after the gen3 arm lands,
rerun per the R2 rule before merging.

## 5. Runbook — gen3 / 2080 Ti box (EXECUTED 2026-07-29, see §5a)

On the 2080 Ti server, branch `v2-isa-isolation`:

```sh
# 1. build (Turing = sm_75; per-TU ISA flags matter, README recipe)
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX2.cpp -mavx2 -mfma -mf16c -o QuantAVX2.o
g++ -O3 -DNDEBUG -std=c++17 -fPIC -DRELOC_QUANT_HAVE_X86_SIMD=1 \
  -Ilibreloc/include -c libreloc/quant/QuantAVX512.cpp -mavx512f -mavx512bw -o QuantAVX512.o
nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -DRELOC_QUANT_HAVE_X86_SIMD=1 -Ilibreloc/include -Ibench \
  bench/rtrack/rtrack_bench.cu libreloc/src/*.cpp libreloc/quant/Quant.cpp \
  QuantAVX2.o QuantAVX512.o libreloc/cuda/CudaBackend.cu libreloc/cuda/CudaKernels.cu \
  -o bench-rtrack -Xcompiler -pthread
g++ -O3 -DNDEBUG -std=c++17 -DRELOC_QUANT_HAVE_X86_SIMD=1 -Ilibreloc/include -Ibench \
  bench/rtrack/cpu_rooflines.cpp libreloc/src/*.cpp libreloc/quant/Quant.cpp \
  QuantAVX2.o QuantAVX512.o -o bench-cpu-rooflines -pthread

# 2. session ritual (bare metal): performance governor, persistence mode,
#    pin to the GPU's NUMA cores (docs/m0-2080ti-bringup.md if rebel-gpu1)

# 3. set the machine slug in the prepared configs
M=<machine-slug>   # e.g. epyc7351-2080ti
sed -i "s/gen3-2080ti-SETME/$M/; s/_MACHINE/_$M/" bench/rtrack/configs/v2_isa_gen3_*.json

# 4. calibration (records the real link state under load)
python3 bench/rtrack/calibrate.py --out calibration.json --load-bin ./bench-rtrack

# 5. arms — depends on the host:
#    AVX-512 host (grep avx512f /proc/cpuinfo): run all six configs.
#    AVX2-only host (e.g. rebel-gpu1 EPYC): the avx512 arm is impossible —
#    run only the *_avx2 and *_bfair configs and record the empty cell.
for cfg in bench/rtrack/configs/v2_isa_gen3_*.json; do
  python3 bench/rtrack/run_rtrack.py --config "$cfg"
done

# 6. per-ISA rooflines (add avx512 runs if the host supports it)
for T in 8 1; do
  ./bench-cpu-rooflines --kernel all --plan identity --n 8192 --threads $T \
    --variant avx2 --json bench/results/v2_rooflines/identity_avx2_t${T}_$M.json
done
./bench-cpu-rooflines --kernel all --plan blocked --n 8192 --threads 8 \
  --variant avx2 --json bench/results/v2_rooflines/blocked_avx2_t8_$M.json

# 7. r* per arm against this box's own b_fair anchor
python3 bench/rtrack/figure_rstar.py \
  --csv bench/results/v2_isa_gen3_rsweep_avx2_$M.csv bench/results/v2_isa_gen3_bfair_rsweep_$M.csv \
  --rooflines bench/results/v2_rooflines/*_avx2_t8_$M.json \
  --b-method b_fair --json bench/results/v2_isa_gen3_rstar_avx2_$M.json \
  --out bench/results/v2_isa_gen3_figure_rstar_avx2_$M.png

# 8. copy calibration.json to bench/results/v2_isa_gen3_calibration_$M.json,
#    commit CSVs/JSONs/figures onto this branch, push.
```

## 5a. Gen3 row — executed 2026-07-29 on `rebel-gpu1`

Runbook §5 followed as written (session: performance governor,
persistence mode ×4, pinned to GPU0's NUMA cores 4-7,20-23; fresh
calibration `v2_isa_gen3_calibration_epyc7351-2080ti.json`, gen3 x16
under load, b_fair anchor H2D 13.08 GB/s — matching V1's 13.07). Host is
the EPYC 7351: **no `avx512f`**, so per the runbook's step-5 fallback
only the `*_avx2` and `*_bfair` configs ran (4 arms, every config
bit-exact verified, 0 failures) and the gen3×avx512 cell is recorded as
**impossible on this host** rather than silently absent. `Auto` ≡ AVX2
for `quantize_pack`/`convert_f32_f16` on this host, so R1/V1's committed
auto-arm numbers double as the avx2-arm cross-check (rooflines agree:
quantize_pack 22.76 here vs R1's 23.21 GB/s at T=8).

**Gen3 avx2 r\*** (vs this box's own b_fair anchor,
`v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json`): **quant 0.636**;
**blocked_transpose: no crossing** — A/B_fair is 0.50 at r=0.5 and 0.68
at r=1.0, i.e. Method A loses at every measured r on this host (the R1
gather wall; shipping fewer bytes cannot rescue a CPU-bound transform,
so the curve moves *away* from 1.0 as r shrinks).

**The cross-box contrast at FIXED ISA** (both arms avx2, per-kernel
stage rooflines, in-GB/s):

| kernel | gen3 EPYC, T=8 | gen4 7800X3D, T=8 | ×gap | ×gap at T=1 |
|---|---|---|---|---|
| gather_f32 (no SIMD path) | 11.47 | 17.15 | 1.49 | 1.99 |
| quantize_pack | 22.76 | 38.44 | 1.69 | 2.03 |
| convert_f32_f16 | 17.26 | 26.75 | 1.55 | 2.05 |
| contig_read (no SIMD path) | 37.28 | 62.17 | 1.67 | 1.86 |

The 1.5–2.1× gap survives at identical AVX2 dispatch — and is largest at
T=1, where memory latency and per-core width dominate. Combined with the
within-box ISA null (§2), the attribution closes from both directions:
same box + different ISA → no gap; same ISA + different box → full gap.

## 6. The 2×2 and final attribution (issue #96 acceptance)

| r\* (quant / blocked) | AVX-512 dispatch | AVX2 dispatch |
|---|---|---|
| gen4 x16 (7800X3D) | 0.597 / 0.396 | 0.579 / <0.5 (unbracketable) |
| gen3 x16 (EPYC/2080Ti) | **impossible — host has no AVX-512** | 0.636 / none (A loses at all measured r) |

Figure 1, re-drawn with series labelled by the isolated variable:
`bench/results/v2_figure1_isolated_variables.png`
(`bench/rtrack/v2_figure.py`). Same-color pair = ISA contrast at fixed
box/link (curves superimposed — the null); color change = box contrast
at fixed ISA (the cross-box row, honestly labelled link+host-generation
per the recorded plan change).

**Final attribution table:**

| R1/R3 claim | verdict after V2 |
|---|---|
| "A stronger transform host flips the CPU-bound outcomes" | **Supported** — 1.5–2.1× per-kernel at fixed ISA; quant r\* moves 0.60→0.64 across boxes |
| "…because it has AVX-512 (vs AVX2)" | **Refuted as stated** — ISA arms are 0.97–1.03 at T=8 on the box that has both; gather kernels have no vector paths at all |
| the correct wording for reports | **host generation (memory system / core)**, optionally noting Zen4 double-pumps AVX-512 so the wider ISA is not the mechanism |
| link generation as the r\*-mover | **Weakly moving on quant** (0.60 gen4 → 0.64 gen3 measured, direction consistent with the halved link raising B's cost more than A's), but cross-box confounded — a single-host link downgrade remains the open follow-up if BIOS access materializes |

**Untestable by dispatch** (unchanged from §3): `gather_quantize` /
`pack_s8_s4` exist only as AVX512/Scalar — no AVX2 comparison is
possible anywhere; on this gen3 host they run Scalar via `Auto`.
