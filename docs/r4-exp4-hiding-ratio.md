# R4 / EXP-4 — Hiding-ratio model validation on Turing (issue #85)

**Verdict: model validated. On the 2080 Ti under Gen3 PCIe, every kernel in
the R0.2 transform set hides under the transfer (predicted and observed),
and the isolated kernel time predicts the in-pipeline exposure to 1.3–2.5 %
— inside the ±20 % bar.**

Measured 2026-07-27 on `rebel-gpu1` (bare-metal EPYC 7351, 1× RTX 2080 Ti,
PCIe gen3 x16), performance governor, persistence mode, pinned to GPU0's
NUMA cores. Tool: `bench-hiding-ratio` (`bench/rtrack/hiding_ratio.cu`),
model: `bench/rtrack/hiding_model.py`. Raw data + artifacts in
`bench/results/r4_*`. Reproduce the analysis with
`python3 bench/rtrack/hiding_model.py --json bench/results/r4_hiding_ratio_epyc_2080ti.json --pipeline-csv bench/results/r1_gen3_nsweep_epyc_2080ti.csv`.

## The model

A Method-B GPU transform moves `m` tensor-passes through HBM, so it runs in
`m·S/HBM_BW`; the PCIe transfer feeding it takes `S/PCIe_BW`. The transform
hides under the transfer iff

```
    m < ratio = HBM_BW / PCIe_BW.
```

`HBM_BW` is the measured `copy_f32` ceiling; the per-kernel multiplier is
`m = HBM_BW / kernel_BW` on a read+write traffic basis (so `copy_f32` is
`m = 1.0` by construction). `PCIe_BW = 13.06 GB/s` (M0/R1 measured Gen3).

## Turing kernel bandwidths (the literature-gap fill)

Median GB/s (read+write traffic), 5 warmup + 30 timed, IQR/median < 1.5 %:

| kernel | N=8192 | N=16384 | m (N=16384) |
|---|---|---|---|
| `copy_f32` (HBM ceiling) | 544.6 | 543.0 | 1.00 |
| `transpose_smem_padded` (tile[32][33]) | 386.1 | 380.2 | 1.43 |
| `transpose_smem_unpadded` (tile[32][32]) | 375.0 | 371.5 | 1.46 |
| `relocate_naive_f32` (1 thread/elem) | 172.6 | 182.8 | 2.97 |
| `scatter_random` blk=1 (identity) | 278.6 | 280.1 | 1.94 |
| `scatter_random` blk=1024 | 262.1 | 279.0 | 1.95 |
| `scatter_random` blk=1 Mi-elem | 82.4 | 84.6 | 6.42 |
| `scatter_random` blk=full (max entropy) | 20.5 | 20.2 | 26.84 |

Measured HBM ceiling **544 GB/s = 88 % of the 616 GB/s theoretical peak**
(7000 MHz × 352-bit), stable across N. No published 2080 Ti transpose GB/s
existed; **386 GB/s padded / 375 GB/s unpadded** is the novel datum.

Two honest readings for the paper:

- **Bank-conflict cost is size-dependent.** Padded vs unpadded SMEM
  transpose differ only ~3 % at N ≥ 8192 (386 vs 375) — at these sizes the
  transpose is DRAM-bound and the SMEM conflicts hide behind memory
  latency. At the small N=2048 smoke the gap was ~30 % (431 vs 302), where
  the kernel is SMEM/compute-bound. The +1 pad matters at small tiles, not
  at the R-track's working sizes.
- **The entropy sweep spans a 14× bandwidth collapse** (280 → 20 GB/s) as
  scatter locality falls from identity to a full random permutation — the
  write-amplification curve that sets the scatter multiplier.

## Hiding ratio and verdicts

Measured `ratio = 544 / 13.06 ≈ 41.7` (theoretical `616 / 13.06 ≈ 47.2`).

**Every kernel hides**: the largest multiplier in the set is the full-random
scatter at `m = 26.8`, still below the ratio of ~42. So on Turing under a
Gen3 link, the GPU-side transform is *never* the bottleneck — the transfer
always dominates. This is the quantitative explanation of the "Method B is
flat" observation from sym#63: the hiding ratio is large enough (~42) that
the entire R0.2 kernel set, up to and including a maximally-uncoalesced
random scatter, is absorbed under the PCIe transfer. The model also says
where hiding *would* break: a kernel above ~42 copy-passes, which none of
these are.

## Validation against the full Method-B pipeline (±20 % bar)

`transpose_smem_padded` is the identical kernel (`relocateF32` tiled path)
that pipeline T1 Method B runs on the identical rank-2 transpose plan, so it
is the clean prediction-error check:

| N | isolated kernel | pipeline `gpu_kernel_ms` | error | verdict |
|---|---|---|---|---|
| 8192 | 1.391 ms | 1.373 ms | **1.3 %** | PASS |
| 16384 | 5.648 ms | 5.793 ms | **2.5 %** | PASS |

In both, the GPU kernel is 6–7 % of the transfer time — hidden, as the
model predicts (`m ≈ 1.4 ≪ 42`).

**Nsight Systems trace** (`bench/results/r4_methodb_trace_2080ti.nsys-rep`,
digest in `.stats.txt`), Method-B T1 N=8192, 8 iterations: GPU spent
**14.17 ms in the transpose kernel vs 226.6 ms in H2D memcpy** — the kernel
is 6 % of transfer, independent visual/quantitative confirmation of the
hiding claim.

(`relocate_naive_f32` here runs on the rank-2 transpose plan — the
single-element-gather worst case, a roofline floor — which the pipeline
never uses for Method B, so it is not part of the ±20 % match; it is
reported as a bandwidth data point only.)

## Deferred

- **4070 Ti SUPER (ratio ≈ 28).** Adding `copy_f32` + `relocate_naive` there
  is mostly a by-product of the R1/R2 kernel runs; it lands with R2 on the
  Gen4 box (not this machine). The Ada ratio ≈ 28 is still ≫ every
  multiplier here, so the qualitative verdict (everything hides) is expected
  to hold; the point of the Gen4 run is the crossover, not the hiding floor.

## Caveats

- Single 2080 Ti, GPU0, pinned; bare-metal (not WSL2), a clean Turing point.
- The ratio uses the measured `copy_f32` ceiling (544) rather than the
  datasheet 616 — the honest achievable HBM number; both are reported.
