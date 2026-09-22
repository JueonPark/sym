# R7 — End-to-end overlap, Regime 5 (issue #88)

**POST-FREEZE ADDENDUM.** Measured 2026-09-22, after the 2026-09-02 eval
freeze, per #73's own W8 placement of R7 ("Writing + R7 if time"). Every R7
CSV row carries `post_freeze=1`; no frozen CSV or figure was touched; the
paper's main eval quotes pre-freeze data only. Gates (`gates.py --exp r7`)
were committed and registered on #73 before this data existed — commit
order on branch `r7-e2e-overlap`: `843a567` (gates) precedes `db3fc35`
(this session's CSVs + gate report).

## Question

R4 validated kernel hiding on an IDLE GPU
(`docs/r4-exp4-hiding-ratio.md`). In a weight-loading loop the GPU also
computes: Method B's receive kernel contends with model compute for
SMs/HBM, while Method A's host transform contends with nothing on the
GPU. Does concurrent compute move the A/B margin toward A?

## Setup

Tool: `bench-e2e-overlap` (`bench/rtrack/e2e_overlap.cu`) — an L-layer
weight-loading loop with per-layer cuBLAS GEMM compute concurrent to the
load. `N=8192`, `L=16` layers, `gemm_batch=1024`, box `epyc7351-2080ti`
(bare-metal EPYC 7351, RTX 2080 Ti, PCIe Gen3).

Two families, differing only in wire dtype (`r` = wire bytes / f32 bytes)
and GEMM precision:

| family | r | wire dtype | GEMM | plan |
|---|---|---|---|---|
| `quant` | 0.25 | int8 | `cublasGemmEx` (`CUDA_R_8I`/`CUBLAS_COMPUTE_32I`) | contiguous copy+quant |
| `blocked_transpose` | 1.0 | f32 | `cublasSgemm` | blocked transpose (pure relocation) |

Methods (loader × overlap):

| method | loader | overlap | notes |
|---|---|---|---|
| `a` | host transform (CPU), zero GPU kernels | yes | Method A |
| `b_pipelined` | raw f32 H2D + device receive kernel | yes | Method B; two-stream design (BP1) |
| `a_prefold` | DMA of a pre-transformed pinned wire image | yes | V4 semantics, no host transform on the hot path |
| `a_serial` | same as `a` | no (load then compute) | G1 no-overlap anchor |
| `b_serial` | same as `b_pipelined` | no | G1 no-overlap anchor |
| `compute_only` | none (pipeline scaffolding, no loads) | — | G3a anchor |
| `compute_bare` | none (plain GEMM loop, no scaffolding) | — | G3a anchor |

**Per-layer recv granularity.** `b_pipelined`'s receive kernel launches
once per LAYER, not per chunk — no sub-plan slicing. Cross-layer overlap
of recv/H2D/compute is what Regime 5 probes, and BP's `quant` best chunk
was already monolithic (`bench/rtrack/e2e_overlap.cu` header comment);
G3b checks the per-layer cost still matches the frozen BP single-shot
within +10%.

**C sweep.** The overlap ratio `C = compute_only_time / load_only_time` is
targeted at {0.5, 1, 2, 4} via a `repeats` (GEMMs-per-layer) grid computed
per family from a quick timing pass; because `compute_only`/`load_only`
differ slightly by method, the *measured* `C` at a given `repeats` value
differs slightly between `a` and `b_pipelined` (e.g. `quant` `repeats=9`
measures `C=1.860` for `a` vs `C=1.267` for `b_pipelined`) — this is
expected, not a bug, and is exactly why gates key on `measured_C`, not on
`repeats`.

**Ritual / manifest** (from the CSV headers, both files identical):
bare-metal, `governor: performance` (sysfs already read `performance`, no
change made — `cpupower` binary not installed), `persistence: Enabled`
(re-confirmed via `sudo nvidia-smi -pm 1`), `pinned: numactl
--physcpubind=4-7,20-23 --membind=1` — **corrected from the brief's
`--membind=0`**: GPU0's die is NUMA node 1 (62 GiB local DRAM); node 0 has
0 MB local DRAM per `docs/m0-2080ti-bringup.md`, and the literal
`--membind=0` fails outright (`node argument 0 is out of range`, exit 1,
no rows). Binary: standalone nvcc `sm_75` build (recipe below), rebuilt
from HEAD `843a567`.

**Protocol.** 5 warmup + 30 timed iterations per config
(`bench/rtrack/rstats.h`); CSV reports median/min/p95 and
`iqr_over_median_pct` (0 rows flagged `unstable` across both files).

**Verification.** Every full-method row (`a`, `b_pipelined`, `a_prefold`,
`a_serial`, `b_serial`) was run with `--verify`: a `memcmp`-exact check of
the last-written weight slot against the host reference wire image, plus
the GEMM output against a reference `y`
(`bench/rtrack/e2e_overlap.cu:593-609`). All 53+53 rows carry
`verified=1`. The `*_load_only` anchor rows also carry `verified=1`, but
that column is written unconditionally `true` for them regardless of
`--verify` (`e2e_overlap.cu`, the `_load_only` `writeRow` call) — they are
a **timing-only** pass, not an independently re-checked weight load.
`a_prefold`'s own verify is close to tautological: it DMAs the reference
wire directly, so the check confirms the DMA copied correctly, not that
the prefold transform (computed once, off the hot path) was correct.

## Verdicts (`gates.py --exp r7`, bars pre-registered)

Quoted verbatim from `bench/results/r7_gate_report.txt`:

```
R7-G1: PASS  ({'n_cells': 8, 'fails': []})
R7-G2: FAIL  ({'n_pairs': 5, 'fails': [('blocked_transpose', '7', 41.41000000000008, 40.26999999999998)]})
R7-G3a: PASS  ({'n': 8, 'fails': []})
R7-G3b: PASS  ({'n': 10, 'fails': []})
R7 OVERALL: FAIL
```

Gate bars (`bench/rtrack/gates.py`, fixed before this data existed):
G1 `wall(overlapped) <= 0.75 x wall(serial)` on cells with `measured_C` in
`[0.5, 2.0]`; G2 `delta_b > delta_a` per `(family, repeats)` best-chunk
pair where **both** cells' `measured_C >= 1.0` (`delta_m = wall_m -
max(load_only_m, compute_only_m)`); G3a `|compute_only - compute_bare| /
compute_bare <= 5%`; G3b per-layer `load_only` within +10% of the frozen
BP single-shot best-chunk median (or, for `a_prefold`, within +10% of
`wire_bytes / pcie.h2d_gbps` from calibration).

## Results

Best-chunk (min `median_ms` across `chunk_req_mib` at each `repeats`)
tables, generated from the two committed CSVs:

### `quant` (r=0.25, int8 GEMM)

| method | repeats | measured_C | wall (ms) | load_only (ms) | compute_only (ms) | delta (ms) | exposed_load (ms) |
|---|---|---|---|---|---|---|---|
| a | 2 | 0.409 | 224.375 | 218.766 | 89.377 | 5.608 | 134.998 |
| a | 4 | 0.830 | 230.801 | 218.766 | 181.555 | 12.035 | 49.246 |
| a | 9 | 1.860 | 434.115 | 224.221 | 416.951 | 17.164 | 17.164 |
| a | 17 | 3.582 | 815.928 | 224.221 | 803.131 | 12.797 | 12.797 |
| b_pipelined | 2 | 0.272 | 334.204 | 329.167 | 89.377 | 5.036 | 244.827 |
| b_pipelined | 4 | 0.552 | 339.340 | 329.167 | 181.555 | 10.173 | 157.785 |
| b_pipelined | 9 | 1.267 | 449.865 | 329.167 | 416.951 | 32.914 | 32.914 |
| b_pipelined | 17 | 2.440 | 831.998 | 329.167 | 803.131 | 28.867 | 28.867 |
| a_prefold | 2 | 1.088 | 98.526 | 82.130 | 89.377 | 9.149 | 9.149 |
| a_prefold | 4 | 2.211 | 192.756 | 82.130 | 181.555 | 11.200 | 11.200 |
| a_prefold | 9 | 5.077 | 429.990 | 82.130 | 416.951 | 13.039 | 13.039 |
| a_prefold | 17 | 9.779 | 809.778 | 82.130 | 803.131 | 6.647 | 6.647 |

### `blocked_transpose` (r=1.0, SGEMM, pure relocation)

| method | repeats | measured_C | wall (ms) | load_only (ms) | compute_only (ms) | delta (ms) | exposed_load (ms) |
|---|---|---|---|---|---|---|---|
| a | 2 | 0.698 | 519.339 | 497.443 | 347.266 | 21.897 | 172.073 |
| a | 3 | 1.041 | 584.552 | 517.082 | 538.323 | 46.229 | 46.229 |
| a | 7 | 2.501 | 1334.400 | 517.082 | 1292.990 | 41.405 | 41.405 |
| a | 13 | 4.679 | 2454.010 | 517.082 | 2419.630 | 34.386 | 34.386 |
| b_pipelined | 2 | 1.053 | 399.427 | 329.654 | 347.266 | 52.161 | 52.161 |
| b_pipelined | 3 | 1.631 | 585.801 | 329.992 | 538.323 | 47.478 | 47.478 |
| b_pipelined | 7 | 3.922 | 1333.260 | 329.654 | 1292.990 | 40.266 | 40.266 |
| b_pipelined | 13 | 7.340 | 2455.070 | 329.654 | 2419.630 | 35.443 | 35.443 |
| a_prefold | 2 | 1.057 | 387.036 | 328.496 | 347.266 | 39.770 | 39.770 |
| a_prefold | 3 | 1.639 | 573.235 | 328.496 | 538.323 | 34.912 | 34.912 |
| a_prefold | 7 | 3.936 | 1319.840 | 328.496 | 1292.990 | 26.852 | 26.852 |
| a_prefold | 13 | 7.366 | 2443.510 | 328.496 | 2419.630 | 23.885 | 23.885 |

**C-sweep narrative.** At low C (`quant` `repeats=2,4`), the load is not
fully hidden: `exposed_load_ms` is large (up to 245 ms for
`b_pipelined`) because `compute_only` (89–182 ms) is smaller than
`load_only` (218–329 ms) — the load, not the compute, is the wall-clock
floor. Once `compute_only` exceeds `load_only` (`quant` `repeats=9,17`;
all four `blocked_transpose` repeats except `a` at `repeats=2`),
`exposed_load_ms == delta_ms`: the load is now fully covered by compute
and only the residual (contention, launch, or pipeline-fill overhead)
shows up as `delta`.

**a_prefold vs compute_only.** `a_prefold`'s `delta` (a DMA-only load
racing pure compute, V4 semantics) stays small relative to
`compute_only` at every repeats value. Per family (ms ranges are not
comparable across families, so kept separate): `quant`'s delta ranges
6.6–13.0 ms out of 89.4–803.1 ms of compute; `blocked_transpose`'s delta
ranges 23.9–39.8 ms out of 347.3–2419.6 ms of compute. Expressed as a
percentage of `compute_only` **per row** (not the min/min-vs-max/max of
two different rows used previously), the true range across all 8 rows
is 0.83–11.45%: the low end is `quant` `repeats=17` (6.647/803.131 =
0.83%) and the high end is `blocked_transpose` `repeats=2` (39.770/
347.266 = 11.45%) — both ends of the range sit within a single family,
not stitched together from different families' extremes. Its
`load_only` floor (82.1 ms `quant`, 328.5 ms `blocked_transpose`) is the
lowest of any method in its family, consistent with R4's hiding-ratio
result carried into a loaded-GPU setting.

**A/B e2e ratio vs. the isolated BP ratio.** `quant` `wall_b`/`wall_a` at
`repeats=9,17` (the two G2-eligible pairs) is 449.865/434.115 = 1.036 and
831.998/815.928 = 1.020 (equivalently `wall_a`/`wall_b` = 0.965 and
0.981) — Method A finishes faster end-to-end in both cells, the same
*direction* as the isolated BP-pipelined ratio already on record in
`docs/claim-ledger.md` (Gen3 `quant`, `b_pipelined`: 1.40–1.48x, "below
bar at every N" row).

**The pre-registered magnitude expectation was NOT met.** The spec
pre-registered that "the e2e A/B wall ratio at r=0.25, C≥1 should be ≥
the isolated BP A/B ratio (1.40–1.48)". The measured values — 1.036
(`repeats=9`) and 1.020 (`repeats=17`) — fall far below that band: the
direction is right, the magnitude is not, and this pre-registered bar
was **not met**. The comparison was ill-posed from the start: once
`measured_C >= 1`, `wall` for both methods converges toward
`compute_only` as the GEMM comes to dominate the critical path (at
`repeats=17`, `wall_a`=815.928 ms and `wall_b`=831.998 ms sit only 1.6%
and 3.6% above `compute_only`=803.131 ms respectively), so the *wall*
ratio is compressed toward 1 by construction — it cannot reflect the
isolated transfer-leg advantage measured off the critical path in the
BP track. The *load-leg* ratio, which isolates the transfer cost that
the wall ratio buries under compute, does reproduce the isolated band:
at the same best-chunk configuration used in the Results table for
these two pairs (`b_pipelined` `chunk_req_mib`=256, `a`
`chunk_req_mib`=4), `b_pipelined_load_only`/`a_load_only` =
329.167/224.221 = **1.468** — squarely inside the isolated 1.40–1.48x
band. (Pairing each method's globally-lowest `load_only` instead —
`a` at `chunk_req_mib`=16, 218.77 ms, vs. `b_pipelined`'s
`chunk_req_mib`=256, 329.167 ms, its minimum either way — gives
329.167/218.766 = 1.505, just above the band; both readings put the
load-leg ratio at ~1.5x, well clear of the wall ratio's ~1.02.) The
transfer-leg advantage is intact; it is the e2e wall metric, saturated
by concurrent compute at C≥1, that cannot see it.

## Interpretation

**G2 is the Regime-5 contention claim.** It states: at every C≥1 pair,
`delta_b > delta_a` — i.e., Method B's device receive kernel is exposed
(loses more wall-clock to contention with compute) than Method A's
host-side transform, which touches no GPU resource. **It FAILED overall,
on 1 of 5 pairs.** No retuning was done in response to this result; the
root cause below is diagnostic, not corrective.

- **`quant` (r=0.25): both C≥1 pairs pass decisively.** At `repeats=9`,
  `delta_b=32.914` vs `delta_a=17.164` (`delta_b` ≈1.92x `delta_a`); at
  `repeats=17`, `delta_b=28.867` vs `delta_a=12.797` (≈2.26x). Where
  dtype reduction gives Method A a real byte-count advantage — A ships
  r=0.25 bytes over the wire, B ships raw f32 plus runs a device receive
  kernel that contends with the concurrent GEMM for SMs/HBM — the
  Regime-5 contention direction is confirmed strongly, by roughly a
  factor of two.
- **`blocked_transpose` (r=1.0, pure relocation, both methods ship
  identical bytes): the two methods sit at the wall-time noise floor.**
  `wall_b/wall_a` across all three C≥1 pairs is 585.801/584.552=1.00214
  (`repeats=3`), 1333.260/1334.400=0.99915 (`repeats=7`), and
  2455.070/2454.010=1.00043 (`repeats=13`) — a 0.999–1.002 band. Two of
  the three pairs pass narrowly in the predicted direction
  (`repeats=3`: `delta_b=47.478 > delta_a=46.229`, `repeats=13`:
  `delta_b=35.443 > delta_a=34.386`); the third flips by ~2.7%
  (`repeats=7`: `delta_a=41.405 > delta_b=40.266`, the gate's reported
  fail cell, `41.41` vs `40.27` after floating-point printing). The
  single FAIL cell lives entirely inside this null regime, where neither
  method ships fewer bytes than the other and there is no first-order
  reason to expect either direction to win reliably.
- **G3a PASS and G3b PASS together rule out a ritual/setup artifact as
  the cause.** G3a confirms the pipeline scaffolding adds no measurable
  overhead over a bare GEMM loop (≤1.32% observed, bar 5%); G3b confirms
  the per-layer load cost is consistent with the frozen BP single-shot
  measurement (within 10%). The FAIL is a property of the
  `blocked_transpose` noise floor itself, not of how R7 was run.

**Framing.** The strict all-cells direction gate FAILS. The underlying
physics is confirmed where dtype reduction applies (`quant`, a clean 2x
margin at both eligible C) and is null — neither confirmed nor
contradicted beyond measurement noise — at the pure-relocation r=1.0
floor. This is an honest negative/narrowed result: the Regime-5
contention claim does not hold as a universal, dtype-independent
direction gate, but it is not refuted where it has a mechanism (a real
byte-count difference) to act through.

## Caveats

- **Per-layer recv granularity**: `b_pipelined`'s receive kernel launches
  once per layer, not per chunk (design choice, see Setup); this is the
  granularity G3b validates against the frozen BP data, not a
  finer-grained in-pipeline recv.
- **Single-source weights**: both families reuse one host weight buffer
  across layers/repeats (the fixture's `refWire`), not L independent
  tensors — appropriate for an overlap-timing harness, not a
  representativeness claim about real model weights.
- **Static activation quantization**: the `quant` family's GEMM
  activations are quantized once with a single symmetric whole-tensor
  scale (RNE), not per-layer or per-channel calibrated.
- **Verification is bit-exact for full methods, timing-only for
  `*_load_only` anchors** — see Setup; `a_prefold`'s check is
  near-tautological (see Setup).
- **0 unstable rows**: both CSVs report `unstable=0` (IQR/median ≤ 5%)
  on every row; no targeted reruns were needed this session.
- **Gen4 pending**: no 7800X3D/4070 Ti SUPER data exists yet for R7; see
  the runbook in `bench/rtrack/README.md`.
- **Post-freeze status**: this entire document and its data are a
  post-freeze addendum (see header); nothing here supersedes or is
  quoted by the frozen main-eval numbers.
- **No timeline capture**: this session did not capture an Nsight
  Systems timeline; a trace is optional future work, not a missing
  committed artifact.

## Reproduction

Standalone build: the `bench-rtrack` recipe in `bench/rtrack/README.md`
("Build" section) with `bench/rtrack/e2e_overlap.cu` as the source and
`-lcublas` appended (CMake target: `bench-e2e-overlap`).

Exact session commands (from the committed CSV headers):

```sh
./bench-e2e-overlap --machine epyc7351-2080ti --family quant \
  --csv bench/results/r7_e2e_quant_epyc7351-2080ti.csv --verify

./bench-e2e-overlap --machine epyc7351-2080ti --family blocked_transpose \
  --csv bench/results/r7_e2e_blocked_transpose_epyc7351-2080ti.csv --verify
```

Ritual: pin to the GPU's NUMA node before running (see Setup's `numactl`
note — `--membind=1`, not the brief's `--membind=0`, on this box).

Gates:

```sh
python3 bench/rtrack/gates.py --exp r7 \
  --csv bench/results/r7_e2e_quant_epyc7351-2080ti.csv \
        bench/results/r7_e2e_blocked_transpose_epyc7351-2080ti.csv
```

## Cross-links

#88 (this track), #73 (post-freeze W8 registration comment),
`docs/r4-exp4-hiding-ratio.md` (the idle-GPU hiding-ratio model this
track extends to a loaded GPU), `docs/claim-ledger.md` (Post-freeze
addenda row below; boundary-law and BP-pipelined ratio rows this doc
cites), `docs/v4-prefold.md` (`a_prefold` semantics), BP track
(`docs/claim-ledger.md` Machinery/boundary-law rows;
`bench/rtrack/README.md` §BP3 runbook).
