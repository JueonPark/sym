# Claim ledger

Formalizes Build Document v3 Appendix C (issue [#117](https://github.com/JueonPark/sym/issues/117)): after three baseline
generations (staged `b` → `b_fair` [#95](https://github.com/JueonPark/sym/issues/95) → `b_pipelined` [#108](https://github.com/JueonPark/sym/issues/108)), this table is
the single authoritative place a claim's current number and standing live.
Quote claims from here, not from the frozen experiment reports — those keep
their as-measured history, including numbers later withdrawn.

**Status vocabulary** (this document is the definition): `survives` — the
claim holds as originally stated under the strongest baseline; `narrowed` —
the direction holds but the originally quoted margin/bar does not;
`withdrawn` — the number must not be quoted (baseline artifact);
`refuted-as-stated` — the claim's direction failed under a fair baseline.

**A/B convention**: ratio of best effective-input GB/s per method
(equivalently t_B/t_A of best-chunk medians), stabler-preference merged
(docs/r2-exp2-gen4-crossover.md:48-58) for all `b_pipelined` figures —
derived from bench/results/cm5_eval_report.json's regen-locked machinery.

## The ledger

| claim | box | staged `b` | `b_fair` | `b_pipelined` | status | authoritative source |
|---|---|---|---|---|---|---|
| R1-G2: dtype reduction wins ≥1.5× (quant) | Gen3 | 1.74–2.43× PASS | 1.44/1.43/1.49/1.53× (issue [#95](https://github.com/JueonPark/sym/issues/95), comment 2026-07-28) | 1.40–1.48× (below bar at every N) | narrowed | docs/r1-exp1-gen3-gates.md §BP restatement; boundary-law row below |
| R2-G2: dtype reduction wins ≥1.5× (quant) | Gen4 | 4.18/3.66/2.12/2.12× (R2); 3.23/3.58/2.21/2.24× (V1 session) | 2.20/2.60/1.46/1.54× | 1.94/2.39/1.52/1.42× (below bar at N=16384) | narrowed | docs/r2-exp2-gen4-crossover.md §BP restatement |
| R2-G4: pure relocation A/B ∈ [0.40, 0.80] (blocked_transpose) | Gen4 | **1.40/1.20/1.19/0.91× — WITHDRAWN, do not quote** (staged-baseline artifact; V1 found the baseline inadmissible) | 0.65–0.87× (direction reversed, in-band at N ≥ 8192) | 0.88/0.77/0.76/0.63× (loss direction confirmed, deeper at N ≥ 4096; marginally shallower at N=2048, 0.8796 vs 0.874) | withdrawn (staged number) / narrowed (loss direction holds; in-band at N ≥ 8192 (b_fair) / N ≥ 4096 (b_pipelined)) | docs/r2-exp2-gen4-crossover.md §§V1 + BP restatements |
| r\* (blocked_transpose) | Gen4 | 0.499 (R2 rsweep, Jul 22) / 0.604 (V1-session staged) | 0.374 (V1 rsweep) / 0.3956 (BP rsweep, serial; pred 0.3247) | 0.3605 (BP rsweep, overlapped; pred 0.2914) | narrowed (each column is a DIFFERENT dataset — see note) | r2 doc r\*-tables; cm5_eval_report.json rstar_rows |
| r\* (quant) | Gen4 | 0.992 (R2) / none (V1 session) | 0.541 (V1 rsweep) / 0.6107 (BP rsweep, serial; pred none — one-sided) | 0.6150 (BP rsweep, overlapped; pred none — one-sided) | narrowed (dataset caveat as above) | same |
| r\* (quant) | Gen3 | — | 0.6356 (V2 rsweep, vs b_fair — docs/v2-isolation.md) / 0.7025 (BP rsweep, serial; pred none — one-sided) | 0.7024 (BP rsweep, overlapped; pred 0.9966, Δ 0.2942) | narrowed (dataset caveat) | docs/v3-costmodel.md; cm5_eval_report.json rstar_rows |
| Boundary law: A/B ≈ BW_cpu/BW_link at the largest N | both | — | — | Gen3 1.4817, Gen4 1.4153 (quant, N=16384) | survives | this document, section below |
| cost-model v1 quality (MISCLASS / RSTAR rule-v1 / REGRET-p90, all-cells + held-out) | both | — | (Serial) MISCLASS PASS (2/48=0.0417; held-out 1/24=0.0417); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (serial max\|Δ\|=0.0709 + 2 one-sided mismatches; overall FAIL) | (Overlapped) MISCLASS PASS (2/40=0.0500; held-out 1/20=0.0500); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (overlapped max\|Δ\|=0.2942 + 1 one-sided mismatch; overall FAIL) | narrowed | `bench/results/cm5_eval_report.json` ([#113](https://github.com/JueonPark/sym/issues/113)) |
| cost-model v2 quality (post-hoc refinement — MISCLASS / RSTAR rule-v1 / REGRET-p90) | both | — | (Serial) MISCLASS PASS (2/48=0.0417; held-out 1/24=0.0417); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (serial max\|Δ\|=0.0768 + 2 one-sided mismatches; overall FAIL) | (Overlapped) MISCLASS PASS (2/40=0.0500; held-out 1/20=0.0500); REGRET-p90 PASS (0.0000, all four splits); RSTAR (rule v1) FAIL (overlapped max\|Δ\|=0.0673 + 2 one-sided mismatches, was 1 under v1; overall FAIL) | narrowed | `bench/results/cm6_eval_report.json` ([#125](https://github.com/JueonPark/sym/issues/125)); post-hoc, not pre-registered |

**r\* dataset note**: the three r\* columns come from three different
measurement campaigns (R2 rsweep → V1 rsweep → BP rsweep), not re-fits of
one dataset; cross-column movement mixes baseline change with session
variance. Each cell is labeled accordingly.

## Fused host stacks (`torch.stack`, 2026-10)

| claim | box | measured | status | authoritative source |
|---|---|---|---|---|
| A host `torch.stack(xs, dim).to("cuda")` fused by `RelocBackend()` with its default transfer options (one gather thread, pageable staging) is faster than the same code in eager PyTorch and Inductor at >= 32 MiB of inputs, the default `min_stack_bytes` | EPYC 7351 / RTX 2080 Ti, PCIe Gen3 | FBD (default transfer options; 8 Torch threads on CPUs 4–7 and 20–23, warm retained resources): 0.212–0.941× of min(B1, B3) in every configuration at >= 32 MiB (gate 4, default options: 32 MiB); narrowest for two-input last-dim stacks, 0.938×, 0.941× and 0.916× at 32, 64 and 128 MiB; at 24 MiB, 1.013–1.672× in eight of nine configurations | survives | bench/results/stack-fusion/README.md |
| The same stack fused with the tuned options `transfer_options={"pinning": "auto", "min_pinned_bytes": 8 << 20, "gather_threads": 8}` is faster than eager PyTorch and Inductor at >= 24 MiB of inputs | same | FB (tuned options; same threads, CPUs and resources): 0.122–0.914× of min(B1, B3) in every configuration at >= 24 MiB (gate 3 at >= 32 MiB; gate 4, tuned options: 24 MiB) | survives | same |
| ...and faster than rewriting it as copy-then-GPU-stack | same | FB (tuned options) at >= 24 MiB: 0.980–1.714× of min(B2, B4), below 1× only for 16-input last-dim stacks at 24, 32 and 64 MiB (0.980×, 0.995×, 0.986×); FBD (default options) at >= 32 MiB: 1.554–3.847× | refuted-as-stated | same |
| The input-pointer table costs nothing significant against a single-source transfer of the same bytes, >= 8 MiB | same | FD/S1 with the tuned options: dims 0 and 1, 0.992–1.058× (gate 2); last dim (dim 2), 1.041–1.123×, above the 10% bar at 16 MiB N=4 (1.123×); one disturbed configuration left out (16 MiB N=16 last dim, per the source's Deviations) | narrowed | same |

Part of the as-written margin depends on the host allocator: PyTorch
builds a fresh pageable stacked host tensor on every call, and glibc serves
allocations of 32 MiB and more with a new `mmap`, so each call page-faults it.
Allocators that reuse memory shrink that part of the margin.

## Boundary law — the headline

"host-side transform wins by the margin host memory bandwidth exceeds link
bandwidth — a margin that shrinks each PCIe generation."

| box | BW_cpu | BW_link (pinned, under load) | law BW_cpu/BW_link | measured A/B_pipelined (quant, N=16384) | residual |
|---|---|---|---|---|---|
| Gen3 | 23.2 ([#108](https://github.com/JueonPark/sym/issues/108) pre-registered) / 23.01 (r1 rooflines, quantize_pack t8) | 13.06 (R1 gate figure; BP session read 13.08) | ≈1.78 / ≈1.76 | 1.4817 | −17% / −16% |
| Gen4 | 38.4 ([#108](https://github.com/JueonPark/sym/issues/108) pre-registered) / 36.61 (r2 rooflines, quantize_pack t8) | 26.87 (BP session; [#108](https://github.com/JueonPark/sym/issues/108) used 26.9) | ≈1.43 / ≈1.36 | 1.4153 | −1% / +4% |

Both BW_cpu readings are shown — [#108](https://github.com/JueonPark/sym/issues/108)'s pre-registered arithmetic verbatim
and the committed-roofline recomputation — precisely so no post-hoc choice
between them can tune the residual.

1. The margin shrink is measured (Gen3 1.48 → Gen4 1.42), matching the
   pre-registered direction.
2. Gen3's −16/−17% residual is a finding, not hidden: the naive law's
   BW_cpu term (isolated roofline) overstates the pipeline's effective CPU
   bandwidth. BP-G3's fair-baseline-corrected predictions (÷1.06:
   1.35–1.44) sit 3–5% below the measured 1.40–1.48 at every N — a
   residual ~4x smaller than the naive law's −17% and of the opposite
   sign — so the law's shape holds with BW_cpu read as *pipelined
   effective* bandwidth, recorded as a definition refinement (no number
   adjusted; not a refit). Gen4's V-cache makes the two readings nearly
   coincide (−1/+4%).

Status: **survives** — both-box consistency in the pre-registered form;
the track's closing claim, in contrast to the narrowed G2 rows above.

## Machinery

| claim | boxes | result | status | authoritative source |
|---|---|---|---|---|
| Same folded plan, runtime symbol-bind, bind-time auto-placement — correct choice on both boxes, no recompilation | Gen3 + Gen4 | 23/24 decisions match measured winners (1 small-N miss disclosed); r=0.25 row flips correctly (Gen3 `b` ×4, Gen4 `a` ×4); artifacts byte-equal CI regeneration | survives | `docs/r6-crossbox-bind.md`; `bench/results/r6_bind_demo_*.json` ([#87](https://github.com/JueonPark/sym/issues/87)) |

## Engineering completion is not a research claim

The finalization (issues [#131](https://github.com/JueonPark/sym/issues/131)–[#133](https://github.com/JueonPark/sym/issues/133), completed 2026-09-24,
[runtime-integration.md](runtime-integration.md)) delivers a correct,
reproducible compiler-to-runtime path. It adds no row above and changes no
status: [#63](https://github.com/JueonPark/sym/issues/63)'s performance criterion stays unmet, [#73](https://github.com/JueonPark/sym/issues/73)'s withdrawn and
narrowed claims stand, and [#88](https://github.com/JueonPark/sym/issues/88)'s regime-5 result stays `narrowed`. The
integration's own end-to-end latency is recorded descriptively in
`runtime-evidence/cuda.json`; in that configuration the supported blocking
path is slower than PyTorch's copy, and no claim is made from it.

## Post-freeze addenda (measured after the 2026-09-02 freeze; never quoted in the main eval)

| claim | box | result | status | authoritative source |
|---|---|---|---|---|
| Regime 5: concurrent GPU compute moves the A/B margin toward A (B's recv kernel contends; A's host transform does not) | Gen3 | R7-G2 FAIL overall (1/5 pairs): `quant` passes decisively at both eligible C≥1 pairs (delta_b/delta_a ≈ 1.92x at repeats=9, ≈2.26x at repeats=17); `blocked_transpose` (r=1.0, identical bytes both methods) sits at the wall-time noise floor (wall_b/wall_a = 0.999–1.002) and flips at one of three eligible pairs (repeats=7: delta_a=41.41 > delta_b=40.27) | narrowed (holds decisively for dtype-reduction families; null at the pure-relocation r=1.0 noise floor) — refuted-as-stated as a strict universal-direction gate | docs/r7-e2e-overlap.md; bench/results/r7_e2e_quant_epyc7351-2080ti.csv, bench/results/r7_e2e_blocked_transpose_epyc7351-2080ti.csv ([#88](https://github.com/JueonPark/sym/issues/88)) |
