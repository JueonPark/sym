# Historical completed-request decisions

The region planner selected the fastest observed feasible arm in **16 of 24** historical deployment cases. All 24 primary selections had completed successfully in their original campaigns. The largest loss was **28.11%** when a 32 MiB safety margin excluded packed residency under a real 512 MiB allocation limit. Removing that margin selected a full-resident arm that exhausted device memory in three of three rounds. These are retrospective CPU planner decisions over archived GPU timings. The fresh region benchmark exercises kernel and schedule execution separately.

## Reproduce the comparison

From the repository root, use Python 3.10 or later. The script checks SHA-256 for the original published tables and the code that declares E5 resource configurations. It checks all 30 primary and sensitivity decisions against an independent exhaustive calibration ranking and the tables below. The JSON includes the core plan, its byte floor and rejections, the calibration prices, every target arm, observed OOMs, measured costs, oracle, regret and selected prediction error. Missing measurements have JSON value `null`. An OOM row's peak is the sampled increment at failure.

```sh
python -m bench.issue271.historical --check
python -m bench.issue271.historical --json > /tmp/sym-271-historical.json
```

The exact table inputs are pinned at source revision `e9b2ce2d0c9c334a246d6eeecd993a4d7a08f8c5`. The underlying one-host campaigns used an RTX 2080 Ti and FP16 workloads. Issue 263 used runtime revision `2617ba166f98a7476a5f472d829a573317fc0cfb`. Issue 266 qualified its memory policies at `119bd7f587213967c1c3fd770dd6a71dcd14cc33`. The three campaigns used different runtime/configuration contracts. All rankings compare arms within one campaign, endpoint, hardware configuration and compute mode. The linked archives include raw samples, integrity manifests and the original analysis instructions. [#263 results](https://github.com/JueonPark/sym/blob/e9b2ce2d0c9c334a246d6eeecd993a4d7a08f8c5/bench/issue263/RESULTS.md) · [#263 archive](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-263) · [#266 tables](https://github.com/JueonPark/sym/blob/e9b2ce2d0c9c334a246d6eeecd993a4d7a08f8c5/bench/issue266/TABLES.md) · [#266 archive](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-266) · [#269 tables](https://github.com/JueonPark/sym/blob/e9b2ce2d0c9c334a246d6eeecd993a4d7a08f8c5/bench/issue269/TABLES.md) · [#269 archive](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-269)

## How a decision is scored

The same `choose_plan` entry point used in the fresh qualification receives one indivisible event per executable archival arm. Its duration is the completed endpoint price from a different calibration condition. Each projected physical increment, including any declared margin, becomes a byte floor rounded up to a whole byte. A configured context quota restricts execution qualification. Folds without a physical constraint use zero for the modeled byte floor and budget, which carries no claim about their actual footprint. The core minimizes duration, then projected bytes, then lexicographic arm ID on ties. [Core planner](planner.py) · [Historical adapter and independent ranking](historical.py)

The lifecycle fold sets reuse to one complete 120-request sequence whose aggregate event already includes first use, updates and retirement. The burst fold ranks an aggregate queue-inclusive p95 event using the core's duration objective. Both retain the measured interactions within the event. The archive comparison neither reconstructs individual kernels and overlap nor generates an arrival process. An independent direct scan ranks the calibration prices and checks the core result before evaluation. Only then does the measured feasible oracle rank every eligible arm actually exercised in the target cell, including target-only attribution arms, by completed endpoint, sampled peak where available and arm ID. Regret is `100 × (selected / oracle − 1)`. An actual OOM has no latency or finite regret. The inputs were public when this retrospective rule was written. Case separation prevents direct lookup of target prices during selection.

| Endpoint | Calibration → target | Objective and eligible alternatives |
|---|---|---|
| E2 warm, #263 | Other compute mode for the same workload, eager ↔ Inductor | Median process p50 of completed requests. Matched Torch and Sym with the same kernels, grouping and depth two. |
| E5 lifecycle, #263 | Other workload at the same fixed compute mode, decode ↔ sparse MoE | Accounted time for startup, all 120 requests, shape/scale revisions and retirement. Includes cold compilation where incurred. Validation gaps are excluded. |
| E2 layout, #269 | Prompt-8 decode → prompt-64 decode or prefill-256, same compute mode | Median of three process p50s. Six offload arms with equal replay-off opportunity, covering native, materialized and lowered KV consumers in Torch and Sym. The full-resident control has a different deployment contract. Prefill has no KV restore. |
| E4, #266 | Same layer/context shape at 1024 MiB → 512 MiB free | Completed request p50. Eight calibration arms cover offload, eight-layer partial residency and full residency. Admit a price if its 1024 MiB deployment peak plus 32 MiB fits. The 32 MiB margin is the original E4 model's safety allowance, reused here as a heuristic. Adaptive policies keep their own budget-dependent configuration. |
| E5 bursts, #266 | One caller → two, then two → four in the same workload | Queue-inclusive request sojourn p95 for 30 bursts of eight requests. Only fresh/adaptive Torch/Sym have registered multi-caller implementations in this matrix. A peak projection must fit the actual 1024 MiB fixture. Completed burst duration divided by eight and request service p95 are different endpoints. |

For E5, the two-caller peak estimate conservatively doubles the one-caller observed whole-application increment. For four callers it extrapolates the one-to-two caller increment twice from the two-caller peak. If that linear projection overcounts native contexts under adaptive depth, it subtracts the excess times the last measured warm retained native device MiB per configured context. The measured arena lives inside Torch allocations. The estimate counts the arena only through the measured physical increment. Tail prices carry the previous caller count's *completed burst sojourn percentile*. This permits both interference and queueing errors. An isolated transfer duration cannot price either.

## Held-out decisions

| Target | Selected | Feasible oracle | Pred. ms | Measured ms | Best ms | Regret |
|---|---|---|---:|---:|---:|---:|
| E2 warm llm_decode eager | torch | torch | 17.539 | 19.200 | 19.200 | 0.00% |
| E2 warm llm_decode inductor | torch | torch | 19.200 | 17.539 | 17.539 | 0.00% |
| E2 warm llm_prefill eager | torch | torch | 4.812 | 4.792 | 4.792 | 0.00% |
| E2 warm llm_prefill inductor | torch | torch | 4.792 | 4.812 | 4.812 | 0.00% |
| E2 warm moe_sparse eager | torch | torch | 3.725 | 2.790 | 2.790 | 0.00% |
| E2 warm moe_sparse inductor | torch | torch | 2.790 | 3.725 | 3.725 | 0.00% |
| E2 warm moe_dense eager | torch | torch | 12.340 | 9.360 | 9.360 | 0.00% |
| E2 warm moe_dense inductor | torch | torch | 9.360 | 12.340 | 12.340 | 0.00% |
| E5 lifecycle llm_decode eager | sym | torch | 5428.000 | 7292.000 | 7062.000 | 3.26% |
| E5 lifecycle llm_decode inductor | torch | torch | 12905.000 | 24556.000 | 24556.000 | 0.00% |
| E5 lifecycle moe_sparse eager | torch | sym | 7062.000 | 5453.000 | 5428.000 | 0.46% |
| E5 lifecycle moe_sparse inductor | torch | torch | 24556.000 | 12905.000 | 12905.000 | 0.00% |
| E2 layout decode64 eager | sym lowered | sym lowered | 21.961 | 22.265 | 22.265 | 0.00% |
| E2 layout decode64 inductor | torch native | torch lowered | 17.479 | 17.745 | 17.717 | 0.16% |
| E2 layout prefill256 eager | sym lowered | sym lowered | 21.961 | 5.593 | 5.593 | 0.00% |
| E2 layout prefill256 inductor | torch native | torch materialized | 17.479 | 4.871 | 4.851 | 0.41% |
| E4 L16/S16 512 MiB | partial sym fresh | resident torch packed | 13.839 | 13.836 | 10.800 | 28.11% |
| E4 L16/S512 512 MiB | partial torch fresh | partial sym fresh | 15.484 | 15.627 | 15.022 | 4.02% |
| E4 L32/S16 512 MiB | partial sym fresh | partial sym fresh | 30.506 | 30.474 | 30.474 | 0.00% |
| E4 L32/S512 512 MiB | offload sym fresh | partial torch fresh | 35.599 | 36.168 | 33.565 | 7.75% |
| E5 burst llm_decode C2 | sym fresh | sym fresh | 152.936 | 140.126 | 140.126 | 0.00% |
| E5 burst llm_decode C4 | sym fresh | sym fresh | 140.126 | 157.825 | 157.825 | 0.00% |
| E5 burst moe_sparse C2 | torch fresh | sym fresh | 22.462 | 20.177 | 19.719 | 2.32% |
| E5 burst moe_sparse C4 | sym fresh | sym fresh | 19.719 | 18.480 | 18.480 | 0.00% |

### E2 completed requests and E5 lifecycle

Torch is faster than baseline Sym in all eight matched #263 warm cells. In eager decode, report extraction and accounting for 22 groups lie on the consumer-launch critical path. The separate diagnostic report bypass saved 4.36% on average, short of the 7.75% reduction needed for parity. Prefill exposes less of this work. The bypasses disable behavior and are ineligible for deployment here. At lifecycle scope, sparse/eager's small advantage for Sym transfers poorly to decode/eager. The latter pays 7.292 s using Sym versus 7.062 s using Torch. The full sequence comprises 100 initial requests, ten after a shape change and ten after scale replacement. The one-time costs matter even for implementations with a warm saving. Sparse/Inductor's diagnostic report bypass improved initial warm p50 by 2.96% while full accounted time regressed 0.20%, with its paired interval crossing zero. [Attribution and lifecycle measurements](../issue263/RESULTS.md)

In #269, eager prompt-8 decode chooses lowered Sym and keeps that rank on prompt-64 decode. The Inductor decision misses the best prompt-64 arm by 0.16%. Inductor already fused the materialized input in the diagnostic traces. Eager prefill's winning arm label denotes an unchanged compute path for this rewrite. It supplies no evidence of a fusion gain. Training on decode and applying its absolute price to prefill produces errors up to **292.65%** despite small ranking regrets. None of #269's own 24 registered targets met its 5% advantage over both matched Torch controls across all rounds. [Lowering and unchanged-kernel controls](../issue269/RESULTS.md)

### E4 capacity and OOM

The target archive physically reserved the GPU until at most 512 MiB remained. Its original 72 E4 arm cells across both budgets contain 192 successes and 24 OOM rounds. All OOMs occurred at 512 MiB. Twenty-one were resident Torch and three were L32/context512 partial Sym fresh. The 2 ms NVML values are sampled lower bounds above the post-reservation/context baseline. Packed full residency for L32 requested 768 MiB and failed with only 90 or 94 MiB sampled. Failure of the pending allocation is decisive. [Physical budget and outcome protocol](../issue266/PROTOCOL.md) · [OOM and successful arms](../issue266/TABLES.md)

At L16/context16 the 32 MiB margin excludes packed resident Torch, which actually completed at 496 MiB in 10.800 ms. The selected 444 MiB partial Sym arm costs 13.836 ms. At L32/context512 the same guard chooses full offload Sym (336 MiB, 36.168 ms). Partial Torch fresh is the feasible oracle (504 MiB, 33.565 ms). The corresponding partial Sym fresh actually OOMed at first request in all three rounds. Packed and recycled partial Sym with packed resident preload fit at 484 MiB and take 39.539 and 38.520 ms. Adaptive partial Sym fits at 444 MiB and takes 48.275 ms. Dropping from two prefetch slots/contexts to one cuts native device retention from 30.82 to 17.04 MiB relative to packed/recycled partial Sym. This slower measured capacity option remains in the oracle's candidate set. [Fixed and adaptive controls](../issue266/RESULTS.md)

Removing only the 32 MiB margin gives the following four decisions. The high-budget 512 MiB sampled peak for L16/context16 resident Torch fresh predicts an exact fit. That selection failed in every target round. The other three selections match the fastest measured arms. The primary policy's safer tradeoff costs up to 28.11% on completed latency.

| Target | Selected | Feasible oracle | Pred. ms | Measured ms | Best ms | Regret |
|---|---|---|---:|---:|---:|---:|
| E4 L16/S16 512 MiB | resident torch fresh | resident torch packed | 10.826 | OOM | 10.800 | — |
| E4 L16/S512 512 MiB | partial sym fresh | partial sym fresh | 14.682 | 15.022 | 15.022 | 0.00% |
| E4 L32/S16 512 MiB | partial sym fresh | partial sym fresh | 30.506 | 30.474 | 30.474 | 0.00% |
| E4 L32/S512 512 MiB | partial torch fresh | partial torch fresh | 33.593 | 33.565 | 33.565 | 0.00% |

### E5 held outputs, resource ownership and burst tails

The E5 1024 MiB limit was physically imposed on every process. Issue 266's matrix registers 1024 MiB for each burst arm. Its acceptance analyzer requires a valid physical meter, the requested free budget and `1008 MiB ≤ free after reservation ≤ 1024 MiB` for each E5 record. Hence the measured oracle admits arms by actual successful completion under that reservation and any declared context quota. The 2 ms NVML increments above the context/reservation baseline are sampled lower bounds. The adaptive policy's separate **256 MiB** transfer allocation was a modeling allowance shared by callers within that physical envelope. [Campaign matrix](../issue266/support.py#L52) · [Physical reservation gate](../issue266/analyze.py#L53) · [Sampling and adaptive policy](../issue266/PROTOCOL.md)

Four-caller decode selects Sym fresh with 157.825 ms sojourn p95. The previous eight outputs remain held across the next burst. The two-to-four caller peak projection for fresh Sym is 956 MiB, equal to the median sampled target increment. Adaptive reduces the configured native transfer contexts per caller from two to one and stages one group member at a time. Its sampled 526 MiB warm peak is lower than the 669 MiB projection, while its 243.967 ms p95 is 54.6% higher than fresh. Four-caller sparse MoE retains two contexts per caller under both policies. Adaptive leaves its sampled physical peak at 442 MiB and worsens sojourn p95 from 18.480 to 27.344 ms. The small two-caller sparse loss in the main table is a rank reversal relative to one caller. [Measured ownership and tails](../issue266/RESULTS.md)

| Four-caller decode at 1024 MiB | Configured native transfer contexts, all callers | Warm retained native device MiB | Torch reserved MiB | Logical arena MiB | Sampled warm peak MiB | Sojourn p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Sym fresh | 8 | 110.28 | 726 | 0 | 956 | 157.825 |
| Sym adaptive | 4 | 32.05 | 398 | 192 | 526 | 243.967 |
| Torch fresh | 0 | 0 | 606 | 0 | 694 | 164.067 |
| Torch adaptive | 0 | 0 | 478 | 192 | 566 | 192.129 |

Context counts describe configured Sym transfer-owner contexts. Torch also uses CUDA runtime, streams and allocator state. A Torch row with zero Sym contexts still consumes device memory. The native retention column is exact at warm end and the physical peak covers the burst loop including validation gaps. The arena is already part of Torch storage. Adding the columns would double-count storage and mix observation windows. All 96 E5 processes verified held outputs, including 768 checks after closure. Native owners retired with zero retained/live bytes and active contexts. [Source owner and allocator counters](../issue266/TABLES.md) · [Closure checks](../issue266/RESULTS.md)

As a separate deployment sensitivity, impose a total quota of four **configured Sym transfer contexts** across four callers. The comparison filters configurations that already ran under the 1024 MiB physical budget. Both four-caller Sym MoE arms exceed that quota. Decode's Sym adaptive arm fits the quota, but Torch fresh has lower p95.

| Target | Selected | Feasible oracle | Pred. ms | Measured ms | Best ms | Regret |
|---|---|---|---:|---:|---:|---:|
| E5 burst llm_decode C4 quota 4 | torch fresh | torch fresh | 150.365 | 164.067 | 164.067 | 0.00% |
| E5 burst moe_sparse C4 quota 4 | torch fresh | torch fresh | 20.177 | 19.531 | 19.531 | 0.00% |

## Prediction limits and prior experimental selector

Median absolute percentage errors for the selected arm were 16.81% for #263 E2, 38.48% for #263 lifecycle, 130.17% for #269 layouts, 0.51% for E4 and 10.18% for E5 bursts. Carrying a measured price into a different workload can preserve a ranking while missing the magnitude of the objective. The tables use the original published precision. Sub-percent differences are descriptive. The archive provides complete request/burst observations for its specific deployment contracts. It provides no common per-stage timestamps for all hypothetical joint plans, no independent GPU replication and no sustained arrival-rate/SLO experiment. The separate fresh qualification supplies direct planner execution and exhaustive arm measurements in its declared region family.

On 2026-11-27 GitHub reported PR #260 **closed, unmerged**, at head `85a889de1842a54d9ee436a2f6c872e2c476d9bb`, closed 2026-10-29 14:43:07 UTC. Its registered NL1 criterion failed on DLRM 512 (+9.5%), DLRM 2048 (+4.7%) and GPT decode (+3.9%). Decode also failed NL2. Decode regressed from 19.986 ms unadvised to 20.767 ms advised. Its review fixed an unexecutable/unsupported split suppressing a valid cheaper direct arm. The corrected branch passed correctness tests, but its sweep and end-to-end CUDA performance qualification were never rerun. The record attributes poor streaming prediction to using isolated group latency without the effects of prefetch overlap and exposed host overhead. The branch's experimental `FastTransfers(advice=True)` has a separate history from main's existing blocking per-transfer selector. [PR status](https://github.com/JueonPark/sym/pull/260) · [Pinned failed gates and timings](https://github.com/JueonPark/sym/blob/85a889de1842a54d9ee436a2f6c872e2c476d9bb/bench/beat_torch/README.md#when-the-runtime-pays-cost-advice)

```sh
gh api repos/JueonPark/sym/pulls/260 --jq '{state,merged,closed_at,head:.head.sha}'
```
