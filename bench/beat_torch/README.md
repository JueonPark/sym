# Beating eager Torch and Inductor on the #162 workloads

## Outcome

These measurements evaluate optimized producer/dequantization kernels together
with an opt-in transfer runtime. They do not isolate AST folding as the source
of the improvement. In particular, DLRM's fused EmbeddingBag producer is built
natively outside the compiler recipe; the recipe does not express the pooling
reduction. Grouped KV transfers and execution management are additional system
contributions. The ablation in stacked PR #256 separates these effects more
closely; its contrasts are not independent additive contributions.

With the explicit fast-lane API (`reloc_torch.fast`, [docs](../../docs/fast-transfers.md))
and the native changes behind it, the `sym_fast` arm has lower completed
model latency than **both** eager Torch and Inductor in seven of the eight
#225 regimes, and lower than Inductor in the eighth. 4 regimes meet the
project's stricter criterion (at least 5% below both controls in every
independent round): DLRM 512, DLRM 2048, GPT decode, GPT prefill. GraphSAGE 1024 and
the two MoE regimes beat both controls by 2–3%, and GraphSAGE 256 beats
Inductor by 9% but trails eager Torch by 19%.

Milliseconds, **median of three independent round medians [min–max round
median]**, 30 samples per round after 5 warmups, rotated path order. Lower
is better; the last column compares `sym_fast` with the faster of the two
Torch controls.

| Regime | Eager Torch | Inductor | Sym + Inductor (previous arm) | `sym_fast` | vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM, batch 512 | 3.761 [3.708–3.784] | 3.737 [3.680–3.883] | 4.530 [4.427–4.547] | **2.250 [2.230–2.633]** | -39.8% (every round) |
| DLRM, batch 2048 | 7.880 [7.878–7.960] | 8.057 [7.996–8.081] | 8.559 [8.548–8.804] | **4.738 [4.735–4.762]** | -39.9% (every round) |
| GraphSAGE, 256 seeds | **1.018 [1.011–1.023]** | 1.335 [1.330–1.340] | 1.905 [1.884–1.937] | 1.210 [1.204–1.285] | +18.9% (-9.3% vs Inductor) |
| GraphSAGE, 1024 seeds | 4.120 [4.116–4.134] | 2.955 [2.953–2.964] | 3.652 [3.403–3.811] | **2.896 [2.758–2.980]** | -2.0% |
| GPT, four decode steps | 26.211 [26.197–27.090] | 30.069 [30.044–30.389] | 43.312 [43.078–43.669] | **20.485 [20.434–20.641]** | -21.8% (every round) |
| GPT, 256-token prefill | 6.949 [6.946–6.966] | 6.450 [6.442–6.472] | 6.450 [6.438–6.464] | **5.993 [5.981–6.004]** | -7.1% (every round) |
| MoE, one token | 3.120 [3.109–3.131] | 4.409 [4.405–4.428] | 4.648 [4.641–4.663] | **3.055 [3.043–3.066]** | -2.1% |
| MoE, 512 tokens | 10.470 [10.447–10.513] | 16.394 [16.359–16.462] | 17.422 [17.398–17.477] | **10.162 [10.100–10.198]** | -2.9% |

Every timed `sym_fast` output equals eager Torch's bit for bit (maximum
absolute difference 0 in all 720 samples); the Inductor arms differ by the
usual fusion rounding (≤ 2.5e-6). No warm compilation callbacks, no implicit
graph breaks, no foreign GPU or build processes were observed during timing.

Tail latency, median of the three per-round p95s:

| Regime | Eager Torch | Inductor | `sym_fast` |
| --- | ---: | ---: | ---: |
| DLRM 512 | 4.223 | 4.231 | 2.876 |
| DLRM 2048 | 8.017 | 8.687 | 5.091 |
| GraphSAGE 256 | 1.615 | 1.647 | 1.769 |
| GraphSAGE 1024 | 4.474 | 3.263 | 3.395 |
| GPT decode | 26.430 | 30.972 | 21.015 |
| GPT prefill | 7.072 | 6.574 | 6.062 |
| MoE sparse | 3.166 | 4.454 | 3.096 |
| MoE dense | 10.626 | 16.472 | 10.269 |

### Transparency control: the whole-request window alone is not the win

`torch_whole` is the eager Torch control given `sym_fast`'s whole-request
weight prefetch window. It measures 25.933 ms for decode (Torch 26.211,
`sym_fast` 20.485), 6.925 ms for prefill (Torch 6.949) and is within 1% of
Torch elsewhere. The decode win therefore comes from the runtime (grouped KV
moves, the native prefetch submission and the fused/vectorized dequantize),
not from the scheduling opportunity by itself.

## What `sym_fast` changes, and what it keeps

The arms share the model mathematics, wire precision (INT8 weights with FP32
scales, FP16 feature wires, FP32 KV), shapes, offline prepacking and the
two-slot prefetch budget of the #225 protocol. `sym_fast` runs the same
downstream eager GPU model compute as the `torch` control. CPU producers,
value-conversion kernels and transfer execution also change:

- **GPT decode/prefill, MoE.** Weight groups are submitted natively from
  prepared INT8 snapshots (`FastWeights`, one native call per group, the
  libtorch helper reads tensor metadata); the prefetch window opens with a
  single submission so the first DMA starts earlier, spans the whole request
  in decode, and performs any slot wait at the start of the next iteration
  rather than between handing out a group and its first use. The eight K/V
  caches of a decode step move as one `KVCacheSwap` group: one batched
  permute kernel, one copy into one pinned block, one completion. The
  per-channel dequantize kernel now processes four elements per thread
  (bit-identical results).
- **DLRM.** The CPU producer is fused: per-table bag sums, the `[B, T, 64]`
  layout, the FP16 cast and the chunked H2D are one pipelined native call
  (`PooledProgram`), replacing 26 `EmbeddingBag` calls, `stack`, `transpose`,
  `to(float16)` and the copy. Sums are FP32 in index order; the result is bit
  for bit the `torch` control's.
- **GraphSAGE.** The row gather, cast and H2D are one chunked native call
  (`IndexedProgram`) with the index bound from the tensor's memory, row
  prefetching and all eight workers fed per chunk.

Known limitation: GraphSAGE 256 (9k random rows of a 256 MiB table) trails
eager Torch under the #225 protocol although Sym's transfer is not slower
in matched conditions. With the harness's own inputs on an idle host, Sym's
fused gather/cast/H2D takes 0.63 ms against Torch's 0.72 ms back to back and
1.16 vs 1.21 ms fully cold, and `compare.py --cache-state warm` (each path
run once untimed right before its timed run) gives 1.01 ms for `sym_fast`
against 1.19 ms for Torch (−15%). The protocol's rotated median lands on a
different sample for each path: the three paths run back to back on the same
inputs, Torch's `index_select` materializes a 4.6 MiB temporary plus a
2.3 MiB wire that evict the selected rows from the 8 MiB L3, so the path
that follows Torch (Sym or Inductor, one rotation in three) pays a cold
gather, whereas Sym's 2.3 MiB footprint leaves Torch's rows warm. Under
`--cache-state cold` (caches swept before every path) `sym_fast` is 1.81 ms
against Torch's 1.64 (+10%): a standalone microbenchmark of the same random
row gather shows both runtimes within ~0.4 ms of the machine's cold floor
(0.6–0.8 ms for 9k rows after an eight-thread cache sweep), and prefetch
distance (0–128 rows), worker count (4/8), transparent huge pages, pinned
or write-combined staging and an asynchronous submission did not move the
rotated or cold medians. The 1024-seed regime, where the gather is large
enough to overlap its DMA, wins in every cache state.

## Inside `torch.compile`: the `sym_fast_inductor` arm

The fast lane's prepared programs are also `torch.compile` primitives
(`program.op`, [docs](../../docs/fast-transfers.md#inside-torchcompile)):
functional custom ops with fake kernels that return the program's destination
metadata, so Dynamo captures them whole and Inductor schedules the compute
around them. `sym_fast_inductor` is `sym_fast` with the `inductor` control's
compute: the DLRM pooled producer and the GraphSAGE gather are ops inside the
compiled model regions; the GPT/MoE weight windows and the KV swaps stay
between regions, exactly where the Inductor control keeps its `TorchQueue`.
Per-call overhead of the op entry points: `op_overhead.py` (table in the
docs: +70 µs over the direct call in eager mode, +170 µs inside a compiled
region, against +360–460 µs for the previous generic custom-op route).

Three independent rounds on revision `04a5eff` (same protocol as above,
arms `torch`, `inductor`, `sym_fast`, `sym_fast_inductor`; evidence in
`evidence-compile-op/`), milliseconds, median of three round medians
[min–max round median]; the last two columns compare each fast arm with the
faster Torch control of this run:

| Regime | Eager Torch | Inductor | `sym_fast` | `sym_fast_inductor` | `sym_fast` | `sym_fast_inductor` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLRM, batch 512 | 3.690 [3.595–4.069] | 3.909 [3.869–4.249] | 2.396 [2.393–2.444] | 2.442 [2.419–2.479] | -35.1% (every round) | -33.8% (every round) |
| DLRM, batch 2048 | 7.808 [7.807–7.884] | 8.042 [7.990–8.081] | 4.866 [4.830–6.238] | 4.837 [4.829–6.693] | -37.7% (every round) | -38.0% (every round) |
| GraphSAGE, 256 seeds | 1.215 [1.196–1.265] | 1.324 [1.315–1.326] | 1.332 [1.316–1.435] | 1.412 [1.400–1.450] | +9.7% | +16.3% |
| GraphSAGE, 1024 seeds | 4.030 [4.011–4.048] | 2.965 [2.965–2.967] | 2.693 [2.684–2.962] | 2.884 [2.866–3.249] | -9.2% | -2.8% |
| GPT, four decode steps | 26.300 [26.257–26.364] | 30.023 [29.986–30.226] | 20.637 [20.557–21.060] | **19.356 [19.199–19.449]** | -21.5% (every round) | -26.4% (every round) |
| GPT, 256-token prefill | 6.971 [6.963–6.999] | 6.394 [6.388–6.409] | 6.022 [6.018–6.037] | **5.817 [5.802–5.826]** | -5.8% (every round) | -9.0% (every round) |
| MoE, one token | 3.175 [3.172–3.208] | 4.356 [4.322–4.360] | 3.032 [3.013–3.073] | 3.773 [3.764–3.815] | -4.5% | +18.8% |
| MoE, 512 tokens | 10.678 [10.666–10.760] | 16.337 [16.306–16.339] | 10.221 [10.126–10.259] | 13.982 [13.953–14.045] | -4.3% | +31.0% |

Reading the arm:

- **The fast lane composes with Inductor's compute.** Where Inductor's
  fused kernels beat eager compute (GPT decode and prefill), the fast lane
  inside the compiled model is the best arm: 19.36 ms for four decode steps
  (−26.4% against eager Torch, −35.5% against the Inductor control) and
  5.82 ms for prefill (−9.0% / −9.0%), in every round. `sym_fast`'s eager
  compute was leaving that on the table.
- **The ops cost what `op_overhead.py` says.** DLRM is unchanged (the op is
  called once per request against a 2–5 ms budget); GraphSAGE 256 pays the
  compiled-function call (about +0.1 ms on a 1.2 ms request) on top of the
  ordering effect documented above.
- **It inherits Inductor's losses.** On MoE the Inductor control itself is
  37–53% slower than eager Torch (per-expert graphs around the routing
  synchronizations), and `sym_fast_inductor` moves exactly the same bytes as
  `sym_fast`, so it lands between the two: the transfer path cannot repay a
  compute regression. The right arm per workload is therefore the control's
  own compute plus the fast lane, which is what both `sym_fast` arms now
  offer.
- Zero graph breaks in every arm (the harness asserts it), outputs within
  the Inductor rounding band (≤ 2.5e-6), `sym_fast` bit-exact; the isolation
  audit saw no foreign process in any round.

## Where the time goes (Nsight Systems)

`profile.sh` captures `torch` and `sym_fast` in one process under NVTX
ranges (`--trace=cuda,nvtx`); `analyze.py` attributes, per timed request,
the union of DMA activity, the union of kernel activity, the time with
neither, bytes per direction, copy/kernel counts and DMA gaps. Medians over
five traced samples; tracing adds overhead, so compare rows, not absolute
values against the table above.

| Regime / path | wall | DMA busy | kernel busy | GPU idle | H2D MiB | copies | kernels | first DMA | max DMA gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dlrm_512/sym_fast | 3.14 | 0.13 | 0.23 | 2.77 | 1.6 | 1 | 18 | 1.74 | 0.00 |
| dlrm_512/torch | 4.59 | 0.13 | 0.23 | 4.23 | 1.6 | 1 | 18 | 3.43 | 0.00 |
| dlrm_2048/sym_fast | 6.20 | 0.53 | 0.65 | 5.02 | 6.5 | 7 | 17 | 1.37 | 0.62 |
| dlrm_2048/torch | 10.84 | 0.52 | 0.65 | 9.66 | 6.5 | 1 | 17 | 9.60 | 0.00 |
| gnn_256/sym_fast | 1.70 | 0.18 | 0.12 | 1.40 | 2.2 | 1 | 14 | 0.79 | 0.00 |
| gnn_256/torch | 1.94 | 0.18 | 0.12 | 1.64 | 2.2 | 1 | 14 | 1.16 | 0.00 |
| gnn_1024/sym_fast | 3.25 | 0.73 | 0.36 | 2.16 | 8.6 | 9 | 14 | 0.68 | 0.15 |
| gnn_1024/torch | 4.59 | 0.69 | 0.36 | 3.55 | 8.6 | 1 | 14 | 3.51 | 0.00 |
| llm_decode/sym_fast | 26.02 | 15.98 | 4.67 | 7.94 | 193.4 | 154 | 411 | 0.37 | 0.94 |
| llm_decode/torch | 31.04 | 15.85 | 8.00 | 7.42 | 193.4 | 196 | 493 | 0.24 | 0.39 |
| llm_prefill/sym_fast | 6.71 | 4.10 | 4.17 | 0.71 | 48.1 | 37 | 101 | 0.33 | 0.93 |
| llm_prefill/torch | 8.20 | 3.96 | 5.07 | 0.57 | 48.1 | 37 | 117 | 0.23 | 0.95 |
| moe_sparse/sym_fast | 4.07 | 1.33 | 0.53 | 2.34 | 16.0 | 24 | 78 | 0.33 | 0.68 |
| moe_sparse/torch | 4.43 | 1.33 | 0.85 | 2.25 | 16.0 | 24 | 86 | 0.35 | 0.56 |
| moe_dense/sym_fast | 14.17 | 5.34 | 4.38 | 6.66 | 64.2 | 84 | 262 | 0.36 | 1.25 |
| moe_dense/torch | 16.09 | 5.29 | 5.47 | 5.33 | 64.2 | 84 | 294 | 0.36 | 0.68 |

Reading the rows (ms per timed request, traced):

- **GPT decode.** Both paths move the same 193.4 MiB and keep the DMA engine
  busy for ~15.9 ms; the difference is on the SMs and in the launch stream.
  `sym_fast` runs 4.67 ms of kernels against Torch's 8.00 ms: the vectorized
  per-channel dequantize (`dequantS8F32Quad`, 1.92 ms for 64 matrices) replaces
  Torch's two elementwise passes (`q.float()` then `* scale`, 3.21 + 2.20 ms),
  and the K/V swaps are 6 batched permutes plus 6 copies instead of 48
  transposes and 48 copies (154 copies and 411 kernels per request versus
  196 and 493).
- **GPT prefill.** Same DMA (4.1 ms, 48 MiB) and the same sgemms; kernel time
  drops from 5.07 to 4.17 ms for the same reason, and the request finishes
  as soon as the last layer's compute does.
- **DLRM.** The first byte reaches the GPU after 1.37 ms (batch 2048) instead
  of 9.60 ms: the fused producer streams the pooled rows through 7 chunk
  copies while the remaining bags are still being summed, whereas the Torch
  control must finish 26 `EmbeddingBag` calls, the stack, the transpose/cast
  and only then issues one 6.5 MiB copy. GPU kernel work is identical.
- **GraphSAGE 1024.** First DMA at 0.68 ms versus 3.51 ms, nine chunk copies
  overlapping the gather versus one copy after it; identical kernels.
- **MoE.** Same DMA volume and counts; `sym_fast` saves 0.3–1.1 ms of kernel
  time (fused dequantize) but shows longer maximum DMA gaps (1.25 vs 0.68 ms
  dense), i.e. its remaining cost is CPU-side submission latency between
  expert groups around the routing synchronizations that both paths pay.
- **GraphSAGE 256.** The only regime where eager Torch stays ahead in the
  median: the traced GPU work is identical and tiny (0.18 ms DMA, 0.12 ms
  kernels); the difference is CPU time in a gather that is DRAM/TLB-latency
  bound for both paths, and that Sym runs with a colder cache because it
  follows Torch's larger footprint in one rotation out of three (see the
  limitation above).


## First requests

`first_completed_ms` ranges over the rounds (fresh process, after common
setup): eager Torch 1–30 ms, `sym_fast` 5–66 ms (artifact decoding and
native context creation), Inductor 0.5–22 s (compilation). Prepared
snapshots for GPT weights retain 48 MiB pinned; native typed scratch stays
below the 64 MiB budget per context (two contexts).

## Validation

On the measured tree: CUDA Python suite 1,256 passed (4 skipped, including
the new `test_fast.py`: bit-exact typed/indexed/pooled/KV/weight paths,
asynchronous producers, index range checks, pageable D2H, the libtorch view
helper), workload tests 58 passed, CTest 12/12 (CUDA) and 3/3 (CPU-only
build), compiler lit 41/41, `git diff --check` and clang-format 21 clean.
The pre-existing `bench-dma-engine` CUDA bench target does not compile with
this toolkit (unrelated `atomicAdd` overload) and is not part of the gates.

## Protocol

- Harness: `compare.py` = bench/issue225's harness plus the `sym_fast` and
  `torch_whole` arms; `run.py` runs three fresh processes with rotated path
  order and the post-roadmap environment (GPU 0, CPUs `4-7,20-23`, eight
  Torch/OpenMP threads, one interop thread, passive OpenMP waits, TF32 and
  CUDA graphs off, Release build with NVTX off, foreign-work audit every 3 s).
- Controls are unchanged from #225/#246: pinned staging, prepacked INT8
  images, `TorchQueue` two-slot prefetch per step, eager KV moves; Inductor
  compiles the matched compute regions and host transforms.
- `--cache-state {rotated,cold,warm}` selects the #225 rotation (default), a
  cache sweep before every path (an involutive mutation keeps the values), or
  an untimed run of each path right before its timed run. The committed
  evidence is `rotated`; the other two are reported only for GraphSAGE 256.
- Correctness: tokens and expert routes exact; outputs compared against
  eager Torch with `rtol=2e-4, atol=2e-5` (the `sym_fast` transfers are byte
  exact and its compute is the control's).
- Evidence: `evidence/workloads-r{1,2,3}.json.gz` (every sample, counters,
  resource reports, hashes), `evidence/summary.csv`, `evidence/manifest.json`.
- Provenance: the three rounds ran on the committed revision `e85f076`
  (`evidence/manifest.json`) on an idle host (isolation audit clean). The
  Nsight captures below were taken on the same tree two commits earlier
  (before the static gather schedule and one-call indexed request), whose
  measured paths are otherwise unchanged.



```sh
export BUILD=build/perf   # Release, RELOC_ENABLE_CUDA=ON, RELOC_ENABLE_TORCH_EXT=ON
python bench/beat_torch/run.py --build $BUILD --output /tmp/beat-torch-rounds
python bench/beat_torch/summarize.py /tmp/beat-torch-rounds bench/beat_torch/evidence
bench/beat_torch/profile.sh $BUILD /tmp/sym-torch-cuda/bin/python llm_decode /tmp/decode-trace 5
python bench/beat_torch/analyze.py /tmp/decode-trace.sqlite
```

Environment: AMD EPYC 7351, RTX 2080 Ti (PCIe 3 x16), driver 595.71.05,
CPython 3.14.7, Torch 2.14.0+cu126, CUDA toolkit 12.5 for the native build,
clocks not locked.
