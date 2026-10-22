# Beating eager Torch and Inductor on the #162 workloads

## Outcome

With the explicit fast-lane API (`reloc_torch.fast`, [docs](../../docs/fast-transfers.md))
and the native changes behind it, the `sym_fast` arm has lower completed
model latency than **both** eager Torch and Inductor in seven of the eight
#225 regimes, and lower than Inductor in the eighth. Five regimes meet the
project's stricter criterion (at least 5% below both controls in every
independent round): DLRM 512 and 2048, GraphSAGE 1024, GPT decode and GPT
prefill. The two MoE regimes beat both controls by 2–3%, and GraphSAGE 256
beats Inductor by 2% but trails eager Torch by 28%.

Milliseconds, **median of three independent round medians [min–max round
median]**, 30 samples per round after 5 warmups, rotated path order. Lower
is better; the last column compares `sym_fast` with the faster of the two
Torch controls.

| Regime | Eager Torch | Inductor | Sym + Inductor (previous arm) | `sym_fast` | vs best control |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM, batch 512 | 3.705 [3.660–3.761] | 3.852 [3.791–3.887] | 4.463 [4.451–4.570] | **2.350** [2.314–2.357] | −36.6% (every round) |
| DLRM, batch 2048 | 8.063 [7.876–8.204] | 8.203 [7.972–8.302] | 8.997 [8.767–9.107] | **5.287** [4.822–5.462] | −34.4% (every round) |
| GraphSAGE, 256 seeds | **1.018** [1.001–1.030] | 1.339 [1.335–1.354] | 1.909 [1.892–1.941] | 1.307 [1.305–1.343] | +28.4% (−2.4% vs Inductor) |
| GraphSAGE, 1024 seeds | 4.095 [4.091–4.124] | 2.976 [2.968–2.976] | 3.427 [3.423–3.460] | **2.739** [2.646–2.743] | −8.0% (every round) |
| GPT, four decode steps | 27.067 [26.213–27.117] | 30.176 [30.084–30.550] | 43.663 [43.656–43.683] | **20.378** [20.373–20.825] | −24.7% (every round) |
| GPT, 256-token prefill | 6.931 [6.930–6.963] | 6.476 [6.453–6.497] | 6.420 [6.409–6.453] | **5.968** [5.967–5.993] | −7.8% (every round) |
| MoE, one token | 3.132 [3.113–3.134] | 4.439 [4.413–4.466] | 4.653 [4.648–4.756] | **3.060** [3.047–3.103] | −2.3% |
| MoE, 512 tokens | 10.483 [10.438–10.491] | 16.440 [16.283–16.451] | 17.391 [17.321–17.665] | **10.131** [10.118–10.219] | −3.4% |

Every timed `sym_fast` output equals eager Torch's bit for bit (maximum
absolute difference 0 in all 720 samples); the Inductor arms differ by the
usual fusion rounding (≤ 2.5e-6). No warm compilation callbacks, no implicit
graph breaks, no foreign GPU or build processes were observed during timing.

Tail latency, median of the three per-round p95s:

| Regime | Eager Torch | Inductor | `sym_fast` |
| --- | ---: | ---: | ---: |
| DLRM 512 | 4.038 | 4.232 | 2.914 |
| DLRM 2048 | 8.851 | 9.235 | 5.360 |
| GraphSAGE 256 | 1.631 | 1.681 | 1.833 |
| GraphSAGE 1024 | 4.475 | 3.271 | 3.248 |
| GPT decode | 27.164 | 31.374 | 21.083 |
| GPT prefill | 7.030 | 6.538 | 6.070 |
| MoE sparse | 3.181 | 4.502 | 3.125 |
| MoE dense | 10.631 | 16.519 | 10.197 |

### Transparency control: the whole-request window alone is not the win

`torch_whole` is the eager Torch control given `sym_fast`'s whole-request
weight prefetch window. It measures 25.936 ms for decode (Torch 27.067,
`sym_fast` 20.378), 6.912 ms for prefill (Torch 6.931) and is within 1% of
Torch elsewhere. The decode win therefore comes from the runtime (grouped KV
moves, the native prefetch submission and the fused/vectorized dequantize),
not from the scheduling opportunity by itself.

## What `sym_fast` changes, and what it keeps

The arms share the model mathematics, wire precision (INT8 weights with FP32
scales, FP16 feature wires, FP32 KV), shapes, offline prepacking and the
two-slot prefetch budget of the #225 protocol. `sym_fast` runs **the same
eager PyTorch compute as the `torch` control**; only the way bytes move
differs:

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

Known limitation: GraphSAGE 256 (9k random rows out of a 256 MiB table,
read cache-cold after the harness mutates the source) is bound by the same
DRAM/TLB-latency gather in both paths (~1.2 ms for the gather alone), and
Sym's fixed per-call cost (~0.3 ms) is not recovered by the small overlap.

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
- **GraphSAGE 256.** The only regime where eager Torch stays ahead: the
  traced GPU work is identical and tiny (0.18 ms DMA, 0.12 ms kernels); the
  ~0.3 ms difference is CPU time around a gather that is DRAM/TLB-latency
  bound for both paths.


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
- Correctness: tokens and expert routes exact; outputs compared against
  eager Torch with `rtol=2e-4, atol=2e-5` (the `sym_fast` transfers are byte
  exact and its compute is the control's).
- Evidence: `evidence/workloads-r{1,2,3}.json.gz` (every sample, counters,
  resource reports, hashes), `evidence/summary.csv`, `evidence/manifest.json`.

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
