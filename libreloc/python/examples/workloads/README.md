# Workload examples: DLRM, GNN, LLM offloading, MoE

Four small synthetic models from families that move a lot of data between
the CPU and the GPU, each with Sym placed on the transfers it can handle:

- **DLRM** with embedding tables in CPU memory;
- **GraphSAGE** mini-batches from a CPU feature store;
- a **GPT-style decoder** whose int8 weights and KV cache live on the CPU;
- **mixture-of-experts** blocks whose int8 experts live on the CPU.

Every example runs its workload twice on the same inputs, once with plain
PyTorch transfers and once with Sym, and checks that every tensor Sym moved
and every model output is bit-identical. The examples show how Sym plugs into
these workloads. They do **not** show a speedup: on the development box every
Sym transfer is slower than the equivalent PyTorch code today (see
[Status](#status)).

## What runs where

Sym relocates layouts (reshape, transpose, permute, pad) and converts values
(float32 ↔ float16, int8 quantize and dequantize) as part of a CPU ↔ GPU
transfer, with symbolic shapes so that one compiled plan serves many sizes.
Embedding lookups, neighbour sampling and expert routing are data-dependent
gathers that Sym cannot express, so they stay in PyTorch.

| Example | Model (default size) | PyTorch does | Sym does | Sym API |
| --- | --- | --- | --- | --- |
| `dlrm_embeddings.py` | 26 tables × 100k rows × 64, bottom and top MLP | CPU `EmbeddingBag` lookups and pooling; all GPU compute | pooled `[T, B, 64]` float32 → `[B, T, 64]` float16 on the GPU; one plan for every batch size | `torch.compile(backend=RelocBackend())` |
| `gnn_minibatch.py` | 500k nodes, 128 features, two SAGE-mean layers | sampling, the CPU feature gather, GPU aggregation | gathered `[n, 128]` float32 → float16 on the GPU; one plan while `n` changes every batch | `torch.compile(backend=RelocBackend())` |
| `llm_offload.py` | GPT-style, d=1024, 8 layers, int8 weights on the CPU | attention, MLP, greedy decoding | each layer's int8 `[in, out]` weights → float32 `[out, in]` just before use (one recipe for all four shapes); KV cache `[H, S, Dh]` ↔ `[S, H, Dh]` between steps while `S` grows | `WeightFetcher` (typed recipe through `reloc_torch.dispatch`); `RelocBackend` for the KV cache |
| `moe_experts.py` | 16 experts × 2 blocks, top-2, int8 experts on the CPU | routing, token dispatch, GPU-to-GPU token moves, combine | only the selected experts, int8 → float32 `[out, in]`, to each expert's GPU | `WeightFetcher` |

`WeightFetcher` (in `common.py`) compiles one symbolic typed recipe,
`int8[s0, s1] → dequantize(scale[s1]) → transpose → float32[s1, s0]`, and
binds it for every matrix shape. With `policy="auto"` and a calibration, the
runtime's cost model decides whether the dequantize runs on the CPU or the
GPU; the report lists the rows it chose.

## Run

You need the qualified CUDA environment and a CUDA-enabled build; see
[Use with PyTorch](../../../../docs/getting-started.md#use-with-pytorch).
The runner itself needs only the Python standard library:

```bash
export SYM_PYTHON=/tmp/sym-torch-cuda/bin/python   # the defaults
export SYM_BUILD="$PWD/build/torch-cuda"
python3 libreloc/python/examples/workloads/run_workloads.py --quick
python3 libreloc/python/examples/workloads/run_workloads.py          # default sizes
python3 libreloc/python/examples/workloads/run_workloads.py --only moe --devices cuda:0,cuda:1,cuda:2,cuda:3
```

| Option | Meaning |
| --- | --- |
| `--quick` | small sizes, seconds per example |
| `--only dlrm gnn llm moe` | run a subset |
| `--devices cuda:0,...` | dlrm, gnn and llm use the first device; moe places expert `e` on device `e % n` |
| `--calibration auto\|none\|PATH` | cost-model calibration for llm and moe; `auto` picks the committed file for this GPU, if there is one |
| `--build`, `--python` | override `SYM_BUILD` and `SYM_PYTHON` |
| `--output-dir DIR` | where `<name>.json`, `<name>.log` and `summary.json` go (default: a fresh temporary directory) |
| `--timeout SECONDS` | per example (default 900) |

Exit status: 0 when every example passed, 1 when a check failed or an example
crashed or timed out, 2 when a prerequisite is missing.

To run one example directly, set the environment the runner would set:

```bash
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
"$SYM_PYTHON" libreloc/python/examples/workloads/llm_offload.py --quick --output /tmp/llm.json
```

## Reading the report

Each example prints a summary and, with `--output`, writes JSON:

| Field | Meaning |
| --- | --- |
| `checks` | per check, how many observations passed and failed, and the first failure's detail; the example passes only if every check passed |
| `counters` | the `RelocBackend` counters: Dynamo and plan compiles, symbol binds, executions, typed payload, fallbacks and exclusions |
| `plan_compiles` | artifacts the example compiled (backend plans plus the `WeightFetcher` recipe) |
| `bytes` | per transfer kind: Sym transfers, source bytes, **wire** bytes (the tensor that crossed the link) and destination bytes; typed weight transfers add `payload`, the runtime's observed link traffic (wire plus scale uploads) |
| `dispatches` | typed dispatch rows that ran (`cpu_reference`, `cuda_dequant_relocate`, ...) |
| `timings_ms` | per transfer kind and path (`torch`, `sym`): the first call (includes compilation), the median of the rest, the total |
| `workload` | example-specific facts: batch sizes, rows per mini-batch, generated tokens, active experts |
| `summary` | what the runner's table shows |

Wire bytes follow from the recipe's dtype choice. PyTorch code that performs
the same conversion moves the same bytes (`x.to("cuda", torch.float16)`
converts on the CPU before copying; moving int8 and dequantizing on the GPU
moves the int8 bytes), so a smaller wire is not an advantage over equivalent
PyTorch code.

## Status

What the examples establish on the development box (EPYC 7351 with four RTX
2080 Ti on PCIe gen3):

- every tensor Sym moved and every model output is bit-identical to PyTorch;
- one compiled plan serves every batch size, mini-batch row count and KV
  length, and one weight recipe serves every matrix shape;
- no fallbacks or exclusions occur in these configurations.

They do not establish a speed advantage. Probe measurements from 2026-09-25,
warm, per call:

| Path | Sym | PyTorch |
| --- | --- | --- |
| `[8, 8192, 64]` transpose + float16, CPU → GPU (16 MB source) | 131 ms | 2.7 ms |
| the same transpose, layout only, low-level `pyreloc.h2d` | 26.5 ms | 3.0 ms |
| int8 `[1024, 4096]` dequantize + transpose, CPU → GPU (4 MB wire) | 28.7 ms | 1.3 ms (move int8, dequantize on the GPU); 20.7 ms (dequantize on the CPU, move float32) |

Known contributors, not profiled: float32 → float16 has only the scalar
`cpu_reference` row, and every call constructs a CUDA backend and its pinned
staging (per-call amortization is deferred).

### Sample run

`run_workloads.py` at default sizes on the development box (2026-09-25, commit 4ee147d):

```
example  result  checks  plans  sym transfers  wire/dest MiB  steady sym/torch ms  fallbacks
-------  ------  ------  -----  -------------  -------------  -------------------  ---------
dlrm     pass    7/7     1      4              19.5/19.5      235.6/5.6            0
gnn      pass    8/8     1      5              43.5/43.5      569.6/6.9            0
llm      pass    9/9     3      480            798.6/3102.6   5137.7/178.8         0
moe      pass    6/6     1      170            340.0/1360.0   2363.0/76.0          0
```

## Limitations

- Transfers are blocking (`non_blocking=True` falls back to PyTorch), so weight
  streaming and KV movement never overlap compute.
- Index-based gathers (embedding lookups, sampling, routing) stay in PyTorch.
- float32 → float16 has only the CPU reference row; there is no GPU narrowing
  kernel.
- The importer excludes a 4D permute with a batch dimension of 1
  (`[1, S, H, Dh]`) as `conditional_materialization`, so the LLM example keeps
  its KV cache 3D.
- Typed dispatch launches its CUDA kernels on the caller's current device;
  `WeightFetcher` wraps each call in `torch.cuda.device(target)` until the
  runtime is fixed.
- Several GPUs in the MoE example show placement and correctness only.

## About the feedback these examples answer

| Claim | What the project's own evidence says |
| --- | --- |
| DLRM gains from `gather_bw.cpp`, `GatherPool` and `gather_quantize` on embedding traffic | These are strided *layout* gathers over a plan, not index gathers; the lookups stay in PyTorch and Sym moves the pooled block (`dlrm_embeddings.py`) |
| LLM offloading gains from asynchronous H2D (`m0_h2d_gpu`, `dma_engine.cu`) and AVX-512 quantization overlapping transfers with compute | `m0_h2d_gpu*.json` are bring-up bandwidth measurements; the DMA-engine experiment was a negative result ([R5](../../../../docs/r5-exp5-dma-engine.md)); the PyTorch integration is blocking; the AVX-512 attribution was refuted ([V2](../../../../docs/v2-isolation.md)), and the Gen3 box has no AVX-512 |
| GNN gains from `PlanBuilder` and `ChunkSchedule` pre-relocating scatter/gather | Plans describe affine layouts only; neighbour gathers stay in PyTorch and Sym moves the gathered block with a symbolic row count (`gnn_minibatch.py`) |
| MoE gains from `multigpu_reloc.cu` and the cost model broadcasting or scattering experts | The multi-GPU gate failed on the four-2080 Ti box, 0.92× ([R3](../../../../docs/r3-exp3-multigpu.md)); the cost model places each transform on the CPU or the GPU (used here through `policy="auto"`); `cm4_registered_predictions.json` is a pre-registered evaluation artifact, not a runtime input |

## Tests

CPU tests cover the helpers and the runner; tests marked `gpu` run each
example through the runner. CI does not collect this directory:

```bash
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
"$SYM_PYTHON" -m pytest libreloc/python/examples/workloads/tests -q
```
