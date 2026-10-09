# Prepared INT8 wire weights (#224)

[API and ownership contract](../../docs/prepared-wire-weights.md).
The measured feature snapshots INT8 `[in,out]` into private INT8 `[out,in]`
storage once, retains validated FP32 channel scales and dispatch metadata,
and dequantizes on the GPU after each compact upload. It removes repeated
permutation and preparation work; it does not reduce wire bytes relative to
an equivalent INT8 Torch transfer.

## Results

Median of three round medians, in milliseconds; lower is better. Each warm
distribution has 30 samples per round. All nine accepted processes and all
samples for the reported comparisons are retained, including the slower first
candidate round.

| Matrix `[in,out]` | Sym raw | Sym prepared | Torch raw | Torch prepared | Inductor raw | Inductor prepared |
|---|---:|---:|---:|---:|---:|---:|
| (256, 1024) | 0.577 | 0.393 | 0.268 | 0.231 | 0.394 | 0.325 |
| (1024, 4096) | 1.182 | 0.962 | 0.787 | 0.675 | 0.621 | 0.621 |
| (4096, 1024) | 1.135 | 0.939 | 0.768 | 0.663 | 0.612 | 0.620 |

Warm Sym matrix latency falls **17–32%** relative to its raw path. It still
trails both equally prepacked Torch and Inductor in all three matrix cases.

| Model | Sym raw | Sym prepared | Torch raw | Torch prepared | Inductor raw | Inductor prepared |
|---|---:|---:|---:|---:|---:|---:|
| llm_decode | 32.40 | 25.11 | 31.18 | 26.21 | 26.66 | 26.57 |
| moe_sparse | 4.62 | 3.53 | 3.45 | 3.06 | 3.64 | 3.55 |

Prepared Sym reduces GPT decode latency by **22.5%** and sparse MoE by
**23.6%** relative to raw Sym (median reductions). Every round improves. This
does **not** establish a consistent win over Torch: prepared Torch remains
faster for sparse MoE, and the small median GPT lead reverses in the slow
first round. Inductor prepacking is essentially neutral for warm 4 MiB
matrix loads, and its preparation cost makes the total reuse time worse.

Round medians for the close GPT comparison (ms):

| Method | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| sym_raw | 43.24 | 32.40 | 32.38 |
| sym_packed | 33.60 | 25.10 | 25.11 |
| torch_packed | 28.88 | 26.17 | 26.21 |
| inductor_packed | 35.89 | 26.46 | 26.57 |

The first candidate process was slower across several matrix/GPT methods;
the cause was not isolated. It is retained rather than replaced based on
its timings. Median cross-framework differences of a few percent should
not be treated as stable wins. Paired Sym preparation gains and reuse
crossovers are consistent across all three rounds.

Median per-round p95 for prepared methods (ms):

| Case | Sym | Torch | Inductor |
|---|---:|---:|---:|
| (256, 1024) | 0.462 | 0.254 | 0.395 |
| (1024, 4096) | 0.975 | 0.683 | 0.654 |
| (4096, 1024) | 0.954 | 0.674 | 0.631 |
| llm_decode | 26.219 | 26.291 | 26.880 |
| moe_sparse | 3.597 | 3.089 | 3.709 |

The separate raw-only control changes existing Sym p50 versus main by
**−1.2% to +3.7%** across the three shapes. Its main/control medians are:

| Matrix | Main | Candidate raw-only |
|---|---:|---:|
| (256, 1024) | 0.540 | 0.560 |
| (1024, 4096) | 1.184 | 1.169 |
| (4096, 1024) | 1.111 | 1.122 |

## Preparation, replacement and amortization

These costs are separate from the warm latency above (ms). First load is
the first completed prepared load for that shape/path. Full per-path first
calls, including Inductor initialization, are preserved in diagnostics.

| Matrix | First prepare Sym / Torch | First load Sym / Torch | Replace Sym / Torch | Invalidate + prepare Sym / Torch |
|---|---:|---:|---:|---:|
| (256, 1024) | 0.605 / 0.489 | 1.172 / 0.761 | 0.422 / 0.313 | 0.421 / 0.312 |
| (1024, 4096) | 9.162 / 9.257 | 1.490 / 0.980 | 4.902 / 4.649 | 4.869 / 4.574 |
| (4096, 1024) | 5.269 / 5.117 | 0.905 / 0.629 | 5.094 / 4.798 | 5.056 / 4.730 |

Combined symbolic artifact setup for the two Sym recipes is 12.91 ms
(recorded separately per process). First snapshot creation can allocate
new pinned pages; later replacement can reuse allocator caches. Atomic
replacement temporarily reserves both revisions. Checkpoint source
mutation itself is not timed; re-preparation always copies and validates.

**Measured break-even:** the first faster sampled reuse count is **8**
for `[256,1024]`, and **64** for both 4 MiB matrices, in every round.
The measured crossing is therefore between 4 and 8 loads for the small
matrix, and between 32 and 64 for the large matrices. These are empirical
brackets, not exact thresholds. One-shot preparation loses substantially.

Total time including prepare, completed loads and invalidation (ms):

| Matrix | Loads | Sym raw | Sym prepared | Torch raw | Torch prepared | Inductor raw | Inductor prepared |
|---|---:|---:|---:|---:|---:|---:|---:|
| (256, 1024) | 1 | 0.46 | 0.79 | 0.22 | 0.53 | 0.29 | 0.61 |
| (256, 1024) | 8 | 3.30 | 2.99 | 1.55 | 1.78 | 2.09 | 2.41 |
| (256, 1024) | 32 | 13.09 | 10.65 | 6.16 | 6.07 | 8.27 | 8.54 |
| (256, 1024) | 64 | 26.15 | 20.66 | 12.27 | 11.75 | 16.51 | 16.72 |
| (1024, 4096) | 1 | 0.86 | 5.81 | 0.66 | 5.25 | 0.50 | 5.27 |
| (1024, 4096) | 8 | 6.39 | 10.41 | 5.16 | 8.98 | 3.78 | 8.48 |
| (1024, 4096) | 32 | 25.45 | 26.13 | 19.79 | 21.58 | 15.01 | 19.49 |
| (1024, 4096) | 64 | 50.90 | 47.08 | 39.59 | 38.38 | 29.87 | 34.17 |
| (4096, 1024) | 1 | 0.83 | 6.03 | 0.67 | 5.52 | 0.49 | 5.47 |
| (4096, 1024) | 8 | 6.31 | 10.65 | 5.22 | 9.18 | 3.75 | 8.74 |
| (4096, 1024) | 32 | 25.14 | 26.09 | 20.03 | 21.79 | 14.97 | 19.75 |
| (4096, 1024) | 64 | 50.31 | 46.81 | 40.00 | 38.50 | 29.91 | 34.48 |

All seven reuse counts and raw samples are in `evidence/timings.csv`.
The total-time loop has different cache/allocation behavior from the
interleaved warm loop, which performs exact comparisons between calls;
do not infer its crossover by multiplying the warm table alone.

Preparing the whole model checkpoint costs **116.6 ms Sym / 112.8 ms
Torch** for GPT and **152.4 / 145.8 ms** for MoE. A simple
`prepare_cost / (raw_model_time - prepared_model_time)` estimate gives
**about 16 four-token GPT requests or 140 one-token MoE requests** to
repay Sym checkpoint preparation, with artifacts already initialized.
These model estimates are hypotheses, not measured model reuse sweeps.
MoE prepares all experts but transfers only routed experts; unused
expert preparation and storage are included.

## Representation and bounded storage

| Matrix | INT8 wire bytes | Scale wire bytes | Prepared owned bytes | Prepared pinned bytes | FP32 output bytes |
|---|---:|---:|---:|---:|---:|
| (256, 1024) | 262,144 | 4,096 | 270,340 | 262,144 | 1,048,576 |
| (1024, 4096) | 4,194,304 | 16,384 | 4,227,076 | 4,194,304 | 16,777,216 |
| (4096, 1024) | 4,194,304 | 4,096 | 4,202,500 | 4,194,304 | 16,777,216 |

Torch has the same wire bytes; its prepared host payload is INT8 plus one
FP32 scale copy. Sym also charges the native arithmetic scale copy and
zero point. The original checkpoint remains caller-owned. See the API
document for metadata, transient binding and allocator-cache exclusions.

| Model | Prepared owned MiB | Prepared pinned MiB | Peak queue output MiB | Peak native scratch MiB |
|---|---:|---:|---:|---:|
| llm_decode | 48.281 | 48.000 | 96.000 | 13.820 |
| moe_sparse | 64.375 | 64.000 | 32.000 | 4.523 |

Queues have zero outstanding handles after each window, and create one
retained native context per model owner. The DMA trace independently
confirms compact weight/scale traffic and one `dequantS8F32Kernel` per
load. Native host-transform and packing counters are zero for this path.

## Protocol and limits

- EPYC 7351, RTX 2080 Ti (GPU 0), CUDA toolkit 12.5, driver 595.71.05,
  Torch 2.14.0+cu126, CPython 3.14.7. Affinity `4-7,20-23` (four physical
  cores, eight hardware threads); Torch/OMP/MKL threads 8, interop 1;
  `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, Inductor compile threads 1.
  TF32 is disabled. Clocks use normal boost, with no locked-clock claim.
- Implementation `ac607d9`; native runtime changes are identical to `2b95104`.
  Merged-main baseline `4ffe900` uses the preserved #223 build (its runtime and
  frontend sources match merged main). Full revisions, binary/frontend hashes,
  harness hash, environment and path order are in `evidence/provenance.jsonl`.
- Three fresh processes per revision/control, five warmups, 30 completed warm
  samples per method/case. Path order rotates between candidate processes and
  within warm loops. Main and candidate rounds alternate, followed by a
  candidate process running only the same three raw methods as main. This
  isolates the existing path from the different allocation/cache footprint of
  the six-method process. All local GPU jobs run serially; competing builds and
  packaging jobs must finish before qualification. Interrupted/contended
  attempts are excluded. No profiler or tests run during these timings.
- Shapes and cases were chosen before qualification: matrices `[256,1024]`,
  `[1024,4096]`, `[4096,1024]`; four-layer GPT decode (four tokens, prompt 8,
  width 1024, FFN 4096); top-2 MoE (one token, two blocks, eight experts/block,
  width 1024, FFN 2048). Checkpoints are immutable during timed reuse.
  Model compute is common Torch eager. **Inductor compiles weight conversion,
  not the whole model.** All candidate outputs and model tokens/routes compare
  exactly, outside timing. Tests separately cover explicit checkpoint changes.
- All methods use the same signed INT8 values, FP32 per-channel scales and
  contiguous FP32 GPU outputs. Original checkpoint generation/pinning is common
  untimed setup. Torch and Inductor receive their own equally prepacked INT8
  snapshots and scale copies. No method is given an FP32-expanded checkpoint.
- Each queue has two output slots within 128 MiB and 64 MiB transfer scratch.
  Sym's prepared and pinned limits are separately 1 GiB. Counters record live
  payload ownership, not total RSS or allocator reservations. Torch controls
  use the same queue/output/scratch limits as #223; their scratch gauge is a
  conservative upper bound. Original checkpoints, model state, caller-retained
  outputs and allocator caches are outside those owner limits.
- Matrix warm timing includes fresh preparation of the invocation, allocations,
  uploads, dequantization, completed wait and handle retirement. Both Sym matrix
  paths use `wait()`; Torch likewise host-waits and records output allocator use
  on the consumer stream. Models use the same two-slot lookahead on all paths.
- First preparation and first completed load are recorded per shape/path, with
  artifact setup separate. Earlier cases can warm allocators; first Inductor
  calls can hit disk caches. These are not fully cold-system/compiler results.
  Reuse sweeps at 1,2,4,8,16,32,64 loads have five total-time samples each and
  include preparation, completed loads and invalidation, with artifacts/JIT
  already initialized. Replacement and invalidate/reprepare have separate
  30-sample series. Reported break-even concerns this measured reuse context.
- Lossless samples, p50 and interpolated p95 are compact CSV rows containing
  sample arrays. Unused reuse sweeps from the raw-only regression controls are omitted.
  Scalar metadata/counters are JSONL. Logs, generated kernels,
  raw benchmark JSON and profiler databases are not included.

## Reproduction

Configure/build the candidate and keep the main build separate. Use the
qualified interpreter, exporter and Python package from the selected build:

```sh
export PYTHONPATH=/tmp/sym-224-build/python
export SYM_RELOC_EXPORT=/tmp/sym-224-build/sym/tools/sym-reloc-export
export SYM_RUNTIME_REVISION=ac607d9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0 TORCHINDUCTOR_COMPILE_THREADS=1
taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python bench/issue224/prepared_wire.py \
  --output /tmp/sym-224-final-candidate1.json --samples 30 --sweep-samples 5 --models
```

Run three independent rounds, rotating the six `--paths` by two places each
round. Run main and the candidate's separate existing-path control with
`--paths sym_raw torch_raw inductor_raw` and without `--models`. Wait for an
otherwise idle CPU/GPU host; do not overlap these commands. Compact the nine
accepted files with `compact_evidence.py --output bench/issue224/evidence`.
The diagnostic DMA trace uses Nsight Systems 2024.2.3.38, `--trace=cuda,nvtx
--sample=none --cpuctxsw=none --capture-range=cudaProfilerApi
--capture-range-end=repeat:3`, and the harness options `--paths sym_packed
--trace --samples 1 --sweep-samples 1`. Join GPU copies/kernels to runtime calls
by correlation ID, retaining calls contained in the named `issue224/...` NVTX
ranges. The compact trace records database hashes and correlations. It confirms
INT8/scale bytes and one dequantization kernel per load; its timings under
competing CPU load are not performance evidence. The native binary is unchanged
by the later Python diagnostic-cache refinement.

## Validation

The full CUDA runtime/frontend suite passed 1,176 tests before the six additional
wire-failure cases; all six new cases passed. The final refinement passed the
42-test ownership/fault subset, five model tests and nine CPU-only snapshot tests.
The broader workload suite passed 58 tests, CPU-only suite 831 tests, CTest 6/6
for both builds, and compiler lit 41/41. GitHub's build, native tests, Torch CPU
frontend, CPU wheel, clang-format, clang-tidy and cost-model checks also passed
on the frozen implementation.
