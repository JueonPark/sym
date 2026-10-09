# Completed-cost placement (#227)

## Protocol fixed before acceptance runs

The selector ranks qualified paths using whole completed calls. Calibration
and held-out evaluation use separate sizes. Matrix calibration uses rows
64/256/1024/4096 and columns 256/1024; held-out shapes are 128×512, 512×512,
2048×512 and 2048×1024. KV calibration uses sequence lengths 4/8/32/128/512;
held-out lengths are 12/24/64/256, with 16 heads of width 64. Both directions
are measured, with matrix layout-only and FP32→FP16 variants.

Each of three fresh processes calibrates independently. Each forced path gets
three cold-owner samples and 30 warm samples, after three warmups. Cold-owner
means `TransferResources.clear()` before the call, excluding clear itself;
compiler and global Torch allocator are already initialized. This is not a
fresh-process allocation measurement. Warm automatic selection first warms
all qualified alternatives at the held-out shape. The first automatic call
after owner clear is also retained separately. Model inference exercises
natural repeated-call reuse rather than warming unchosen alternatives.

Controls include the original eager Torch region, explicit CPU-transform
Inductor and explicit GPU-transform Inductor, and the existing unprofiled Sym
backend. Each has identical input/output precision and fresh output ownership.
CPU/GPU cuts can send different wire bytes; each has a matching Inductor cut.
The largest source is 16 MiB and each candidate's transform/transfer temporary
storage is bounded within the common 64 MiB scratch allowance. Output size is
at most 16 MiB. All Sym owners use 64 MiB retained/live scratch limits. Torch
uses its caching allocator; the bound follows from these fixed shapes.

All timings synchronize the request's current stream and include allocation
and frontend costs. Correctness checks happen outside timing. Inductor
compilation happens before timing and CUDA graphs/TF32 are disabled. Controls
rotate within samples; forced-path order rotates between rounds. Selection
alone is timed separately without binding, copying or GPU work. Held-out forced
paths and automatic selection also run in the same rotated interleaved loop,
so regret compares equal cache/allocator pressure. Regret is automatic median /
best qualified interleaved forced warm median − 1, including selector overhead;
path regret excludes that overhead. Report all misses; exploratory runs are
excluded from acceptance. An initial run using consecutive forced samples for
regret was discarded because its scheduling differed from automatic samples.

Hypotheses: reduce small KV latency by at least 30% versus the current Sym
backend; keep median held-out regret within 10% and p95 within 20%. These are
targets, not prerequisites for claiming an implementation or guaranteed gains.
Also evaluate a held-out GPU/thread profile and explicitly tagged load/context
changes. No builds, tests or unrelated GPU jobs run during headline timing.
Three GPT decode comparisons reuse #225's model and matched weight delivery.

```sh
export PYTHONPATH=/tmp/sym-227-build/python
export SYM_RELOC_EXPORT=/tmp/sym-227-build/sym/tools/sym-reloc-export
export SYM_OPT=/tmp/sym-227-build/sym/tools/sym-opt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/sym227-round1 taskset -c 4-7,20-23 \
  /tmp/sym-210-venv/bin/python bench/issue227/placement.py \
  --round 1 --output /tmp/sym227-round1.json
```

Raw timing arrays and scalar metadata are retained compactly. Build/test logs,
compiler caches and profiler databases are excluded from the PR.

## Results (2026-10-09)

Measured implementation: `cb647e2`, based on main `1c27503`. Release CUDA build,
EPYC 7351, RTX 2080 Ti GPU 0, driver 595.71.05, CUDA toolkit 12.5,
Torch 2.14.0+cu126, CPython 3.14.7. Affinity 4–7,20–23, eight Torch/OpenMP
threads, interop one, passive OpenMP waits; GPU clocks were not locked.
All 15 final invocations observed no foreign build/packaging or GPU processes.
The three main sweeps each took 59 seconds; each decode comparison took 44
seconds. Hardware/build hashes and complete settings are in
[provenance.jsonl](evidence/provenance.jsonl).

Values below are medians of the three round medians, in **milliseconds**.
Torch is the original eager region. The two Inductor columns explicitly place
the transform on CPU or GPU. Old Sym is the existing unprofiled backend on the
same build. Automatic is the new opt-in Sym frontend policy, including its
guards, selection, allocation and completion overhead. Torch alternatives use
Torch's allocator (including pageable host intermediates); Sym uses the stated
pinned scratch configuration. These are complete implementations with the same
memory bound, not an assertion that their allocation strategies are identical.

| Transfer / source shape | Torch | Inductor CPU | Inductor GPU | Old Sym | Automatic | vs old Sym |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KV H2D, 12×16×64 | 0.076 | 0.322 | 0.191 | 0.489 | 0.457 | −6.4% |
| KV D2H, 16×12×64 | 0.095 | 0.327 | 0.202 | 0.478 | 0.467 | −2.5% |
| FP32 transpose H2D, 128×512 | 0.182 | 0.380 | 0.220 | 0.543 | 0.501 | −7.7% |
| FP32 transpose H2D, 2048×1024 | 4.887 | 2.672 | 1.357 | 2.414 | 1.721 | −28.7% |
| FP32 transpose D2H, 2048×1024 | 1.085 | 2.564 | 1.320 | 2.508 | 1.766 | −29.6% |
| Transpose + FP16 cast H2D, 128×512 | 0.272 | 0.268 | 0.245 | 1.705 | 0.599 | −64.8% |
| Transpose + FP16 cast H2D, 2048×1024 | 5.339 | 1.896 | 1.349 | 26.430 | 1.769 | −93.3% |
| Transpose + FP16 cast D2H, 2048×1024 | 0.712 | 2.220 | 0.848 | 16.722 | 1.330 | −92.0% |

The best-placement gains are large for typed transposes, but automatic still
loses to the best Torch/Inductor control in these rows. Small KV gains miss the
30% target. Of 72 main held-out decisions, 56 choose `torch_gpu`, 12 choose
`torch_cpu`, and four choose `native`; no Sym kernel row wins the calibrated
cost comparison on this machine. Thus these gains measure placement and Torch
bypass, not a claim that libreloc's kernels outperform Torch.

### Selection accuracy and overhead

| Round | Total regret p50 / p95 | Path-only regret p50 / p95 | Selector p50 |
| --- | ---: | ---: | ---: |
| 1 | 9.43% / 14.97% | 0% / 0.29% | 52.13 µs |
| 2 | 9.52% / 15.15% | 0% / 0.56% | 50.82 µs |
| 3 | 9.97% / 14.51% | 0% / 0% | 50.97 µs |

Across all 72 decisions, total regret p50/p95/max is **9.71% / 15.76% / 18.82%**;
path-only regret is **0% / 0.53% / 3.87%**. The 10% median / 20% p95 targets
pass. Median selector p50/p95 across cases is **51.21 / 59.63 µs**. The warm
prediction cache avoids repeating interpolation, while hardware settings,
context and owner generation are checked each time. Guards/custom-op dispatch
and output validation remain significant additional overhead, visible in the
gap between the Torch path inside Sym and direct eager Torch.

`p95` for the transfer harness is the empirical lower quantile,
`sorted[floor(0.95*(n−1))]`; model timings retain #225's interpolated percentile.
Three-sample cold sets describe allocation behavior, not a reliable tail
estimate. Cold-owner automatic medians are 0.654 / 0.675 ms for the small KV
H2D/D2H cases, and 2.019 / 1.569 ms for the 2048×1024 narrowing cases. The
warm automatic p95s for those four cases are 0.470 / 0.475 / 1.795 / 1.362 ms.
Every forced cold/warm distribution and cold automatic decision is retained.

### Crossovers, GPU narrowing and context compatibility

Forced warm calibration exposes the size crossover. For KV H2D, CPU/GPU Torch
transforms cost **0.388/0.409 ms** at sequence length four, then **0.860/0.638 ms**
at length 512. For narrowing H2D, the CPU/GPU Torch pair changes from
**0.418/0.431 ms** at 64×256 to **8.891/2.499 ms** at 4096×1024. CPU narrowing
sends two bytes per element; upload-first GPU narrowing sends four. The output
remains the same FP16 tensor in both cases.

The newly eligible Sym GPU narrowing row at 4096×1024 H2D costs **2.728 ms**
versus **42.159 ms** for its CPU reference, a **15.5×** forced-path speedup. It
still trails Torch GPU transformation at 2.499 ms. D2H improves only from
46.507 to 36.005 ms because that Sym row retains CPU layout work; Torch GPU
transformation takes 1.765 ms. GPU cast availability alone does not remove the
remaining D2H layout bottleneck.

Controlled contention uses a single NumPy addition worker on the benchmark's
CPU affinity, or a separate CUDA stream repeatedly executing 2048² FP32 GEMM
with one operation outstanding. Each context is calibrated independently at
64×512 and 256×512, then tested at held-out 128×512. Three repetitions give:

| Context | Idle-profile fallback | Matching-profile automatic | Choice | Forced Torch CPU / GPU |
| --- | ---: | ---: | --- | ---: |
| CPU worker busy | 0.622 ms | 0.501 ms | GPU, all three | 0.567 / 0.459 ms |
| GPU GEMM busy | 0.669 ms | 0.662 ms | CPU, all three | 0.555 / 0.729 ms |

An idle-only profile reports `profile_missing_native_coverage` under either
busy context and uses the controlled CPU replay. Load calibration changes the
choice where justified. These forced context samples are consecutive and
describe the crossover; the headline regret statistics use the separately
interleaved idle measurements. Busy labels describe these measured loads, not
all possible contention levels. No overlap speedup is claimed: requesting
`overlap=consumer` with a `none` profile also falls back (0.588 ms).

Applying GPU 0's profile to held-out GPU 1 reports
`profile_hardware_mismatch` and conservatively uses Torch CPU (0.601 ms).
Independently calibrating GPU 1 with **one** Torch/Sym worker and evaluating its
24 held-out shapes gives total regret p50/p95/max **9.12% / 12.88% / 16.67%**,
and path-only p50/p95/max **0% / 0% / 0.15%**. Both GPUs are RTX 2080 Ti; this
demonstrates device/thread-profile handling, not generalization to a different
GPU architecture. Hardware, execution-option, wire, context, owner-clear and
missing-shape rejection also have focused tests.

### End-to-end GPT decode

This reuses #225's four-layer GPT, four decode steps, identical compact INT8
weight snapshots, two-slot prefetch, model arithmetic and exact-token checks.
Placement applies to the 48 KV transfers; grouped/prepared weight delivery keeps
its existing policy. All 1,728 timed/warm/setup decisions per placement path
choose `torch_cpu` in each round. Outputs agree, with no warm recompilation,
graph breaks or runtime fallbacks.

| Path | Round 1 p50 | Round 2 p50 | Round 3 p50 | Median p50 | Median p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch | 27.012 | 26.202 | 27.075 | 27.012 | 27.125 |
| Inductor | 29.556 | 29.895 | 29.977 | 29.895 | 30.343 |
| Sym + Inductor | 43.504 | 42.709 | 43.479 | 43.479 | 43.801 |
| Sym placement + Inductor | 41.871 | 42.837 | 42.071 | 42.071 | 42.660 |

Aggregate latency improves **3.2%**, with per-round changes **−3.75%, +0.30%,
−3.24%**. This is a modest aggregate gain with a regression in round two,
and it remains slower than both controls. The separate 48-transfer KV diagnostic
changes from 19.040 to 18.393 ms (Torch 2.175 ms, Inductor 7.515 ms). These
serialized diagnostics cannot be subtracted from end-to-end latency.

First completed model calls, including compilation but after common model and
snapshot setup, have medians 30.1 ms / 11.377 s / 2.447 s / 1.907 s in the table's
path order. These are not independently cold-process measurements; path order
rotates and shared compiler caches affect later paths. Separate snapshot/owner
preparation, first-call compile data and warm counters are in the evidence.

### Reproduction and retained evidence

Run the full serialized campaign (fresh output directory required):

```sh
/tmp/sym-210-venv/bin/python bench/issue227/run.py \
  --build /tmp/sym-227-build --output /tmp/sym227-results
/tmp/sym-210-venv/bin/python bench/issue227/compact.py \
  --input /tmp/sym227-results --output /tmp/sym227-evidence
gzip -n /tmp/sym227-evidence/timings.csv
```

- [timings.csv.gz](evidence/timings.csv.gz): 64,572 raw timing observations,
  grouped into 3,146 CSV rows with p50/p95; read with `gzip -dc` or Python's
  `gzip.open(..., 'rt')` and `csv.DictReader`.
- [decisions.jsonl](evidence/decisions.jsonl): predictions, actual wire bytes,
  cold decisions, selection overhead, regret and model correctness/counters.
- [provenance.jsonl](evidence/provenance.jsonl): 15 invocation fingerprints,
  source revision, settings, controlled-load evidence and isolation checks.

Validation: 1,203 runtime/frontend tests pass across the full run and targeted
reruns (four skips); 58 workload tests; all six CTest targets; 41 compiler tests.
CUDA narrowing checks 14 boundary witnesses and 65,539 random FP32 bit patterns.
All seven CI checks passed for the measured implementation. Larger gains over
native Torch still require reducing frontend/call overhead and improving the
remaining transform implementations; this change does not establish the
project-level performance win over both Torch controls.
