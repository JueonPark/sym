# Typed transform/H2D pipeline — issue #221

On the qualified EPYC 7351 / RTX 2080 Ti host, two buffers reduce large-transfer
warm p50 by **26–37% against the prior typed runtime**, **25–36% against the
same-build whole-buffer control**, and **21–32% against Torch with retained
pinned staging**. Native traces establish CPU-transform/actual-DMA overlap in
both CPU-reference and CPU/GPU-split rows. Small transfers remain slower than
Torch. These are completed transfer components, **not model inference or a
general win over full-model `torch.compile`**.

## Protocol and versions

- Runtime implementation: `79254f81032600b71b6e3a70aa8340fb82d34c2e`, based on
  merged main `4ede4bc98a871707e333504727a2e3552b36b973`.
  Prior binary: `dbb229cd3632eb301799dbdcbfe18332eb477b67`;
  `git diff dbb229c..4ede4bc -- libreloc sym` is empty. It was not rebuilt.
- Release, GCC 11.4.0, CUDA toolkit 12.5.82, native architectures 75/89;
  NVTX **off** for timings, separate NVTX-on build for Nsight 2024.2.3.38.
  Python 3.14.7, Torch 2.14.0+cu126, driver 595.71.05, Linux 6.8.0-124-generic.
- GPU 0, CPU affinity `4-7,20-23` (four physical cores, eight hardware threads
  on GPU-local NUMA node 1), Torch/gather/OMP/MKL eight threads, interop one.
  GPU clocks were unlocked; recorded starting clocks are in provenance.
- Three independent processes per revision, eight warmups, 100 samples per
  case/path/round. Candidate path order rotates between rounds and samples;
  revision order is old/new, new/old, old/new. GPU jobs are serial; builds,
  tests and profiling do not overlap headline runs.
- One stream, pinned staging, identical 128 MiB live/retained native scratch
  budgets per owner. Large default chunks are 1 MiB. One/two/four-buffer
  controls have the same schedule: 33 chunks for cast/split (six-byte tail),
  32 for transpose. Small calls take the whole-buffer fallback.
- Time includes fresh output allocation, preparation/binding/dispatch, CPU
  work, copies and completion. Input sign changes, byte-exact oracle checks,
  checking previously retained outputs, recipe compilation and Inductor
  compilation/priming are outside timing. All oracle checks passed.
- Rows are explicitly selected to isolate this feature. `cast` is f32→f16,
  16,777,219 elements; `transpose` swaps axes 0/1 of `[128,128,1024]` then
  casts f32→f16; `split` casts f32→f16 on CPU, uploads f16, widens on GPU.
  Large source/wire sizes are about 64/32 MiB. `small` is a 65,539-element
  cast, about 128 KiB on the wire. Both rounding stages are preserved.
- Torch eager materializes the CPU wire result then uploads it. Torch pinned
  converts directly into its retained pinned destination before uploading.
  Inductor compiles the CPU transformation; upload and GPU widening remain
  ordinary Torch operations. All controls use the same wire precision/bytes.
  Scalar quantization/dequantization performance and full workloads are not
  measured here; their chunk semantics are covered by correctness tests.

## Warm completed latency

Milliseconds, **p50 / p95**; each entry is the median of the three independent
rounds' percentiles, not a pooled percentile or a confidence interval.

| Path | Small cast | Large cast | Transpose + cast | CPU/GPU split |
|---|---:|---:|---:|---:|
| Prior runtime | 0.504 / 0.711 | 7.086 / 7.167 | 7.147 / 7.209 | 7.312 / 7.388 |
| Whole-buffer control | 0.287 / 0.296 | 6.955 / 7.129 | 7.100 / 7.220 | 7.213 / 7.382 |
| One buffer | 0.279 / 0.291 | 6.305 / 6.576 | 7.185 / 7.366 | 6.417 / 6.696 |
| **Two buffers** | **0.282 / 0.293** | **4.472 / 4.719** | **5.309 / 5.473** | **4.586 / 4.808** |
| Four buffers | 0.281 / 0.292 | 4.761 / 4.961 | 5.919 / 6.103 | 4.844 / 5.160 |
| Torch eager | 0.077 / 0.085 | 13.861 / 14.217 | 23.401 / 23.882 | 13.894 / 14.479 |
| Torch retained pinned | 0.053 / 0.057 | 6.523 / 6.612 | 6.696 / 6.803 | 6.761 / 6.940 |
| CPU Inductor + upload | 0.173 / 0.188 | 14.158 / 14.577 | 14.391 / 14.870 | 14.280 / 14.681 |

Two versus one buffer improves the large calls by 26–29%. Whole versus one
buffer also changes cache behavior and scheduling costs, so the entire
old/new gain must not be attributed to overlap alone. One-buffer transpose is
slightly slower than whole execution. Four buffers consume more memory and
are slower than two in these cases.

The small old-runtime process runs only the legacy path, whereas the candidate
interleaves all controls. Its different duty cycle/process state and unlocked
clocks confound that cross-process small-call difference. The matched
whole/two-buffer numbers show **no small-call pipeline gain**; both execute
whole buffers and remain about five times slower than retained-pinned Torch.

## First completed calls

Milliseconds, min–max of the first call in each of three rounds. These are
first uses of each method/case with fresh Sym owners and independent recipe
caches, after compiler/CUDA initialization. Process-wide Torch allocation
caches can persist from earlier cases: notably, the split Torch-pinned call
reuses cached host capacity. These are **not cold-process measurements**.
Inductor compile/prime observations are recorded separately in the CSV and
may load existing compiler caches.

| Path | Small cast | Large cast | Transpose + cast | CPU/GPU split |
|---|---:|---:|---:|---:|
| Prior runtime | 4.78–4.87 | 42.96–43.11 | 43.85–44.07 | 44.60–44.67 |
| Whole-buffer control | 1.09–4.19 | 41.67–43.41 | 42.69–43.10 | 42.59–44.27 |
| One buffer | 1.04–1.26 | 7.06–9.45 | 7.99–10.19 | 7.47–10.11 |
| Two buffers | 1.05–4.16 | 7.59–9.52 | 8.58–10.94 | 8.75–10.10 |
| Four buffers | 0.99–1.21 | 13.79–14.41 | 15.34–15.60 | 14.29–14.71 |
| Torch eager | 0.10–0.65 | 13.17–13.43 | 22.25–23.20 | 34.54–35.73 |
| Torch retained pinned | 0.14–2.65 | 67.01–68.18 | 37.12–37.64 | 6.82–6.84 |
| CPU Inductor + upload | 0.30–0.31 | 13.43–13.50 | 13.62–13.83 | 13.82–14.09 |

## Memory and overlap

Measured retained host allocation capacities, MiB:

| Path | Large cast / split | Transpose + cast |
|---|---:|---:|
| Prior / whole | 36.25 | 36.00 |
| One buffer | 1.25 | 1.25 |
| Two buffers | 2.50 | 2.50 |
| Four buffers | 5.00 | 5.00 |
| Torch retained pinned | 64.00 | 32.00 |

Native capacities include arena growth/rounding. Torch's isolated host
allocator counters show its different rounding for the six-byte tail; see
`torch-memory.json`. In `timings.csv`, the Torch-pinned host byte column is
the logical tensor size, not that rounded allocator capacity. The native
split additionally retains **36.25 MiB device scratch** in every variant;
its GPU stages still wait for all uploads. Logical outputs are about 32 MiB
(cast/transpose), 64 MiB (split), and 128 KiB (small), separately allocated.
These are scratch/live-output figures, not process RSS or total allocator
caches; the timing harness also retains old logical outputs for correctness.

Actual H2D intervals are joined to submission ranges by CUDA correlation ID.
CPU ranges execute inside workers and are unioned before intersection with
**DMA(n) / transform(n+1)**. Five traced calls per case/control use identical
33-chunk schedules:

| Case | One-buffer overlap | Two-buffer overlap |
|---|---:|---:|
| CPU-reference cast | 0 in every call | 1.960–2.046 ms per call |
| CPU/GPU split | 0 in every call | 1.909–2.028 ms per call |

Trace timings are excluded from headline latency. All-call summaries and
SQLite hashes are retained. CSVs retain the first call of each control with
native worker unions, actual DMA timestamps, correlation IDs and chunk bytes;
this selection is deterministic. The analyzer can emit every interval when
`--first-per-path` is omitted. Profiler databases are not committed.

## Configuration exploration

The 12-sample prototype sweep informed the defaults; it is **not** part of the
three-round headline claim. These exploratory binaries predate the committed
default tuning/fallback refactor; their hashes are preserved, and their
metadata's repository HEAD is the parent, not an assertion of main-runtime
identity. Reproduce the configurations against the committed implementation
with `--chunk-size`, `--threads`, `--pinning` and `--min-pinned-bytes`.

Large-cast warm p50 in ms, pinned/eight workers unless specified:

| Chunk | Configuration | Whole | One buffer | Two buffers | Four buffers |
|---|---|---:|---:|---:|---:|
| 256 KiB | pinned, 8 workers | 6.956 | 8.846 | 6.780 | 6.384 |
| 1 MiB | pinned, 8 workers | 6.973 | 6.288 | 4.546 | 4.719 |
| 2 MiB | pinned, 8 workers | 6.975 | 5.784 | 4.726 | 5.584 |
| 4 MiB | pinned, 8 workers | 6.961 | 6.019 | 5.704 | 5.748 |
| 8 MiB | pinned, 8 workers | 6.973 | 7.032 | 5.909 | 5.874 |
| 4 MiB | pinned, 1 worker | 8.713 | 8.794 | 7.108 | 7.085 |
| 4 MiB | pinned, 4 workers | 6.883 | 5.762 | 5.486 | 5.588 |
| 1 MiB | pinned, 4 workers | 6.911 | 5.995 | 4.229 | 4.275 |
| 1 MiB | pageable, 8 workers | 7.753 | 8.291 | 6.631 | 8.326 |
| 4 MiB | pageable, 8 workers | 7.700 | 6.860 | 7.291 | 7.401 |
| 4 MiB | unconfigured auto | 7.719 | 6.861 | 7.304 | 7.395 |
| 4 MiB | auto, 1 MiB threshold | 6.969 | 5.916 | 5.648 | 5.734 |

Smaller chunks have a scheduling cost; four pageable slots can regress.
Two slots bound memory without that regression in the sampled settings.
Worker counts remain explicit: four workers were best for this cast pilot;
the final comparison holds all paths at eight. The transpose pilot favors
2 MiB/eight workers (4.763 ms with two slots) over 1 MiB (5.309 ms in final
rounds); an override is useful because physical row width limits parallelism.
These observations motivate a configurable heuristic, not a universal optimum.

## Reproduction and validation

With the chosen CUDA build staged in `PYTHONPATH` and its exporter selected:

```sh
export PYTHONPATH=/tmp/sym-221-build/python
export SYM_RELOC_EXPORT=/tmp/sym-221-build/sym/tools/sym-reloc-export
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 4-7,20-23 /tmp/sym-210-venv/bin/python \
  bench/issue221/typed_pipeline.py --output /tmp/round.json --samples 100
```

Use `--paths legacy` with the prior build; rotate candidate paths by two
positions per round. Use `--case cast` or `split`, `--paths ring1 ring2`,
`--samples 1 --trace` with the NVTX build under:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/trace <command>
nsys export --type=sqlite --output=/tmp/trace.sqlite /tmp/trace.nsys-rep
python bench/issue221/analyze_trace.py /tmp/trace.sqlite /tmp/overlap --first-per-path
python bench/issue221/torch_memory.py --case cast --output /tmp/memory.json
python bench/issue221/compact_evidence.py --output-dir /tmp/compact /tmp/round*.json
```

`timings.csv` preserves every warm sample, first calls, percentiles, schedule
and native memory counters. `provenance.jsonl` and `trace-provenance.jsonl`
preserve run configuration and binary/frontend/compiler/recipe hashes.

Validation on the committed implementation: **1,042 Python tests passed**
(four skips), **53 workload tests passed**, **6/6 CTest checks passed**
(321 runtime tests and 61 plan-builder tests; two unavailable AVX-512 skips).
CPU-only build: **746 Python tests passed** (one skip, 299 GPU deselections),
**299 native tests passed** (two AVX-512 skips). New qualification includes
byte-exact changing shapes/values/channel parameters, pad/cast/quantization
boundaries, tails, one/two/four slots, budget limits, caller-stream ordering,
retained output ownership and injected event/copy failures with successful
drain or owner quarantine. Verbose logs are not part of this PR.
