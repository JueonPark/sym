# Completed-request attribution for issue #263

This directory contains the benchmark-only investigation of [#263](https://github.com/JueonPark/sym/issues/263). It compares Sym with the matched Torch control using the frozen FP16 E2/E5 workloads. The production runtime remains at `2617ba166f98a7476a5f472d829a573317fc0cfb`.

**The 150-process campaign is complete.** Read the [results and limitations](RESULTS.md), [compact result index](result-index.json), and [public artifact release](https://github.com/JueonPark/sym/releases/tag/profile-issue263-2026-11-07-v1). Pilot timings are excluded from acceptance.

## What the experiment changes

`no_report` suppresses completion report materialization and accounting. It preserves native completion, output stream recording, ownership, and retirement. Its accounting counters are intentionally incomplete; this is a diagnostic intervention, not a qualified replacement API.

`no_check` suppresses repeated Python source-signature checks for the already-qualified metadata while retaining native validation. `both` measures their combined effect directly. Neither bypass changes kernels, grouping, prefetch depth, compute precision, or the mathematical workload. Production validation must be retained in any future optimization.

`off` installs no phase wrappers and runs without a CUDA profiler. `counters` adds exclusive host timing. `nvtx` adds request/group identifiers and CUDA tracing. These modes have separate measurements: diagnostic timings are not used as production performance estimates.

## Reproduce

Download and verify the [frozen E1–E5 artifact](https://github.com/JueonPark/sym/releases/tag/experiments-2026-11-04-v1). Its native helper and the published runtime hashes correspond to the recorded Python 3.14 / Torch 2.14 / CUDA 12.6 environment. Preserve the extracted artifact unchanged. The new release also bundles the verified runtime under `runtime/`. Use a Git checkout of the release tag for new runs (benchmark provenance reads the Git revision); the bundled `source/` supports offline inspection and reanalysis. Supply explicit paths to the frozen input artifact, runtime build, Python interpreter, Nsight executable, and an empty output directory:

```sh
python3 bench/issue263/run.py \
  --artifact /path/to/sym_experiments_261104 \
  --build /path/to/sym/build/perf \
  --python /path/to/venv/bin/python \
  --nsys /path/to/nsys \
  --output /path/to/new-campaign

python3 bench/issue263/analyze.py /path/to/new-campaign \
  --output /path/to/summary.json --traces
```

The runner defaults to GPU 0 and CPU affinity `4-7,20-23`, the original machine's eight logical CPUs. `--device` and `--affinity` configure the launch, but the frozen concurrency runner also fixes its internal caller CPU partition to the original topology. Requalify that runner and record the changed artifact for a different topology; do not silently claim the original protocol on another machine. Nsight must capture CUDA runtime APIs, device activity, and NVTX.

A new runtime build normally has different binary hashes. Individual benchmark CLIs accept `--allow-new-build` to record that difference, but the registered baseline campaign and its analyzer require the original hashes. Rebuilding the historical native helper for a different Python/Torch ABI also requires a separately recorded artifact manifest. This strict mode is intentional: it prevents a changed baseline from being presented as a reproduction of this attribution result.

The runner uses an allowlisted launch environment, serializes jobs, rotates arm order, assigns fresh Inductor caches, and audits foreign GPU/build processes every three seconds. A failed or contaminated run is preserved and stops the campaign. Completed jobs can be resumed against the same `matrix.json`; incomplete jobs must be investigated and preserved separately before rerunning. The matrix includes all named interventions, counter-overhead contrasts, and diagnostic traces. See [PROTOCOL.md](PROTOCOL.md) for the registered counts and selection rule.

## Evidence and aggregation

- `models.py` runs E2 with common compute functions and numerical oracles. Separate post-timing allocator snapshots distinguish Torch allocation requests, live Python blocks, and native retained resources.
- `lifecycle.py` and `concurrency.py` execute the frozen E5 programs through `e5.py`; their numerical, revision, physical-budget, and held-output checks are retained.
- `trace.py` assigns GPU work by process-scoped CUDA correlation ID and launch thread, not temporal containment. It keeps event-record/wait evidence, group IDs, streams, host-exclusive phases, and interval unions.
- `paired.py` bootstraps independent process pairs. `analyze.py` retains per-process p50/p95 and all raw records; request samples are not treated as independent process replicates.
- `timeline.py` uses matplotlib (an optional reporting dependency) to export an SVG or PNG for a selected request from a compressed trace-analysis JSON. The raw `.nsys-rep` and SQLite remain the authoritative interactive trace.

Host phases and GPU intervals overlap. Their durations must not be added, and host time coincident with GPU idle is not automatically removable. Native CPU internals remain opaque because the unchanged runtime was built without native NVTX. Sampled physical-memory peaks are lower bounds; Python live blocks are net counts, Torch allocation requests can reuse cached storage, and native counters have a different scope. None forms an additive memory decomposition.
