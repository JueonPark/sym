# Workload profiling after the pinning-policy patches

These tools compare unchanged PR #162 workload bodies with the current Sym
runtime. They do not merge the examples or change production execution.
Results from main `519f6a1` are in
[`../results/main-workload-profile-20261002`](../results/main-workload-profile-20261002).

`run.py` runs one phase at a time, serially on GPU 0 with CPU affinity
`4-7,20-23`. That affinity is specific to the EPYC 7351 machine in the report;
adapt it for another host. It records the commands and samples CPU utilization
and the background compiler/simulator processes observed during this session.
It does not reserve the host or detect every kind of external load.

```sh
export SYM_RELOC_EXPORT=/path/to/build/sym/tools/sym-reloc-export
export SYM_OPT=/path/to/build/sym/tools/sym-opt
python bench/workload_profile/run.py --phase controls \
  --workloads /path/to/pr162 --build /path/to/normal-build --output results/controls
python bench/workload_profile/run.py --phase measure --rounds 3 \
  --workloads /path/to/pr162 --build /path/to/normal-build --output results/measurements
python bench/workload_profile/run.py --phase profile \
  --workloads /path/to/pr162 --build /path/to/trace-build --output results/profiles \
  --nsys /path/to/nsys
python bench/workload_profile/analyze.py results/profiles
python bench/workload_profile/report.py results
```

Use a Python environment with CUDA-enabled Torch and a Sym Python extension
built for that interpreter. `run.py` selects the build's Python packages. The
normal build must disable NVTX; the diagnostic build enables NVTX and applies
the archived `build/native-annotations.patch`. Build both with Release and CUDA
enabled, targeting `reloc_runtime` and `pyreloc_ext`. The captured CMake caches
and build logs record the exact toolchain used. The report generator requires
Matplotlib and NumPy; it consumes three rounds and the six named captures.

- `capture.py` annotates transfers, untimed oracle checks and model regions. Its
  Python annotations and the extra native ranges affect latency, so its timings
  are for attribution only. It preserves the examples' Torch-first order.
- `analyze.py` correlates each GPU activity with its CUDA API using process and
  correlation IDs. It partitions caller wall time into disjoint phases and
  checks ordered model kernel signatures. GPU time is a separate view, never an
  extra term added to a blocking host API.
- `controls.py` measures complete transfers with fresh outputs and checked
  results. Its explicitly named `sym_prepared_outside_timer` rows exclude
  preparation and are diagnostic bounds, not end-to-end speedups. Its CPU call
  profiles use calling-thread CPU time and incur profiler overhead.
- `report.py` produces the summary, a standalone HTML report with embedded
  figures, and PNG/SVG/PDF exports. The narrative describes the archived
  2026-10-02 run; revise its numerical interpretation for a different run.

The workload matrix reuses `bench/issue189/run_matrix.py` and
`measure_example.py`, with `default,typed_reuse` variants. The latter injects an
explicit resource owner into direct typed dispatch calls. Frontend-managed
resources remain automatic in both variants. Pinning is auto with no configured
threshold, which selects pageable staging on this main revision.
