# Torch-side preparation profiling

This diagnostic experiment makes native Torch preparation visible alongside
Sym's request preparation. It uses the same phase definitions for both paths,
and leaves unclassified Python/C++ work explicit. It does not alter production
runtime sources, remove guards, or optimize the measured operations.

Results and images are in
[`../results/torch-preparation-profile-20261002`](../results/torch-preparation-profile-20261002).
The report distinguishes profiler-free completed-transfer measurements from
instrumented attribution. Do not use the preload library for headline timings.

## Native observer

`native_hooks.cpp` interposes the installed Torch wheel's exported
`TensorIteratorBase` methods. Every wrapper forwards once to the original
method, with the same arguments and return value. It records preparation,
overlap/shape/type checks, output setup, dimension handling and CPU iteration.
The native `RecordFunction` callback also identifies `aten::copy_` with actual
CPU source and destination devices, covering CPU conversion loops that do not
call the exported `for_each` method. Operator regions still contain some
uninstrumented checks and dispatch work; they are not pure kernel busy time.

This probe is specific to the installed ELF symbols and C++ ABI. It was built
against Torch `08187d9e0fba026dc8217405802ab5381dc88d90` using that wheel's
headers. `-DNDEBUG` is necessary to match its Release `RecordFunction` layout.
The report links to the exact upstream source revision. Missing original
symbols fail loudly rather than silently forwarding elsewhere.

```sh
python bench/torch_profile/build_hooks.py \
  --nvtx-include /path/to/nvtx/include --output /tmp/torch-hooks.so
LD_PRELOAD=/tmp/torch-hooks.so python bench/torch_profile/hook_smoke.py
```

The smoke check requires correct values, preserved invalid-shape/overlap errors,
and nonzero native and CPU-copy callback counts. Also check that stderr contains
no callback warnings. The workload captures subsequently check transfer/model
results, and the analyzer checks native callback coverage and GPU correlations.

## Reproduce the experiment

Use the runtime builds and unchanged workload checkout described in
[`../workload_profile/README.md`](../workload_profile/README.md). Main runtime
revision is `519f6a1`; workload head is `cdc44e2`. Normal measurements use NVTX
off. Diagnostic captures use the separately built Sym native annotations from
the archived patch, NVTX on, and the Torch probe. Set `SYM_RELOC_EXPORT` and
`SYM_OPT` to the appropriate compiler binaries. The CPU affinity in `run.py` is
machine-specific and should be adapted for other hosts.

```sh
python bench/workload_profile/run.py --phase measure --rounds 3 \
  --workloads /path/to/pr162 --build /path/to/normal-build --output results/measurements
python bench/workload_profile/run.py --phase controls \
  --workloads /path/to/pr162 --build /path/to/normal-build --output results/controls
python bench/workload_profile/run.py --phase profile --retained-only --aten-nvtx \
  --torch-hooks /tmp/torch-hooks.so --workloads /path/to/pr162 \
  --build /path/to/trace-build --output results/profiles --nsys /path/to/nsys
SYMPROF_NATIVE_DETAIL=build python bench/workload_profile/run.py \
  --phase profile --retained-only --examples llm --aten-nvtx \
  --torch-hooks /tmp/torch-hooks.so --workloads /path/to/pr162 \
  --build /path/to/trace-build --output results/coarse --nsys /path/to/nsys
python bench/torch_profile/analyze.py results/profiles
python bench/torch_profile/analyze.py results/coarse --coarse
python bench/torch_profile/report.py results
```

Run all GPU work serially. `run.py` records commands and samples the host every
two seconds. The monitor counts the background compiler/simulator processes
observed in the earlier experiment; it does not establish exclusive access.

The build-only control disables inner native range recording, leaving build
and CPU-loop ranges plus the same ATen and CPU-copy observers. It exposes how
much the additional instrumentation changes observed build durations. Both
modes still forward through every wrapper and both incur observer overhead.
Differences between separate captures are not an exact overhead subtraction.

## Analysis contract

`analyze.py` reuses the earlier GPU correlation/model-signature checks. It
assigns each caller-wall interval to one category; CUDA API time takes priority
over enclosing scopes, output setup is separated from preparation, and observed
preparation is subtracted from CPU operator regions. GPU durations remain
separate from blocking host time.

The graph combines residual frontend, Torch operator, native and outer host
buckets into **Other Python / C++ host work** for both implementations. It does
not imply that one side has zero execution overhead. Preparation is an observed
subset, not a census of every inline check or instruction. Uninstrumented guards
may remain in the residual categories. Small native function times can be
dominated by instrumentation; use the uninstrumented runs to compare latency.

`report.py` requires Matplotlib and consumes three measurement rounds, four full
captures and one LLM build-only capture. Archived SQLite exports are gzipped;
decompress them before rerunning the analyzer. Detailed per-transfer analyses
are also gzipped. The report's narrative describes this particular experiment;
review it when reusing the renderer for another environment.
