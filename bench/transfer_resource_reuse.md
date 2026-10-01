# Frontend resource reuse benchmark (#171)

`transfer_resource_reuse.py` measures completed FP32 transpose transfers through
`torch.compile(..., backend=RelocBackend(...))`. It compares explicit retained
resources with `None` using identical buffers, streams, gather budgets, recipes
and native scheduling. Every call binds/validates its current input, allocates a
fresh output and completes before the timer stops. Resource retention is opt-in.

Run each thread budget in a separate process, with an explicit CPU affinity local
to the selected GPU. The example matches GPU 0 of the EPYC 7351 / RTX 2080 Ti host;
select appropriate CPUs on another machine. Build the normal runtime with
`RELOC_ENABLE_CUDA=ON` and `RELOC_ENABLE_NVTX=OFF`, and set these environment paths:

```sh
export PYTHONPATH="$PWD/build/torch-cuda/python"
export SYM_RELOC_EXPORT="$PWD/build/torch-cuda/sym/tools/sym-reloc-export"
export SYM_OPT="$PWD/build/torch-cuda/sym/tools/sym-opt"
python bench/transfer_resource_reuse.py --threads 1 --cpus 4 --output /tmp/reuse-t1.json
python bench/transfer_resource_reuse.py --threads 8 --cpus 4,5,6,7,20,21,22,23 --output /tmp/reuse-t8.json
```

The default matrix covers 4096x1024, 1024x4096, 256x1024 and 1009x4093; H2D,
forward D2H, alternating shapes/payloads, two simultaneous callers, and a fixed
chunk one/four-buffer control. CPU Inductor transpose + H2D and H2D + Torch GPU
transpose allocate their outputs and intermediates normally. Forward D2H also
compares Torch GPU transpose + D2H. This is not the historical prototype that
reused output storage. All inputs are pageable CPU tensors or dense CUDA tensors.

Five warmups, three shuffled method rounds and 30 samples per round are the
defaults. Competing methods' caches are cleared between blocks so one policy
does not benefit from another policy's live pinned allocations. The first
transfer after compilation and cache clearing is reported
separately in each round: the process, CUDA context and Torch allocators have
already been initialized. Warmed samples include caller-stream synchronization.
Byte-exact checks run outside each latency sample and touch the CPU data; there
is no cache flushing or clock locking. An earlier output remains live and is
checked after the round. Concurrent throughput includes the release barrier and
worker completion; input creation/compilation/warmup and output checks are
outside that window. First/last outputs are retained per caller for validation.
Each caller has its own CUDA stream and the full gather budget, sharing the
specified affinity; two callers with eight participants can oversubscribe those
CPUs. Throughput is reported separately from per-call latency.

JSON contains raw samples, p50, nearest-rank p95, first-call latency, configuration,
source/binary hashes, versions, affinity, hardware/clock snapshots, resource
limits, native before/after counters, CUDA allocator peak and process peak RSS.
The latter is a cumulative process high-water mark, not an isolated method peak.
Cache counters are unavailable for `None` and Torch methods; null never means
zero allocations. Retained staging bytes are separate from fresh tensor memory.
Measured Sym execution counts must match requested calls, compilation must stay
outside timing, and warmed retained calls must create no new staging, streams
or workers; otherwise the harness fails instead of emitting a passing row.

The schedule metadata is derived for dense FP32 transpose using the documented
native clamp, and is labeled as derived. The GPU trace independently records
actual copy sizes. On the reference control, both one and four buffers use four
4 MiB chunks. Large arbitrary shapes can change the automatic target when buffer
count changes; do not assume every one-buffer comparison isolates overlap.

## CPU/GPU timeline

Configure a separate build with `RELOC_ENABLE_CUDA=ON` and
`RELOC_ENABLE_NVTX=ON`. The latter adds optional worker gather and H2D submission
ranges to the existing pipeline; it defaults off and adds no checks/timestamps
in disabled builds. Use this build only for profiling, keeping the normal build
for latency measurements. Point `PYTHONPATH` to its `python/` directory; compiler
binaries can remain those of the normal build.

```sh
export PYTHONPATH="$PWD/build/torch-cuda-trace/python"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/reuse-4 \
  python bench/transfer_resource_reuse.py --trace --trace-buffers 4 \
  --threads 8 --cpus 4,5,6,7,20,21,22,23 --samples 5 --output /tmp/reuse-4.json
nsys export --type=sqlite --output=/tmp/reuse-4.sqlite /tmp/reuse-4.nsys-rep
python bench/analyze_resource_overlap.py /tmp/reuse-4.sqlite \
  --output /tmp/reuse-4-overlap.json --chrome /tmp/reuse-4-timeline.json
```

Repeat with `--trace-buffers 1` and distinct output names. Retain both `.nsys-rep`
files, metadata and analyzer JSON; Chrome trace JSON can be opened in Perfetto.
Compare the actual `chunks[].bytes` sequences between captures before treating
one buffer as a control. Profiling durations are not benchmark results.

The analyzer joins each native submission to a CUDA runtime correlation ID and
then to GPU H2D activity. It intersects DMA(n) with the union of actual worker
`gatherChunk(n+1)` intervals, excluding the driver's barrier wait and avoiding
multiple counting when workers overlap each other. It checks gather-before-
submission and completion within the request. A four-buffer request must show
more than 1 microsecond of overlap; a one-buffer request must show none above
that timestamp tolerance. Missing GPU activities, annotations, correlations or
multi-chunk requests produce `status=unavailable` and nonzero exit. A missing
profiler/hardware capability is never a passing overlap result.

## Smoke checks

```sh
python -m pytest -q libreloc/python/tests/torch_frontend/test_resource_benchmark.py
```

CPU cases check fixed-chunk controls and synthetic trace evidence, including
false/missing overlap. The GPU subprocess runs every benchmark scenario with
small sample counts, validates outputs and checks counters. No performance
threshold is enforced by these tests. Larger stress/failure qualification is
tracked separately in #172; automatic enablement remains gated in #173.
