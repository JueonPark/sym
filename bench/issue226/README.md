# Typed CPU kernels — issue #226

The kernels use bounded ahead-of-time specialization: 32×32 dense transpose
tiles and 256-element value-stage buffers, with runtime ISA qualification and
checked generic fallback. They preserve intermediate rounding and parameter
mapping. No shape-specific code or code cache is generated.

## Measurement protocol

`kernels.py` measures prebound native `executeHost` separately from completed
Torch frontend transfers. Native samples exclude binding, allocation, pool
construction and Python overhead. Effective bandwidth is logical source plus
result bytes divided by kernel time; it excludes intermediate/cache traffic.
Completed samples include preparation, result allocation, transformation,
DMA and current-stream completion. Compilation, checks, parameter creation
and input mutation are outside timing. Every output is checked byte for byte
against the Torch eager CPU transformation. Source values change each call;
separate correctness tests reuse symbolic recipes across changing odd shapes.

CPU-reference placement is explicit. Torch eager and Inductor controls perform
the same transformation on the CPU with the same direction, precision and
wire bytes. This isolates CPU kernel work; it is not a GPU-placement or model
end-to-end benchmark. Inductor uses `emulate_precision_casts=True`: its default
can remove the f32→f16→f32 rounding, which fails the required bit comparison.
Compile-and-prime time is recorded separately.

Warm rows have five completed warmups and 30 samples per process. Cold-owner
rows have three samples after clearing Sym's retained resources, with the
recipe and process still warm; they do not represent cold compilation or
fresh-process startup. Each owner permits 64 MiB of retained/live typed
staging and uses pinned memory, one transfer stream and two buffers. Results
compare whole-buffer execution and the default 1 MiB H2D ring. Selected
recipes also sweep 32 KiB/4 MiB chunk targets and 1/4/8 CPU threads. D2H uses
the existing whole-buffer path even with `pipeline=True`.

`run.py` runs three independent process rounds, rotates method order and
alternates baseline/candidate order. Both use affinity `4-7,20-23` (four
physical EPYC 7351 cores and their SMT siblings), GPU 0 (RTX 2080 Ti), Torch
2.14.0+cu126, one interop thread, passive OpenMP waiting and one Inductor
compiler worker. A process-tree-aware monitor rejects foreign builds or GPU
work. Inductor compilation belongs to the benchmark process and finishes
before timing samples. No performance claim is made for other CPUs/ISAs.

## Reproduce

Build baseline and candidate in separate Release directories with CUDA,
Python and `SYM_BUILD_TESTS=ON`. Candidate also needs the build-only
`typed_host_test` target (not installed). The helper calls unchanged
`TypedBoundPlan`/`Program`/`executeHost` interfaces and can load either runtime;
`LD_LIBRARY_PATH` selects the frozen baseline. `/proc/self/maps` verifies the
loaded library and the metadata records library, extension, helper, exporter
and frontend hashes. Baseline/candidate must have the same compiler/toolchain.

```sh
python bench/issue226/run.py --baseline /path/to/baseline-build \
  --candidate /path/to/candidate-build --baseline-revision BASELINE_SHA \
  --output /tmp/sym-226-acceptance
python bench/issue226/compact.py --input /tmp/sym-226-acceptance \
  --output bench/issue226/evidence
```

Only compressed raw samples and scalar provenance are kept in `evidence/`;
execution logs and compiler caches remain outside the repository.
