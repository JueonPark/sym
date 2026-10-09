# Typed CPU kernels — issue #226

The kernels use bounded ahead-of-time specialization: 32×32 dense transpose
tiles and 256-element value-stage buffers, with runtime ISA qualification and
checked generic fallback. They preserve intermediate rounding and parameter
mapping. No shape-specific code or code cache is generated.

## Measurement protocol

`kernels.py` measures prebound native `executeHost` separately from completed
Torch frontend transfers. Native samples exclude binding, external buffer
allocation, pool construction and Python overhead; `executeHost`'s internal
bookkeeping is timed. Effective bandwidth is logical source plus
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

Warm rows have five completed warmups (three for isolated kernels) and 30
samples per process. Cold-owner
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

## Results (2026-10-09)

Baseline: merged main `82b6a74`, using the frozen #227 build (`cb647e2` has
identical `libreloc/` and `sym/` sources). Candidate: `2d0103a`. All 18 runs
passed byte comparisons and observed no foreign build/packaging or GPU
processes. Driver 595.71.05, CUDA toolkit 12.5, CPython 3.14.7; CPU/GPU clocks
were not locked. [Raw samples](evidence/samples.csv.gz) contain 29,052 timings;
[run provenance](evidence/runs.json.gz) contains hashes, loaded library paths,
per-round distributions, recipe coverage and resource/dispatch reports.

Tables use the **median of three round medians**, in milliseconds. A `p50/p95`
cell gives the median of each round's corresponding statistic, not a pooled
percentile. Unless stated otherwise, the CPU budget is eight threads. Torch
and Inductor columns come from the interleaved candidate runs.

### Kernel coverage and effective bandwidth

The cast cases narrow f32→f16. `chain` preserves f32→f16→f32; `channel` applies
s8→f32 dequantization with runtime per-channel scales, transpose, then f16
narrowing. Shapes below are source shapes. Baseline uses its existing
contiguous SIMD cast only for `cast_inner`; the other cases use scalar stage
execution. The new report exposes the actual selected family.

| Case / shape | New family | Baseline p50/p95 | New p50/p95 | Kernel speedup | New effective GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| cast_small, 65×129 | tiled_transpose | 0.088/0.095 | 0.020/0.020 | 4.40× | 2.50 |
| cast_medium, 512×512 | tiled_transpose | 1.236/1.995 | 0.226/0.250 | 5.48× | 6.97 |
| cast_large, 2048×1024 | tiled_transpose | 16.003/16.053 | 1.132/1.173 | 14.14× | 11.12 |
| cast_odd, 2049×1025 | tiled_transpose | 9.781/9.914 | 1.321/1.434 | 7.40× | 9.54 |
| cast_inner, 128×128×128, permute(1,0,2) | contiguous_cast | 0.296/0.485 | 0.293/0.383 | 1.01× | 42.97 |
| chain_vector, 2,097,155 | contiguous_stages | 11.845/12.113 | 0.634/0.691 | 18.69× | 26.48 |
| chain_transpose, 2049×1025 | tiled_transpose | 14.011/14.085 | 1.959/2.132 | 7.15× | 8.57 |
| channel_outer, 1025×2049, source axis 1 | tiled_transpose | 38.061/38.321 | 1.751/1.792 | 21.73× | 3.60 |
| channel_inner, 257×513, source axis 0 | generic | 1.476/1.713 | 2.352/2.378 | **0.63×** | 0.17 |

The generic fallback's isolated time regresses in this acceptance sample;
do not treat it as an optimized kernel. Its native round medians are
0.912/2.352/2.436 ms, versus baseline 1.404/1.476/2.416 ms. Both builds have
bimodal samples, so thread scheduling is a possible contributor, not an
established root cause. Completed calls below improve, but their p95 remains
noisy. Inner-channel vectorization and generic scheduling remain open work.
Outer-channel native baseline medians also vary (15.974–40.810 ms); its
completed-call gain is the more conservative result.

### Completed frontend calls

`Whole` disables H2D chunking; `Ring` requests the default 1 MiB target. D2H
still executes whole-buffer work, so differences between its Whole/Ring
columns are sampling variation. All columns use the same wire payload and
CPU transformation location. The optimized kernels do not imply an
automatic-placement or whole-model speedup.

| Case / direction | Baseline Whole p50 | New Whole p50/p95 | New Ring p50 | Torch p50 | Inductor p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cast_small H2D | 0.334 | 0.254/0.276 | 0.254 | 0.085 | 0.296 |
| cast_medium H2D | 1.588 | 0.595/0.642 | 0.560 | 0.546 | 0.389 |
| cast_large H2D | 16.789 | 2.075/2.276 | 2.056 | 3.681 | 1.679 |
| cast_odd H2D | 10.836 | 2.326/2.480 | 2.055 | 3.669 | 1.662 |
| cast_inner H2D | 1.276 | 1.273/1.496 | 1.324 | 1.244 | 1.297 |
| chain_vector H2D | 13.392 | 1.866/2.095 | 1.545 | 1.732 | 1.838 |
| chain_transpose H2D | 15.347 | 3.299/3.498 | 4.064 | 5.225 | 2.428 |
| channel_outer H2D | 17.202 | 2.669/2.846 | 3.876 | 4.734 | 1.444 |
| channel_inner H2D | 1.732 | 1.283/2.884 | 1.310 | 0.491 | 0.306 |
| cast_small D2H | 0.332 | 0.271/0.298 | 0.259 | 0.096 | 0.273 |
| cast_medium D2H | 1.707 | 0.764/0.852 | 0.654 | 0.699 | 0.549 |
| cast_large D2H | 16.782 | 2.457/2.689 | 2.419 | 4.437 | 2.341 |
| cast_odd D2H | 11.116 | 2.809/3.022 | 2.757 | 4.239 | 2.392 |
| cast_inner D2H | 1.685 | 1.669/1.857 | 1.678 | 1.885 | 1.869 |
| chain_vector D2H | 13.365 | 2.040/2.235 | 2.053 | 1.706 | 2.202 |
| chain_transpose D2H | 15.426 | 3.465/3.698 | 3.495 | 5.559 | 2.607 |
| channel_outer D2H | 17.242 | 2.537/2.811 | 2.444 | 4.706 | 1.439 |
| channel_inner D2H | 1.790 | 1.306/1.787 | 1.319 | 0.496 | 0.301 |

Large transpose/cast Whole H2D improves **8.09×**, with new round medians
2.078/2.068/2.075 ms versus 17.093/16.789/16.474 ms. The contiguous chain
improves **7.18×** (1.922/1.866/1.823 versus 13.411/13.392/13.388 ms), and the
outer-channel chain improves **6.45×**. The already-SIMD control is essentially
unchanged. Default-ring contiguous-chain H2D beats both CPU-transform controls
here, but Sym still trails Inductor on most transpose cases and eager Torch
on tiny transfers. Results do not establish a general advantage over Torch.

Cold-owner Whole H2D remains substantially slower than warm execution:

| Case | Baseline cold p50/p95 | New cold p50/p95 |
| --- | ---: | ---: |
| cast_small | 0.658/0.683 | 0.545/0.553 |
| cast_large | 28.501/32.767 | 8.370/8.401 |
| chain_vector | 23.318/27.980 | 11.850/11.969 |
| channel_outer | 27.560/30.155 | 9.237/9.493 |

### Worker and chunk tradeoffs

New H2D p50s; thread cells list **1 / 4 / 8** workers:

| Case | Kernel, 1/4/8 workers | Whole, 1/4/8 workers | Ring 32 KiB, 8 workers | Ring 1 MiB, 8 workers | Ring 4 MiB, 8 workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| cast_small | 0.020/0.020/0.020 | 0.248/0.249/0.254 | 0.249 | 0.254 | 0.246 |
| cast_medium | 0.438/0.221/0.226 | 0.792/0.601/0.595 | 0.860 | 0.560 | 0.551 |
| cast_large | 4.573/1.171/1.132 | 5.475/2.079/2.075 | 7.939 | 2.056 | 2.063 |
| chain_vector | 2.441/0.660/0.634 | 3.587/2.289/1.866 | 4.227 | 1.545 | 1.668 |

Small kernels correctly remain inline. Four physical cores capture most of
the large transpose gain; eight workers add little there and do not improve
the medium kernel. Tiny wire chunks lose to scheduling/copy overhead even
when their CPU work stays inline. The 256 KiB input-plus-output worker floor
lets a default 1 MiB chunk use multiple workers, without imposing that
bandwidth-oriented floor on generic scalar evaluation. Whole execution is
still preferable for the tested transpose and outer-channel stage chains;
chunk size is a caller option, not a universal speedup.

### Cost, validation and remaining limits

GNU `size`'s text/read-only column grows 877,239→885,160 bytes (+7,921, 0.90%);
data grows by 72 bytes and BSS is unchanged. Tile plus stage scratch is
7.5 KiB per active worker, independent of tensor shape. There is no JIT,
executable allocation, shape-code cache or intermediate whole-tensor buffer.

Validation: six CTest targets, all 41 compiler tests, 1,205 Python tests
(four skips), and 58 workload tests pass. New tests cover random f32 bit
patterns, all 65,536 half encodings including NaN payloads, lossy intermediate
rounding, zero-point extremes, odd tails, unaligned buffers with guards,
per-channel mapping/wrapping, changing padded shapes, seven-row chunks,
nondefault streams, and H2D/D2H CPU/GPU stage cuts. CI also passes CPU-only
build/tests, wheel packaging, formatting and clang-tidy.

This targets CPU kernel costs. Generic inner-channel maps, frontend overhead,
cold resource creation and GPU-placement choices still matter. Existing
[#227 placement profiles](../../docs/completed-placement.md) contain binary
fingerprints and need fresh calibration for this runtime. This benchmark
does not repeat the model end-to-end study or claim its gains carry over.
