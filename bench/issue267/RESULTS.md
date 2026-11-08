# Issue 267: automatic specialization of ordered layout/value chains

**Qualification passed: 36 processes, 93,600 timed producer/transfer checks and 3,600 additional consumer-output checks.** No failed or contaminated acceptance attempt occurred.

The candidate improves all **36/36 completed-H2D E1 medians**, beats true materialization in 36/36, manual fusion in 30/36, and the fastest manual/eager/Inductor alternative in **28/36**. Producer-only medians improve in 36/36 and beat the fastest alternative in 25/36. Equal-case geometric-mean speedups over the matched old folded runtime are 1.92× completed H2D and 2.59× producer-only. These summaries weight each case equally; they are not workload-throughput estimates.

Held-out completed medians improve in 14/16 comparisons. Completed consumer latency improves in 3/4; the eight-worker dequant consumer regresses by **7.5%**. All losses remain below and in the [complete tables](TABLES.md).

## Original E1: all large cases

N=8192, M=128; rank3 outer extent 8. Times are completed producer+H2D milliseconds. Speedup is old folded / candidate folded. Candidate / fastest compares the candidate with the fastest manual/eager/Inductor control in its own processes; values below one favor Sym.

| Family | Workers | Old ms | Candidate ms | Speedup | Candidate / fastest | Fastest alternative |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dequant_transpose | 1 | 4.586 | 3.147 | 1.46× | 1.350 | torch_inductor |
| dequant_transpose | 8 | 1.811 | 1.330 | 1.36× | 1.413 | torch_inductor |
| padded_roundtrip | 1 | 8.866 | 2.358 | 3.76× | 0.624 | torch_inductor |
| padded_roundtrip | 8 | 7.129 | 1.809 | 3.94× | 0.965 | torch_inductor |
| rank3_chain | 1 | 79.065 | 9.089 | 8.70× | 0.981 | torch_inductor |
| rank3_chain | 8 | 16.745 | 5.590 | 3.00× | 0.955 | torch_inductor |
| roundtrip1 | 1 | 1.620 | 0.816 | 1.99× | 0.946 | manual_fused |
| roundtrip1 | 8 | 0.838 | 0.749 | 1.12× | 0.969 | manual_fused |
| roundtrip2 | 1 | 2.379 | 0.894 | 2.66× | 1.019 | manual_fused |
| roundtrip2 | 8 | 0.903 | 0.744 | 1.21× | 0.997 | manual_fused |
| roundtrip4 | 1 | 3.435 | 1.419 | 2.42× | 1.624 | manual_fused |
| roundtrip4 | 8 | 1.142 | 0.748 | 1.53× | 0.998 | manual_fused |

The large rank3 case improves 8.70×/3.00× with one/eight workers; large padding improves 3.76×/3.94×. The large dequantization cases improve 1.46×/1.36× but remain 35.0%/41.3% slower than Inductor. The one-worker large roundtrip4 case remains 62.4% slower than manual fusion. Near-ties below a few percent are descriptive comparisons, not significance claims.

The preserved original manual roundtrip control performs one F32→F16→F32
roundtrip for every roundtrip family, exploiting idempotence under the checked
value contract. Sym retains the requested stage count, including roundtrip4
and held-out roundtrip7. Their manual gaps therefore include an algebraic
simplification advantage, not just loop/dispatch quality. Identical output and
H2D contracts do not imply identical conversion counts. No manual control was
weakened for this comparison.

### Every remaining E1 completed-latency loss to an alternative

| Case | Workers | Candidate ms | Candidate / fastest | Fastest alternative |
| --- | ---: | ---: | ---: | --- |
| dequant_transpose_n8192_m128_o8 | 1 | 3.147023 | 1.350 | torch_inductor |
| dequant_transpose_n8192_m128_o8 | 8 | 1.329566 | 1.413 | torch_inductor |
| roundtrip1_n512_m128_o8 | 8 | 0.102035 | 1.045 | manual_fused |
| roundtrip2_n8192_m128_o8 | 1 | 0.894315 | 1.019 | manual_fused |
| roundtrip4_n33_m128_o8 | 8 | 0.045562 | 1.002 | manual_fused |
| roundtrip4_n512_m128_o8 | 1 | 0.117731 | 1.307 | manual_fused |
| roundtrip4_n512_m128_o8 | 8 | 0.121748 | 1.130 | manual_fused |
| roundtrip4_n8192_m128_o8 | 1 | 1.418932 | 1.624 | manual_fused |

## Held-out quality

Producer medians improve in 15/16 and completed-H2D medians in 14/16. The only completed regressions against the old runtime are the unsupported quantize/dequantize chain: 1.4% with one worker and 1.9% with eight. The eight-worker direction varies by round. No held-out median regression exceeds 5%.

Candidate folded execution beats materialization in 15/16 held-out completed comparisons and the fastest available alternative in 12/16. BF16 fallback remains 1.89×/1.36× the eager latency; roundtrip7 remains 1.48×/1.84× the manual latency (one/eight workers). The fixed held-out set is evidence across additional shapes/chains, not a claim that all affine layouts or stage families are accelerated.

## Completed GPU consumers

These transfer transformed weights into the same batch-32 linear+GELU computation. All delivered weights and consumer outputs are checked. Times include producer, H2D, GPU compute and synchronization.

| Family | Workers | Old ms | Candidate ms | Speedup | Candidate / fastest | Fastest alternative |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dequant_transpose_n1024_m512_o8 | 1 | 2.498483 | 1.636619 | 1.527× | 1.178 | torch_inductor |
| dequant_transpose_n1024_m512_o8 | 8 | 0.726163 | 0.780518 | 0.930× | 1.128 | torch_inductor |
| padded_roundtrip_n1024_m256_o8 | 1 | 2.275040 | 0.688030 | 3.307× | 0.680 | torch_eager |
| padded_roundtrip_n1024_m256_o8 | 8 | 1.927822 | 0.597217 | 3.228× | 0.782 | torch_inductor |

Padding gains survive consumption: 3.31×/3.23× improvements, with both candidate medians faster than the alternatives. Dequantization improves with one worker but regresses with eight. For that regression, old per-round medians are 1.101519/0.724100/0.726163 ms and candidate medians are 0.772217/0.782007/0.780518 ms. The direction changes across rounds; the median-of-medians regression is retained, not dismissed or rerun. These measurements do not isolate a CPU/GPU cause for that variation. They prevent a blanket claim that producer gains improve every consumed workload.

## Compilation, preparation and cache measurements

Across the 168 candidate folded case records, source-plan compilation is 4.125–5.593 ms (median 4.378 ms), binding 0.016–0.050 ms (median 0.024 ms), and native preparation 0.005–0.408 ms (median 0.138 ms). The unchanged baseline compiler median is also 4.378 ms. These are distributions across the fixed cases, not per-case confidence intervals. Folded plans are 418–1,078 bytes, compared with 994–8,834 bytes for the sum of materialized plans. All **2,520 saved plan files** match their measured hashes, operation boundaries and paired runtime/thread/round counterparts.

Candidate folded first completed calls span 0.042–139.576 ms, with a 0.233 ms median. Consumer library startup contributes the large first-call values; common initialization can land on whichever arm runs first. Inductor first completed calls, including lazy compilation and per-process cache setup, span 854–7,124 ms (median 991 ms). These are shared-process first calls, not independent per-arm deployment cold starts.

The following large one-worker rank3 example retains shared allocation and first-call costs. Values are medians of complete per-process sums, in ms; they need not equal the sum of separately reported medians.

| Candidate-runtime arm | Preparation | First completed | Total(1) | Total(100) | Total(1000) |
| --- | ---: | ---: | ---: | ---: | ---: |
| manual_fused | 31.616 | 17.479 | 49.159 | 1778.433 | 17493.951 |
| sym_folded | 35.915 | 9.126 | 45.041 | 944.850 | 9124.933 |
| sym_unfused | 49.411 | 129.912 | 179.190 | 10309.328 | 102352.244 |
| torch_eager | 31.614 | 83.351 | 114.981 | 9213.529 | 91921.827 |
| torch_inductor | 32.180 | 908.285 | 941.135 | 1858.552 | 10198.705 |

All cases, request counts and warm p95s are in [TABLES.md](TABLES.md); raw records retain each preparation component and warmup cost.

| Fresh-process suite | Final Inductor cache bytes | Generated .py/.cpp/.so bytes | Generated files |
| --- | ---: | ---: | ---: |
| main | 4,511,180–4,575,050 | 1,441,182–1,498,226 | 56 |
| heldout | 2,104,956–2,118,562 | 714,481–724,461 | 26 |
| consumer | 574,337–579,659 | 225,211–229,039 | 8 |

All 36 final on-disk cache inventories match the recorded file/byte counters. Per-case deltas are retained in raw data. The benchmark helper ELF `.text` is 189,946 bytes for baseline and 190,890 for candidate; the extra diagnostics are benchmark code, separate from production runtime growth. No per-shape native executable cache is created.


## What changed

The checked host plan now selects bounded ahead-of-time (AOT) kernels for
compatible ordered F32/F16 cast chains and leading S8→F32 dequantization followed
by those casts. Intermediate values remain in registers across numerical
stages. Six finite vector variants use runtime-gated AVX2/F16C; scalar execution
and unsupported-chain buffered fallbacks remain. The source compiler is
unchanged. This is automatic plan-driven specialization, not a new JIT or
arbitrary machine-code generator.

An initial profiling hypothesis was rejected: E1's rank3 chain already coalesces
to a rank-two transpose with eight output columns. The existing tile path now
batches adjacent short output rows when checked strides prove contiguity.
Production dispatch uses layout/type/stage properties, never E1 family names.
Existing generic, indexed and strided-run paths remain available.

Every FP32 multiply and FP16 rounding stage is retained. A lossy roundtrip is
not replaced by a copy. Logical channel coordinates and global chunk origins
remain authoritative; channel runs split at their boundaries. Initial half
signaling-NaN payload behavior, padding entry stages and unaligned tails retain
the existing CPU semantics. BF16/quantize chains use buffered fallbacks.
Empty or nondense source descriptors still follow the frontend rejection/
Torch fallback contract before native execution; this work adds no new native
empty-plan support.

## Attribution and bounded costs

| Work | Before | After | Work retained |
| --- | --- | --- | --- |
| Eligible contiguous value chain | Input scratch copy, a read/write stage pass per conversion, final scratch copy | One input load and final store per lane group, with ordered register conversions | Every numerical stage and scalar tail |
| Short contiguous transpose rows | One stage invocation per output row of a gathered tile | One stage invocation for the contiguous tile span | Tile gather/transpose and channel splits |
| Padding initialization | Variable-width per-element `memcpy` | Fixed-width 1/2/4/8-byte store loops | Full output fill pass and subsequent valid-region writes |
| E1 layout indexing | Existing coalesced contiguous/transpose layout | Same checked layout; short-row batching added | No claim of newly eliminating a generic E1 index walk |

For a full 32-row, eight-column tile, stage entry falls from 32 calls to one.
For a chain of k conversions, the arithmetic still executes k conversions;
the removed passes are bounded staging loads/stores and entry/dispatch overhead.
The transpose tile is at most 4,096 bytes. Buffered fallback scratch remains
bounded independently of tensor shape. The stage descriptor is 40 bytes on
the qualified build and is prepared once per execution range.

GNU ELF `.text` grows from 666,448 to 669,776 bytes (**+3,328 bytes, 0.50%**).
The broader GNU `size` text column grows by 4,295 bytes; it also includes
read-only sections and is not the `.text` section alone. The runtime creates
**zero per-shape executable code/cache bytes**. Generated Inductor code and
compiled source-plan sizes are reported separately.

The artifact preserves 16 debugger stack snapshots per target family for both
baseline and candidate, plus separate unprofiled diagnostic producer timings.
These support the stage/copy/transpose attribution but do not enter accepted
latency results. `perf` is unavailable; no hardware-counter or measured DRAM
bandwidth claim is made. Static byte/pass counts are not DRAM traffic counters.
For example, candidate rank3 snapshots include transpose SIMD in 8/16 top
frames and register conversion in 5/16; candidate dequantization includes tile
traversal in 10/16. These small debugger samples indicate remaining layout and
conversion work, not a quantified bandwidth/compute utilization split.

## Measurement contract

The [protocol](PROTOCOL.md) freezes 36 processes: three rounds, one/eight CPU
workers, two runtimes, and main/held-out/consumer suites. Baseline is merged
`c5085cd9`; candidate native source is `3c008646`; the final measurement harness
is `18e12e3`. Both runtimes use the same unchanged compiler binary.
The machine is an AMD EPYC 7351 with an RTX 2080 Ti (driver 595.71.05),
Python 3.14.7 and Torch 2.14.0+cu126. CPU affinity 4–7,20–23 selects four
physical cores and their eight SMT threads on NUMA node 1. Eight workers
therefore means eight software workers, not eight independent physical cores.

The six original E1 families retain true per-source-operation materialization,
whole-chain folding, the original manual fused control, eager Torch and
Inductor. `--no-canonicalize` is not used as a no-fold control. New held-out
chains without an original manual implementation omit that arm explicitly.
Each arm writes the same pinned wire and uses one identical H2D copy. Producer
latency and completed H2D latency are separate. The consumer suite adds the
same batch-32 GPU linear+GELU to transferred dequantized or padded weights;
it is a scoped consumer experiment, not a complete model/server result.

Inputs change before each sample, arm order rotates, and all timed outputs are
checked outside timing with exact non-NaN bits and NaN classification. Each
case has five warmups and 30 samples. Inductor preserves intermediate casts,
TF32 is disabled, and disk caches are fresh per process. Runtime order
alternates by round. Mandatory observed 60-second cooldowns and continuous
foreign CPU/build/GPU audits separate qualification from excluded pilots.

Tables report the median of three process medians and the median of three
process p95s. These are descriptive summaries on one machine, not confidence
intervals or evidence of universal performance.

## Setup and total-cost interpretation

Raw records retain source compilation, binding, runner/setup and intermediate
storage allocation separately, plus the first completed call, warmup cost,
plan sizes and Inductor file/byte growth. Inductor lazy compilation is charged
to its first completed call. Unfused intermediate storage is charged to that
arm only. A first completed call is that arm's first invocation in the shared
process after CUDA initialization; it does not imply a cold operating-system
cache or a separate fresh GPU context for every arm. The complete tables
include derived totals for 1, 100 and 1,000 calls:

`Total(N) = common allocation + arm-specific compile/bind/setup/storage + first completed call + (N−1) × warm median`.

The formula excludes benchmark warmups, CPU input generation and CUDA context
initialization. Consumer activation upload and matmul-output storage are also
preconditions: their setup was not separately timed. Thus these totals describe
the measured transfer/producer setup boundary, not full deployment startup.
This is a retained review limitation, not a backfilled measurement.
Building the AOT runtime and manual-control extension is outside these request
totals for both native paths. A zero manual compilation entry means the control
is already compiled; it does not measure the cost of authoring/building a new
manual kernel. Sym's source-plan compilation and Inductor's lazy compilation
are measured within their stated preparation/first-call boundaries.

## Correctness and review

- Native runtime: 346 tests passed, two ISA-specific skips.
- Full Python suite: 1,448 passed, 13 backend/environment skips; two expected
  fork deprecation warnings.
- Native/manual/eager/Inductor controls: 40 passed, including exceptional values.
- Harness validation: 32 passed. Nine added cooldown regressions failed before
  the review fix and passed afterward.
- One independent AI whole-change review found no runtime correctness defect.
  Its mandatory cooldown finding was fixed; consumer setup accounting remains
  explicitly scoped. See [review disposition](REVIEW.md).

Tests include all 65,536 half bit patterns at widening stage cuts, repeated
casts, binary32-before-half witnesses, per-channel mapping, unaligned buffers,
sentinels, tile tails, chunk origins, worker partitions and transfer pipelines.
Unsupported/non-x86 execution was inspected but not run on separate hardware.

## Public evidence

The immutable package is available through
[the issue 267 Wiki artifact page](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-267).
It includes exact source/runtimes, original E1 controls, complete accepted and
excluded evidence, profiles, build/test/CI records and standard-library-only
verification/reanalysis instructions. No experiment Release or source SVG is
used. This work is part of #261, #270, #272 and #274; it does not claim the
handwritten DLRM producer as an automatic-codegen result.
