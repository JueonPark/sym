# Padded native H2D campaign · 2026-11-27

**Closed with `baseline_improved` after one of ten proposals.** P01 reduced completed transfer latency from 7.5547 to 0.9979 ms (7.57×). All six matched pairs improved on the 16 MiB padded transfer. The candidate passed all 349 tests in the CUDA-enabled native test binary.

## Fixed comparison and gates

Baseline and starting implementation are b2b183c754338b9b3b86504da5bb8f9d1effcf02. No runtime changes preceded the pilot. We chose the standalone blocking CUDA pipeline and scaled the repository's `libreloc/test/corpus/pad_2d_outer.json` operation (same axis, fill bits and one row on each edge) to a 16 MiB output. This measures a supported layout transfer in isolation. The input is a hand-built bound plan with 4094 × 1024 contiguous four-byte elements. The GPU receives 4096 × 1024, with -2.5f padding. The control delivers the same byte count with 4096 source rows and no padding.

Measure warmed host wall time in ms across the complete caller-owned-pool `executeH2DPipelined` call, including chunk planning, staging, submission and completion. Reuse two pinned 4 MiB buffers and two streams on CUDA device 0, gather on the calling thread. Input generation, allocation, source mutation, and correctness readbacks are outside this call for both versions. The oracle independently constructs the valid region and both pad rows, and compares all device bytes after the first warmup and last measured call. A source word changes each call. See [driver](transfer.cpp) and [runner](measure.py).

The host is a Ryzen 7 7800X3D under WSL2 kernel 6.18.58.1, RTX 4070 Ti SUPER, CUDA toolkit 13.2.78, GCC 11.4.0, CMake 3.22.1, Ninja, Release, CPU affinity 2. No clock locks. Use 10 warmups and 50 samples per fresh process. Three baseline pilot runs, then six matched baseline/candidate process pairs with order alternating by block. For each metric, take the median per process then the median across six processes per version.

[Pilot samples](pilot.csv) give padded baseline medians 8.0981, 7.5456, 7.6539 ms (7.3% max/min spread), and unpadded control 0.9068, 0.8637, 0.9027 ms (5.0%). The separate diagnostic times the actual staging-fill call on each chunk and verifies all four timers fired. Median fill occupies 92–94% of each diagnostic process's median per-call time fraction. The diagnostic executable compiles a second copy of the pipeline with timers, which changes absolute timings. Phase measurements are comparable across the two diagnostic builds.

The gates fixed before candidate measurements require at least 15% lower aggregate padded wall time, and a lower padded median in at least five of six pairs. Allow at most 10% higher aggregate unpadded wall time. Device oracles, relevant existing native pipeline tests, and reviewer cases must pass. The campaign stops at overall improvement or ten closed proposals.

### P01 · Fixed-width four-byte staging fill

- Status `succeeded`. The preceding implementation was the fixed baseline above.
- The diagnostic fill medians are 9.4347, 9.8299 and 9.3490 ms. The uninstrumented library's `detail::executeH2DPrepared` calls `memcpy@plt` inside the per-element fill loop (objdump offsets 0x997e8–0x997fd). In `fillStagingWindow` in `libreloc/src/Pipeline.cpp`, dispatch once on four-byte padded elements and copy the first four bytes of the existing fill representation with fixed-size memcpy on each iteration. Process complete elements only, without typed or aligned pointer assumptions. Preserve the existing no-pad and other-width paths. This should remove the runtime-sized libc call per element while writing exactly the same staging bytes.
- Test against the preceding implementation with the same timed transfer, fill phase and plain control. Require at least 50% reduction in aggregate fill time and inspect optimized assembly for elimination of the call on the four-byte path. Reject the local hypothesis if the reduction misses 50% in a valid run. A byte mismatch or regression beyond the gate fails the item. Exercise arbitrary nonuniform and high-bit fill representations, pad-only and partial chunks, other widths and unpadded execution with independent expectations.

#### Independent reviews before implementation

- Soundness **pass**. The reviewer confirmed 4,194,304 dynamic copies and the call in the optimized baseline. The paired production timings, four separate phase intervals, optimized disassembly, and independent readback distinguish fill cost from moving work outside the timer.
- Functionality and generality **pass**. Fixed-size byte copying from the same representation preserves fill bits and alignment semantics. Check both padded axes on a 2 × 3 valid region, one reused slot, pad-only and mixed chunks, and a shorter final chunk. Verify destination guards, arbitrary high bits, other widths and unpadded execution.
- Contribution **low**. This is a conventional compiler specialization. The experiment quantifies its effect on this particular transfer and leaves the fill volume at 16 MiB. Only 2,048 of 4,194,304 elements retain pad values after gather.

#### Measured outcome

| Metric (median of six process medians) | Fixed baseline | P01 | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Completed padded transfer | 7.5547 ms | 0.9979 ms | −86.79% | ≥15% reduction and faster in ≥5/6 pairs. Passed in 6/6. |
| Staging fill, diagnostic executable | 9.3734 ms | 0.3334 ms | −96.44% | ≥50% reduction. Passed. |
| Completed unpadded control | 0.8026 ms | 0.8074 ms | +0.60% | ≤10% regression. Passed. |

The ordinary candidate library branches on width four at 0x997a6. GCC broadcasts the pattern with `pshufd` at 0x99cb5 and writes 16 bytes per `movups` in the loop at 0x99cc0. The scalar tail handles remaining whole elements. Every timing process passed the device readback checks. The independent mixed-padding test passed against both implementations. The candidate's full native run passed 349/349 tests across 53 suites with zero skipped. [Raw transfer and phase samples](p01.csv) · [Native test results](tests.xml.gz)

The adopted [pipeline change](../../../libreloc/src/Pipeline.cpp) has Git blob `02790f0d2417448443702ed68841d1c1bb7ccf81`. The [test](../../../libreloc/test/PipelineTest.cpp) covers fill representations and storage reuse. The baseline and candidate shared-library SHA-256 digests were respectively `e62f7fbda4c928b1ff28c71d9c9dc98f84e060048014d2d680efac6ae21e6903` and `0adaa0eb3c5333c6d0144b27080214e9309ffd6280eaefdf84ac9d292c72e34a`.

The padded transfer still took 0.1905 ms longer than the candidate's equal-byte unpadded control. The change still fills the entire staging window, including valid rows that gather subsequently overwrites. The measured speedup covers this warmed standalone transfer on this machine. Application-level latency and performance on other layouts or machines remain unmeasured.

## Reproduction

Run from the repository root with a CUDA-capable device and the recorded toolchain.
The standalone CMake driver builds the repository's runtime target. The test
target also requires the GoogleTest CMake package (1.11.0 on the measurement
host, extracted from the Ubuntu `libgtest-dev` package). Use the prefix option
below if GoogleTest is installed outside the default search paths.

```sh
art=bench/results/padded-h2d-20261127
run=$(mktemp -d)
git archive b2b183c754338b9b3b86504da5bb8f9d1effcf02 libreloc | tar -x -C "$run"
cmake -S "$art" -B "$run/base" -G Ninja -DRUNTIME_ROOT="$run" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$run/base" --parallel 6
cmake -S "$art" -B "$run/p01" -G Ninja -DRUNTIME_ROOT="$PWD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DSYM_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="${GTEST_PREFIX:-/usr}"
cmake --build "$run/p01" --parallel 6
"$run/p01/libreloc/test/libreloc-test" --gtest_brief=1
taskset -c 2 python3 "$art/measure.py" "$run/pilot.csv" "$run/base"
taskset -c 2 python3 "$art/measure.py" "$run/p01.csv" "$run/base" "$run/p01"
python3 "$art/summarize.py" "$run/p01.csv"
objdump -dC "$run/p01/libreloc/libreloc_runtime.so" | less
```
