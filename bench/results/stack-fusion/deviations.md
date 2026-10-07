## Deviations

- Toolchain: measured with CUDA 12.5 `nvcc`, not the qualified 12.6.3.
- Inductor compile failures: none (`latency.json` records no `error` rows).
- Gate 2 originally covered every configuration. On 2026-10-07 it was narrowed to stacks at dim 0 or 1, and last-dim
  stacks are now reported descriptively. In this run, one last-dim configuration at ≥ 8 MiB is above the original 10%
  bound: 16 MiB N=4, FD/S1 1.123.
- One configuration was disturbed: 16 MiB N=16 dim 2. The cause was not investigated, and the configuration was not
  re-run.
  - **S1.** Every round measured 4.234–4.271 ms (median 4.238 ms). The previous run measured 2.934–2.989 ms (median
    2.987 ms), so this run is 42% slower. It is even slower than the same stack at 24 MiB (3.997 ms).
  - **The last-dim range.** This configuration's FD/S1 of 0.756 is the low end of the last-dim range above. Without it,
    last-dim FD/S1 at ≥ 8 MiB is 1.041–1.123.
  - **The host-stack paths.** They were erratic between rounds:

    | Method | Round medians (ms) |
    |---|---|
    | B1 | 17.989, 9.353, 8.811 |
    | B3 | 9.346, 12.766, 9.026 |
    | GB | 18.645, 13.485, 9.673 |

    Its GB − B1 of 4.131 ms therefore comes from those rounds, not from GB's own cost. In the rounds where B1 and GB
    moved together, GB − B1 was 0.656 and 0.862 ms.
  - **The fused paths.** FD, FB and FBD were steady in all three rounds.
- The library default, `DEFAULT_MIN_STACK_BYTES`, is set from FBD by gate 4 (default options): 32 MiB, qualified for
  `RelocBackend()`'s default transfer options. FB's crossover, gate 4 (tuned options), is 24 MiB. It is published for
  callers that pass `transfer_options={"pinning": "auto", "min_pinned_bytes": 8 << 20, "gather_threads": 8}`.
- The default's history:
  - it started at 32 MiB, before any measurement;
  - the previous run, measured with the tuned options only, set it to 24 MiB;
  - this run sets it back to 32 MiB, from the default-options measurement.

  `latency.json`'s `default_min_stack_bytes` (24 MiB) records the library default at the time of this run. No method
  depends on it.
