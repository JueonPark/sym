# Grouped qualification for issue 264

All 25 registered processes completed with clean interference audits. The 75 case/policy/process records include 22,500 primary completed samples, 7,500 separate CPU-submission samples, and 750 separate construction diagnostics. Each policy has five independent process rounds.

**Selection:** keep `prepared_groups=True, report_mode="native"` for full E2/E5 qualification. It gives the largest completed p50 reduction in all three grouped cases, with positive process-bootstrap intervals against both the frozen baseline and the new-build legacy control. These measurements do not establish a model-level win.

Values below are medians of the five process medians, in microseconds. Effects are separately computed as means of process-paired percentage reductions.

| Case | Policy | Completed p50 | Completed p95 | CPU submission p50 |
|---|---|---:|---:|---:|
| weights_small | baseline | 222.1 | 232.6 | 165.6 |
| weights_small | legacy-eager | 235.2 | 245.1 | 173.3 |
| weights_small | legacy-native | 220.9 | 231.2 | 171.1 |
| weights_small | prepared-eager | 195.5 | 206.6 | 138.1 |
| weights_small | prepared-native | 183.0 | 193.0 | 136.8 |
| weights_decode | baseline | 1386.6 | 1397.2 | 314.7 |
| weights_decode | legacy-eager | 1396.5 | 1406.6 | 321.9 |
| weights_decode | legacy-native | 1376.5 | 1387.1 | 319.5 |
| weights_decode | prepared-eager | 1298.7 | 1310.4 | 226.7 |
| weights_decode | prepared-native | 1277.6 | 1286.8 | 224.8 |
| kv_restore | baseline | 209.5 | 219.7 | 151.2 |
| kv_restore | legacy-eager | 215.0 | 225.2 | 158.0 |
| kv_restore | legacy-native | 202.4 | 212.4 | 155.9 |
| kv_restore | prepared-eager | 190.8 | 200.8 | 134.7 |
| kv_restore | prepared-native | 178.9 | 188.9 | 133.6 |

| Prepared/native versus frozen baseline | Completed p50 reduction (95% process interval) | p95 reduction | CPU submission p50 reduction | Estimated break-even requests across five pairs |
|---|---:|---:|---:|---|
| kv_restore | 14.80% [14.62, 14.98] | 14.07% | 11.46% | 3, 1, 1, 9, 1 |
| weights_decode | 8.07% [7.85, 8.33] | 8.10% | 28.68% | 3, 2, 1, 1, 1 |
| weights_small | 17.88% [17.25, 18.52] | 17.13% | 17.64% | 13, 15, 37, 16, 13 |

Setup and first-request costs are included below (milliseconds). First small-weight calls include lazy native/CUDA setup; explicit-handle timings are separate diagnostics after all timing and do not replace the charged implicit first request.

| Case | Baseline setup / first | Prepared/native setup / first | Explicit handle preparation | Native retained scratch (both policies) | Prepared live slots |
|---|---:|---:|---:|---:|---:|
| weights_small | 12.542 / 16.785 | 12.455 / 17.365 | 0.098 | 17,408 B | 1 |
| weights_decode | 34.984 / 2.902 | 35.072 / 2.900 | 0.120 | 14,721,024 B | 1 |
| kv_restore | 7.801 / 1.013 | 7.828 / 1.054 | 0.066 | 262,144 B | 1 |

Break-even is a steady-state extrapolation of setup plus first-request premium, not a directly measured latency for every reuse count. It varies with cold preparation; small weights need 13–37 requests in these pairs. Decode weights need 1–3, and KV 1–9. Explicit handle preparation includes metadata only; its first native slot construction is additionally charged in the raw diagnostic.

Across ten warmed requests, legacy construction builds ten groups and 40 view descriptors for two-member cases (120 for six weights). Prepared policies build zero groups/descriptors and perform ten rebinds. Eager reporting materializes ten detailed reports; native accounting materializes zero. The native retained scratch counts are identical, warmed native allocation/context deltas are zero, and retained/live native bytes are zero after close.

Whole-process median RSS is roughly 826–925 MiB across all policies/cases and includes CUDA/Torch loaders, CPU allocator caches, snapshots, and benchmark state. Paired RSS differences do not isolate metadata size: prepared/native minus baseline medians are approximately +0.43 MiB (small weights), +0.04 MiB (decode weights), and +3.79 MiB (KV), while retained native scratch is identical. Treat these as whole-process observations, not exact prepared-plan byte costs. The structural cache is bounded by handles/slots/member counts; it has no separate host-byte budget.

The new-build legacy control is slower than the frozen baseline, especially for small groups. This exposes the cost of locking/lifetime changes instead of attributing all differences to reusable structure. Native report accounting alone does not remove that entire small-group penalty. Full policy tables and all unfavorable effects are retained in the raw analysis.

See [protocol](PROTOCOL.md) and [reproduction entry point](README.md). See [final E2/E5 results](RESULTS.md) and the [versioned Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264) for raw results and checksum-verified build/input provenance.
