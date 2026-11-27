# Joint region planner results

These results score the complete measured catalog. The oracle can choose any executable plan in the catalog. For the arrival experiment it can also choose the offered load.

## Warm request, five-call lifecycle and paced throughput

| Objective | Workload | Budget MiB | Joint plan | Predicted | Measured | Oracle | Regret % | Outcome |
|---|---|---:|---|---:|---:|---:|---:|---|
| warm | train | 7 | cpu-r0-c1-d2-reuse | 0.52 | 0.59 | 0.59 | 0.00 | ok |
| lifecycle | train | 7 | cpu-r0-c1-d2-reuse | 3.18 | 6.24 | 4.29 | 45.25 | ok |
| warm | train | 12 | fused-r2-c1-d2-fresh | 0.39 | 0.85 | 0.21 | 313.21 | ok |
| lifecycle | train | 12 | fused-r2-c1-d2-fresh | 2.73 | 9.59 | 3.54 | 170.44 | ok |
| warm | held_capacity | 11 | cpu-r0-c1-d1-reuse | 0.70 | 1.25 | 1.10 | 13.23 | ok |
| lifecycle | held_capacity | 11 | cpu-r0-c1-d1-reuse | 4.09 | 9.82 | 7.51 | 30.77 | ok |
| warm | held_capacity | 14 | cpu-r2-c1-d1-reuse | 0.39 | 1.28 | 0.74 | 72.34 | ok |
| lifecycle | held_capacity | 14 | cpu-r2-c1-d1-reuse | 2.93 | 13.05 | 5.37 | 143.02 | ok |
| warm | held_capacity | 18 | cpu-r4-c1-d1-fresh | 0.10 | 0.10 | 0.10 | 0.00 | ok |
| lifecycle | held_capacity | 18 | cpu-r4-c1-d1-fresh | 1.88 | 4.20 | 4.20 | 0.00 | ok |
| warm | held_tail | 10.5 | fused-r0-c2-d2-reuse | 0.89 | 0.90 | 0.79 | 13.35 | ok |
| lifecycle | held_tail | 10.5 | fused-r0-c2-d2-reuse | 5.03 | 5.77 | 5.43 | 6.30 | ok |
| slo | held_tail | 10.5 | fused-r0-c2-d2-reuse @ load 1 | 673.49 | — | 671.95 | — | slo_miss |
| warm | held_tail | 14 | fused-r0-c1-d2-reuse | 0.73 | 1.58 | 0.41 | 284.05 | ok |
| lifecycle | held_tail | 14 | fused-r0-c1-d2-reuse | 4.23 | 4.93 | 4.93 | 0.00 | ok |
| slo | held_tail | 14 | fused-r0-c1-d2-reuse @ load 0 | 1334.18 | — | 1330.29 | — | slo_miss |
| warm | held_final | 17 | cpu-r0-c1-d1-reuse | 1.05 | 1.35 | 1.35 | 0.00 | ok |
| lifecycle | held_final | 17 | cpu-r0-c1-d1-reuse | 5.81 | 10.98 | 9.10 | 20.66 | ok |
| slo | held_final | 17 | cpu-r0-c1-d1-reuse @ load 1 | 671.74 | — | 665.72 | — | slo_miss |
| warm | held_final | 23.5 | cpu-r6-c1-d1-fresh | 0.39 | 0.21 | 0.21 | 0.00 | ok |
| lifecycle | held_final | 23.5 | cpu-r6-c1-d1-fresh | 3.24 | 5.16 | 5.16 | 0.00 | ok |
| slo | held_final | 23.5 | cpu-r6-c1-d1-fresh @ load 0 | 1349.69 | 1332.57 | 1333.82 | 0.09 | ok |

Latency is in ms. Paced throughput is in completed requests/s. A failed deadline renders a selection infeasible at that load. Infeasible and refused selections have no numerical regret.

## Same-engine controls

| Objective | Policy | Scored / total | Invalid admissions | Missed feasible | No oracle | Median regret % | Worst regret % |
|---|---|---:|---:|---:|---:|---:|---:|
| warm | fixed-cpu | 9 / 9 | 0 | 0 | 0 | 103.28 | 1142.48 |
| warm | fixed-gpu | 6 / 9 | 0 | 3 | 0 | 428.93 | 3078.75 |
| warm | fixed-forward | 7 / 9 | 0 | 2 | 0 | 205.20 | 2989.76 |
| warm | fixed-fused | 7 / 9 | 0 | 2 | 0 | 325.77 | 3150.95 |
| warm | resident | 4 / 9 | 0 | 5 | 0 | 0.00 | 0.00 |
| warm | per-transfer | 9 / 9 | 0 | 0 | 0 | 103.28 | 1142.48 |
| warm | placement-only | 9 / 9 | 0 | 0 | 0 | 24.81 | 225.71 |
| warm | schedule-only | 9 / 9 | 0 | 0 | 0 | 110.37 | 1522.67 |
| warm | buffer-only | 9 / 9 | 0 | 0 | 0 | 212.33 | 1115.78 |
| warm | joint | 9 / 9 | 0 | 0 | 0 | 13.23 | 313.21 |
| lifecycle | fixed-cpu | 9 / 9 | 0 | 0 | 0 | 105.05 | 149.38 |
| lifecycle | fixed-gpu | 6 / 9 | 0 | 3 | 0 | 169.48 | 389.57 |
| lifecycle | fixed-forward | 7 / 9 | 0 | 2 | 0 | 92.15 | 334.17 |
| lifecycle | fixed-fused | 7 / 9 | 0 | 2 | 0 | 56.74 | 346.59 |
| lifecycle | resident | 4 / 9 | 0 | 5 | 0 | 3.10 | 88.88 |
| lifecycle | per-transfer | 9 / 9 | 0 | 0 | 0 | 105.05 | 149.38 |
| lifecycle | placement-only | 9 / 9 | 0 | 0 | 0 | 19.63 | 124.76 |
| lifecycle | schedule-only | 9 / 9 | 0 | 0 | 0 | 143.88 | 281.10 |
| lifecycle | buffer-only | 9 / 9 | 0 | 0 | 0 | 100.62 | 229.50 |
| lifecycle | joint | 9 / 9 | 0 | 0 | 0 | 20.66 | 170.44 |
| slo | fixed-cpu | 0 / 4 | 2 | 2 | 0 | — | — |
| slo | fixed-gpu | 1 / 4 | 0 | 3 | 0 | 100.04 | 100.04 |
| slo | fixed-forward | 0 / 4 | 1 | 3 | 0 | — | — |
| slo | fixed-fused | 2 / 4 | 1 | 1 | 0 | 100.56 | 103.02 |
| slo | resident | 2 / 4 | 0 | 2 | 0 | 49.09 | 98.10 |
| slo | per-transfer | 0 / 4 | 2 | 2 | 0 | — | — |
| slo | placement-only | 2 / 4 | 2 | 0 | 0 | 49.10 | 98.10 |
| slo | schedule-only | 0 / 4 | 2 | 2 | 0 | — | — |
| slo | buffer-only | 0 / 4 | 2 | 2 | 0 | — | — |
| slo | joint | 1 / 4 | 3 | 0 | 0 | 0.09 | 0.09 |

## Held-out prediction and feasibility

Warm error is the absolute percentage error for each admitted, successful catalog row. The accuracy columns also count refusals against the independent execution result.

| Workload | Catalog rows | Successful | Arena refusals | Driver OOM | False admit | False reject | Peak mismatches | Median error % | Worst error % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 130 | 101 | 29 | 0 | 0 | 0 | 0 | 31.24 | 154.65 |
| held_capacity | 195 | 105 | 90 | 0 | 0 | 0 | 0 | 38.64 | 81.07 |
| held_tail | 130 | 81 | 49 | 0 | 0 | 0 | 0 | 41.11 | 232.62 |
| held_final | 130 | 71 | 59 | 0 | 0 | 0 | 0 | 31.00 | 99.05 |

## Paced sojourn

| Workload | Budget MiB | Joint period ms | Predicted p95 ms | Measured p95 ms | Completed/s at selected load | Feasible pairs | Feasible oracle plan / period ms |
|---|---:|---:|---:|---:|---:|---:|---|
| held_tail | 10.5 | 1.50 | 0.89 | 11.48 | 565.77 | 2 | fused-r0-c1-d1-reuse / 1.5 |
| held_tail | 14 | 0.75 | 0.73 | 35.33 | 600.92 | 18 | gpu-r3-c2-d2-reuse / 0.75 |
| held_final | 17 | 1.50 | 1.05 | 8.32 | 589.73 | 2 | fused-r0-c2-d1-reuse / 1.5 |
| held_final | 23.5 | 0.75 | 0.39 | 0.79 | 1332.57 | 28 | forward-r3-c1-d1-reuse / 0.75 |

The no-oracle count includes every cell with zero measured feasible choices, including a cell in which the policy admitted a plan that missed its deadline.

## Warm amortization against the current per-transfer selector

This estimate charges all independent component calibration to a single deployment plus the observed difference in setup, first completion and release. Subsequent calls use the measured warm medians. The estimate assumes stationary service on this host. The last column additionally charges the full cold Triton import and qualification. A dash means the selected plan has no warm saving. CPU search and common CUDA initialization are itemized in the [planner report](README.md).

| Workload | Budget MiB | Per-transfer ms | Joint ms | Saved per warm call ms | Calls to recover calibration and lifecycle difference | Calls including cold qualification |
|---|---:|---:|---:|---:|---:|---:|
| train | 7 | 0.82 | 0.59 | 0.23 | 3934 | 7524 |
| train | 12 | 1.03 | 0.85 | 0.18 | 4877 | 9324 |
| held_capacity | 11 | 1.58 | 1.25 | 0.33 | 2730 | 5222 |
| held_capacity | 14 | 1.51 | 1.28 | 0.23 | 3922 | 7485 |
| held_capacity | 18 | 1.26 | 0.10 | 1.16 | 780 | 1490 |
| held_tail | 10.5 | 1.59 | 0.90 | 0.69 | 1298 | 2482 |
| held_tail | 14 | 1.74 | 1.58 | 0.16 | 5473 | 10484 |
| held_final | 17 | 1.64 | 1.35 | 0.30 | 3023 | 5782 |
| held_final | 23.5 | 2.10 | 0.21 | 1.89 | 478 | 913 |
