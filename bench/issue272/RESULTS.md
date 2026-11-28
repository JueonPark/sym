# Reuse on the original E3 shape trace

On the RTX 4070 Ti SUPER, caching a prepared layout-cast program saved 40 of
45 preparations and reduced time across the repeated requests by 8.74%.
Changing dequantization parameters on every eligible call yielded zero cache
hits. Reusing groups increased repeated time for that case by 2.45%.
Both outcomes include preflight checks, fresh output allocation, transfer and
host-observed completion.

## Cold, total and repeated costs

Each cell is the median of three independent process values in milliseconds.
Cold includes import/configuration, CUDA initialization, owner setup and the
first completed request. First cycle and all requests sum 12 and 60 completed
calls respectively. Lifecycle total adds setup and owner retirement to all 60
calls. Warm sums the last 48 calls. Warm p50 is the median per-call latency
within that window. Input generation and verification are outside the timers.
Each process saw the same original request order. [Protocol](PROTOCOL.md)

| Original trace | Arm | Cold | First cycle | All requests | Lifecycle total | Warm | Warm p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layout cast | Cache off | 1785.46 | 275.12 | 302.13 | 1955.02 | 27.01 | 0.60 |
| Layout cast | Groups off | 1771.23 | 263.63 | 287.49 | 1936.45 | 24.86 | 0.55 |
| Layout cast | All on | 1696.69 | 267.38 | 292.17 | 1863.63 | 24.30 | 0.54 |
| Layout cast | Specialization off | 1770.36 | 270.95 | 295.07 | 1933.81 | 24.11 | 0.54 |
| KV restore | Cache off | 1705.29 | 242.29 | 262.99 | 1854.20 | 20.77 | 0.46 |
| KV restore | Groups off | 1702.49 | 243.52 | 264.09 | 1855.11 | 19.93 | 0.45 |
| KV restore | All on | 1754.91 | 254.99 | 275.62 | 1916.61 | 19.96 | 0.44 |
| KV restore | Specialization off | 1642.23 | 238.58 | 257.31 | 1790.93 | 19.22 | 0.42 |
| Live dequantization | Cache off | 1765.09 | 201.68 | 222.33 | 1858.74 | 20.65 | 0.46 |
| Live dequantization | Groups off | 1659.62 | 199.04 | 219.32 | 1756.49 | 20.66 | 0.45 |
| Live dequantization | All on | 1683.56 | 195.91 | 216.88 | 1776.17 | 21.15 | 0.46 |
| Live dequantization | Specialization off | 1716.16 | 212.19 | 235.64 | 1816.36 | 21.74 | 0.49 |
| Live indices | Cache off | 1746.31 | 224.44 | 247.56 | 1875.14 | 20.53 | 0.44 |
| Live indices | Groups off | 1756.48 | 227.04 | 246.84 | 1889.26 | 19.80 | 0.41 |
| Live indices | All on | 1698.34 | 220.41 | 239.07 | 1828.51 | 19.10 | 0.39 |
| Live indices | Specialization off | 1657.92 | 215.62 | 236.10 | 1784.73 | 20.47 | 0.42 |

## Which work changed

Each cell gives the mean paired percentage reduction in warm time, followed by
the three individual round effects. A negative value means enabling the factor
took longer. Cache-off and groups-off both construct each group per call.
Comparing these two arms isolates program reuse. Groups-off and all-on share
the same program cache and kernels. Specialization-off and all-on share the
same cache, groups and compiler, and load different native kernel builds.

| Original trace | Program cache | Prepared groups | Specialized kernel build |
| --- | ---: | ---: | ---: |
| Layout cast | **8.74%** (+11.6, +6.3, +8.3) | 1.31% (−3.9, +2.3, +5.6) | 0.34% (−2.8, −1.3, +5.1) |
| KV restore | **4.17%** (+0.9, +7.5, +4.1) | 0.03% (+4.1, −3.5, −0.5) | −3.67% (−5.3, −7.3, +1.6) |
| Live dequantization | 1.82% (+1.0, +4.5, −0.0) | **−2.45%** (−3.4, −1.6, −2.4) | **3.43%** (+3.5, +4.1, +2.7) |
| Live indices | 5.19% (+9.1, −7.9, +14.4) | 1.08% (+0.0, −0.3, +3.5) | 2.86% (+8.8, −9.7, +9.5) |

All arms used immediate admission with the forced CPU row for typed plans.
KV restore used its ordinary layout CUDA row, providing a negative control for
the typed-kernel build. Live indices rebound all 55 concrete index programs
in every arm. Neither group-reuse nor kernel-build timings on those indexed
requests demonstrate reuse of a native group or activation of a register
kernel. The separate probe records actual kernel execution on held-out chains.
[Execution witness](WITNESS.md)

These counters describe each all-on process. The first preparation count in
each pair is the cache-off control. The group counts are constructions and
rebinds of a **prepared** binder. Per-call legacy groups are counted under
native submissions. Key bytes measure copied immutable parameters. The active
program, live outputs and staging allocations have separate lifetimes.

| Original trace | Native executions / requests | Preparations off → on | Hits | Entries / retained key bytes | Prepared constructions / rebinds | Plan compiles | Preparation time, ms¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Layout cast | 45 / 60 | 45 → 5 | 40 | 5 / 0 | 5 / 40 | 1 | 6.49 |
| KV restore | 45 / 60 | 45 → 5 | 40 | 5 / 0 | 5 / 40 | 1 | 6.31 |
| Live dequantization | 50 / 60 | 50 → 50 | 0 | 32 / 33,024 | 50 / 0 | 1 | 10.46 |
| Live indices | 55 / 60 | 55 → 7 | 48 | 7 / 0 | 0 / 0 | 3 | 8.03 |

¹ Median cumulative preflight preparation across three processes, already
included in completed-call times. The live dequantization cache evicted 18
revisions. The cache-off arm retained zero entries and parameter key bytes in
all four cases. All 48 processes had zero additional Dynamo callbacks after
the first cycle.

Each of the 2,880 requests passed the archived exact-output oracle. Each
process also rechecked a held output after closing the owner. Empty tensors,
conditional materializations and unsupported symbolic expressions account for
the recorded capture exclusions, depending on the trace. Every paired step
had matching operand metadata, actual dispatch, payload, logical transfers,
fallback/exclusion counters and Dynamo callback count. The evidence covers
blocking inference on one CPU/GPU combination with three process pairs.
[Original E3 arms and archived matched Torch controls](CONTROLS.md#cache-reuse-and-the-original-e3-request-trace)

## Replay without CUDA

The archive contains the frozen contract and all 48 pairs of raw worker and
contamination-audit JSON files. Its SHA-256 is
`9dfdbc3c93ee5d141cdd32a37700c6b0b8d527fae3ea2e32bb921be05b753c07`.
Re-scoring the archive reproduces the complete analysis, including all request
deltas, absolute times, process pairs and memory gauges. [Raw records](e3-records.tar.gz)

```sh
mkdir -p /tmp/sym272-e3-replay
tar -xzf bench/issue272/e3-records.tar.gz -C /tmp/sym272-e3-replay
python -m bench.issue272.ablate score /tmp/sym272-e3-replay
```
