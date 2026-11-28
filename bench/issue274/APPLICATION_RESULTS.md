# Pretrained SqueezeNet 1.1 with a live image boundary

On a Ryzen 7 7800X3D and RTX 4070 Ti SUPER, Sym + Inductor reduced warm
completed p95 for this application relative to both Torch controls. Against
Inductor the paired p95 ratio was 0.795, with a 95% paired bootstrap interval
of 0.634–0.904. This met the frozen latency criterion. Eager Torch completed
the first request sooner and had a lower total lifecycle cost over 112 calls.
[Protocol](APPLICATION_PROTOCOL.md) · [Raw records](application-records.tar.gz)

The endpoint starts from normalized dense CPU HWC float32 and finishes when
pretrained SqueezeNet's GPU logits complete. The request trace uses 112
distinct crop/flip variants of one pinned photo. The same weights, full
application callable, fp32 semantics and Inductor settings apply to the two
compiled arms. JPEG decoding and input preparation are charged in setup
where applicable. Both compiled arms use fresh local caches per subprocess
and run without CUDA Graph replay. [Frozen input manifest](application-assets.json)

## Completed latency

Each entry is a median across six independent processes for that arm. The
p50 and p95 entries are medians of the process percentiles, each calculated
from 96 warm completions. First completion includes on-demand compilation.
Lifecycle is the sum of recorded setup, all 112 completed calls and owner
retirement. It excludes correctness checks and diagnostic replays.

| Arm | First completed, ms | Warm p50, ms | Warm p95, ms | Warm sum, ms | Lifecycle, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch eager | 233.11 | 2.989 | 5.523 | 315.58 | 1,906.93 |
| Torch Inductor | 11,711.22 | 2.207 | 5.022 | 246.16 | 14,237.88 |
| Sym + Inductor | 5,505.59 | 1.804 | 4.154 | 219.49 | 7,892.61 |

The table below gives the geometric mean of the six **within-round** latency
ratios and the paired bootstrap interval. An interval below 1 favors Sym in
this trace. All six Sym/Inductor p95 ratios were below 1. The secondary p50 metric had one losing pair against Inductor (1.081).
Sym took 22.49× as long to complete the first request and 3.97× as long
for the measured lifecycle as eager Torch.

| Metric | Sym / Torch eager [95% interval] | Sym / Torch Inductor [95% interval] |
| --- | ---: | ---: |
| Warm p50 | 0.618 [0.557, 0.690] | 0.789 [0.650, 0.928] |
| Warm p95 | 0.734 [0.606, 0.887] | **0.795 [0.634, 0.904]** |
| First completed | 22.488 [21.156, 23.873] | 0.449 [0.420, 0.472] |
| Lifecycle | 3.974 [3.746, 4.181] | 0.535 [0.503, 0.558] |

In arm order eager / Inductor / Sym, median startup and configuration costs
were 933 / 968 / 941 ms, input and CPU model setup took 101 / 96 / 101 ms,
CUDA initialization and model upload took 298 / 283 / 277 ms, and callable
setup took <0.01 / 824 / 818 ms. Each subprocess uses the common harness
imports, including the Sym runtime Python modules. The Sym cold capture
callback retains references to the captured and rewritten graphs for later
inspection. Detailed reports, comparisons and hashes are outside request
timing.

## Execution, correctness and memory

- All 18 process cells and 2,016 requests passed. The largest absolute
  compiled-logit error was 1.34 × 10⁻⁵ against eager Torch. Every request had
  the same top-1 class as its eager oracle. The initial result stayed valid
  through subsequent requests and after owner close.
- Each Sym process had one Dynamo callback, one 281-byte layout plan and
  one Inductor compile in the measured trace. The first request used the
  saved Torch region. The next constructed a native group, and 110 rebinds
  followed with new live input storage. The 666 total prepared executions
  completed `layout_cuda` through the ordinary adapter. The dispatch moves
  602,112 bytes per request and the detailed report records one layout
  kernel launch for the observed group. No native report was materialized
  during timing.
- The untimed audit matched three observed prepared NCHW tensors bit for
  bit against eager materialization in each Sym process. Their separate
  buffers remained exact after retirement. The unsupported input with
  strides (1344, 6, 2) added a `source_layout` capture exclusion, returned
  matching logits, and submitted no native group. The earlier smoke record
  preserves the audit failure caused by comparing against CHW before
  accounting for the folded batch-dimension view. The final smoke passed
  under the unchanged workload and protocol before timing began.
- Median Torch peak allocated memory through the measured trace was 10.95 MiB
  eager and 58.73 MiB in both compiled arms. Peak reserved medians were
  28, 76, and 78 MiB in arm order. Host peak RSS medians were 1,011, 1,276,
  and 1,279 MiB. Sym's native counters separately record a 602,112-byte
  device scratch allocation and zero bytes of retained host scratch.
  Device-wide snapshots include other device usage and can miss transient
  peaks. The Torch allocator, process RSS and Sym-owned native scratch cover
  different scopes. The run used default clocks with the display attached.

This is the aggregate application effect of rewriting a layout boundary and
using prepared native execution with an Inductor consumer. The comparison
also includes whatever effect the rewrite has on Inductor's generated graph.
It supports the fixed-shape application result on this machine. A separate
mechanism campaign isolates layout-group reuse and typed stage specialization.
[Mechanism controls](../issue272/README.md)

## Reproduce and audit

The protocol and asset manifest were committed before accepted timing at
`9a986f34a1ef9e14d693e5236a49a7567e1cd130`. The run manifest records that
revision, the protocol and runner SHA-256 values, model/photo SHA-256,
runtime file hashes, and machine snapshots. The CPU-only scorer needs
the repository sources and Python, independent of Torch and CUDA.

```sh
out=$(mktemp -d)
tar -xzf bench/issue274/application-records.tar.gz -C "$out"
python -m bench.issue274.application score "$out/application-records/accepted"
```

The archive includes all per-request records, stdout/stderr logs, and both
untimed smoke attempts. Its SHA-256 is
`10d9fb8fd82f2189445e5f75409a15755f6d3935569a3add8f0bc0cba8881014`.
[Archive](application-records.tar.gz)

To rerun on CUDA, use the qualified Python 3.14.7 and Torch 2.14.0+cu126
environment with NumPy 2.5.3 and Pillow 12.2.0. Rebuild the Release runtime
with CUDA architecture 89 and the SHA-verified #265 compiler bundle using
the [specialized build instructions](../issue272/README.md#reproduce). Set
`BUILD` to the resulting `on` directory. It includes the compiler exporter.
Fetch the three files listed in the input manifest into `INPUTS`; the runner
verifies every SHA-256 before loading them. The runner loads the original
TorchVision classes and suppresses the constructor's usage telemetry hook.
The architecture and forward methods are unchanged. Keep the Git freeze
commit available and choose fresh output paths.

```sh
python -m bench.issue274.application preflight \
  --build "$BUILD" --inputs "$INPUTS" --output "$SMOKE"
python -m bench.issue274.application run \
  --build "$BUILD" --inputs "$INPUTS" --preflight "$SMOKE" --output "$RESULTS" \
  --freeze-revision 9a986f34a1ef9e14d693e5236a49a7567e1cd130
```
