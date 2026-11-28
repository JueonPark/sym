# Kernel quality from a captured artifact

The same typed artifact extracted from a real Dynamo call took **0.156 ms**
with the current native kernel and **1.044 ms** with the prior kernel. The
current kernel also completed faster than the manual, eager Torch and Inductor
controls in all three rounds. The automatic frontend endpoint completed in
**0.467 ms** with the current kernel. It uses its own binding and submission
path. [Runner](captured_quality.py) · [Raw records](captured-quality-records.tar.gz)

## Matched operations and resources

The caller narrows F32 to F16 on the host, permutes an `8 × 5 × 29 × 151` tensor
to `5 × 29 × 151 × 8`, and transfers the result to CUDA as F32. The shape and
operation chain were outside the #267 tuning set. The ordinary capture and
lowering produced a 620-byte typed plan. All six processes saved the same raw
bytes, with SHA-256
`aa935c728b4a2823d0d7cef45168ebeb0219eebdd6cd1f8bd681544c7643fae3`.
The isolated native producer reloads those bytes and binds the actual input
metadata. The separate automatic endpoint executes the same plan through the
frontend and prepared dispatch. Its completed native report confirms the CPU
row, cut after both casts, one 700,640-byte host chunk, no parameter traffic and
700,640 bytes transferred.

All five isolated producers write to the same 700,640-byte pinned wire tensor
within a 1 MiB wire budget. Each call allocates one fresh CUDA destination,
copies the wire once and synchronizes. Each producer has one host worker.
The materialized arm compiles three separate plans and retains the host
results of narrowing and dense permutation in two separate F16 buffers totaling
700,640 bytes. Widening produces the final wire. The manual loop orders the
permutation and F16 rounding in one pass. Eager and Inductor execute the same
expression and write the common wire. Inductor uses precision-cast emulation
and a fresh cache per process. The automatic endpoint has its own host staging
and one-worker resource owner, so it appears separately.

The prior runtime executes the tiled transpose. The current runtime uses the
short-row tile path and one ordered register cast pair. In a separate CPU-only
probe on the exact saved plan and current runtime, the instrumented kernel
executed **685** successful register runs across all **175,160** elements. Each
run covered stages 0–2, preserving F16 rounding before widening. The probe
records execution counts. The six timing processes verified **1,296** completed
outputs including first calls and warmups, all shapes and strides, independent
wire results, and one held output per arm across retirement. The inputs change
on every measured sample and include half-rounding ties, signed zeros and a
value that overflows to infinity at the narrowing stage.

## Completed latency

Values are milliseconds and report medians across three fresh processes per
runtime. Warm denotes the process median of 30 completed calls after five
warmups. The prior runtime row uses the same captured artifact and isolated
copy path with the older #267 kernel. All timing used one RTX 4070 Ti SUPER,
Torch 2.14.0+cu126, CPU affinity 0–7 and one worker.

| Path | Cold first | Accounted total | Warm p50 |
| --- | ---: | ---: | ---: |
| Materialized, current runtime | 1559.922 | 1579.342 | 0.541 |
| Folded captured artifact, prior kernel | 1997.863 | 2037.218 | 1.044 |
| Folded captured artifact, current kernel | 1906.418 | 1911.781 | 0.156 |
| Manual fused loop | 1550.321 | 1557.577 | 0.217 |
| Eager Torch | 1550.346 | 1557.809 | 0.204 |
| Torch/Inductor | 7936.760 | 7947.552 | 0.273 |
| Automatic endpoint, prior kernel | 1804.578 | 1854.048 | 1.361 |
| Automatic endpoint, current kernel | 1724.633 | 1740.469 | 0.467 |

Cold includes shared campaign imports and CUDA initialization plus arm setup
and the first completed call. The captured-artifact arms also charge the actual
automatic call used to acquire the artifact. Inductor charges its lazy compile
to the first call. Total includes warmups and measured calls, and charges
automatic backend retirement where applicable. Inputs, oracle work, reporting
and binary builds sit outside those timers. These are shared-process first-call
costs, with a fresh process and Inductor cache per round and runtime. The
current artifact path saves warm time against eager Torch but incurs more total
time across the measured 36 calls. The automatic endpoint remains 2.29× the
matched eager warm median. The result applies to this one capture, shape and
machine.

## Reproduction

Build the host-only benchmark helpers against each matched native build and
run the GPU campaign serially. The prior native build reverses the #267 kernel
changes while retaining the current frontend and all other runtime changes.
The helper diagnostic macro enables metadata reporting. Both runner variants
call their linked runtime's production host executor. [Native helper](../issue267/build_native.py) ·
[Matched runtime builds](build_controls.py)

```sh
python -m bench.issue267.build_native --runtime "$on" --output "$helper_on" --specialized
python -m bench.issue267.build_native --runtime "$off" --output "$helper_off"
python -m bench.issue272.captured_quality run \
  --specialized-build "$on" --generic-build "$off" \
  --specialized-helper "$helper_on" --generic-helper "$helper_off" --output "$records"
python -m bench.issue272.captured_quality score "$records"
```

The first preflight used the `.half()` spelling and stopped at frontend
capture before measuring an arm. The published chain uses `.to(torch.float16)`,
which the importer supports. The three recorded rounds used this same caller.
