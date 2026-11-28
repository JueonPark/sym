# Automatic lowering and reuse qualification

Use the four original E3 source functions, inputs and bit-level oracle from the
[November archive](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04).
Keep the 12 shapes in their original order for five cycles, including empty,
singleton, strided, changing-index and changing-parameter requests. The caller supplies
its ordinary Torch function to Dynamo and RelocBackend. All arms use eager
surrounding compute, blocking transfers and fresh outputs. No explicit recipe,
prefetch program, pooled producer or manually fused kernel is an automatic arm.

Freeze four arms per case before inspecting their timings. Use three fresh
processes per arm with rotated arm order, CPU affinity 0–7, eight workers and
one interop thread, two pinned buffers, one transfer stream, and identical
resource limits. Admit eligible source sizes immediately and force the typed CPU
row to expose host kernels. Layout-only and indexed transfers keep their normal
dispatch. The default admission and device policy receive a separate witness.
Use one compiler binary pair in all arms. Only the native runtime shared
library may differ in the kernel comparison.

| Contrast | Control | Treatment | Fixed factors |
| --- | --- | --- | --- |
| Bound specialization reuse | Rebind and prepare every admitted call | Cache the bound program | Per-call groups, current kernels |
| Native group reuse | Construct each native group per call | Rebind an existing group to current storage | Cached programs, current kernels |
| Bounded kernel specialization | Current runtime with only the native changes of b319a0a reversed | Current runtime | Identical cached programs, group reuse and compiler |

Charge import/configuration, CUDA initialization, owner setup and the first
completed request to cold time. Accounted total covers setup, all 60 completed
calls and owner retirement. The repeated window sums the last 48 completed
calls. Input generation, diagnostics and correctness checks sit outside
these timers. Compare times within matched process rounds. Report losses
and the absolute costs of all arms. These descriptive process comparisons
cover one host. The original E1 manual/eager/Inductor controls measure kernel
quality separately. Changing the binary without changing the executed kernel
is a negative control.

Each process must preserve exact outputs and a previously held output across
calls and retirement. At each trace step require matching actual dispatch,
payload, logical transfers, fallback and callback counts across each pair.
Reject incomplete or contaminated processes. Retain the source, environment
and binary hashes, per-call cumulative stats and completion times so that a
CPU-only reanalysis can reconstruct the comparisons. Publication claims
about automatic code selection require an observed native execution on
held-out input. A guard fallback or an explicit hand-written recipe counts
only under its own path.
