# Issue 269 qualification protocol

This document declares a candidate for inverse-KV layout materialization removal,
not a predicted performance result. The immutable runner manifest will pin exact
sources, binaries, original workloads, membership and settings before timing.

## Main matrix

Three rounds × two compute backends × three model cases × seven arms =126
fresh-process arms. Cases are original decode(prompt8,new_tokens4), held-out
decode(prompt64,new_tokens4), and original prefill(prompt256,new_tokens1).
The six offload arms are Torch/Sym × native-materialized, optimized-materialized,
and lowered. A shared resident-weight/cache baseline is the seventh arm; it has
no offload obligation and is reported separately from the matched offload target.

All three variants use the same static-shape compilation policy for their GPU
regions; eager executes FX operations, Inductor compiles them. This is an explicit
compiler-execution mode, not an unchanged replay of historical dynamic compilation
results. The native-materialized arm keeps deployed grouped native inverse-KV
conversion. The other two copy identical packed wire bytes with one shared Torch
helper and place a transparent inverse permutation/copy inside the decoder.
Only the lowered arm enables the conservative copy-into-cat rule. Inductor has
its ordinary optimization opportunity in both. Record any identical generated
kernels rather than assigning existing Inductor fusion to Sym.

A replay-enabled original-decode subset adds three rounds × two compute backends
× six offload arms =36 fresh-process arms. Every arm uses the same #268 wrapper,
two slots, eight buckets and256MiB retention admission per GPU region. CPU
transforms/routing/copies remain outside capture. Explicitly prepare every bucket
and freeze before measurement. Compare against the unwrapped main runs as well.

Total model membership:162arms. Rotate three variant blocks by round and reverse
Torch/Sym order in round2; rotate the resident arm's insertion position. Freeze
job IDs/order before accepted timing. Parent acceptance is >=5% lower completed
latency than both fresh eager and Inductor Torch controls in every round of a
named mode, evaluated on unrounded values. Also report fastest offload controls
across variants/replay modes; residency is a separate deployment comparison.

## Numerical and ownership checks

Original FP16 functions/weights, INT8→FP32→FP16 rounding, native dequantization,
matmul accumulation policy and TF32-off flags remain unchanged. Wire layout is
[S,H,D]. The candidate bypasses its private inverse copy before fresh cat;
attention consumes the same contiguous[H,S,D]cache. Compare cache bits exactly,
model output with the original tolerance, and generated token decisions exactly.
No bitwise cross-compiler whole-model arithmetic equivalence is claimed.

Use changing seeded inputs during correctness preparation and repeated samples;
retain outputs across subsequent requests and after owner close. Record actual
lowering selections/exclusions and warm compile/capture counters. Cases with
mixed/visible consumers must retain materialization. No favorable selection of
shapes, samples, variants or failed attempts is allowed.

## Measurements and isolation

Pinned platform: Python3.14.7, Torch2.14.0+cu126, RTX2080Ti; CPU mask4–7,20–23,
eight compute threads and one interop thread, passive OpenMP. Original native
runtime/control binaries remain identical. Two-context/two-slot weight prefetch,
grouped KV, immutable snapshots and native scratch128MiB are shared. The main
matrix does not artificially reserve VRAM; report observed physical memory and
residency. Each process gets a fresh Inductor cache. Compilation and graph setup
are measured separately from warm timings, never hidden.

Observe60 quiet seconds before each matched cell, then audit each process for
foreign GPU/build/busy CPU activity. No local build/test/profile overlaps accepted
timing. Contaminated attempts remain excluded with hashes before any retry.
Unexpected errors stop execution; actual OOMs remain named outcomes.

Five warmup requests and30 completed samples per process. Record preparation,
first request, all warmups, submission, completed p50/p95, close, retained memory,
Torch allocator peaks and2ms sampled physical peaks with explicit phase windows.
Missing physical observations are null, not zero. Samples are lower bounds and
memory scopes overlap; never add them as disjoint quantities. Total(N) retains
recorded intercepts and warm completed slope, with break-even only for positive
warm advantages. Fixture/oracle timing boundaries must be explicit in the report.

## Supplements and profiles

Additional frozen micro membership covers rank3 KV concatenation at canonical and
held-out head/dimension/sequence shapes, positive packed offsets, mixed-consumer
fallback and declared raw-byte/physical-memory limits. Each round must preserve
all variants including budget rejection/OOM. Freeze exact shapes and budgets in
the manifest before qualification; these are mechanism/generalization cases,
not additional model wins. Separate diagnostic runs measure copy kernels,
allocator requests and observed traffic, plus consumer-only timing/slowdowns.
Logical eliminated byte estimates are checked against allocation/kernel evidence.
Diagnostic timings do not supply model acceptance samples or additive phases.

Saved-data analysis must reject incomplete/duplicate membership, mismatched
source/runtime/input hashes, settings, incorrect or missing checks, warm compile/
capture changes, contaminated audits and inadequate observed cooldown. Preserve
all per-round/raw observations and unfavorable outcomes. Public Wiki packaging
includes exact executed sources/runtime, controls, logs/caches/profiles, excluded
pilots, checksums, analysis, reports and independent review disposition.

The six supplement processes (three rounds × eager/Inductor) each run all15
shape/variant rows. Shapes [S,H,D] are [8,16,64], [31,4,32], [4096,16,64],
[512,16,64] with a visible inverse copy, and [8192,16,64] with128MiB free
physical VRAM after reservation. Each uses two adjacent packed members and four
changing inputs. One process shares compiler/library caches among its15rows;
report preparation without claiming independent cold-start comparisons. Rotate
variant order each round and clear inactive Torch cache between preparations.
Measure100 complete requests and100 separate consumer/transfer intervals per
successful row. Preserve actual allocation OOMs as outcomes.

Every successful model arm also performs an untimed, post-measurement cache
boundary audit over all eight inputs. Each actual decoder invocation is compared
to a materialized reference with identical arguments and the same compute backend.
Check both returned caches bitwise and require256 checks for decode or64 for
prefill. Audit compilation/allocation occurs after warm memory snapshots and is
reported as a separate phase; it cannot warm the measured first call. This is
not a bitwise comparison between eager and Inductor arithmetic.
