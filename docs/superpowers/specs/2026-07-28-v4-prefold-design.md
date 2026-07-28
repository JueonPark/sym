# V4 — P4 pre-folded transform path (issue #98) — design

**Status**: approved 2026-07-28.
**Scope**: sub-issue #V4 of #94 (validity hardening). Implements P4 — the
load-time amortized/precomputed transform — as a libreloc component, and
measures it against the pre-registered gates V4-G1..G3 on the Gen3 box
(EPYC 7351 / 4× RTX 2080 Ti).

## Why

R3 (#92, `docs/r3-exp3-multigpu.md`) measured scatter int8 K=4 N=8192:
Method A wall 16.81 ms, of which the int8 DMA is 4.29 ms vs B_xK's
15.21 ms. A would win ~3.5× if the CPU transform were off the critical
path; it scored 0.92× because the transform re-runs every load. A
static-shape weight transformed once at load time and reused across many
transfers is exactly what the plan machinery expresses, and it removes the
host-transform term that failed G3/G4/G5 on this AVX2 host. P4 has never
been implemented.

## Architecture

Two units in libreloc, one consumer in the bench. Mechanism and policy are
separate so the policy is unit-testable without hardware and does not
block on #V3's cost model.

### 1. Library: `libreloc/include/reloc/Prefold.h` + `libreloc/src/Prefold.cpp`

**Mechanism — `prefoldArtifact`.** Given a `BoundPlan` (all symbols
bound), the fp32 source, and an output spec, materialize the transformed
artifact once into a pinned host buffer and return it. The artifact is the
transfer source: DMAs read it directly.

- Output specs (only what the R3 scenarios need — YAGNI; an f32
  relocate-only spec is a natural later addition but nothing in V4
  measures it):
  - `S8GatherQuant` — fused relocate+quantize (broadcast's
    blocked-transpose plan), via `gatherQuantizeF32S8Parallel`;
  - `S8QuantPack` — contiguous quantize (scatter's identity plan), via
    `quantizePackF32S8Parallel`.
- No new kernels; the component only sequences existing executors into a
  pinned allocation.
- `PrefoldArtifact`: move-only owner of `{pinned ptr, bytes, dtype}`;
  frees the pinned allocation on destruction. Cold-path costs (the
  `cudaHostAlloc` and the fold itself) are the caller's to time — the API
  does not hide them.
- Parallelism: takes a caller-owned `GatherPool` + `Variant`, same
  convention as the parallel quant wrappers.
- Preconditions (checked unconditionally, the #63 lesson): packed dst,
  pad-free plan; quant specs additionally need per-channel invScales sized
  to the plan's outer axis.
- Builds and runs without CUDA: pinned allocation goes through a tiny
  alloc hook so the CPU-only CI tree (`RELOC_ENABLE_CUDA=OFF`) exercises
  the whole component with plain aligned allocation. (Precedent:
  HostBackend exists so Strategy-4 runs in CI.)

**Policy — `shouldPrefold`.** The standalone amortization rule as a pure
function:

```
prefoldWins(nReuse, tTransformMs, tPrefoldMs, penaltyMs)
    := nReuse * tTransformMs > tPrefoldMs + penaltyMs
```

`tPrefoldMs` is the one-time fold; `penaltyMs` carries memory-side costs
the caller supplies (cold pinned allocation; a memory-budget term). Pure
arithmetic, no measurement dependencies. With `tPrefold ≈ tTransform` and
zero penalty, prefold wins for any `nReuse ≥ 2` — the losing
configurations live entirely in the penalty term and in `nReuse = 1`,
which is why the counter-cases below are what they are.

### 2. Bench: extend `bench/rtrack/multigpu_reloc.cu`

- New method `aprefold` alongside `a` / `bxk` / `bstaged`: identical
  delivery to `iterA` (K concurrent int8 DMAs, spin-barrier start) but the
  transform and the artifact's pinned allocation are hoisted to setup via
  `prefoldArtifact`. Reported against both `bxk` and `a` in the same JSON.
- **Reuse sweep** `--reuse 1,2,4,16`: amortized per-load wall for
  A (`n × (transform + DMA)`) vs prefold (`fold once + n × DMA`), so the
  amortization rule is tested against measured reuse counts.
- **Counter-case modes** (both predicted losses):
  - `--cold`: `nReuse = 1` with the artifact's `cudaHostAlloc` + fold
    inside the timed path — the cold single-use load.
  - `--streaming`: the source mutates every iteration (touch one value
    per channel, then re-fold) — weight streaming / model swapping, where
    the artifact is never reusable and prefold degenerates to Method A
    plus artifact-management overhead.
- Memory-budget regime: analytical paragraph in the report (no eviction
  harness) — holding source + artifact costs `(1 + r) × S` of pinned
  memory; state where that breaks even rather than contriving a harness.
- Verification unchanged: every method bit-exact against the CPU scalar
  reference before timing.

### 3. Gates: `bench/rtrack/exp4v_gate.py`

Bars fixed in code before the measurement run (V1 discipline, `a051a5a`):

- **V4-G1**: `A_prefold / B_xK ≥ 3.0×` on scatter int8 K=4 N=8192 (the
  DMA-only column predicts ~3.5×; the bar allows pipeline overhead).
- **V4-G2**: admissibility in reverse — the prefold int8 DMA leg reaches
  ≥ 0.90 × the box's measured pinned H2D for `r·S` bytes, so the win is
  transfer-bound, not a measurement artifact. (Gen3 pinned H2D = 13.07
  GB/s, committed in `bench/results/v1_*`.)
- **V4-G3**: both measured counter-cases lose, and `shouldPrefold`
  predicts each verdict from the measured `tTransform` / `tPrefold` /
  penalty inputs.

Measurement matrix: scatter + broadcast, N=8192 (+ N=16384 scatter,
matching R3), K=1,2,4; reuse sweep at the G1 cell. The report ties in R4
with one statement: the prefold path has no host-transform term left, so
it is trivially transfer-bound under the hiding-ratio model.

## Testing

- **Unit (CPU-only CI)**: `shouldPrefold` truth table incl. boundary
  (`nReuse·tT == tP + penalty` → not worth it); `PrefoldArtifact`
  move/ownership (no double free).
- **Correctness**: prefold artifact bit-exact vs the scalar reference for
  both output specs (RtrackTest pattern, hand-authored plans — never the
  frozen golden blob, issue #63).
- **Hardware**: the bench's existing per-config verify gate.

## Error handling

Existing conventions: `CUDA_CHECK`/stderr fail-fast on allocation
failure; unconditional precondition checks (not asserts) for plan shape,
since benchmarking builds compile with `-DNDEBUG`.

## Out of scope

- Wiring the decision into the bind/strategy-selection hook (that is #V3,
  which consumes this component's policy function later).
- New workload families, dtypes beyond {f32, s8}, D2H prefold, artifact
  caching/keying (P5 territory).

## Size estimate

~250 LOC library + ~100 LOC tests + ~250 LOC bench/gate ≈ #98's ~0.6k.

## Acceptance (from #98)

Gates V4-G1..G3 reported from committed JSON; amortization rule tested
against measured reuse counts; counter-cases documented, including the
analytical memory-budget statement.
