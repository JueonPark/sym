# Review and validation disposition (#269)

The independent whole-change source, evidence and public-artifact assessments are
complete; four source findings were addressed in one executor fix pass. No material
findings remain. Public-download verification is recorded in the
[Wiki receipt](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-269).
No second source review of fixes is claimed.

## Findings and verification

1. **Hidden mutation in high-level FX calls (Important).** The reviewer reproduced
   a source overwrite through `out=` and a custom `.clone()` receiver between
   the private copy and cat. The pass now rejects non-`None` outputs, unknown
   operands/receivers, subclasses and attribute access that could hide effects.
   Four mutation regressions failed before the fix, then passed; additional
   subclass coverage also passes. Unsupported graphs keep their original copy.
2. **Eager callable after backend closure (Important).** A changed layout-only
   eager graph previously survived owner closure because it had no native handles.
   Rewritten eager callables now retain the backend lifetime guard. The regression
   failed before the fix and passed afterward; Inductor retains its existing guard.
3. **Unmeasured model cache boundary (Important).** Token/logit checks and standalone
   cat tests did not prove actual decoder cache equality. Every successful model
   arm now performs an untimed post-measurement audit of eight full requests
   against a materialized reference using the same compute backend and identical
   arguments. The analyzer requires 256 exact cache checks for decode or 64 for
   prefill. Missing-count regressions failed before enforcement, then passed.
   Eager, Inductor, replay and prefill correctness pilots all passed this audit.
4. **No-op contiguous cost (regraded Important).** An already-contiguous view could
   report nonexistent allocation savings and incorrectly meet `min_saved_bytes`.
   Such candidates now report zero bytes and `already_contiguous`. The regression
   failed before the fix and passed afterward; a true clone still reports and
   eliminates its allocation. The public planner-admission effect justified
   treating this as more than a reporting nit.

The final local production/analyzer run passed **1,571 tests**, with 16 skips and
five existing warnings (fork, forward-AD/Python, sparse and nested APIs).
A separate two-GPU wire-device check passed. The warnings do not add new findings.

CI then exposed three test-fixture portability failures on the pinned CPU-only
Torch wheel: `Tensor.contiguous()` rejected a fake CUDA tensor before dispatch.
All three were reproduced locally. The fixtures now construct the identical
materializing clone IR directly; five CPU-only checks and all 28 CUDA rule tests
passed. This changed test construction only; frozen runtime/measurement hashes
remain unchanged. All seven GitHub checks pass on `9f7265a`.

## Completed evidence assessment

All 168 qualification processes completed: 162 successful model arms and 90
supplementary rows, including six actual native-path pressure OOMs. The strict
analyzer passed; its 30 regression tests also pass. The reviewer independently
recomputed all four analysis JSON files, matching the saved results exactly,
and reproduced both Markdown reports. None of the 24 parent targets or eight
lowered regimes against the strongest Torch alternative qualifies.

The reviewer independently parsed all 27 compressed profiler traces and matched
kernel counts, positive CUDA allocation events/bytes, and H2D bytes against the
profile summary. Eager lowering removes 24 kernels relative to transparent
copies, but only three relative to native batched conversion. Inductor shows no
additional kernel/allocation saving. Both transparent materialized and lowered
paths fit the pressure budget, so that capacity result is not unique to lowering.
Slower eager consumer cases remain reported.

Every model arm passed eight complete cache-audit requests: 256 exact cache
checks for decode or 64 for prefill. All 36 lowered decode arms selected six
static decisions; prefill and controls selected none. Qualification audits
contain zero exclusions and at least 62.69 observed quiet seconds per group,
with no foreign activity. The 17 accepted diagnostic jobs are separate from
timing; their 12 contaminated attempts remain preserved. Frozen qualification
and diagnostic provenance match, including the executed source hashes.

## Public artifact verification

The anonymous 217,212,575-byte download matches archive SHA-256
`65a412945300825929c82022620ebbee88cd2449a7882439b95761d655ac52f9`.
All 46,036 inventoried files verify. Four analysis JSON files, the profile summary,
and both reports reproduce byte-for-byte. The relocated runtime passed 39 tests
with one expected skip; loaded Python modules and native libraries belong to the
extracted package. The initial external launcher's interpreter-path error and
successful correction remain in the public logs; no packaged source changed.

The same reviewer independently retrieved and hash-verified all 13 receipt evidence
files, checked the downloaded archive checksum/size, reran the package inventory
verification, and matched the seven reproduced files to the receipt. No material
artifact finding remains. The final recommendation permits delivery after the
CI/PR mechanical gate, retaining the default-off setting and negative performance
conclusion. The immutable package preserves source/report revision `fc2c1bf` and
executed revision `510889b`; subsequent source changes record delivery status only.

## Scope judgments

- Inverse KV layout is the selected first consumer target. Dequantization inside
  GEMM remains a later experiment, with its own arithmetic qualification.
- A generic transparent FX rule and explicit wire-restoration API provide the
  integration boundary. Automatic whole-program region discovery is not claimed.
- Valid dense packed views with nonzero offsets are supported. Source alias
  mutation and asynchronous source lifetime remain guarded; no extra zero-offset
  device copy is inserted.
- Static-shape compilation is matched across all model arms. Unknown/symbolic
  metadata is excluded; historical dynamic-compiler timings are not paired data.
- Training/transforms are outside the optimization targets. Existing fallback
  and owner-lifetime requirements remain in scope.
- Performance claims are restricted to the declared single-GPU/runtime setup.
  The separate two-GPU check establishes a device guard, not multi-GPU speedup.
- Host source retention protects lifetime, not concurrent mutation. Callers must
  wait for copy completion before changing source bytes, as the API specifies.

The single independent source/harness review began before the long frozen
campaign to avoid measuring known defects. The same reviewer completed the
evidence and public-download assessments. There are no deferred minor findings.
