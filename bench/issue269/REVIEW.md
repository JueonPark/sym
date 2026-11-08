# Review and validation disposition (#269)

The independent whole-change review is continuing through evidence qualification.
The source review is complete; four findings were addressed in one executor fix
pass. Final performance/artifact assessment is pending. No second source review
of fixes is claimed.

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
campaign to avoid measuring known defects. Its final assessment will incorporate
complete data, strongest-control comparisons and public artifact verification.
There are no deferred minor findings at this stage.
