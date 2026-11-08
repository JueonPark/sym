# Independent implementation and harness review

One independent read-only review examined `ed7a68b..6b9936c` before the expensive
acceptance campaign. It found five Important issues and one Minor issue. All
Important findings were accepted for a single fix-and-regression pass. There
was no second reviewer pass; final validation uses the regressions, complete
suite, frozen campaign and independent download/reanalysis.

| Finding | Correction and regression |
| --- | --- |
| Close replaced a published DataPtr and invalidated retained original backing | Preserve the original allocator DataPtr/StorageImpl throughout. Reuse checks strong TensorImpl and Storage ownership, accounting for Torch's preserved canonical Python wrappers. Close drops only owner references. Retained-backing tests failed before the fix; detached/storage aliases, concurrent consumers, post-close stream use, allocator pressure and quarantine tests cover the resulting contract. |
| Packed Torch changed GPU staging lifetime | Release each GPU input after its kernel enqueue, preserving the original control's same-stream allocation reuse. Completions retain pinned sources. The producer also avoids retaining completed output groups between requests. Both staging-lifetime and completed-output weak-reference regressions failed before correction. |
| Analyzer accepted invalid execution records | Require the qualified environment, executed workload and selected policy, finite positive timings, request/input/caller identities, readiness, group/previous-burst/after-close checks and lifecycle costs. Invalid-record mutations failed before correction. |
| Physical analysis trusted peaks without validating their windows | Validate ordered finite trace samples, require actual samples within measured intervals, and reconstruct warm/deployment peaks from raw traces and reservation baselines. Missing observations remain invalid rather than becoming zero memory use. |
| E5 omitted worker lifecycle costs | Record each caller's construction, first request, warmup samples and retirement, with sampler-relative boundaries and retired arena statistics for both runtimes. Validation rejects missing or inconsistent lifecycle records. |

Exported storage is non-resizable before publication. A new growth test failed
for the earlier packing-only path and now enforces the declared capacity.
Recycling still performs a conservative device completion barrier; the change
does not infer consumer release from transfer completion.

The deferred Minor finding is the group iterator's initial submission, which
occurs when the prefetch iterator is constructed, before delivery timing starts.
The reported group metric is completed **iterator delivery latency**, not a full
per-group submission cost. Per-iterator initialization instrumentation is deferred;
this limits interpretation of very small group timings.

## Scope decisions

- Performance/OOM conclusions come from the subsequent complete campaign;
  pilots remain excluded. An implementation review cannot establish a speedup.
- Final reports, Wiki publication and downloads are verified after measurement.
  Their correctness depends on frozen hashes, complete raw data and successful
  anonymous reproduction, rather than this earlier source review.
- The concurrent close finding rests on unsynchronized DataPtr mutation in the
  code and a deterministic retained-backing failure. No claim is made that the
  original CUDA race was dynamically reproduced. The replacement avoids that
  mutation and is exercised by concurrent and post-close consumer tests.
- Unchanged compiler/conversion kernels are pinned dependencies. This work does
  not re-prove their implementation; numerical qualification checks their use
  in the measured workloads.
- Weak-storage resurrection and raw-pointer reuse after the final strong
  consumer releases ownership are outside the contract. Such a handle is not
  a lease and must not be used to read an earlier request's bytes.

The final results report records regression totals and qualification outcomes.
