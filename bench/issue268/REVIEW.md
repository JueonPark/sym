# Issue 268 review disposition

Independent reviewer: `/root/review_268`, GPT-6 Astra, fresh context. The reviewer
read the whole branch without changing source or running GPU work. Review began
during the early measurement campaign to catch implementation defects before
finishing expensive experiments; final evidence reconciliation found no remaining
Important or Critical finding.

## Fixes with regressions

- Centralize the compute settings signature. Eight changes that previously
  reused an incompatible frozen bucket now fall back: high/medium matmul
  precision, FP16 accumulation, FP16/BF16 split-K policy, math-SDPA reduction,
  SDPA priority/overrideable enablement and deterministic warn-only mode.
  Valid transitions failed before the fix and pass after it. The signature also
  includes available BLAS/linalg preferences and cuDNN selection controls.
- Retire abandoned owners. In both replay and slots-off modes, deleting an owner
  previously returned before its pending private-storage use completed. The
  destructor now uses normal close/retirement in the owning process. Both
  immediate-destruction completion-fence regressions failed before and pass
  afterward. Explicit close remains the way to surface cleanup errors.
- The earlier qualification preparation uncovered a fallback input-lifetime
  defect, independently of this review. A delayed consumer plus allocator reuse
  returned -99 instead of 10. Input stream recording fixed the reproduced
  corruption, and all pre-fix timings were excluded.

- Backward autograd retention was reproduced in eight Parameter/nonleaf × outer
  no-grad/inference × replay/off cases. `inference_mode(False)` alone enabled
  grad for static clones; pairing it with no-grad removes CloneBackward and lets
  caller weakrefs release. The tests failed before and pass after the fix.
- Active forward AD and torch.func transforms are rejected before copying, so
  hidden tangent storage cannot bypass owner accounting or stream lifetimes.
- Input containers are builtin tuple/list/dict trees with scalar keys. Opaque
  pytree contexts, tensor dictionary keys and cycles are explicitly rejected.
  Repeated alias containers remain valid. Tensor-valued output keys disable
  capture and run the original callable, preserving held-key values. These
  regressions failed before and pass afterward.
- Sparse COO/CSR and nested layouts were reproduced as invalid fallback stream
  recording/metadata paths. They, tensor subclasses and named tensors are now
  explicitly outside the accepted input domain. Ordinary dense metadata, shape,
  device, slot and budget fallbacks retain their stream-lifetime handling.

The definitive source-only ownership assessment for `1f0578c` found no remaining
source blocker in this set. The completed evidence assessment independently checked
all 288 records/audits, saved hashes, quiet gates, matching provenance, all six
supplements, profile launch/kernel counts and reported cost/memory examples.

## Compatibility and reporting decisions

Named tensor axes were raised as a specialization concern. Both installed Torch
2.13 and the qualified 2.14 build have removed the named-tensor API, so a genuine
named-axis arithmetic reproduction is unavailable here. The wrapper explicitly rejects
named tensors when a build exposes names, and a compatibility test explicitly
skips otherwise. This is not a claim of qualification on older named-tensor builds.

Supplemental mode/scenario rows reset Dynamo while sharing an Inductor disk cache
inside a process. Their preparation/first timings are not independent cold-process
costs. Varying-shape timings include first-shape compilation penalties; maxima and
totals remain reported. Main matched arms use independent fresh processes/caches.

## Verification

Full Python suite: 1,502 passed, 15 skipped, five documented warnings (two fork,
one forward-AD/Python-version, and sparse/nested API warnings).
Focused replay suite: 54 passed, two skips (second device hidden; named API absent).
Separate two-GPU device check: one passed. Saved-data analyzer: 30 passed.
Final qualification: 288 arms (282 successes and six actual OOMs), all six
supplemental processes and strict saved-data analysis passed. Publication verification
is recorded separately on the [Wiki artifact page](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-268).

## Design rulings and their costs

1. Use explicit pure GPU-region boundaries. Automatic arbitrary FX partitioning is
   outside this implementation; callers must identify eligible regions themselves.
2. Leave mutating MoE combine outside capture to preserve original semantics. This
   limits the number of launches that replay can remove.
3. Document the API in `docs/gpu-replay.md` and link existing Torch integration docs;
   the planned Python README did not exist. There is no behavior change.
4. Increase E2 warmup from three to five requests to satisfy parent #247. All arms
   pay the same extra setup cost; the earlier 18-arm campaign is excluded wholesale.
5. Start the single independent review during the long campaign, then reconcile
   completed evidence before delivery. Early source findings caused full restarts;
   no preliminary source assessment substitutes for final evidence review.
6. Explicitly reject named tensors when available and skip compatibility tests on
   builds without their API. Older named-tensor builds remain unqualified.
7. Reject forward AD/torch.func, opaque or cyclic inputs, nonscalar dictionary keys,
   tensor subclasses and sparse/nested/named layouts. Their hidden storage cannot
   share the dense-input lifetime/accounting contract; users must call the original
   function outside GPUReplay. Dense shape/device/budget fallback remains supported.

The E5 prose was corrected to say each caller warms its assigned subset of eight
inputs, matching the executed `24 // callers` warmup count. No measurement changed.
Earlier order/phase limitations were superseded before the final freeze by reversed
runtime pairs in round two and explicit phase markers.

## Evidence interpretation

Sym replay beats slots-off in 45/45 successful per-round comparisons but beats
plain Sym in only 4/48. Zero parent E2 targets pass. Six four-caller slots-off
OOMs remain; graph-enabled four-caller decode admits no graphs and is a fallback
result, while plain already fits. The eager decode profile retains original kernel
counts and adds 160 copy events; identical total kernel work is not claimed.
Warmup costs include validation, E5 latencies are finite-burst amortizations,
physical peaks are sampled lower bounds, and supplementary shape JIT spikes remain.
See [RESULTS.md](RESULTS.md) and [TABLES.md](TABLES.md) for complete interpretation.

The review identified one minor report wording issue: TABLES.md contains cross-round
aggregates, while the package preserves per-round fields. The final report makes
that distinction explicit. No minor finding is deferred. Publication verification
is a delivery check, not a new timing campaign or another source review.
