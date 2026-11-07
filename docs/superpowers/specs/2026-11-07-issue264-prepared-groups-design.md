# #264: reusable native groups and completion accounting

Resolve [#264](https://github.com/JueonPark/sym/issues/264), following the measured report cost in #263 and sharing the persistent-group mechanism requested by #248. The implementation extends existing dispatch groups/resources; it does not add an independent execution engine. Baseline is `2617ba166f98a7476a5f472d829a573317fc0cfb`. #263's PR remains open, so this branch starts from main and uses its published evidence/harness as a separate benchmark input.

## Constraints and approach

The issue and prior session authorize implementation, experiments, an isolated PR, and public evidence publication. The choices below are implementation decisions, not additional user requirements. No merge into main is included.

Three approaches were considered:

1. Only defer detailed Python reports. This addresses a measured cost but leaves invariant group construction in the warm path and does not satisfy the full issue.
2. Share immutable plans but reconstruct every native request. This simplifies lifetimes, but retains deep program/parameter copies on every submission.
3. **Use bounded reusable native request slots and independent report snapshots.** Validate invariant metadata at preparation, validate live tensor metadata/storage/aliases on every binding, and recycle a slot only after successful completion. This also supports native accounting with lazy detailed reports. Selected because it removes the identified repeated work while preserving existing kernels/resources/ownership.

The structural handle owns program identities and immutable parameter revisions. Shape specializations remain in the existing program/KV caches; immutable compact weight snapshots remain in PreparedWireWeights. Mutable input contents are never cached by address. Every invocation reads current values and allocates fresh outputs unless the caller supplies valid outputs.

## Native group rebinding

Extend the Torch-free dispatch API with compact buffer bindings (base, capacity, offset) for an already validated GroupRequest. Retain its item/program/selected-row structure, initial report skeletons, view shape/stride/type/device metadata, and validated source/destination spans. Rebinding checks counts, pending/executing/failed state, capacity, byte offsets, integer overflow, and cross-member aliases before modifying any request. It changes only pointers/storage metadata and resets per-invocation reports and consumed flags. Validation failure is atomic and allows a later valid binding; execution/completion failure poisons the slot permanently, including unknown completion.

The optional libtorch bridge supplies a prepared binder owning immutable member plans, directions, and expected source/destination dtypes. Initial slot construction calls the existing complete group validator. Warm binding checks each actual tensor's shape, stride, dtype, device/ordinal, inference/plain-tensor status, and storage, then passes compact bindings into the core. Shape/stride changes are rejected by an explicit handle; callers prepare the appropriate specialization. Storage-offset changes are allowed only when the same view fits its current storage. Both inputs and user-supplied outputs are checked. The Torch-free fallback retains the existing one-shot submission path.

Rebinding never mutates a pending request. GroupRequest owns its report skeleton; no pointers to caller tensor values or outputs are cached in the binder. Native execution continues through the existing Resources/Completion implementation and its bounded scratch, contexts, admission, event ordering, and failure quarantine.

## Python handle and cache

`FastTransfers.prepare_group(programs)` returns `PreparedGroup`; `PreparedGroup.submit(sources, outputs=None, ...)` returns FastCompletion. The handle owns its programs, native binder, and at most `contexts` native slots. On saturation it completes the oldest slot before reuse. Slots retire at transfer completion; consumer output lifetime remains separate and is protected by fresh outputs and `record_stream`. Holding a completion/report/output after slot reuse must preserve its original contents.

The existing `FastTransfers.submit(items, ...)` can use a bounded LRU of prepared handles keyed by program identity, so FastWeights/KV callers benefit without changing grouping or prefetch depth. Keep a constructor switch for the original path, and a bounded cache (default 64 handles). One-shot indexed/pooled programs and missing native bridge use the existing path. Explicit handles are registered weakly with the owner so owner close drains them too. Cache eviction closes the handle and releases its metadata only after active submissions complete; it never invalidates returned outputs. Snapshot replacement invalidates affected implicit cached handles; submitted work retains its old revisions until completion. All native metadata retention is bounded by cache/slot/member counts and reported separately from tensor/resource budgets.

An owner RLock serializes structural cache/binding/submit/close transitions; independent native contexts still govern actual transfer overlap. Completion finalization also uses that lock and is idempotent. Avoid holding unrelated snapshot locks while waiting for another thread. Process identity checks reject forked use before touching inherited locks/CUDA state. PreparedGroup.close is idempotent; submission after close fails. Failure does not return an unsafe slot to the free pool.

## Native accounting and reports

A per-owner native accumulator consumes successful completed group reports, updating groups, logical transfers, payload bytes, and implementation counts without per-item Python dictionaries. It returns an immutable native report snapshot for the FastCompletion. `FastCompletion.report` materializes the existing detailed Python schema on demand from that snapshot; late/repeated report access remains valid after subsequent slot reuse and owner close. Single-transfer accounting remains compatible. Owner.stats materializes only compact aggregate counters and includes prepared-cache/slot/rebind/construction/report-materialization counts.

Keep an eager-report control switch for qualification. Default policy is chosen from matched validation; do not silently default to a slower mechanism on conditions the evaluation disproves. Required errors, resource metrics, and strong source/output owners are retained with both policies. A failed operation is not counted as a successfully completed group.

## Correctness and measurement

Required tests cover live in-place value changes, distinct storage addresses, shape/stride/dtype/device/offset changes, invalid output views and aliases, atomic validation failure, parameter/snapshot revision replacement (including in flight), concurrent submissions, held outputs and late reports after many reuses/close, close versus submit, bounded slots/cache eviction, fork guards, and existing failure/quarantine behavior. Exercise legacy and prepared paths and the no-native-bridge fallback. Add core CPU validation coverage and real CUDA integration tests; reuse existing test suites.

Measurement separates preparation, CPU submission, completed group latency, construction/rebind/report counts, and retained memory. Report amortized total and break-even reuse only when measured warm savings are positive. Counters/diagnostics are separate from unprofiled timing. Compare at least legacy/eager-report, legacy/native-accounting, prepared/eager-report, and prepared/native-accounting in focused group experiments before selecting model candidates.

Requalify all four E2 FP16 workloads with eager/Inductor matched controls (five process rounds, 60 warm samples and three warmups), and representative decode/sparse E5 lifecycle and 1/2/4-caller bursts (three rounds). Match the frozen kernels, grouping, prefetch depth, precision, budgets, CPU/thread settings, and source freshness. Compare old runtime and new runtime in rotated separate processes; include a new-build legacy control to distinguish rebuild effects. Report preparation/cold/revision costs, p50/p95, uncertainty over independent process pairs, warm resource deltas, and unfavorable results. Preserve exact inputs/builds/settings, contamination audits, and raw results in a public release; keep repository results compact and do not commit SVG files.

No universal performance claim is promised. If a candidate fails to improve matched completed latency, retain the qualified fallback and report the result. Attribution, correctness, cost accounting, and honest requalification are the issue's completion criteria, not a guaranteed target speedup.
