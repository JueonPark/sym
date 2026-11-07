# Issue 264 Prepared Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve #264 with reusable native group binding, native completion accounting, lifetime-safe integration, and matched E2/E5 requalification.

**Architecture:** Extend existing GroupRequest with explicitly prepared reusable state and atomic compact rebinding. A libtorch binder validates actual metadata, while bounded Python handles manage request-slot lifetimes and native report snapshots keep old completions independent of reused slots. Preserve a legacy path and qualify the default with completed timings.

**Tech Stack:** C++17 core, C++20 libtorch bridge, Python 3.14/Torch 2.14/CUDA 12.6, CMake/Ninja, pytest/GTest, frozen E2/E5 workload artifact.

**Spec:** `docs/superpowers/specs/2026-11-07-issue264-prepared-groups-design.md`; GitHub #264 and #248.

## Global Constraints

- Branch from baseline `2617ba166f98a7476a5f472d829a573317fc0cfb`; build in the isolated worktree. Baseline build and published artifacts are read-only.
- Reuse existing kernels, Resources/Completion, shape programs, and immutable weight snapshots; keep precision, grouping, prefetch and budgets matched.
- Current mutable values must always be read. Rebinding validates actual metadata/capacity/aliases. No pending or failed slot may be reused.
- Fresh outputs and stream recording protect consumer lifetime independently of transfer completion; late detailed reports must be immutable.
- Bound implicit cache to 64 handles and each handle to the owner's context count; no unbounded caching of dynamic one-shot programs.
- Keep report schema, aggregate accounting, errors, close/fork behavior, and fallback paths. Choose the default from qualification.
- Publish raw evidence with checksums; do not commit SVGs. User authorization covers an isolated PR/release, not merging main.

## Review Focus

- GPU completion may occur before Python retirement; a raw native handle can still exist. Reuse must not change an old completion's report or expose quarantined buffers (Tasks 1–2).
- Metadata can change while addresses remain stable, and output dtype aliases can have equal byte widths. Check both actual tensor metadata and capacity/alias bounds (Tasks 1–2).
- Owner close/eviction can race with submission and snapshot replacement. Keep lock ordering explicit and preserve strong in-flight owners (Task 2).
- Legacy controls must not pay new reusable-skeleton/report overhead. Prepare reusable metadata only for reusable groups; measure new-build legacy as well as the frozen baseline (Tasks 1, 3–4).
- Warm gains can hide cold costs or cache/parameter retention. Measure preparation, break-even, retained memory, revisions, and caller-dependent p95 (Tasks 3–4).

### Task 1: Native reusable bindings, Torch binder, and report snapshots

**Files:** `libreloc/include/reloc/Dispatch.h`, `libreloc/src/Dispatch.cpp`, `libreloc/src/DispatchResources.cpp`, `libreloc/python/{PyDispatch.cpp,PyTorchViews.cpp,pyreloc/__init__.py}`, `libreloc/python/tests/torch_frontend/test_prepared_groups.py`, `libreloc/test/DispatchTest.cpp`.

**Interfaces:** `prepareReusableDispatchGroup(entries)` seeds invariant views/spans/report skeletons; `rebindDispatchGroup(group, bindings)` takes compact allocation base/capacity/offset pairs, validates atomically, and resets invocation state. Native `PreparedTensorGroup(members)` exposes `prepare(sources, outputs)` and `rebind(group, sources, outputs)` with identity and metadata guards. `GroupCounters.capture(group)` returns `GroupReportSnapshot`; `snapshot.report` lazily builds the existing schema; `counters.stats()` returns compact aggregates and materialization counts.

- [x] Write low-level tests for binder presence, valid source/offset replacement, bad shape/stride/dtype/alias/capacity rejection with unchanged group, cross-binder rejection, reusable-state bounds, and pending/failed-slot guards. Confirm missing-API/behavior failures on baseline.
- [x] Implement compact core binding validation and explicit reusable initialization without adding skeleton-copy work to legacy group construction. Poison group state on execution/completion failure.
- [x] Implement the libtorch binder and immutable native report snapshots/counters. Validate same-width output dtype and plain/inference tensor requirements.
- [x] Build isolated `reloc_runtime`, `pyreloc_ext`, `pyreloc_torch_ext`, compiler/test/fault targets. Run new low-level tests, core Dispatch/Transfer tests, and existing group tests; commit after passing.

### Task 2: PreparedGroup lifecycle and fast-path integration

**Files:** new `libreloc/python/reloc_torch/prepared_group.py`; modify `libreloc/python/reloc_torch/fast.py`; extend `test_prepared_groups.py` and existing fast/snapshot tests as needed.

**Interfaces:** `FastTransfers.prepare_group(programs) -> PreparedGroup`; `PreparedGroup.submit(sources, outputs=None, ...) -> FastCompletion`; `PreparedGroup.close()`. `FastTransfers(..., prepared_groups=True, report_mode='native', max_prepared_groups=64)` retains legacy/eager switches. `FastCompletion.report` preserves the existing schema lazily; `stats()['prepared_groups']` reports constructions/rebinds/slots/cache counts and `stats()['report_materializations']` reports detailed conversion count.

- [x] Add failing tests for same-address mutations, fresh values/outputs, held completions/reports across reuses/close, shape/stride/offset/output errors, source/parameter revisions including in flight, concurrent callers, eviction, close/failure/fork, and no-bridge/one-shot fallback.
- [x] Implement bounded slots, owner locking, idempotent retirement, independent report snapshots, and cache eviction. Snapshot replacement invalidates affected implicit cache entries; explicit handles retain their chosen revision.
- [x] Connect FastWeights/KV calls through the same owner path and preserve eager-report/legacy controls. Keep structural, shape, and snapshot caches distinct.
- [x] Run new tests plus fast, wire_weights, fast_ops, group, async transfers, resources/context/fault suites. Check no warmed native resource growth and zero retained resources after close; commit.

### Task 3: Focused costs and qualification harness

**Files:** create `bench/issue264/{README.md,PROTOCOL.md,support.py,groups.py,models.py,e5.py,run.py,analyze.py,test_analysis.py}` as needed; reuse published #263/frozen workload helpers rather than copy model mathematics.

**Interfaces:** explicit artifact/harness/baseline-build/candidate-build/python/output paths; immutable campaign contract and per-process hashes/audits; separate unprofiled samples and diagnostic counts. Compare legacy/eager, legacy/native-accounting, prepared/eager, and prepared/native-accounting in grouped tests.

- [ ] Test paired aggregation/missing rounds and provenance rejection before implementing analysis/runner. Verify exact kernels/grouping/precision, numeric oracles, and output lifetime in pilots.
- [ ] Measure preparation, cold and warm completed latency, CPU submission, actual native group/descriptor/report counts, retained memory, and positive-savings break-even with process uncertainty.
- [ ] Select the model candidate/default policy from independent focused results. Preserve unfavorable outcomes and any design adjustment in the ledger; commit finalized protocol before acceptance runs.

### Task 4: Matched E2/E5 campaign and report

**Files:** finalize `bench/issue264/run.py`, `analyze.py`, `RESULTS.md`, compact result index; raw evidence outside source control.

- [ ] Register and run E2 decode/prefill/sparse/dense with eager/Inductor, five rounds, 60 samples/three warmups; include original-build Sym, candidate, and candidate-build legacy control with matched Torch.
- [ ] Run representative E5 decode/sparse lifecycle including revisions and 1/2/4-caller bursts, three rounds, original resource/budget/freshness checks. Serialize GPU/build work and audit contamination.
- [ ] Report preparation/break-even/retention together with matched p50/p95 and independent-process uncertainty. Identify parity, gains, regressions, and default/fallback policy; retain raw inputs/settings/build hashes.
- [ ] Run relevant full core/frontend/fault gates and commit the compact report. No generality or sustained-serving claims beyond the measured protocol.

### Task 5: Independent review, public artifacts, and issue resolution

**Files:** final docs/tests/report fixes only; release bundle/guide/manifests outside source tree.

- [ ] Obtain one independent whole-branch review; address Important/Critical findings with regression tests and final gates. Record deferred limits explicitly.
- [ ] Publish a versioned evidence release with source/build provenance and hashes; verify anonymous download and CPU-only reanalysis.
- [ ] Push the isolated branch and create a PR, cross-link #248/#261/#272/#274, and map all #264 acceptance items to evidence. Close #264 only when the criteria are satisfied; preserve the worktree for review.
