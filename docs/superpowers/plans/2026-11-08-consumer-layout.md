# Consumer Layout Elimination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline implementation. Track each checkbox.

**Goal:** Deliver guarded consumer layout elimination and matched qualification for #269/#251.
**Architecture:** A conservative FX rule bypasses private permutation copies before cat. Explicit packed wire restoration feeds transparent consumers; untouched original graphs supply exclusions/fallback.
**Tech Stack:** Python, Torch FX/Dynamo/Inductor, existing CUDA/native runtime, pytest, stdlib evidence tools.
**Spec:** docs/superpowers/specs/2026-11-08-consumer-layout-design.md

## Global Constraints

- Default off; rank2/rank3 dense CUDA permutation inputs, including valid offsets.
- Only fresh functional cat consumers; no arithmetic/precision/value-stage change.
- Unknown/symbolic metadata, observable intermediates, mutation and unsupported layouts retain materialization.
- Preserve held outputs, host source/stream lifetimes, same wire traffic and budgets.
- Three rounds, five warmups, >=30 completed samples,60 observed quiet seconds, rotated arms.
- English PR/Wiki, no Releases/SVG, issue closure only on merge.

## Review Focus

- Packed source views alias storage at nonzero offsets; no reads outside the logical view.
- Mutation through another alias between clone and cat; preserve original snapshot semantics.
- Cat output dtype/layout/ownership with multiple inputs and held outputs; no exposed intermediate changes.
- Backend close/training/transform fallback; optimization must not bypass original execution restrictions.
- Async wire copy source destruction/cross-stream handoff/partial failure; no premature release.

### Task 1: General guarded graph rule and backend integration

**Files:** `libreloc/python/reloc_torch/consumer_layout.py`, `backend.py`, `__init__.py`; `libreloc/python/tests/torch_frontend/test_consumer_layout.py`; `docs/consumer-layout.md`.
**Interfaces:** `ConsumerLayoutPolicy(min_saved_bytes=0)`; `lower_consumer_layout(gm, example_inputs, *, policy=None) -> (GraphModule, tuple[ConsumerLayoutDecision,...])`; immutable decisions expose node/source/users, eligible/selected/reason, eliminated_bytes/read_write_bytes. `RelocBackend(consumer_layout=False|policy)` reports decisions without retaining tensor references.

- [x] Write GPU/FX tests: rank2/3 permutations→copy→cat removed; fresh values/NaN payloads/signed zeros and held outputs exact; source graph untouched; threshold/default off; same dtype/contiguous cat output.
- [x] Add exclusions: visible/alias/mixed/CPU consumers; mutation before cat through source alias; noncontiguous/symbolic metadata; channel gather/pad/cast; unsupported cat out; backend close/autograd behavior. Dense positive-offset packed inputs must optimize correctly.
- [x] Run new test file and observe missing-feature failure. Implement rule and backend integration; run focused and existing backend tests, then full Python suite. Commit.

### Task 2: Explicit wire restore and ownership

**Files:** `consumer_layout.py`, `fast.py`, focused tests and API docs.
**Interfaces:** `PackedWireRestore(device)` with `restore(stored)`, `drain()`, `close()`, `stats()`; `KVCacheSwap.restore_wire(stored)` owns helper and drain. Equal helper available to Torch controls. Returns fresh dense [S,H,D] tensors, with one packed copy for adjacent pinned views.

- [x] Test changing wire inputs, offsets/shared storage, grouped/safe separate copies, wrong dtype/rank/device/layout, held outputs after reuse/close, producer→consumer stream and temporary host input lifetime; existing restore still yields original layout.
- [x] Observe failures, implement all preflight before submission and completion-retained host references; handle partial error by fencing submitted work before release. Validate current owning process/device; close idempotent.
- [x] Run focused wire/fast tests and full Python suite. Commit.

### Task 3: Qualification harness, diagnostics and complete evidence

**Files:** `bench/issue269/{contract,worker,run,analyze,test_analysis}.py`, protocol/README/report/tables.
**Interfaces:** Frozen matrix and inventories; materialized/native, optimized-materialized and lowered variants; resident and matched replay subsets. Records keep actual eligibility, allocation/copy/kernel evidence, cold/warm costs and numerical checks.

- [ ] Declare exact matrix/protocol; analyzer regressions reject missing/duplicate data, unequal settings/provenance, bad correctness, contamination, missing cooldown or physical samples. Observe failure then implement strict analysis.
- [ ] Build transparent wire decoder wrapper around unchanged original model functions; equal lowering available to both runtimes. Run pilots on canonical/held-out/mixed-fallback/low-budget shapes. Keep every excluded attempt and fix observed defects with regressions.
- [ ] Freeze source/runtime; run all fresh-process rounds serially under isolation, no concurrent test/build/profile. Run separate launch/allocation diagnostics and complete numerical/ownership checks.
- [ ] Reproduce reports from raw data, run analyzer tests, report setup/compile/first/p50/p95/memory/break-even and parent target passes or misses. Commit.

### Task 4: Independent review and public delivery

**Files:** REVIEW/report corrections, immutable Wiki package and receipts.
**Interfaces:** Whole branch against3a62084; one independent most-capable reviewer; public checksummed artifact; closing PR.

- [ ] Dispatch one independent whole-change review, address Important/Critical with regressions and whole-suite verification; ledger scope rulings/deferred minors.
- [ ] Publish Wiki evidence; anonymous download, inventory, exact stdlib reanalysis and relocated functional verification.
- [ ] Push English PR linking #269/#251; verify CI, no SVG/generated binaries in source diff, original checkout clean and automatic closing linkage. Preserve worktree, never merge/close manually.
