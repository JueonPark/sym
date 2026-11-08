# GPU Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver bounded GPU-region replay and matched completed-request evidence for #268.
**Architecture:** Pure GPU callables consume refreshed static slots after existing transfers. Graph pools are private; cloned outputs and completion events protect lifetimes. The same wrapper and kernels qualify eager and Inductor controls.
**Tech Stack:** Python, PyTorch CUDA Graphs, existing native Sym runtime, pytest, stdlib analysis.
**Spec:** docs/superpowers/specs/2026-11-08-cuda-graph-replay-design.md

## Global Constraints

- Inference only; mutable parameters explicit; CPU/routing/transfers outside capture.
- Default slots=2, max_buckets=8, max_bytes=256<<20; bounded retention and fallback.
- Freeze all regions before concurrent serving; no lazy capture during serving.
- Clone returned tensors; do not retry replay failures; include private pool reservation.
- Unchanged E2/E5 mathematics, kernels, grouping and prefetch; three rounds and 60-second quiet gate.
- English PR and Wiki; no Releases/SVG; close issues only on PR merge.

## Review Focus

- Producer and consumer streams differ: current-stream dependencies and input allocator lifetime.
- Failed capture or clone after submitted GPU work: no double execution or premature retirement.
- Retained outputs alias via views/storage: survive replay and owner close.
- Capture settings or tensor metadata change: never reuse stale specialization.
- Concurrent frozen callers exceed slot count: correct fallback without capture or reuse races.

### Task 1: Bounded replay runtime

**Files:** Create `libreloc/python/reloc_torch/gpu_replay.py` and `libreloc/python/tests/torch_frontend/test_gpu_replay.py`; modify package exports.
**Interfaces:** Produces `GPUReplayPolicy`, `GPUReplay(function, *, policy=None, device=None)` with `prepare`, `freeze`, `invalidate`, `stats`, `close`, context manager and callable.

- [ ] Write tests for fresh parameters/addresses, held views, bucket/slot limits, low budget, failed capture, streams, retirement, inference guard and settings fallback.
- [ ] Run `bash /home/jueonpark/sym-issue268-20261108/test.sh -q libreloc/python/tests/torch_frontend/test_gpu_replay.py`; expected missing implementation failures.
- [ ] Implement validated policy, exact metadata keys, static input copies, graph capture and pool accounting, cloned outputs, events and bounded fallback. Detect input mutation in warmup and retain exclusion reason. Capture failures are recoverable only after safe cleanup; runtime failures poison owner.
- [ ] Run the same test command; expected pass. Commit runtime/tests.

### Task 2: Prepared backend integration and documented contract

**Files:** Modify `libreloc/python/reloc_torch/backend.py`, replay tests and `libreloc/python/README.md`.
**Interfaces:** Consumes Task1; produces backend-owned `prepare_gpu_region(function, *example_args, policy=None, device=None, **example_kwargs)`. It prepares slots and returns wrapper; user freezes after all example buckets. Backend stats/close include wrappers.

- [ ] Write a failing GPU test using automatically prepared CPU→GPU transformation followed by owned GPU replay; verify changed source values, held result and close. Add process mismatch/failed creation cleanup tests.
- [ ] Run replay test file; expected new failures.
- [ ] Implement integration without weakening whole-graph capture guard; document pure boundary, preparation, explicit mutable state, budgeting and fallback.
- [ ] Run replay tests and full Python suite with existing native runtime; expected pass or explicitly reported existing skips. Commit integration/docs.

### Task 3: Reproducible qualification and analysis

**Files:** Create `bench/issue268/` protocol, worker, runner, analyzer and analyzer tests; results/review/README.
**Interfaces:** Uses Task1 at unchanged original E2/E5 GPU boundaries; immutable job IDs and hashed input/runtime source inventories; stdlib saved-data reproduction.

- [ ] Define exact matrix and schema before acceptance. Write failing analyzer tests for missing/duplicate cases, mismatched kernel/settings, wrong counts, bad provenance, contamination and missing cooldown/memory observations.
- [ ] Run analyzer tests, then implement strict analyzer and isolated runner.
- [ ] Run correctness pilots, fix observed defects with regression tests. Freeze measurement source and matrix.
- [ ] Execute three-round E2/E5 campaign with equal wrapper/storage opportunity; preserve shape/budget failures. Profile launches separately. Report capture/first/submission/completed p50/p95, retained/peak memory and break-even; retain negative results.
- [ ] Run analyzer tests and reproduce reports from raw artifacts; expected complete matrix and exact tables. Commit harness/results.

### Task 4: Independent review and delivery

**Files:** Review disposition and final public artifact references.
**Interfaces:** Whole branch against b319a0ab94dbc8c8264e4fa093ffbbb796e2c046; Wiki immutable manifest and verification receipts.

- [ ] Dispatch one final whole-change reviewer per executing-plans. Fix Important/Critical once via RED→GREEN; ledger other decisions.
- [ ] Run relevant final checks; verify no SVG/native/generated artifact blobs in source PR.
- [ ] Publish checksummed Wiki artifact, anonymously download and reproduce saved-data results; update Wiki catalog.
- [ ] Push branch and create ready English PR linking #268 and #252 as appropriate; verify CI and automatic closing linkage. Do not merge or close issues.
