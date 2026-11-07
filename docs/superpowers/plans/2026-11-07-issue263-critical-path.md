# Issue 263 Critical-Path Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Resolve #263 with request-correlated CPU/CUDA evidence, controlled bypass experiments, unprofiled uncertainty, and a publicly accessible artifact.

**Architecture:** A benchmark-only adapter imports the frozen E2/E5 artifact and the unchanged runtime. Thread-local host phases and NVTX request/group labels feed Nsight SQLite analysis; separate unprofiled runs qualify counter overhead and diagnostic interventions. Runtime behavior changes are confined to explicitly named experimental contexts, restored on exit.

**Tech Stack:** Python 3.14 / Torch 2.14, the existing libreloc CUDA build, Nsight Systems 2024.2, SQLite, pytest, GitHub releases.

**Spec:** https://github.com/JueonPark/sym/issues/263; frozen input artifact `experiments-2026-11-04-v1`; experiment protocol `bench/issue263/PROTOCOL.md`.

## Global Constraints

- Baseline runtime/source: `2617ba166f98a7476a5f472d829a573317fc0cfb`; verify the published 41-file provenance before/after measurement.
- Keep E2/E5 kernels, grouping, two-slot prefetch, FP16 rounding, TF32/graphs disabled, input freshness, output ownership, and resource budgets matched.
- Cover decode/prefill, sparse/dense MoE, eager/Inductor; E5 initial/warm/shape/scale changes and 1/2/4 callers.
- Trace runs are diagnostics; completed latency and uncertainty come from unprofiled independent process rounds. Never sum overlapping host/GPU intervals.
- Native warm allocation/context delta zero is a starting observation, not a claim about Python allocations or total memory.
- Preserve failed pilots and negative results outside the source tree. Publish raw evidence and checksums; keep committed evidence compact.
- Execute in the managed worktree; do not change the recorded experiment snapshot or the baseline runtime build.

## Review Focus

- Concurrent thread scopes must not mix request/group identities or charge another request's CUDA activity.
- Nested host phases, asynchronous GPU work, and event waits must not be summed as independent causal shares.
- Diagnostic bypasses must preserve completion, output lifetime, live inputs, and parameter-revision behavior; report disabled observability explicitly.
- Profile instrumentation, compile costs, and sampler overhead must not contaminate the unprofiled speedup estimate.
- Reproduction must accept explicit artifact/build paths and reject stale inputs, mismatched provenance, missing traces, or contaminated runs.

### Task 1: Portable artifact adapter, instrumentation, and diagnostic interventions

**Files:** Create `bench/issue263/{support.py,instrument.py,interventions.py,models.py,test_instrument.py,test_interventions.py}`.

**Interfaces:** `load_artifact(root, repo)` returns the E2/E5 modules without editing the artifact; `Instrumentation(mode)` supplies thread-local `request(label)` / `phase(name)` contexts and JSON records; `Intervention(name)` restores all changed callables on exit. `models.py` exposes a CLI for original E2 matched native/grouped/whole-window paths.

- [x] Add failing tests for nested exclusive attribution, concurrent scope isolation, restoration after exceptions, and real CUDA numerical/held-output behavior under interventions.
- [x] Implement the minimal adapter and instrumentation, retaining original source/binary hashes and an explicit mode (`off`, `counters`, `nvtx`).
- [x] Probe Nsight capture/export on this runtime and run the current 21 E2 controls as the clean baseline.
- [x] Run focused pytest gates and small E2 pilots; preserve all failures. Commit the harness and tests after successful gates.

### Task 2: Correlated trace and paired-statistics analysis

**Files:** Create `bench/issue263/{trace.py,statistics.py,test_trace.py,test_statistics.py}`.

**Interfaces:** `analyze_trace(sqlite_path)` emits host-exclusive phases, GPU interval unions, CPU-to-GPU correlation, and representative dependency witnesses. `paired_summary(rows)` reports process-paired effects and bootstrap intervals, separately from request percentiles.

- [x] Add failing hand-checked fixtures for overlapping intervals, nested/threaded requests, missing GPU evidence, paired aggregation, and unmatched/incomplete rounds.
- [x] Implement request/thread/correlation joins; distinguish serialized request analysis from concurrent burst unions and flag unassigned activity.
- [x] Validate against real pilot SQLite exports and unprofiled records. Report unsupported trace fields or residuals explicitly rather than inventing dependency edges.
- [x] Run CPU analysis tests and commit.

### Task 3: E2/E5 qualification campaign

**Files:** Create `bench/issue263/{lifecycle.py,concurrency.py,run.py,analyze.py}`; finalize `PROTOCOL.md` before accepted timings.

**Interfaces:** A serialized runner accepts `--artifact`, `--build`, `--python`, `--output`, and `--nsys`; every process records configuration, provenance, isolation, correctness, and raw samples. Lifecycle/concurrency adapters use the frozen workload and retained-output checks.

- [x] Add integration gates for original controls versus interventions, fresh values/revisions, retained outputs, counter-off execution, and 1/2/4-caller isolation.
- [x] Capture diagnostic traces for the required workload/compute/phase/caller coverage; establish explicit hypotheses and select interventions before acceptance.
- [x] Run the registered unprofiled rounds and instrumentation-overhead contrasts with no concurrent GPU/build jobs. Record contamination as failed evidence, never accepted timing.
- [x] Analyze actual critical-path evidence, attributable savings with uncertainty, parity requirements, memory counters, and unexplained residuals. Commit runner/analysis and compact results.

### Task 4: Review, publish, and reconcile the issue

**Files:** Create `bench/issue263/README.md` and a compact machine-readable result index.

- [x] Map each #263 acceptance criterion to evidence and state whether the original regression reproduces under the current protocol.
- [x] Run the focused harness suite and relevant existing frontend/lifetime controls; obtain an independent whole-branch review and address important findings.
- [x] Publish a versioned release containing the raw records, Nsight traces/SQLite, source snapshot, report, and checksum manifest; verify an anonymous download.
- [x] Push the isolated branch and create a reviewable PR. Post the evidence and prioritized follow-ups on #263; close it only if all attribution acceptance criteria are met, otherwise state the remaining work precisely.

Completion: [PR #275](https://github.com/JueonPark/sym/pull/275) is open for review; [#263](https://github.com/JueonPark/sym/issues/263#issuecomment-6703223362) is closed with all four attribution criteria satisfied. The [public release](https://github.com/JueonPark/sym/releases/tag/profile-issue263-2026-11-07-v1) pins implementation commit `351c4664b6796ae7cd58fed0854e56ec2e9fbecd`. Anonymous download verified all 18,400 file digests and reproduced all 570 rows / 122 comparisons exactly. Final validation: 134 tests passed (47 harness + 87 original controls/frontend). No production optimization or merge into main is included.
