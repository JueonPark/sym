Evidence paths below refer to the [versioned Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264), publicly available without authentication.

# Issue 264 review disposition

One independent whole-branch review covered baseline `2617ba1` through `9eb1152`. See `review-final.md` for the original report and independent checks. No second reviewer or review loop was used.

## Important finding: binding-error compatibility

Confirmed: supplying aliased outputs through prepared `FastTransfers.submit()` raised `pyreloc.TransferError`, while the legacy policy raised `RuntimeError`. The new default could bypass callers' existing error handling. The regression test covers both policies, first construction and warm rebind, chained cause, successful retry, and unchanged slot/construction accounting.

The initial run failed the two prepared cases and passed both legacy controls (`bootstrap/review-exception-red.log`). The fix restores atomic slot rollback, then translates native `TransferError` to `RuntimeError` with the existing message prefix and cause. The four cases subsequently passed (`bootstrap/review-exception-green.log`). Direct native APIs retain their exception type.

The only production change after performance measurement is this exceptional binding path in `reloc_torch/prepared_group.py`. Original measured binaries/Python remain immutable in `runtime/candidate`; final correctness validation uses `runtime/reviewed`, with identical native binaries and the fixed Python file. Published timings describe the measured revision, not a newly measured post-review revision. The final verification receipt identifies both packages.

## Deferred minor

`libreloc/include/reloc/Dispatch.h:201` still says raw reports are immutable once the pending flag clears. For reusable groups they are stable only until the next rebind, as the rebind API already documents. Retained per-invocation reports require snapshots; the Python completion implementation and tests use independent snapshots correctly. Aligning this redundant field comment remains deferred; no runtime ownership defect was found.

## Behaviors the reviewer declined to extend

- Direct `TypedProgram.execute` / `PreparedWireWeights` optimization remains outside this group-path change; #248 remains open. Those callers receive the existing behavior rather than an implied new speedup.
- Exact host metadata byte accounting and a separate byte cap remain outside the implemented count bounds. RSS is explicitly whole-process, and variable member sets can consume different host bytes under the same handle count.
- Cross-hardware, sustained-serving capacity, and general automatic specialization remain separate research work. The fixed synthetic qualification is not evidence for those broader claims.
- The repository is private. Review verified local evidence integrity; actual release/download verification is a separate completion gate. Repository visibility is not changed, and anonymous access is not claimed.

## Verification records

Original full validation: 12 CTest checks; 1,349 Python tests passed, 13 skipped. Expanded lifecycle/failure run: 29 passed. Independent reviewer: eight analysis tests passed; all 160 saved jobs, 435 rows, physical-trace digests, and eight compact-index hashes validated. Final post-fix validation passed all 12 CTest checks and 1,357 Python tests, with 13 skips and one existing threaded-fork warning (225.65s). The first reviewed-tree run lacked build-only fixtures; its missing-consumer failure and 19 extra skips are preserved in `bootstrap/final-full-missing-test-fixtures.log`. Copying the exact original helpers restored the full coverage. Exact outcomes and source/runtime hashes are in `final-verification.json`.

The optional all-target build's existing sm_52/double-atomicAdd benchmark limitation remains disclosed. Runtime/compiler/test targets built successfully; no architecture changes were used to influence the performance comparison.

## Recorded implementation decisions

The following ledger entries preserve the implementer's decisions and their stated tradeoffs. They include historical task state, not claims that later publication had already completed.

- Ruling: proceed autonomously with concrete design/planning within the issue's supplied acceptance criteria, rather than adding permission gates for reversible implementation choices. User/developer authorization takes precedence over skill workflow menus.
- Ruling: native binder receives immutable plans plus expected source/destination dtype, shape, strides, and device ordinal once. This avoids relying on initial caller tensors to define a program's signature, while validating all actual metadata in C++ on every bind.
- Ruling: prepare reusable report skeletons/spans only for reusable groups, leaving original construction overhead unchanged for the new-build legacy control.
- Task 1: Ruling: preserve the original validator's admitted byte-shifted host storage instead of adding a new alignment restriction only on warm rebinding. A torch.frombuffer regression reproduced the mismatch; retain byte-span/capacity/overflow checks.
- Task 2: Ruling: put frontend lifecycle coverage in test_prepared_fast.py and subprocess fault coverage in prepared_fast_fault_scenario.py, keeping native binder tests separate. This keeps the two API layers independently understandable; no coverage is dropped.
- Task 2: Full default Ninja build hit the unchanged bench/rtrack/dma_engine.cu:57 double atomicAdd kernel, which needs sm_60+ while both the baseline and candidate CMake configuration use sm_52. Ruling: preserve matched runtime architecture and build all test/runtime/compiler targets explicitly; do not change unrelated benchmark source or runtime architecture for this issue. Full-build limitation remains reportable; relevant test targets built successfully.
- Task 2: Ruling: task-done validates a hashed receipt of the just-completed full-suite run rather than repeating 216 seconds of unchanged tests. The developer's instruction forbids redundant repetition after passing; the receipt checks every libreloc source/test file and exact log summaries. Cost if wrong: stale evidence; the per-file hash assertion prevents that.
- Task 3: Ruling: clean candidate builds omit historical unused modules (first observed: reloc_torch/advice.py) present in the frozen build. #263's allow-new-build still assumes every historical file exists. Use a candidate-specific configure adapter with identical numerical/thread settings, explicit absent-file provenance, and full actual-runtime hashes; keep original exact verification for the baseline. Do not copy stale runtime files. Cost if wrong: benchmark build contamination; imported module paths and runtime hashes are recorded and contract-checked.
- Task 3 costs: cold/setup and estimated break-even (small 13–37, decode 1–3, KV 1–9 requests), whole-process RSS, identical native retained scratch, explicit preparation, and bounded slot counts are reported together in GROUPS.md. Ruling: do not label RSS differences as exact metadata allocation bytes; allocator/loader state is included. Metadata is count-bounded, not subject to a separate host-byte budget.
- Task 4: Ruling: retain the full 12 CTest / 1,349 Python receipt for byte-identical runtime/source files and use the fresh 29-test run for the two deliberately expanded test files. The qualification gate verifies both and all measured runtime/source hashes; avoid an unchanged full rerun. Cost if wrong: stale test attribution; hash/file assertions and the explicit separate counts prevent that.
- Task 5: Ruling: GitHub API reports the repository private, and anonymous access to the existing release returns HTTP 404. Publish versioned assets for authorized repository readers and verify authenticated downloads; do not change repository visibility. Correct new documentation that called these public. Cost if wrong: intended readers without repository access cannot download; this access limitation is stated in the guide and user update.
- Final: Ruling: grade the native binding exception mismatch Important because changing default policy breaks existing RuntimeError handlers. Preserve atomic rollback and wrap only native TransferError with the existing prefix/cause. Cost if wrong: callers cannot recover consistently; first/warm bind and successful-retry tests cover both policies.
- Final: Ruling: preserve the measured build unchanged and distribute a separately verified reviewed runtime with only the Python binding-error fix. Report performance under the original measured revision, with no relabeling or inferred new timing. Cost if wrong: readers may conflate packages; manifests, guide, and report explicitly distinguish them.
- Final: Ruling: direct TypedProgram.execute / PreparedWireWeights submission retains legacy behavior; this change targets FastTransfers groups, FastWeights, and KV. Cost if wrong: users expecting direct-API speedup will not receive it; scope is explicit and #248 remains open.
- Final: Ruling: exact host metadata byte accounting/byte budget remains outside the implemented count bounds; RSS is whole-process. Cost if wrong: dynamic member sets may retain more host memory than a caller infers from handle counts; guide/report state no host-byte cap.
- Final: Ruling: cross-hardware, sustained-service SLO capacity, and general automatic specialization remain separate research work. Cost if wrong: fixed synthetic measurements would overstate applicability; no such claims are made.
- Final: Ruling: independent review certifies local evidence integrity, not pending publication. Verify actual release downloads with authorized repository access before declaring issue resolution. Cost if wrong: otherwise-valid evidence could remain inaccessible; final download/hash/reanalysis gate is still required.
