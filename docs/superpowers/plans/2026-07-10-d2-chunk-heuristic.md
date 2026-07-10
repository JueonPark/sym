# D2: Chunk-Size Heuristic Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #66 — retune `planChunks`' default chunk-size heuristic for the D1 parallel producer: target `clamp(totalBytes / (8·nBuffers), 4 MiB, 64 MiB)`, giving ≥ 8 chunks per buffer (fill+drain ≤ ~3% at 2 buffers) and a ≤ 128 MiB pinned pool at the 4 GiB reference scale (down from 512 MiB).

**Architecture:** Pure constant/formula retune inside `ChunkSchedule` plus test updates. `chunkSizeOverride` semantics, the serialized fallback, and every consumer stay untouched. Although issue #66 conceptually depends on D1 (PR #68), it shares no files with it — this branch bases on `main` as an independent PR.

**Tech Stack:** C++17, gtest (`llvm_gtest`), CMake/Ninja.

## Global Constraints

- **MLIR-free contract:** no `#include <mlir/...>` or `<llvm/...>` under `libreloc/src`, `libreloc/include`, `libreloc/cuda`, `libreloc/python` (enforced by the `reloc-runtime-no-mlir-includes` ctest).
- **C++17.**
- **`chunkSizeOverride` semantics unchanged; serialized fallback untouched** (issue #66 acceptance) — proven by the existing `TinyOverrideForcesManyChunks` and `OverlappingRowsFallBackToSingleChunk` tests continuing to pass unmodified.
- **These are defaults, not conclusions** — E8's chunk sweep remains the final arbiter; say so in the header comment.
- The issue's divisor formula `max(8·nBuffers, 2·nBuffers)` reduces to `8·nBuffers` for all `nBuffers ≥ 1`; implement it as `kChunksPerBuffer (= 8) · nBuffers` with a named constant so E8 can retune it.
- **File banners/comments:** repo style; comments state constraints, not narration.
- **Commit style:** `update(libreloc): D2 <what> (#66)` with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Environment (already provisioned by the D1 session): venv at `/home/jueonpark/sym/.venv` — run `export PATH="/home/jueonpark/sym/.venv/bin:$PATH"` in every shell; LLVM/MLIR build at `/home/jueonpark/sym/build/llvm-project/build`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `libreloc/include/reloc/ChunkSchedule.h` | Modify | New clamp constants + `kChunksPerBuffer`, updated doc comments |
| `libreloc/src/ChunkSchedule.cpp` | Modify | New target formula (one expression) |
| `libreloc/test/ChunkScheduleTest.cpp` | Modify | 3 new heuristic tests, 1 comment update |

---

### Task 0: Worktree + build baseline

Environment setup only; nothing committed except the plan document.

- [ ] **Step 1: Create the D2 worktree** (based on `main` — do NOT base on the D1 branch) via the platform worktree tool (`EnterWorktree`, name `d2-chunk-heuristic`). All later paths are relative to that worktree root.

- [ ] **Step 2: Commit the plan document inside the worktree**

```bash
mkdir -p docs/superpowers/plans
cp /home/jueonpark/sym/docs/superpowers/plans/2026-07-10-d2-chunk-heuristic.md docs/superpowers/plans/
git add docs/superpowers/plans/2026-07-10-d2-chunk-heuristic.md
git commit -m "docs: D2 chunk-heuristic retune implementation plan (#66)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Configure and build; verify green baseline**

```bash
export PATH="/home/jueonpark/sym/.venv/bin:$PATH"
cmake -G Ninja -S . -B build/sym \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=/home/jueonpark/sym/build/llvm-project/build/bin/llvm-lit \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DPython_EXECUTABLE=/home/jueonpark/sym/.venv/bin/python
ninja -C build/sym libreloc-test
./build/sym/libreloc/test/libreloc-test
```

Expected: all tests pass (this is main's baseline: 101 tests). If not, STOP and report.

---

### Task 1: Retune the heuristic (constants + formula + tests)

**Files:**
- Modify: `libreloc/include/reloc/ChunkSchedule.h` (lines 22–24 constants + line 44–48 doc comment)
- Modify: `libreloc/src/ChunkSchedule.cpp` (lines 57–62, the `target` expression)
- Test: `libreloc/test/ChunkScheduleTest.cpp`

**Interfaces:**
- Consumes: existing `planChunks(const BoundPlan &, int nBuffers, size_t chunkSizeOverride)` and `ChunkSchedule` fields (unchanged signatures).
- Produces: `constexpr size_t reloc::kMinChunkBytes = 4ull * 1024 * 1024;`, `constexpr size_t reloc::kMaxChunkBytes = 64ull * 1024 * 1024;`, `constexpr size_t reloc::kChunksPerBuffer = 8;` — no other public surface changes.

- [ ] **Step 1: Write the failing tests**

In `libreloc/test/ChunkScheduleTest.cpp`, append after `HeuristicClampsToMinForSmallTensor` (keeping that test itself — only its comment changes in Step 3):

```cpp
TEST(ChunkSchedule, HeuristicTargetsEightChunksPerBuffer) {
  // 64 MiB dst, 2 buffers: target = clamp(64Mi/(8*2) = 4 MiB, 4 MiB,
  // 64 MiB) = 4 MiB -> exactly kChunksPerBuffer * nBuffers = 16 chunks,
  // so pipeline fill+drain is ~1/16 of the run (issue #66's <= ~3% bar
  // at 2 buffers).
  BoundPlan b = makeBound({1024, 16384}, {16384, 1}, {16384, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/2, /*override=*/0);
  EXPECT_FALSE(s.serialized);
  EXPECT_EQ(s.chunks.size(), 16u);
  expectContiguousCover(s, /*paddedOuter=*/1024, b.totalBytes);
}

TEST(ChunkSchedule, HeuristicCapsPoolFootprintAtReferenceScale) {
  // Issue #66 acceptance: 4 GiB dst with defaults and 2 buffers must plan
  // >= 16 chunks and a pinned-pool footprint of at most 2 x 64 MiB =
  // 128 MiB (planChunks is pure planning -- no buffer is allocated here,
  // so a 4 GiB plan is fine in a unit test).
  BoundPlan b = makeBound({65536, 16384}, {16384, 1}, {16384, 1}, 4);
  ASSERT_EQ(b.totalBytes, 4ll << 30);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/2, /*override=*/0);
  EXPECT_FALSE(s.serialized);
  EXPECT_GE(s.chunks.size(), 16u);
  EXPECT_LE(s.maxChunkBytes * 2, 128ull << 20);
  expectContiguousCover(s, /*paddedOuter=*/65536, b.totalBytes);
}

TEST(ChunkSchedule, HeuristicFloorStopsShredding) {
  // 8 MiB dst, 4 buffers: raw target 8Mi/(8*4) = 256 KiB clamps UP to the
  // 4 MiB floor -> 2 chunks, not 32. Pinned-copy bandwidth saturates well
  // above 4 MiB; small tensors must not shred into launch overhead.
  BoundPlan b = makeBound({128, 16384}, {16384, 1}, {16384, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/4, /*override=*/0);
  EXPECT_FALSE(s.serialized);
  EXPECT_EQ(s.chunks.size(), 2u);
  expectContiguousCover(s, /*paddedOuter=*/128, b.totalBytes);
}
```

Derivations pinning the expected values (all rowBytes = 16384·4 = 64 KiB):
- 64 MiB test: rowsPerChunk = 4 MiB / 64 KiB = 64 → 1024/64 = 16 chunks.
- 4 GiB test: target = clamp(4 GiB/16 = 256 MiB → ceiling) = 64 MiB → rowsPerChunk = 1024 → 65536/1024 = 64 chunks; maxChunkBytes = 64 MiB → footprint 128 MiB.
- 8 MiB test: rowsPerChunk = 64 → 128/64 = 2 chunks.

Against the OLD constants these fail as: 64 MiB test → target 16 MiB → 4 chunks; 4 GiB test → target 256 MiB → maxChunkBytes·2 = 512 MiB > 128 MiB; 8 MiB test → floor 8 MiB → 1 chunk.

- [ ] **Step 2: Run to verify they fail**

Run: `export PATH="/home/jueonpark/sym/.venv/bin:$PATH" && ninja -C build/sym libreloc-test && ./build/sym/libreloc/test/libreloc-test --gtest_filter='ChunkSchedule.*'`
Expected: the 3 new tests FAIL with exactly the wrong values listed above (4 ≠ 16; 512 MiB > 128 MiB; 1 ≠ 2); the 4 pre-existing tests still pass.

- [ ] **Step 3: Implement the retune**

In `libreloc/include/reloc/ChunkSchedule.h`, replace lines 22–24:

```cpp
/// Default chunk-byte clamp (design decision 4). Named so P3 can retune them.
constexpr size_t kMinChunkBytes = 8ull * 1024 * 1024;   // 8 MB
constexpr size_t kMaxChunkBytes = 256ull * 1024 * 1024; // 256 MB
```

with:

```cpp
/// Default chunk-byte clamp, retuned by D2 (issue #66) for the D1 parallel
/// producer: with per-chunk gather ~T x faster, finer chunks are affordable
/// and pipeline fill/drain dominates the ends. Floor 4 MiB (pinned-copy
/// bandwidth saturates well below this; the extra ~5-10 us launches are
/// noise), ceiling 64 MiB (a 2-buffer pool tops out at 128 MiB, down from
/// 512 MiB). E8's chunk sweep remains the final arbiter -- these are
/// defaults, not conclusions.
constexpr size_t kMinChunkBytes = 4ull * 1024 * 1024;  // 4 MiB
constexpr size_t kMaxChunkBytes = 64ull * 1024 * 1024; // 64 MiB

/// Unclamped target chunk count per staging buffer: >= 8 chunks/buffer
/// keeps pipeline fill+drain <= ~3% of the run at 2 buffers.
constexpr size_t kChunksPerBuffer = 8;
```

and replace the `planChunks` doc-comment line (currently line 46):

```cpp
/// clamp(totalBytes / (2*nBuffers), kMinChunkBytes, kMaxChunkBytes). Falls
```

with:

```cpp
/// clamp(totalBytes / (kChunksPerBuffer*nBuffers), kMinChunkBytes,
/// kMaxChunkBytes). Falls
```

In `libreloc/src/ChunkSchedule.cpp`, replace the `target` expression (lines 57–62):

```cpp
  size_t target =
      chunkSizeOverride != 0
          ? chunkSizeOverride
          : std::clamp<size_t>(static_cast<size_t>(bound.totalBytes) /
                                   (2u * static_cast<size_t>(nBuffers)),
                               kMinChunkBytes, kMaxChunkBytes);
```

with:

```cpp
  size_t target =
      chunkSizeOverride != 0
          ? chunkSizeOverride
          : std::clamp<size_t>(static_cast<size_t>(bound.totalBytes) /
                                   (kChunksPerBuffer *
                                    static_cast<size_t>(nBuffers)),
                               kMinChunkBytes, kMaxChunkBytes);
```

In `libreloc/test/ChunkScheduleTest.cpp`, update `HeuristicClampsToMinForSmallTensor`'s comment (currently "Small tensor: totalBytes << 8 MB, so the heuristic clamps up to a chunk / bigger than the whole tensor -> a single chunk."):

```cpp
  // Small tensor: totalBytes << the 4 MiB floor, so the heuristic clamps up
  // to a chunk bigger than the whole tensor -> a single chunk.
```

- [ ] **Step 4: Run to verify everything passes**

Run: `ninja -C build/sym libreloc-test && ./build/sym/libreloc/test/libreloc-test --gtest_filter='ChunkSchedule.*'`
Expected: 7/7 ChunkSchedule tests PASS.
Then the full binary once: `./build/sym/libreloc/test/libreloc-test`
Expected: all tests pass (Pipeline matrices use tiny plans whose totals sit below both old and new floors, plus explicit overrides — behavior there is identical).

- [ ] **Step 5: Commit**

```bash
git add libreloc/include/reloc/ChunkSchedule.h libreloc/src/ChunkSchedule.cpp \
        libreloc/test/ChunkScheduleTest.cpp
git commit -m "update(libreloc): D2 retune chunk heuristic for the parallel producer (#66)

target = clamp(total/(8*nBuffers), 4 MiB, 64 MiB): >= 8 chunks/buffer
(fill+drain <= ~3% at 2 buffers) and a <= 128 MiB pinned pool at 4 GiB,
down from 512 MiB. Override semantics and the serialized fallback are
untouched; E8's chunk sweep remains the final arbiter.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Full sweep + PR

- [ ] **Step 1: Full local sweep**

```bash
export PATH="/home/jueonpark/sym/.venv/bin:$PATH"
ninja -C build/sym libreloc-test pyreloc_ext bench-bind-cost bench-protocol-test
ctest --test-dir build/sym -R 'libreloc|reloc-runtime|bench' --output-on-failure
PYTHONPATH=$PWD/build/sym/python python3 -m pytest libreloc/python/tests -q
```

Expected: all green (pytest exercises `relocate` with the `chunked_pipeline` strategy over the corpus — byte-exactness is chunking-independent, so the retune must not move any result).

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin HEAD:refs/heads/d2-chunk-heuristic
gh pr create --repo JueonPark/sym --head d2-chunk-heuristic \
  --title "update(libreloc): D2 chunk-size heuristic retune for the parallel producer (#66)" \
  --body "$(cat <<'EOF'
Implements #66.

- planChunks default target: clamp(totalBytes / (8*nBuffers), 4 MiB, 64 MiB)
  (was clamp(totalBytes / (2*nBuffers), 8 MiB, 256 MiB)); divisor named
  kChunksPerBuffer so E8 can retune it. The issue's max(8*nBuffers,
  2*nBuffers) reduces to 8*nBuffers for all nBuffers >= 1.
- At the 4 GiB reference config with 2 buffers: 64 chunks of 64 MiB
  (>= 16 required) and a 128 MiB pinned pool, down from 16 x 256 MiB / 512 MiB.
- chunkSizeOverride semantics and the serialized fallback are untouched
  (pre-existing override/fallback tests pass unmodified); E8's chunk sweep
  remains the final arbiter -- these are defaults, not conclusions.
- New tests pin the three heuristic regimes: divisor-governed (16 chunks at
  64 MiB), ceiling + pool-footprint at 4 GiB, floor anti-shredding at 8 MiB.

Conceptually depends on #65 (PR #68) -- finer chunks pay off with the
parallel gather -- but shares no code with it; based on main.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance-criteria traceability (issue #66)

| Criterion | Where |
|---|---|
| Reference config defaults ≥ 16 chunks, ≤ 128 MiB pool | Task 1 `HeuristicCapsPoolFootprintAtReferenceScale` |
| Override honored | Existing `TinyOverrideForcesManyChunks` (unmodified) |
| Serialized fallback untouched | Existing `OverlappingRowsFallBackToSingleChunk` (unmodified); no change to that code path |
| New heuristic formula + floors/ceilings | Task 1 constants + `HeuristicTargetsEightChunksPerBuffer` / `HeuristicFloorStopsShredding` |
| ChunkScheduleTest expectations updated | Task 1 Steps 1 & 3 |
