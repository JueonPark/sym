//===- ChunkScheduleTest.cpp - chunk planning tests -----------------------===//

#include "reloc/ChunkSchedule.h"
#include "reloc/Bind.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

namespace {

using reloc::BoundPlan;
using reloc::Chunk;
using reloc::ChunkSchedule;
using reloc::PadRegion;

// Minimal dense row-major BoundPlan builder (mirrors ExecuteTest::makeBound).
BoundPlan makeBound(std::vector<int64_t> extents,
                    std::vector<int64_t> srcStrides,
                    std::vector<int64_t> dstStrides, uint32_t elementSize,
                    std::vector<PadRegion> pads = {}) {
  BoundPlan b;
  b.extents = std::move(extents);
  b.srcStrides = std::move(srcStrides);
  b.dstStrides = std::move(dstStrides);
  b.elementSize = elementSize;
  b.padRegions = std::move(pads);
  int64_t total = elementSize;
  std::vector<int64_t> padded = b.extents;
  for (const PadRegion &p : b.padRegions)
    padded[p.axis] += p.lo + p.hi;
  for (int64_t e : padded)
    total *= e;
  b.totalBytes = total;
  return b;
}

// The chunks must tile [0, paddedOuter) contiguously and cover totalBytes.
void expectContiguousCover(const ChunkSchedule &s, int64_t paddedOuter,
                           int64_t totalBytes) {
  ASSERT_FALSE(s.chunks.empty());
  int64_t expectBegin = 0, coveredBytes = 0;
  for (const Chunk &c : s.chunks) {
    EXPECT_EQ(c.paddedBegin, expectBegin);
    EXPECT_EQ(c.byteOffset, c.paddedBegin * s.rowBytes);
    EXPECT_EQ(c.bytes,
              static_cast<size_t>((c.paddedEnd - c.paddedBegin) * s.rowBytes));
    EXPECT_LE(c.bytes, s.maxChunkBytes);
    expectBegin = c.paddedEnd;
    coveredBytes += static_cast<int64_t>(c.bytes);
  }
  EXPECT_EQ(expectBegin, paddedOuter);
  EXPECT_EQ(coveredBytes, totalBytes);
}

TEST(ChunkSchedule, TinyOverrideForcesManyChunks) {
  // [64, 16] fp32, rowBytes = 16*4 = 64. Override 64 bytes -> 1 row/chunk.
  BoundPlan b = makeBound({64, 16}, {16, 1}, {16, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/2, /*override=*/64);
  EXPECT_FALSE(s.serialized);
  EXPECT_EQ(s.rowBytes, 64);
  EXPECT_EQ(s.chunks.size(), 64u); // one row each
  expectContiguousCover(s, /*paddedOuter=*/64, b.totalBytes);
}

TEST(ChunkSchedule, HeuristicClampsToMinForSmallTensor) {
  // Small tensor: totalBytes << the 4 MiB floor, so the heuristic clamps up
  // to a chunk bigger than the whole tensor -> a single chunk.
  BoundPlan b = makeBound({8, 128}, {128, 1}, {128, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/4, /*override=*/0);
  EXPECT_FALSE(s.serialized);
  EXPECT_EQ(s.chunks.size(), 1u);
  expectContiguousCover(s, /*paddedOuter=*/8, b.totalBytes);
}

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
  // 4 MiB floor -> 2 chunks, not 32. Pinned-copy bandwidth is already
  // saturated by 4 MiB copies; small tensors must not shred into launch
  // overhead.
  BoundPlan b = makeBound({128, 16384}, {16384, 1}, {16384, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/4, /*override=*/0);
  EXPECT_FALSE(s.serialized);
  EXPECT_EQ(s.chunks.size(), 2u);
  expectContiguousCover(s, /*paddedOuter=*/128, b.totalBytes);
}

TEST(ChunkSchedule, OuterPadCountsPadRowsAndClampsValid) {
  // Axis-0 pad lo=1 hi=1 -> paddedOuter = 6+2 = 8. One row per chunk.
  PadRegion pad{0, 1, 1};
  pad.fillBits = 0xDEADBEEF;
  BoundPlan b = makeBound({6, 4}, {4, 1}, {4, 1}, 4, {pad});
  ChunkSchedule s =
      planChunks(b, /*nBuffers=*/2, /*override=*/16); // rowBytes=16
  EXPECT_EQ(s.outerLo, 1);
  expectContiguousCover(s, /*paddedOuter=*/8, b.totalBytes);
  // First chunk is a leading pad row: no valid rows.
  EXPECT_EQ(s.chunks.front().validBegin, s.chunks.front().validEnd);
  // Second chunk (padded row 1) maps to valid row 0.
  EXPECT_EQ(s.chunks[1].validBegin, 0);
  EXPECT_EQ(s.chunks[1].validEnd, 1);
  // Last chunk (padded row 7) is a trailing pad row: no valid rows.
  EXPECT_EQ(s.chunks.back().validBegin, s.chunks.back().validEnd);
}

TEST(ChunkSchedule, OverlappingRowsFallBackToSingleChunk) {
  // dstStrides[0]=1 < inner span (3) -> rows alias -> serialized single chunk.
  BoundPlan b = makeBound({8, 3}, {3, 1}, {1, 1}, 4);
  ChunkSchedule s = planChunks(b, /*nBuffers=*/4, /*override=*/4);
  EXPECT_TRUE(s.serialized);
  ASSERT_EQ(s.chunks.size(), 1u);
  EXPECT_EQ(s.chunks[0].byteOffset, 0);
  EXPECT_EQ(s.chunks[0].bytes, static_cast<size_t>(b.totalBytes));
  EXPECT_EQ(s.chunks[0].validBegin, 0);
  EXPECT_EQ(s.chunks[0].validEnd, 8);
  EXPECT_EQ(s.maxChunkBytes, static_cast<size_t>(b.totalBytes));
}

} // namespace
