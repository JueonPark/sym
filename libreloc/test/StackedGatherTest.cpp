//===- StackedGatherTest.cpp - stacked-source gather equivalence ----------===//
//
// torch.stack support (spec §5): a stacked gather reads N separate inputs as
// one logical row-major source [N, *S]. It must write exactly the bytes the
// single-source gather writes over the inputs' concatenation, for every plan
// shape and every outer-row subrange (pipeline chunks and gather workers).
//
//===----------------------------------------------------------------------===//

#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using reloc::BoundPlan;
using reloc::PadRegion;
using reloc::StackedSource;

// N inputs in separate allocations (no stride can reach one from another)
// plus their concatenation, which is the single-source oracle's input.
struct Inputs {
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<const uint8_t *> bases;
  std::vector<uint8_t> concatenated;

  Inputs(int64_t count, int64_t segmentElements, uint32_t elementSize) {
    for (int64_t i = 0; i < count; ++i) {
      std::vector<uint8_t> buffer(static_cast<size_t>(segmentElements) *
                                  elementSize);
      for (size_t k = 0; k < buffer.size(); ++k)
        buffer[k] = static_cast<uint8_t>((i * 977 + k * 131 + 7) & 0xff);
      concatenated.insert(concatenated.end(), buffer.begin(), buffer.end());
      buffers.push_back(std::move(buffer));
    }
    for (const auto &buffer : buffers)
      bases.push_back(buffer.data());
  }

  StackedSource source(int64_t segmentElements) const {
    return StackedSource{bases.data(), static_cast<int64_t>(bases.size()),
                         segmentElements};
  }
};

// Gather outer rows [begin, end) both ways; untouched cells keep 0xCD.
void expectSameAsConcatenated(const BoundPlan &b, int64_t count,
                              int64_t segmentElements, int64_t begin,
                              int64_t end) {
  Inputs in(count, segmentElements, b.elementSize);
  std::vector<uint8_t> expected(static_cast<size_t>(b.totalBytes), 0xCD);
  std::vector<uint8_t> actual(expected);
  reloc::gatherChunk(b, in.concatenated.data(), expected.data(), begin, end);
  reloc::gatherChunk(b, in.source(segmentElements), actual.data(), begin, end);
  EXPECT_EQ(actual, expected) << "rows [" << begin << ", " << end << ")";
}

void expectAllRanges(const BoundPlan &b, int64_t count,
                     int64_t segmentElements) {
  const int64_t outer = b.extents[0];
  expectSameAsConcatenated(b, count, segmentElements, 0, outer);
  for (int64_t begin = 0; begin < outer; ++begin)
    expectSameAsConcatenated(b, count, segmentElements, begin,
                             std::min(outer, begin + 2));
}

TEST(StackedGather, DimZeroRunsSplitAtInputBoundaries) {
  // stack(3 x [5], 0) is the identity over the logical [3, 5]; the compiler
  // merges it into one contiguous axis of 15, so runs cross inputs.
  for (uint32_t width : {1u, 2u, 4u})
    expectAllRanges(transfer_test::layout({15}, {1}, {1}, width), 3, 5);
}

TEST(StackedGather, MiddleDimCopiesOneInputRowAtATime) {
  // stack(4 x [3, 5], 1): logical [4, 3, 5] -> [3, 4, 5].
  for (uint32_t width : {1u, 2u, 4u})
    expectAllRanges(
        transfer_test::layout({3, 4, 5}, {5, 15, 1}, {20, 5, 1}, width), 4, 15);
}

TEST(StackedGather, LastDimReadsOneElementPerInput) {
  // stack(3 x [2, 4], 2): logical [3, 8] -> [8, 3] after merging [2, 4].
  for (uint32_t width : {1u, 2u})
    expectAllRanges(transfer_test::layout({8, 3}, {1, 8}, {3, 1}, width), 3, 8);
}

TEST(StackedGather, ReshapeAcrossInputsSplitsRows) {
  // stack(3 x [4], 0).reshape(2, 6): rows of 6 straddle inputs of 4.
  expectAllRanges(transfer_test::layout({2, 6}, {6, 1}, {6, 1}, 4), 3, 4);
  // ...and its transpose, whose strided inner runs cross inputs too.
  expectAllRanges(transfer_test::layout({6, 2}, {1, 6}, {2, 1}, 4), 3, 4);
}

TEST(StackedGather, PaddedPlansWriteOnlyValidCells) {
  // stack(3 x [4], 0) with one leading pad row and two trailing pad columns.
  BoundPlan b =
      transfer_test::layout({3, 4}, {4, 1}, {6, 1}, 2,
                            {PadRegion{0, 1, 0, 0}, PadRegion{1, 0, 2, 0}});
  expectAllRanges(b, 3, 4);
}

TEST(StackedGather, SingleInputMatchesPlainGather) {
  expectAllRanges(transfer_test::layout({3, 4}, {1, 3}, {4, 1}, 4), 1, 12);
}

TEST(StackedGather, SixteenInputsWithOddExtents) {
  // stack(16 x [7, 3], 1): logical [16, 7, 3] -> [7, 16, 3].
  expectAllRanges(transfer_test::layout({7, 16, 3}, {3, 21, 1}, {48, 3, 1}, 4),
                  16, 21);
}

} // namespace
