// CPU transpose tests: independent byte oracles and pipeline boundaries.

#include "../src/Transpose.h"
#include "reloc/Bind.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/Pipeline.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

reloc::BoundPlan transposePlan(int64_t rows, int64_t columns) {
  reloc::BoundPlan b;
  b.extents = {rows, columns};
  b.srcStrides = {1, rows};
  b.dstStrides = {columns, 1};
  b.elementSize = 4;
  b.totalBytes = rows * columns * 4;
  return b;
}

// Include distinct element patterns and float encodings that must be moved
// without conversion: signed zero, infinities, quiet/signaling NaNs,
// subnormals.
std::vector<uint8_t> sourceBytes(int64_t count) {
  const uint32_t special[] = {0,           0x80000000u, 0x7f800000u,
                              0xff800000u, 0x7fc12345u, 0x7f800001u,
                              1,           0xffffffffu};
  std::vector<uint8_t> bytes(count * 4);
  for (int64_t i = 0; i < count; ++i) {
    uint32_t bits = static_cast<uint32_t>(i) * 2654435761u;
    if (i % 3 == 0)
      bits = special[(i / 3) % 8];
    std::memcpy(bytes.data() + i * 4, &bits, 4);
  }
  return bytes;
}

// Source-order oracle, independent of the destination-order blocked kernels.
std::vector<uint8_t> reference(const std::vector<uint8_t> &src, int64_t rows,
                               int64_t columns) {
  std::vector<uint8_t> dst(src.size());
  for (int64_t i = 0; i < rows * columns; ++i) {
    const int64_t row = i % rows, column = i / rows;
    std::memcpy(dst.data() + (row * columns + column) * 4, src.data() + i * 4,
                4);
  }
  return dst;
}

TEST(Transpose, TileTailsUnalignedAndSpecialBits) {
  for (int64_t rows : {1, 7, 8, 9, 31, 32, 33, 65}) {
    for (int64_t columns : {1, 7, 8, 9, 31, 32, 33, 65}) {
      const auto b = transposePlan(rows, columns);
      const auto src = sourceBytes(rows * columns);
      const auto expected = reference(src, rows, columns);
      for (size_t offset : {size_t(0), size_t(1), size_t(3), size_t(31)}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << "x" << columns << " offset=" << offset);
        // No readable suffix: ASan catches a vector load beyond the last
        // source element. Destination sentinels also catch excess stores.
        std::vector<uint8_t> storage(src.size() + offset, 0xAB);
        std::copy(src.begin(), src.end(), storage.begin() + offset);
        const auto original = storage;
        // Exercise the portable fallback even when this host supports AVX2.
        for (bool scalar : {false, true}) {
          std::vector<uint8_t> dst(storage.size() + 32, 0xCD);
          auto guardedExpected = dst;
          std::copy(expected.begin(), expected.end(),
                    guardedExpected.begin() + offset);
          if (scalar)
            reloc::detail::transpose32Scalar(storage.data() + offset,
                                             dst.data() + offset, rows, columns,
                                             rows);
          else
            reloc::gatherChunk(b, storage.data() + offset, dst.data() + offset,
                               0, rows);
          EXPECT_EQ(dst, guardedExpected) << "scalar=" << scalar;
          EXPECT_EQ(storage, original);
        }
      }
    }
  }
}

TEST(Transpose, PartialRowsDoNotTouchNeighbors) {
  const int64_t rows = 73, columns = 67;
  const auto b = transposePlan(rows, columns);
  const auto src = sourceBytes(rows * columns);
  const auto expected = reference(src, rows, columns);
  const int64_t cuts[] = {0, 1, 7, 8, 9, 31, 32, 33, 65, 73};
  for (int64_t begin : cuts) {
    for (int64_t end : cuts) {
      if (end < begin)
        continue;
      SCOPED_TRACE(::testing::Message() << begin << ":" << end);
      std::vector<uint8_t> dst(expected.size(), 0xCD), wanted = dst;
      std::copy(expected.begin() + begin * columns * 4,
                expected.begin() + end * columns * 4,
                wanted.begin() + begin * columns * 4);
      reloc::gatherChunk(b, src.data(), dst.data(), begin, end);
      EXPECT_EQ(dst, wanted);
      std::fill(dst.begin(), dst.end(), 0xCD);
      reloc::detail::transpose32Scalar(src.data() + begin * 4,
                                       dst.data() + begin * columns * 4,
                                       end - begin, columns, rows);
      EXPECT_EQ(dst, wanted);
    }
  }
}

TEST(Transpose, RejectUnsupportedPlansBeforeAccessingBuffers) {
  const auto base = transposePlan(33, 35);
  std::vector<reloc::BoundPlan> unsupported;
  for (uint32_t width : {1u, 2u, 8u}) {
    auto b = base;
    b.elementSize = width;
    unsupported.push_back(b);
  }
  auto b = base;
  b.typed = true;
  unsupported.push_back(b);
  b = base;
  b.padRegions.push_back({0, 1, 1});
  unsupported.push_back(b);
  b = base;
  b.extents = {3, 11, 35};
  unsupported.push_back(b);
  for (size_t axis : {size_t(0), size_t(1)}) {
    b = base;
    b.srcStrides[axis] += 1;
    unsupported.push_back(b);
    b = base;
    b.dstStrides[axis] += 1;
    unsupported.push_back(b);
  }
  for (const auto &plan : unsupported)
    EXPECT_FALSE(
        reloc::detail::tryGatherTranspose32(plan, nullptr, nullptr, 0, 33));
}

TEST(Transpose, ThreadedWorkerTailsMatchOracle) {
  const int64_t rows = 97, columns = 67;
  const auto b = transposePlan(rows, columns);
  const auto src = sourceBytes(rows * columns);
  const auto expected = reference(src, rows, columns);
  for (unsigned threads : {3u, 8u}) {
    std::vector<uint8_t> dst(expected.size(), 0xCD);
    reloc::executeH2DThreaded(b, src.data(), dst.data(), threads);
    EXPECT_EQ(dst, expected);
  }
}

TEST(Transpose, PipelineRebasedChunksAndPersistentWorkersMatchOracle) {
  // A 513-row chunk exceeds 2 MiB and engages two workers under the existing
  // 1 MiB floor. Chunk and worker boundaries cut through both tile sizes;
  // the final 25-row chunk also exercises a short rebased staging window.
  const int64_t rows = 1051, columns = 1033;
  const auto b = transposePlan(rows, columns);
  const auto src = sourceBytes(rows * columns);
  const auto expected = reference(src, rows, columns);
  reloc::HostBackend backend(2);
  reloc::GatherPool pool(4);
  for (int buffers : {1, 2}) {
    std::vector<uint8_t> dst(expected.size(), 0xCD);
    reloc::executeH2DPipelined(b, src.data(), dst.data(), backend, buffers,
                               513 * columns * 4, pool);
    EXPECT_EQ(dst, expected);
  }
}

} // namespace
