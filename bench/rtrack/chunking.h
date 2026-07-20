//===- chunking.h - rtrack chunk planning -----------------------*- C++ -*-===//
//
// Method A chunks on the plan's dst outer axis (the staging buffer holds
// TRANSFORMED output rows, so the chunk size is measured in output bytes).
// Method B chunks the contiguous fp32 source by plain bytes. Double
// buffering means 2 x stagingBytes of pinned memory per config.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_CHUNKING_H
#define BENCH_RTRACK_CHUNKING_H

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace bench {
namespace rtrack {

struct RowChunks {
  int64_t rowsPerChunk;
  int64_t nChunks;
  int64_t rowBytes;
  int64_t stagingBytes; // rowsPerChunk * rowBytes; > chunkBytes when one
                        // row alone exceeds the requested chunk
};

inline RowChunks planRowChunks(int64_t rows, int64_t rowBytes,
                               int64_t chunkBytes) {
  assert(rows >= 1 && rowBytes >= 1 && chunkBytes >= 1);
  RowChunks c;
  c.rowBytes = rowBytes;
  c.rowsPerChunk =
      std::min<int64_t>(rows, std::max<int64_t>(1, chunkBytes / rowBytes));
  c.nChunks = (rows + c.rowsPerChunk - 1) / c.rowsPerChunk;
  c.stagingBytes = c.rowsPerChunk * rowBytes;
  return c;
}

struct ByteChunks {
  int64_t bytesPerChunk; // last chunk may be short
  int64_t nChunks;
};

inline ByteChunks planByteChunks(int64_t totalBytes, int64_t chunkBytes) {
  assert(totalBytes >= 1 && chunkBytes >= 1);
  ByteChunks c;
  c.bytesPerChunk = std::min(totalBytes, chunkBytes);
  c.nChunks = (totalBytes + c.bytesPerChunk - 1) / c.bytesPerChunk;
  return c;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_CHUNKING_H
