//===- ChunkSchedule.h - outer-axis chunk planning --------------*- C++ -*-===//
//
// Slice a BoundPlan's outermost coalesced axis into byte-bounded chunks for
// the Strategy-4 pipeline. Each chunk is a contiguous run of physical (padded)
// outer rows; the union of chunks reproduces the whole physical dst, so the
// pipeline is byte-exact vs executeH2D. Size heuristic is design decision 4
// (fixed clamp + override; P3 replaces via the same override).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_CHUNKSCHEDULE_H
#define RELOC_CHUNKSCHEDULE_H

#include "reloc/Bind.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reloc {

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

/// One pipeline chunk: a contiguous run of physical outer (axis-0) rows.
struct Chunk {
  int64_t paddedBegin; // physical outer-row range [begin,end), incl. pad rows
  int64_t paddedEnd;
  int64_t validBegin; // valid (unpadded) outer indices to gather/scatter
  int64_t validEnd;
  int64_t byteOffset; // device byte offset of this chunk's dst window
  size_t bytes;       // window size in bytes
};

struct ChunkSchedule {
  std::vector<Chunk> chunks;
  size_t maxChunkBytes; // pool buffer size (>= every chunk's bytes)
  int64_t outerLo;      // axis-0 leading pad (gather/scatter rebase input)
  int64_t rowBytes;     // dstStrides[0] * elementSize
  bool serialized;      // true => single whole-tensor chunk (fallback)
};

/// Chunk `bound` along the outermost coalesced axis. Target chunk bytes =
/// `chunkSizeOverride` when nonzero, else
/// clamp(totalBytes / (kChunksPerBuffer*nBuffers), kMinChunkBytes,
/// kMaxChunkBytes). Falls
/// back to a single whole-tensor chunk when outer rows are not provably
/// disjoint in dst.
ChunkSchedule planChunks(const BoundPlan &bound, int nBuffers,
                         size_t chunkSizeOverride = 0);

} // namespace reloc

#endif // RELOC_CHUNKSCHEDULE_H
