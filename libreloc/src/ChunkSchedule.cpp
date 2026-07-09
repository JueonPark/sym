//===- ChunkSchedule.cpp - outer-axis chunk planning ----------------------===//

#include "reloc/ChunkSchedule.h"

#include <algorithm>

namespace reloc {

ChunkSchedule planChunks(const BoundPlan &bound, int nBuffers,
                         size_t chunkSizeOverride) {
  if (nBuffers < 1)
    nBuffers = 1;
  ChunkSchedule s;
  const int64_t es = static_cast<int64_t>(bound.elementSize);
  const size_t r = bound.extents.size();

  int64_t outerLo = 0, outerHi = 0;
  std::vector<int64_t> paddedExt = bound.extents;
  for (const PadRegion &p : bound.padRegions) {
    paddedExt[p.axis] += p.lo + p.hi;
    if (p.axis == 0) {
      outerLo = p.lo;
      outerHi = p.hi;
    }
  }
  s.outerLo = outerLo;
  s.rowBytes = bound.dstStrides[0] * es;

  const int64_t validOuter = bound.extents[0];
  const int64_t paddedOuter = validOuter + outerLo + outerHi;

  // Disjointness: distinct outer rows write non-overlapping dst byte ranges
  // iff dstStrides[0] spans the whole physical inner block. Same conservative
  // test as executeH2DThreaded, but over PADDED inner extents so it agrees
  // with the byte windows we cut here.
  int64_t innerSpan = 0;
  for (size_t k = 1; k < r; ++k)
    innerSpan += (paddedExt[k] - 1) * bound.dstStrides[k];
  const bool disjoint = bound.dstStrides[0] >= innerSpan + 1;

  if (!disjoint || paddedOuter <= 0) {
    // Fallback: one whole-tensor chunk == executeH2D.
    s.serialized = true;
    s.maxChunkBytes = static_cast<size_t>(bound.totalBytes);
    Chunk c;
    c.paddedBegin = 0;
    c.paddedEnd = paddedOuter;
    c.validBegin = 0;
    c.validEnd = validOuter;
    c.byteOffset = 0;
    c.bytes = static_cast<size_t>(bound.totalBytes);
    s.chunks.push_back(c);
    return s;
  }

  s.serialized = false;
  size_t target =
      chunkSizeOverride != 0
          ? chunkSizeOverride
          : std::clamp<size_t>(static_cast<size_t>(bound.totalBytes) /
                                   (2u * static_cast<size_t>(nBuffers)),
                               kMinChunkBytes, kMaxChunkBytes);
  int64_t rowsPerChunk =
      s.rowBytes > 0
          ? std::max<int64_t>(1, static_cast<int64_t>(
                                     target / static_cast<size_t>(s.rowBytes)))
          : paddedOuter;
  rowsPerChunk = std::min<int64_t>(rowsPerChunk, paddedOuter);

  for (int64_t pb = 0; pb < paddedOuter; pb += rowsPerChunk) {
    int64_t pe = std::min<int64_t>(pb + rowsPerChunk, paddedOuter);
    int64_t vb = std::max<int64_t>(pb - outerLo, 0);
    int64_t ve = std::min<int64_t>(pe - outerLo, validOuter);
    if (ve < vb)
      ve = vb;
    Chunk c;
    c.paddedBegin = pb;
    c.paddedEnd = pe;
    c.validBegin = vb;
    c.validEnd = ve;
    c.byteOffset = pb * s.rowBytes;
    c.bytes = static_cast<size_t>((pe - pb) * s.rowBytes);
    s.chunks.push_back(c);
  }
  s.maxChunkBytes = static_cast<size_t>(rowsPerChunk * s.rowBytes);
  return s;
}

} // namespace reloc
