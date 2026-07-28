//===- Prefold.cpp - P4 load-time pre-folded transform (issue #98) --------===//

#include "reloc/Prefold.h"

#include "reloc/GatherPool.h"

namespace reloc {
namespace prefold {

PrefoldArtifact prefoldArtifact(const BoundPlan &bound, const float *srcBase,
                                OutputSpec spec, const float *invScales,
                                CopyBackend &backend, GatherPool &pool,
                                quant::Variant v) {
  PrefoldArtifact a;
  // Preconditions checked unconditionally: benchmarking builds compile
  // with -DNDEBUG, and a silently wrong artifact is worse than none
  // (issue #63). Mirrors the rtrack fixture checks.
  if (bound.elementSize != 4 || !bound.padRegions.empty() ||
      bound.extents.empty() || bound.dstStrides.back() != 1 || !invScales ||
      !srcBase)
    return a;
  int64_t innerExtent = 1;
  for (size_t k = 1; k < bound.extents.size(); ++k)
    innerExtent *= bound.extents[k];
  if (bound.dstStrides[0] != innerExtent)
    return a;
  if (spec == OutputSpec::S8QuantPack && bound.srcStrides != bound.dstStrides)
    return a;

  const int64_t totalElems = bound.totalBytes / bound.elementSize;
  const int64_t outBytes = totalElems; // s8 image: 1 byte per element
  void *dst = backend.allocStaging(static_cast<size_t>(outBytes));
  if (!dst)
    return a;

  switch (spec) {
  case OutputSpec::S8GatherQuant:
    quant::gatherQuantizeF32S8Parallel(pool, bound, srcBase,
                                       static_cast<int8_t *>(dst), invScales,
                                       v);
    break;
  case OutputSpec::S8QuantPack:
    quant::quantizePackF32S8Parallel(pool, srcBase,
                                     static_cast<int8_t *>(dst),
                                     bound.extents[0], innerExtent, invScales,
                                     v);
    break;
  }

  a.data_ = dst;
  a.bytes_ = outBytes;
  a.backend_ = &backend;
  return a;
}

bool prefoldWins(int64_t nReuse, double tTransformMs, double tPrefoldMs,
                 double penaltyMs) {
  if (nReuse < 1)
    return false;
  return static_cast<double>(nReuse) * tTransformMs >
         tPrefoldMs + penaltyMs;
}

} // namespace prefold
} // namespace reloc
