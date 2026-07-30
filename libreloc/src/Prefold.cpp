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
  //
  // extents.size() < 2: quant::gatherQuantizeF32S8's contract requires a
  // distinct outer (per-channel-scale) axis -- enforced only by an assert
  // in Quant.cpp (gone under -DNDEBUG) -- so rank-1 plans must be rejected
  // here too (issue #98 final review, finding B).
  if (bound.elementSize != 4 || !bound.padRegions.empty() ||
      bound.extents.size() < 2 || bound.dstStrides.back() != 1 || !invScales ||
      !srcBase)
    return a;
  int64_t innerExtent = 1;
  for (size_t k = 1; k < bound.extents.size(); ++k)
    innerExtent *= bound.extents[k];
  // Packed-dst check, as a running product from the innermost axis
  // outward: dst must be fully row-major contiguous, i.e.
  // dstStrides[k] == prod(extents[k+1..]) for every k. This replaces the
  // old dstStrides[0] == innerExtent check, which only compared the
  // outermost stride against the total inner extent and passes for plans
  // that are contiguous at the front and back but gapped in the middle
  // (issue #98 final review, finding C) -- e.g. extents={2,3,4},
  // dstStrides={12,8,1} (max written index 31, but the alloc below is
  // sized for prod(extents) == 24 elements: a heap overflow). The final
  // value of the running product is prod(extents); cross-check it against
  // totalBytes so a plan whose totalBytes disagrees with
  // prod(extents)*elementSize can't under-allocate what the executors
  // below actually write.
  int64_t packedElems = 1;
  for (size_t k = bound.extents.size(); k-- > 0;) {
    if (bound.dstStrides[k] != packedElems)
      return a;
    packedElems *= bound.extents[k];
  }
  if (packedElems != bound.totalBytes / bound.elementSize)
    return a;
  if (spec == OutputSpec::S8QuantPack && bound.srcStrides != bound.dstStrides)
    return a;

  const int64_t outBytes = packedElems; // s8 image: 1 byte per element
  void *dst = backend.allocStaging(static_cast<size_t>(outBytes));
  if (!dst)
    return a;

  switch (spec) {
  case OutputSpec::S8GatherQuant:
    quant::gatherQuantizeF32S8Parallel(
        pool, bound, srcBase, static_cast<int8_t *>(dst), invScales, v);
    break;
  case OutputSpec::S8QuantPack:
    quant::quantizePackF32S8Parallel(pool, srcBase, static_cast<int8_t *>(dst),
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
  return static_cast<double>(nReuse) * tTransformMs > tPrefoldMs + penaltyMs;
}

} // namespace prefold
} // namespace reloc
