//===- Prefold.h - P4 load-time pre-folded transform (issue #98) -*- C++ -*-==//
//
// Load-time eager evaluation: given a BoundPlan whose symbols are all
// bound, materialize the relocated+quantized artifact ONCE into
// CopyBackend staging memory (pinned under CudaBackend, plain malloc
// under HostBackend so CI exercises the whole component) and publish it
// as the transfer source. Mechanism (prefoldArtifact) and policy
// (prefoldWins, the standalone amortization rule) are separate: the
// policy is pure arithmetic with no dependency on the #97 cost model.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PREFOLD_H
#define RELOC_PREFOLD_H

#include "reloc/Backend.h"
#include "reloc/Bind.h"
#include "reloc/Quant.h"

#include <cstdint>

namespace reloc {

class GatherPool;

namespace prefold {

/// What the fold produces. Only what the R3 scenarios need (an f32
/// relocate-only spec is a natural later addition; nothing in V4
/// measures it).
enum class OutputSpec {
  S8GatherQuant, ///< fused strided gather + per-channel int8 quantize
  S8QuantPack    ///< contiguous per-channel int8 quantize (identity plan)
};

class PrefoldArtifact;

PrefoldArtifact prefoldArtifact(const BoundPlan &bound, const float *srcBase,
                                OutputSpec spec, const float *invScales,
                                CopyBackend &backend, GatherPool &pool,
                                quant::Variant v = quant::Variant::Auto);

/// Move-only owner of the folded artifact. data() is the transfer
/// source; the staging allocation is freed on destruction through the
/// backend that made it. An invalid (default / moved-from / failed)
/// artifact has data() == nullptr.
class PrefoldArtifact {
public:
  PrefoldArtifact() = default;
  PrefoldArtifact(PrefoldArtifact &&o) noexcept
      : data_(o.data_), bytes_(o.bytes_), backend_(o.backend_) {
    o.data_ = nullptr;
    o.bytes_ = 0;
    o.backend_ = nullptr;
  }
  PrefoldArtifact &operator=(PrefoldArtifact &&o) noexcept {
    if (this != &o) {
      release();
      data_ = o.data_;
      bytes_ = o.bytes_;
      backend_ = o.backend_;
      o.data_ = nullptr;
      o.bytes_ = 0;
      o.backend_ = nullptr;
    }
    return *this;
  }
  ~PrefoldArtifact() { release(); }

  PrefoldArtifact(const PrefoldArtifact &) = delete;
  PrefoldArtifact &operator=(const PrefoldArtifact &) = delete;

  const void *data() const { return data_; }
  int64_t bytes() const { return bytes_; }
  bool valid() const { return data_ != nullptr; }

private:
  friend PrefoldArtifact prefoldArtifact(const BoundPlan &, const float *,
                                         OutputSpec, const float *,
                                         CopyBackend &, GatherPool &,
                                         quant::Variant);
  void release() {
    if (data_ && backend_)
      backend_->freeStaging(data_);
    data_ = nullptr;
    bytes_ = 0;
    backend_ = nullptr;
  }
  void *data_ = nullptr;
  int64_t bytes_ = 0;
  CopyBackend *backend_ = nullptr;
};

/// The standalone amortization rule (policy):
///   pre-fold wins iff nReuse * tTransformMs > tPrefoldMs + penaltyMs
/// tPrefoldMs is the one-time fold; penaltyMs carries memory-side costs
/// the caller supplies (cold pinned allocation, budget terms). Strict
/// inequality: a tie is not worth holding (1 + r) * S of pinned memory.
bool prefoldWins(int64_t nReuse, double tTransformMs, double tPrefoldMs,
                 double penaltyMs);

} // namespace prefold
} // namespace reloc

#endif // RELOC_PREFOLD_H
