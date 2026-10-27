//===- CostModel.h - P3b calibrated transfer cost model (#97) ---*- C++ -*-===//
//
// Consolidates the R-track's analytic model into a component: a flat-text
// calibration file (per machine, assembled deterministically from
// committed bench/results artifacts) feeds a two-path cost model
//   T_A = overhead.a_ms + S * max(1/BW_cpu(pattern), wireBytes/S/BW_link)
//   T_B(Overlapped) = overhead.b_ms + S * (max(1/BW_link, m/BW_hbm)
//                     + min(1/BW_link, m/BW_hbm)/n)   [n = total chunk count,
//                     0 if uncalibrated]
//   T_B(Serial)     = overhead.b_ms + S * (1/BW_link + m/BW_hbm)
// (all affine in S), a gather-pattern classifier over BoundPlan
// properties, and a decision with a single-free-symbol threshold
// precompute so bind() only compares the bound size against a stored
// boundary. The prefold arm delegates to reloc::prefold::prefoldWins.
// Units: ms, GB/s (1e9 bytes/s), bytes.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_COSTMODEL_H
#define RELOC_COSTMODEL_H

#include "reloc/Bind.h"
#include "reloc/MethodDecision.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reloc {
namespace costmodel {

Pattern classify(const BoundPlan &b);

class CostModel {
public:
  /// Parse a "# costmodel calibration v0" flat text file. Returns the
  /// model or a diagnostic naming the offending line. Checked
  /// unconditionally (-DNDEBUG builds keep every check).
  static std::variant<CostModel, std::string> parse(const std::string &text);
  static std::variant<CostModel, std::string> load(const std::string &path);

  bool has(const std::string &key) const { return values_.count(key) != 0; }
  double at(const std::string &key) const { return values_.at(key); }
  double get(const std::string &key, double fallback) const {
    auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }
  const std::string &machine() const { return machine_; }

private:
  std::map<std::string, double> values_;
  std::string machine_;
};

/// Source-normalized CPU GB/s of Method A's transform stage for this
/// pattern at wire ratio r (the figure_rstar composition: two-pass
/// stages compose harmonically; pack's source-normalized BW is 4x its
/// input BW). nullopt when the calibration lacks a needed key.
std::optional<double> cpuBw(const CostModel &m, Pattern p, double r,
                            int threads);

struct PathCosts {
  double tAMs = 0, tBMs = 0;
  // Affine decomposition (per-method): t = interceptMs + slopeMsPerByte*S.
  double aInterceptMs = 0, aSlopeMsPerByte = 0;
  double bInterceptMs = 0, bSlopeMsPerByte = 0;
};

/// Both path costs for shipping one source tensor of S bytes with wire
/// ratio r to K receivers (broadcast: every receiver gets the whole
/// tensor; scatter: receivers partition it). nullopt on missing keys.
std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p,
                                   int64_t srcBytes, double r, int threads,
                                   int K = 1, bool broadcast = false,
                                   BPlacement bPlace = BPlacement::Overlapped);

/// The decision. nReuse < 0 disables the prefold arm; nReuse >= 1
/// enables it via reloc::prefold::prefoldWins (V4's validated rule).
std::optional<MethodDecision>
decide(const CostModel &m, Pattern p, int64_t srcBytes, double r,
       int threads = 8, int K = 1, int64_t nReuse = -1, bool broadcast = false,
       BPlacement bPlace = BPlacement::Overlapped);

/// Cost advice pricing (docs/reloc-export.md "Cost advice"). The compiler
/// states each admissible path's structure; the frontend resolves the
/// plan's symbols and the calibration key names (reloc_torch.advice) and
/// this function prices the concrete terms:
///   serial      T = I + sum t_i
///   overlapped  T = I + max t_i + (sum t_i - max t_i) / n   [n = cal[chunks]]
///   grouped     T = I_group + members * I_member + sum t_i
/// with t_i = bytes_i * mult_i / rate_i (ms; GB/s = 1e9 B/s). A missing key
/// is reported in PathCost::missingKey, never guessed; a non-positive rate
/// or multiplier, a negative byte or chunk count, an unknown form or
/// members < 1 throws std::invalid_argument("advice_invalid: ...").
struct AdviceTerm {
  double bytes = 0;
  std::string rateKey;
  std::string multiplierKey; // empty: multiplier 1
};
struct PathAdvice {
  std::string name, form, interceptKey, memberInterceptKey, chunksKey;
  std::vector<AdviceTerm> terms;
};
struct PathCost {
  std::string name;
  double ms = 0; // NaN when missingKey is set
  std::string missingKey;
};
std::vector<PathCost> priceAdvice(const CostModel &m,
                                  const std::vector<PathAdvice> &paths,
                                  int members = 1);

} // namespace costmodel
} // namespace reloc

#endif // RELOC_COSTMODEL_H
