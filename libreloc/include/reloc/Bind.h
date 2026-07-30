//===- Bind.h - symbol evaluation and the concrete BoundPlan ----*- C++ -*-===//
//
// bind() turns a symbolic RelocationPlan plus a {symbol -> value} map into
// a concrete BoundPlan the executors run. It enforces the two-class
// constraint contract (issue #40 design decision 1): correctness
// constraints (divisibility, runtime pad-range) are hard bind errors;
// alignment is a performance concern recorded for execute-time downgrade.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_BIND_H
#define RELOC_BIND_H

#include "reloc/MethodDecision.h"
#include "reloc/Plan.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reloc {

namespace costmodel {
class CostModel;
} // namespace costmodel

/// Symbol values indexed by symbol-table position (bind resolves the
/// caller's name map into this before evaluating).
using SymbolValues = std::vector<int64_t>;

/// Caller-facing symbol binding.
using SymbolMap = std::map<std::string, int64_t>;

/// Evaluate a plan-context expression stream. Returns false and sets
/// `error` on divide/mod by zero, i64 overflow, or a PUSH_DIM opcode
/// (legal only in inverse streams). Symbol indices are trusted from the
/// decoder (already range-checked).
bool evalExpr(const ExprStream &stream, const SymbolValues &symbols,
              int64_t &out, std::string &error);

/// Execution strategy (design decision 4). `Auto` means "let the
/// heuristic choose"; the others force a specific executor.
enum class Strategy {
  Auto,
  ViewNoCopy,
  SingleThreadSimd,
  MultiThreadTiled,
  ChunkedPipeline,
};

/// A concrete pad region in destination coordinates.
struct PadRegion {
  size_t axis; // index into BoundPlan::extents (coalesced dst order)
  int64_t lo;
  int64_t hi;
  uint64_t fillBits = 0; // fill value bit pattern (low elementSize bytes)
};

/// The concrete plan the executors consume. extents/strides are in
/// coalesced destination iteration order.
struct BoundPlan {
  std::vector<int64_t> extents;
  std::vector<int64_t> srcStrides; // element strides into the source buffer
  std::vector<int64_t> dstStrides; // element strides into the dest buffer
  std::vector<uint32_t> perm;      // the plan's perm, carried for reference
  std::vector<PadRegion> padRegions;
  uint32_t elementSize = 0; // bytes per element
  int64_t totalBytes = 0;   // destination footprint
  bool noCopy = false;
  Strategy strategy = Strategy::Auto;
  int64_t L = 1; // innermost coalesced contiguous run length, elements
  // Execute-time downgrade input. Each Alignment::axis is in coalesced
  // BoundPlan::extents index space (remapped through coalescing, matching
  // PadRegion::axis) -- NOT the plan's original axis numbering.
  std::vector<Alignment> requiredAlignments;
  // Populated only when bind() is given a cost model (nullopt otherwise).
  std::optional<costmodel::MethodDecision> decision;
};

struct BindError {
  std::string message;
};

using BindResult = std::variant<BoundPlan, BindError>;

/// Bind `plan` against `symbolMap`. `override` forces a strategy when not
/// Strategy::Auto. Fails (BindError) on: a symbol-map mismatch, an
/// evaluation error, a violated correctness constraint (divisibility /
/// runtime pad-range), or a v0 domain violation (extent < 1, stride < 0).
///
/// `model`, when non-null, populates `BoundPlan::decision` via
/// costmodel::classify + costmodel::decide (wireRatio/K/nReuse forwarded
/// verbatim) and sources the Strategy::Auto size thresholds from the
/// calibration when present, falling back to the built-in constants
/// otherwise.
BindResult bind(const RelocationPlan &plan, const SymbolMap &symbolMap,
                Strategy override = Strategy::Auto,
                const costmodel::CostModel *model = nullptr,
                double wireRatio = 1.0, int K = 1, int64_t nReuse = -1);

} // namespace reloc

#endif // RELOC_BIND_H
