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

/// Evaluate a channel-context stream (C3 typed stages): PUSH_DIM reads the
/// logical result coordinate `dims[i]`, PUSH_SYM a plan symbol. R3 calls
/// this per element to select a per-channel parameter.
bool evalChannel(const ExprStream &stream, const SymbolValues &symbols,
                 const std::vector<int64_t> &dims, int64_t &out,
                 std::string &error);

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
  /// C3: true for the layout part of a TypedBoundPlan. Its source and
  /// destination element widths differ, so the layout-only executors and
  /// transfer validators refuse it (they would copy source-width bytes into
  /// a destination-width allocation); only R3's typed dispatch may run it.
  bool typed = false;
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

//===----------------------------------------------------------------------===//
// Typed binding (C3, issue #143)
//===----------------------------------------------------------------------===//

/// A runtime parameter value handed to bindTyped: element type, extents
/// (empty for rank 0, one entry for rank 1) and the little-endian element
/// bytes. bindTyped copies `bytes` into the bound plan (owned snapshot), so
/// the caller's buffer may be released as soon as bindTyped returns and
/// nothing about it (address, device, framework object) survives binding.
struct ParameterValue {
  ElementType elementType{};
  std::vector<int64_t> extents;
  std::vector<uint8_t> bytes;
};

/// Caller-facing parameter binding, by declared name.
using ParameterMap = std::map<std::string, ParameterValue>;

/// One stage parameter after binding, uniform for inline constants and
/// runtime bindings: validated values, owned bytes.
struct BoundParameter {
  bool present = false;
  ElementType elementType{};
  int64_t length = 0;         // 1 per tensor, the channel extent per channel
  std::vector<uint8_t> bytes; // length * width(elementType) bytes
  std::string bindingName;    // empty for inline constants
};

/// One bound stage: the decoded stage with concrete shape and parameters.
/// `channel` stays an expression over the logical result coordinates
/// (PushDim) and plan symbols (PushSym); R3 evaluates it per element.
struct BoundStage {
  ValueTransformKind transform = ValueTransformKind::Cast;
  NumericPolicyKind policy = NumericPolicyKind::IeeeRne;
  StageType input;
  StageType output;
  std::vector<int64_t> shape; // concrete logical operand shape
  int64_t axis = -1;
  bool hasChannel = false;
  ExprStream channel;
  BoundParameter scale;
  BoundParameter zeroPoint;
};

/// The byte footprint of the tensor at one stage boundary: boundary 0 is
/// the source side, boundary k the output of stage k. Pads entering at or
/// before the boundary are included in `elements`. An execution variant
/// that transfers at boundary k moves exactly `bytes` on the wire.
struct StageFootprint {
  uint32_t boundary = 0;
  ElementType elementType{};
  int64_t elements = 0;
  int64_t bytes = 0;
};

/// The typed bound result. It certifies the representation and its guards,
/// never an executor: `requirements` lists what R3 must still supply.
struct TypedBoundPlan {
  BoundPlan layout; // typed == true: refused by layout-only executors
  std::vector<int64_t> sourceExtents;
  std::vector<int64_t> resultExtents; // padded logical result
  ElementType sourceType{};
  ElementType resultType{};
  SymbolValues symbols;
  std::vector<BoundStage> stages;
  /// The original pad fills with their entry stage (verbatim from the wire),
  /// so an executor that stops at an intermediate boundary can fold each
  /// fill to that boundary's type (R3 stage partitions).
  std::vector<TypedFill> fills;
  std::vector<StageFootprint> cuts; // boundaries 0..stages.size()
  int64_t sourceBytes = 0;          // cuts.front().bytes
  int64_t destinationBytes = 0;     // cuts.back().bytes == layout.totalBytes
  int64_t parameterBytes = 0;       // every bound parameter, recorded apart
  std::vector<std::string> requirements; // e.g. "typed_execution_dispatch"
};

using TypedBindResult = std::variant<TypedBoundPlan, BindError>;

/// Bind a decoded typed plan against symbol values and runtime parameters.
/// Fails (BindError) on: the layout's own bind errors; a source or result
/// descriptor outside the supported subset (dense row-major, zero offset,
/// extents >= 1, byte-multiple element widths); a missing, extra or
/// mismatched parameter (unknown name, wrong element type or rank, length
/// different from the channel extent re-evaluated under these symbols);
/// invalid parameter values (non-finite or non-positive scales, zero points
/// outside [-128, 127], nonzero under symmetric_rne); an inline per-channel
/// parameter whose length disagrees with the now-concrete extent; and any
/// overflow while computing element counts or byte footprints. Every
/// failure happens before anything is copied or executed.
TypedBindResult bindTyped(const TypedRelocationPlan &plan,
                          const SymbolMap &symbolMap,
                          const ParameterMap &parameters);

} // namespace reloc

#endif // RELOC_BIND_H
