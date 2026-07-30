//===- Bind.cpp - symbol evaluation and BoundPlan construction ------------===//

#include "reloc/Bind.h"

#include "reloc/CostModel.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace reloc {
namespace {

bool floorDiv(int64_t a, int64_t b, int64_t &out, std::string &error) {
  if (b == 0)
    return (error = "division by zero"), false;
  if (a == std::numeric_limits<int64_t>::min() && b == -1)
    return (error = "integer overflow in floordiv"), false;
  int64_t q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0)))
    --q;
  out = q;
  return true;
}

} // namespace

namespace {

/// Resolve the caller's name map into position-indexed values, requiring
/// an exact match with the plan's symbol table.
bool resolveSymbols(const RelocationPlan &plan, const SymbolMap &map,
                    SymbolValues &out, std::string &error) {
  for (const auto &kv : map)
    if (std::find(plan.symbols.begin(), plan.symbols.end(), kv.first) ==
        plan.symbols.end())
      return (error = "unknown symbol in binding: " + kv.first), false;
  out.resize(plan.symbols.size());
  for (size_t i = 0; i < plan.symbols.size(); ++i) {
    auto it = map.find(plan.symbols[i]);
    if (it == map.end())
      return (error = "unbound symbol: " + plan.symbols[i]), false;
    out[i] = it->second;
  }
  return true;
}

/// Evaluate one stream or record a contextual error.
bool evalField(const ExprStream &stream, const SymbolValues &symbols,
               const char *what, int64_t &out, std::string &error) {
  std::string inner;
  if (!evalExpr(stream, symbols, out, inner))
    return (error = std::string(what) + ": " + inner), false;
  return true;
}

/// One concrete axis before coalescing.
struct ConcreteAxis {
  int64_t extent;
  int64_t srcStride;
  int64_t dstStride;
  bool padded;
  int64_t lo;
  int64_t hi;
  uint64_t fillBits = 0;
};

bool mulOk(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_mul_overflow(a, b, &out);
}

} // namespace

bool evalExpr(const ExprStream &stream, const SymbolValues &symbols,
              int64_t &out, std::string &error) {
  std::vector<int64_t> stack;
  stack.reserve(stream.size());
  for (const ExprToken &token : stream) {
    switch (token.op) {
    case ExprOp::PushSym:
      // Index range guaranteed by the decoder; guard defensively anyway.
      if (token.value < 0 || static_cast<size_t>(token.value) >= symbols.size())
        return (error = "symbol index out of range"), false;
      stack.push_back(symbols[token.value]);
      break;
    case ExprOp::PushConst:
      stack.push_back(token.value);
      break;
    case ExprOp::PushDim:
      return (error = "PUSH_DIM is not valid in a plan expression"), false;
    case ExprOp::Add:
    case ExprOp::Sub:
    case ExprOp::Mul:
    case ExprOp::FloorDiv:
    case ExprOp::Mod: {
      // The decoder enforces stack discipline, but evalExpr is public and
      // may be handed a hand-built stream; guard defensively (mirrors the
      // PushSym index guard) to avoid pop_back UB on an empty vector.
      if (stack.size() < 2)
        return (error = "expression stack underflow"), false;
      int64_t b = stack.back();
      stack.pop_back();
      int64_t a = stack.back();
      stack.pop_back();
      int64_t r = 0;
      switch (token.op) {
      case ExprOp::Add:
        if (__builtin_add_overflow(a, b, &r))
          return (error = "integer overflow in add"), false;
        break;
      case ExprOp::Sub:
        if (__builtin_sub_overflow(a, b, &r))
          return (error = "integer overflow in sub"), false;
        break;
      case ExprOp::Mul:
        if (__builtin_mul_overflow(a, b, &r))
          return (error = "integer overflow in mul"), false;
        break;
      case ExprOp::FloorDiv:
        if (!floorDiv(a, b, r, error))
          return false;
        break;
      case ExprOp::Mod: {
        int64_t q = 0;
        if (!floorDiv(a, b, q, error))
          return false;
        int64_t prod = 0;
        if (__builtin_mul_overflow(q, b, &prod) ||
            __builtin_sub_overflow(a, prod, &r))
          return (error = "integer overflow in mod"), false;
        break;
      }
      default:
        break;
      }
      stack.push_back(r);
      break;
    }
    }
  }
  if (stack.size() != 1)
    return (error = "expression did not evaluate to a single value"), false;
  out = stack.back();
  return true;
}

BindResult bind(const RelocationPlan &plan, const SymbolMap &symbolMap,
                Strategy override, const costmodel::CostModel *model,
                double wireRatio, int K, int64_t nReuse) {
  std::string error;
  SymbolValues symbols;
  if (!resolveSymbols(plan, symbolMap, symbols, error))
    return BindError{error};

  // 1. Correctness constraint: divisibility (hard error).
  for (const Divisibility &d : plan.divisibility) {
    if (d.divisor <= 0)
      return BindError{"divisibility divisor must be positive"};
    int64_t value = 0;
    if (!evalField(d.expr, symbols, "divisibility expr", value, error))
      return BindError{error};
    if (value % d.divisor != 0)
      return BindError{"divisibility violated: value " + std::to_string(value) +
                       " not divisible by " + std::to_string(d.divisor)};
  }

  if (plan.axes.empty())
    return BindError{"plan must have >= 1 axis (v0)"};

  // 2. Evaluate axes into concrete form; attach pad widths.
  std::vector<ConcreteAxis> axes(plan.axes.size());
  for (size_t k = 0; k < plan.axes.size(); ++k) {
    if (!evalField(plan.axes[k].extent, symbols, "axis extent", axes[k].extent,
                   error) ||
        !evalField(plan.axes[k].srcStride, symbols, "axis src_stride",
                   axes[k].srcStride, error) ||
        !evalField(plan.axes[k].dstStride, symbols, "axis dst_stride",
                   axes[k].dstStride, error))
      return BindError{error};
    if (axes[k].extent < 1)
      return BindError{"axis extent must be >= 1 (v0 rejects zero/negative)"};
    if (axes[k].srcStride < 0 || axes[k].dstStride < 0)
      return BindError{"strides must be non-negative in v0"};
    axes[k].padded = false;
    axes[k].lo = axes[k].hi = 0;
  }

  // 3. Pad ranges. runtime_pad_check => the correctness check the verifier
  // deferred (hard error). Always attach lo/hi to the axis.
  for (const PadFill &pad : plan.padFill) {
    int64_t lo = 0, hi = 0;
    if (!evalField(pad.lo, symbols, "pad lo", lo, error) ||
        !evalField(pad.hi, symbols, "pad hi", hi, error))
      return BindError{error};
    // The decoder rejects an out-of-range pad.dstAxis, but bind is public
    // and may be handed a hand-built plan; guard the index (mirrors the
    // alignment-axis guard) before the OOB WRITE into axes[] below. This
    // also makes the dst.extents access inside the runtime_pad_check block
    // safe against a bad dstAxis.
    if (pad.dstAxis >= plan.axes.size())
      return BindError{"pad dst_axis out of range"};
    ConcreteAxis &axis = axes[pad.dstAxis];
    axis.padded = true;
    axis.lo = lo;
    axis.hi = hi;
    axis.fillBits = pad.fillBits;
    // Negative pad width is a domain invariant (like extent >= 1),
    // independent of runtime_pad_check: a negative width would produce a
    // negative padded extent and negative totalBytes. Reject unconditionally.
    if (lo < 0 || hi < 0)
      return BindError{"pad width negative"};
    if (plan.runtimePadCheck) {
      // The decoder guarantees pad.dstAxis < plan.axes.size() (so the
      // axes[] access above is safe) but not < dst.extents.size(); a
      // hand-crafted blob with dst rank < axes count would OOB here.
      if (pad.dstAxis >= plan.dst.extents.size())
        return BindError{"pad dst_axis out of range for dst extents"};
      int64_t dstExtent = 0;
      if (!evalField(plan.dst.extents[pad.dstAxis], symbols, "dst extent",
                     dstExtent, error))
        return BindError{error};
      int64_t sum = 0;
      if (__builtin_add_overflow(axis.extent, lo, &sum) ||
          __builtin_add_overflow(sum, hi, &sum))
        return BindError{"overflow in pad-range check"};
      if (sum != dstExtent)
        return BindError{"pad range inconsistent: extent + lo + hi (" +
                         std::to_string(sum) + ") != dst extent (" +
                         std::to_string(dstExtent) + ")"};
    }
  }

  // 4. Coalesce adjacent axes contiguous on both sides (padded axes never
  // merge). Fixpoint. `absorbed[c]` tracks how many ORIGINAL axes each
  // surviving coalesced axis covers, so alignment.axis (in original index
  // space) can be remapped to coalesced index space afterward.
  std::vector<size_t> absorbed(axes.size(), 1);
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t k = 0; k + 1 < axes.size(); ++k) {
      const ConcreteAxis &o = axes[k], &i = axes[k + 1];
      if (o.padded || i.padded)
        continue;
      int64_t srcProd = 0, dstProd = 0, extProd = 0;
      if (!mulOk(i.srcStride, i.extent, srcProd) ||
          !mulOk(i.dstStride, i.extent, dstProd) ||
          !mulOk(o.extent, i.extent, extProd))
        continue;
      if (o.srcStride != srcProd || o.dstStride != dstProd)
        continue;
      axes[k] = ConcreteAxis{extProd, i.srcStride, i.dstStride, false, 0, 0};
      axes.erase(axes.begin() + k + 1);
      absorbed[k] += absorbed[k + 1];
      absorbed.erase(absorbed.begin() + k + 1);
      changed = true;
      break;
    }
  }
  // Build the original-axis -> coalesced-axis index map: coalesced axis c
  // spans the next absorbed[c] consecutive original indices. Well-defined
  // because pad-carrying (and any non-merged) axes stay 1:1.
  std::vector<size_t> originalToCoalesced(plan.axes.size());
  for (size_t c = 0, orig = 0; c < absorbed.size(); ++c)
    for (size_t j = 0; j < absorbed[c]; ++j)
      originalToCoalesced[orig++] = c;

  // 5. Assemble BoundPlan.
  BoundPlan bound;
  bound.perm = plan.perm;
  uint32_t bitwidth = plan.dst.elementType.bitwidth;
  if (bitwidth == 0 || bitwidth % 8 != 0)
    return BindError{"element type bitwidth must be a positive multiple of 8 "
                     "in v0"};
  bound.elementSize = bitwidth / 8;
  bound.noCopy = plan.noCopy;
  // alignment.axis is in original-axis index space; remap it to coalesced
  // BoundPlan::extents index space (mirrors PadRegion.axis). The decoder
  // guarantees a.axis < plan.axes.size(), but bind is public and may be
  // handed a hand-built plan; guard the index (same public-trust-boundary
  // class as the pad dst.extents guard). A structurally-malformed axis is
  // a hard error -- the "alignment never fails bind" rule is about
  // semantic alignment shortfalls, not structural malformation.
  for (const Alignment &a : plan.alignment) {
    if (a.axis >= plan.axes.size())
      return BindError{"alignment axis out of range"};
    bound.requiredAlignments.push_back(
        Alignment{(uint32_t)originalToCoalesced[a.axis], a.bytes});
  }
  int64_t totalElements = 1;
  for (size_t k = 0; k < axes.size(); ++k) {
    bound.extents.push_back(axes[k].extent);
    bound.srcStrides.push_back(axes[k].srcStride);
    bound.dstStrides.push_back(axes[k].dstStride);
    int64_t padded = axes[k].extent;
    if (axes[k].padded) {
      bound.padRegions.push_back(
          PadRegion{k, axes[k].lo, axes[k].hi, axes[k].fillBits});
      if (__builtin_add_overflow(padded, axes[k].lo, &padded) ||
          __builtin_add_overflow(padded, axes[k].hi, &padded))
        return BindError{"overflow computing padded extent"};
    }
    if (__builtin_mul_overflow(totalElements, padded, &totalElements))
      return BindError{"overflow computing total element count"};
  }
  if (__builtin_mul_overflow(totalElements, (int64_t)bound.elementSize,
                             &bound.totalBytes))
    return BindError{"overflow computing total bytes"};

  // 6. L = innermost coalesced unit-stride run (valid extent if padded).
  bound.L = 1;
  if (!axes.empty()) {
    const ConcreteAxis &inner = axes.back();
    if (inner.srcStride == 1 && inner.dstStride == 1)
      bound.L = inner.extent;
  }

  // 7. Strategy. Boundaries come from the calibration when present
  // (strategy.single_thread_max_bytes / strategy.multi_thread_max_bytes),
  // falling back to the P2 constants otherwise.
  constexpr int64_t kL2Bytes = 256 * 1024;
  constexpr int64_t kMultiThreadMaxBytes = 256 * 1024 * 1024;
  const double singleMax = model
                               ? model->get("strategy.single_thread_max_bytes",
                                            static_cast<double>(kL2Bytes))
                               : static_cast<double>(kL2Bytes);
  const double multiMax =
      model ? model->get("strategy.multi_thread_max_bytes",
                         static_cast<double>(kMultiThreadMaxBytes))
            : static_cast<double>(kMultiThreadMaxBytes);
  if (override != Strategy::Auto)
    bound.strategy = override;
  else if (bound.noCopy)
    bound.strategy = Strategy::ViewNoCopy;
  else if (static_cast<double>(bound.totalBytes) <= singleMax)
    bound.strategy = Strategy::SingleThreadSimd;
  else if (static_cast<double>(bound.totalBytes) <= multiMax)
    bound.strategy = Strategy::MultiThreadTiled;
  else
    bound.strategy = Strategy::ChunkedPipeline;

  // 8. Cost-model decision (opt-in via `model`).
  if (model) {
    const costmodel::Pattern pat = costmodel::classify(bound);
    if (auto d = costmodel::decide(*model, pat, bound.totalBytes, wireRatio,
                                   /*threads=*/8, K, nReuse))
      bound.decision = *d;
  }

  return bound;
}

} // namespace reloc
