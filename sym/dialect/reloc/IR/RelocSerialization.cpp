//===- RelocSerialization.cpp - Plan wire-format encoder ------------------===//
//
// Implements wire format v0 (docs/reloc-plan-format.md). The body is
// buffered while the symbol table is built by first use, then the header,
// table, and body are concatenated — table/body drift is impossible.
//
//===----------------------------------------------------------------------===//

#include "RelocSerialization.h"
#include "SymDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::reloc;

namespace {

// Expression opcodes (docs/reloc-plan-format.md, "Expressions").
enum : uint8_t {
  kPushSym = 0x00,
  kPushConst = 0x01,
  kAdd = 0x02,
  kSub = 0x03,
  kMul = 0x04,
  kFloorDiv = 0x05,
  kMod = 0x06,
  kPushDim = 0x07,
};

// Element type kinds (docs/reloc-plan-format.md, "Element types").
enum : uint8_t {
  kFloat = 0,
  kBFloat = 1,
  kInt = 2,
  kIndex = 3,
};

class PlanEncoder {
public:
  explicit PlanEncoder(Location loc) : loc(loc) {}

  FailureOr<std::vector<uint8_t>> encode(PlanAttr plan);

private:
  // --- primitive emitters (into `body`) ---
  void emitU8(uint8_t value) { body.push_back(value); }
  void emitU32(uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
      body.push_back(static_cast<uint8_t>(value >> shift));
  }
  void emitI64(int64_t value) {
    auto bits = static_cast<uint64_t>(value);
    for (int shift = 0; shift < 64; shift += 8)
      body.push_back(static_cast<uint8_t>(bits >> shift));
  }
  void emitStr(StringRef str) {
    emitU32(str.size());
    body.insert(body.end(), str.begin(), str.end());
  }

  // --- symbol table (first use in encoding order) ---
  uint32_t symbolIndex(StringRef name) {
    for (auto [index, existing] : llvm::enumerate(symbols))
      if (existing == name)
        return index;
    symbols.push_back(name);
    return symbols.size() - 1;
  }

  // --- composite emitters ---
  LogicalResult emitExpr(Attribute expr);
  LogicalResult emitExprOps(Attribute expr, uint32_t &opCount,
                            std::vector<uint8_t> &ops);
  LogicalResult emitAffineExprOps(AffineExpr expr, uint32_t &opCount,
                                  std::vector<uint8_t> &ops);
  LogicalResult emitInverseResult(AffineExpr expr);
  LogicalResult emitType(Type type);
  LogicalResult emitDesc(TensorDescAttr desc);
  LogicalResult emitTypedValue(TypedAttr value);

  Location loc;
  std::vector<uint8_t> body;
  SmallVector<StringRef> symbols;
};

/// Append postfix ops for a sym expression into `ops`, counting them.
LogicalResult PlanEncoder::emitExprOps(Attribute expr, uint32_t &opCount,
                                       std::vector<uint8_t> &ops) {
  auto pushByte = [&](uint8_t b) { ops.push_back(b); };
  auto pushU32 = [&](uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8)
      ops.push_back(static_cast<uint8_t>(v >> shift));
  };
  auto pushI64 = [&](int64_t v) {
    auto bits = static_cast<uint64_t>(v);
    for (int shift = 0; shift < 64; shift += 8)
      ops.push_back(static_cast<uint8_t>(bits >> shift));
  };

  if (auto constant = dyn_cast<sym::ConstantExprAttr>(expr)) {
    pushByte(kPushConst);
    pushI64(constant.getValue());
    ++opCount;
    return success();
  }
  if (auto symbol = dyn_cast<sym::SymbolExprAttr>(expr)) {
    pushByte(kPushSym);
    pushU32(symbolIndex(symbol.getName()));
    ++opCount;
    return success();
  }
  if (auto binary = dyn_cast<sym::BinaryExprAttr>(expr)) {
    if (failed(emitExprOps(binary.getLhs(), opCount, ops)) ||
        failed(emitExprOps(binary.getRhs(), opCount, ops)))
      return failure();
    uint8_t opcode;
    switch (binary.getOpcode()) {
    case sym::SymbolicExprOp::Add:
      opcode = kAdd;
      break;
    case sym::SymbolicExprOp::Sub:
      opcode = kSub;
      break;
    case sym::SymbolicExprOp::Mul:
      opcode = kMul;
      break;
    case sym::SymbolicExprOp::Div:
      opcode = kFloorDiv;
      break;
    case sym::SymbolicExprOp::Mod:
      opcode = kMod;
      break;
    }
    pushByte(opcode);
    ++opCount;
    return success();
  }
  return emitError(loc) << "cannot encode non-sym expression: " << expr;
}

LogicalResult PlanEncoder::emitExpr(Attribute expr) {
  uint32_t opCount = 0;
  std::vector<uint8_t> ops;
  if (failed(emitExprOps(expr, opCount, ops)))
    return failure();
  emitU32(opCount);
  body.insert(body.end(), ops.begin(), ops.end());
  return success();
}

/// Append postfix ops for an inverse-map affine expression.
LogicalResult PlanEncoder::emitAffineExprOps(AffineExpr expr, uint32_t &opCount,
                                             std::vector<uint8_t> &ops) {
  auto pushByte = [&](uint8_t b) { ops.push_back(b); };
  auto pushU32 = [&](uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8)
      ops.push_back(static_cast<uint8_t>(v >> shift));
  };
  auto pushI64 = [&](int64_t v) {
    auto bits = static_cast<uint64_t>(v);
    for (int shift = 0; shift < 64; shift += 8)
      ops.push_back(static_cast<uint8_t>(bits >> shift));
  };

  if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
    pushByte(kPushConst);
    pushI64(constant.getValue());
    ++opCount;
    return success();
  }
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    pushByte(kPushDim);
    pushU32(dim.getPosition());
    ++opCount;
    return success();
  }
  if (auto binary = dyn_cast<AffineBinaryOpExpr>(expr)) {
    uint8_t opcode;
    switch (binary.getKind()) {
    case AffineExprKind::Add:
      opcode = kAdd;
      break;
    case AffineExprKind::Mul:
      opcode = kMul;
      break;
    case AffineExprKind::FloorDiv:
      opcode = kFloorDiv;
      break;
    case AffineExprKind::Mod:
      opcode = kMod;
      break;
    default:
      return emitError(loc)
             << "inverse map uses an operation not representable in wire "
                "format v0 (ceildiv)";
    }
    if (failed(emitAffineExprOps(binary.getLHS(), opCount, ops)) ||
        failed(emitAffineExprOps(binary.getRHS(), opCount, ops)))
      return failure();
    pushByte(opcode);
    ++opCount;
    return success();
  }
  return emitError(loc) << "inverse map uses an expression not representable "
                           "in wire format v0 (affine symbols)";
}

LogicalResult PlanEncoder::emitInverseResult(AffineExpr expr) {
  uint32_t opCount = 0;
  std::vector<uint8_t> ops;
  if (failed(emitAffineExprOps(expr, opCount, ops)))
    return failure();
  emitU32(opCount);
  body.insert(body.end(), ops.begin(), ops.end());
  return success();
}

LogicalResult PlanEncoder::emitType(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (type.isBF16()) {
      emitU8(kBFloat);
      emitU32(16);
      return success();
    }
    if (type.isF16() || type.isF32() || type.isF64()) {
      emitU8(kFloat);
      emitU32(floatType.getWidth());
      return success();
    }
    return emitError(loc) << "element type not representable in wire format "
                             "v0: "
                          << type;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (!intType.isSignless())
      return emitError(loc) << "signed/unsigned integer element types are "
                               "not representable in wire format v0 "
                               "(signless only): "
                            << type;
    if (intType.getWidth() > 64)
      return emitError(loc) << "integer element type wider than 64 bits not "
                               "representable in wire format v0: "
                            << type;
    emitU8(kInt);
    emitU32(intType.getWidth());
    return success();
  }
  if (isa<IndexType>(type)) {
    emitU8(kIndex);
    emitU32(64);
    return success();
  }
  return emitError(loc) << "element type not representable in wire format "
                           "v0: "
                        << type;
}

LogicalResult PlanEncoder::emitDesc(TensorDescAttr desc) {
  emitU32(desc.getExtents().size());
  for (Attribute extent : desc.getExtents())
    if (failed(emitExpr(extent)))
      return failure();
  emitU32(desc.getStrides().size());
  for (Attribute stride : desc.getStrides())
    if (failed(emitExpr(stride)))
      return failure();
  if (failed(emitExpr(desc.getOffset())))
    return failure();
  return emitType(desc.getElementType());
}

LogicalResult PlanEncoder::emitTypedValue(TypedAttr value) {
  if (failed(emitType(value.getType())))
    return failure();
  uint64_t raw;
  if (auto floatValue = dyn_cast<FloatAttr>(value)) {
    raw = floatValue.getValue().bitcastToAPInt().getZExtValue();
  } else if (auto intValue = dyn_cast<IntegerAttr>(value)) {
    if (intValue.getValue().getBitWidth() > 64)
      return emitError(loc) << "pad value wider than 64 bits not "
                               "representable in wire format v0";
    raw = intValue.getValue().getZExtValue();
  } else {
    return emitError(loc) << "pad value not representable in wire format "
                             "v0: "
                          << value;
  }
  // u64 raw bits, little-endian.
  for (int shift = 0; shift < 64; shift += 8)
    body.push_back(static_cast<uint8_t>(raw >> shift));
  return success();
}

FailureOr<std::vector<uint8_t>> PlanEncoder::encode(PlanAttr plan) {
  // Sections 3..12 into `body`; the symbol table fills up as a side effect.
  if (failed(emitDesc(plan.getSrc())) || failed(emitDesc(plan.getDst())))
    return failure();

  ArrayRef<int64_t> perm = plan.getPerm().asArrayRef();
  emitU32(perm.size());
  for (int64_t value : perm)
    emitU32(static_cast<uint32_t>(value));

  emitU32(plan.getAxes().size());
  for (AxisInfoAttr axis : plan.getAxes()) {
    emitStr(axis.getName());
    if (failed(emitExpr(axis.getExtent())) ||
        failed(emitExpr(axis.getSrcStride())) ||
        failed(emitExpr(axis.getDstStride())))
      return failure();
  }

  emitU32(plan.getPadFill().size());
  for (PadFillAttr pad : plan.getPadFill()) {
    emitU32(static_cast<uint32_t>(pad.getDstAxis()));
    if (failed(emitExpr(pad.getLo())) || failed(emitExpr(pad.getHi())) ||
        failed(emitTypedValue(pad.getValue())))
      return failure();
  }

  emitU32(plan.getDivisibility().size());
  for (DivisibilityAttr divisibility : plan.getDivisibility()) {
    if (failed(emitExpr(divisibility.getExpr())))
      return failure();
    emitI64(divisibility.getDivisor());
  }

  emitU32(plan.getAlignment().size());
  for (AlignmentAttr alignment : plan.getAlignment()) {
    emitU32(static_cast<uint32_t>(alignment.getAxis()));
    emitI64(alignment.getBytes());
  }

  ArrayRef<bool> contiguity = plan.getContiguity().asArrayRef();
  emitU32(contiguity.size());
  for (bool flag : contiguity)
    emitU8(flag ? 1 : 0);

  emitU8(plan.getNoCopy() ? 1 : 0);
  emitU8(plan.getRuntimePadCheck() ? 1 : 0);

  AffineMap inverse = plan.getInverse().getValue();
  emitU32(inverse.getNumDims());
  emitU32(inverse.getNumResults());
  for (AffineExpr result : inverse.getResults())
    if (failed(emitInverseResult(result)))
      return failure();

  // Assemble: header + symbol table + body.
  std::vector<uint8_t> out;
  out.reserve(body.size() + 64);
  const char magic[4] = {'R', 'P', 'L', 'N'};
  out.insert(out.end(), magic, magic + 4);
  for (int shift = 0; shift < 32; shift += 8) // version u32 = 0
    out.push_back(static_cast<uint8_t>(0u >> shift));
  for (int shift = 0; shift < 32; shift += 8) // symbol count
    out.push_back(
        static_cast<uint8_t>(static_cast<uint32_t>(symbols.size()) >> shift));
  for (StringRef name : symbols) {
    for (int shift = 0; shift < 32; shift += 8)
      out.push_back(
          static_cast<uint8_t>(static_cast<uint32_t>(name.size()) >> shift));
    out.insert(out.end(), name.begin(), name.end());
  }
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

} // namespace

FailureOr<std::vector<uint8_t>> mlir::reloc::encodePlan(PlanAttr plan,
                                                        Location loc) {
  if (!plan) {
    emitError(loc) << "cannot encode a null plan";
    return failure();
  }
  return PlanEncoder(loc).encode(plan);
}
