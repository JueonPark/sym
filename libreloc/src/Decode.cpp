//===- Decode.cpp - wire-format v0 decoder --------------------------------===//

#include "reloc/Decode.h"
#include "reloc/TypedValue.h"
#include "reloc/Version.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <utility>

namespace reloc {
namespace {

/// Little-endian bounds-checked reader. All read* methods return false on
/// underflow and record the error at the offset where the read began.
class ByteReader {
public:
  ByteReader(const uint8_t *data, size_t size) : data_(data), size_(size) {}

  size_t offset() const { return offset_; }
  size_t remaining() const { return size_ - offset_; }

  bool fail(size_t at, std::string message) {
    if (!error_)
      error_ = DecodeError{at, std::move(message)};
    return false;
  }
  bool failHere(std::string message) {
    return fail(offset_, std::move(message));
  }
  const DecodeError &error() const { return *error_; }
  bool hasError() const { return error_.has_value(); }

  bool skip(size_t n) {
    if (remaining() < n)
      return failHere("truncated: expected " + std::to_string(n) + " bytes");
    offset_ += n;
    return true;
  }

  bool readU8(uint8_t &out) {
    if (remaining() < 1)
      return failHere("truncated: expected u8");
    out = data_[offset_++];
    return true;
  }
  bool readU32(uint32_t &out) {
    if (remaining() < 4)
      return failHere("truncated: expected u32");
    out = static_cast<uint32_t>(data_[offset_]) |
          static_cast<uint32_t>(data_[offset_ + 1]) << 8 |
          static_cast<uint32_t>(data_[offset_ + 2]) << 16 |
          static_cast<uint32_t>(data_[offset_ + 3]) << 24;
    offset_ += 4;
    return true;
  }
  bool readU64(uint64_t &out) {
    if (remaining() < 8)
      return failHere("truncated: expected u64");
    out = static_cast<uint64_t>(data_[offset_]) |
          static_cast<uint64_t>(data_[offset_ + 1]) << 8 |
          static_cast<uint64_t>(data_[offset_ + 2]) << 16 |
          static_cast<uint64_t>(data_[offset_ + 3]) << 24 |
          static_cast<uint64_t>(data_[offset_ + 4]) << 32 |
          static_cast<uint64_t>(data_[offset_ + 5]) << 40 |
          static_cast<uint64_t>(data_[offset_ + 6]) << 48 |
          static_cast<uint64_t>(data_[offset_ + 7]) << 56;
    offset_ += 8;
    return true;
  }
  bool readI64(int64_t &out) {
    uint64_t raw;
    if (!readU64(raw))
      return false;
    std::memcpy(&out, &raw,
                sizeof(out)); // two's complement, well-defined via memcpy
    return true;
  }
  /// A count that gates an allocation: rejected when it exceeds the
  /// remaining byte budget (every element occupies at least one byte), so
  /// a hostile u32 can never cause a multi-GB reserve.
  bool readCount(uint32_t &out, const char *what) {
    size_t at = offset_;
    if (!readU32(out))
      return false;
    if (out > remaining())
      return fail(at, std::string("count exceeds remaining bytes: ") + what);
    return true;
  }
  bool readString(std::string &out) {
    uint32_t length;
    if (!readCount(length, "string length"))
      return false;
    out.assign(reinterpret_cast<const char *>(data_ + offset_), length);
    offset_ += length;
    return true;
  }

private:
  const uint8_t *data_;
  size_t size_;
  size_t offset_ = 0;
  std::optional<DecodeError> error_; // needs <optional>
};

/// Plan: PUSH_SYM only. Inverse: PUSH_DIM only (dst-axis coordinates).
/// Channel (v1 stage channel maps): both, with PUSH_DIM naming a logical
/// result coordinate.
enum class ExprContext { Plan, Inverse, Channel };

/// Parse one expression stream with full static validation: op_count >= 1,
/// stack discipline (ends at depth exactly 1), context-legal opcodes,
/// in-range PUSH_SYM / PUSH_DIM operands.
bool parseExpr(ByteReader &reader, ExprContext context, size_t symbolCount,
               size_t dimCount, ExprStream &out) {
  size_t start = reader.offset();
  uint32_t opCount;
  if (!reader.readCount(opCount, "expression op_count"))
    return false;
  if (opCount == 0)
    return reader.fail(start, "empty expression stream");
  out.clear();
  out.reserve(opCount);
  int64_t depth = 0;
  for (uint32_t i = 0; i < opCount; ++i) {
    size_t opAt = reader.offset();
    uint8_t opcode;
    if (!reader.readU8(opcode))
      return false;
    ExprToken token{static_cast<ExprOp>(opcode), 0};
    switch (token.op) {
    case ExprOp::PushSym: {
      if (context == ExprContext::Inverse)
        return reader.fail(opAt,
                           "PUSH_SYM is not allowed in inverse expressions");
      uint32_t index;
      if (!reader.readU32(index))
        return false;
      if (index >= symbolCount)
        return reader.fail(opAt, "PUSH_SYM symbol index out of range");
      token.value = index;
      ++depth;
      break;
    }
    case ExprOp::PushConst: {
      int64_t value;
      if (!reader.readI64(value))
        return false;
      token.value = value;
      ++depth;
      break;
    }
    case ExprOp::PushDim: {
      if (context == ExprContext::Plan)
        return reader.fail(opAt, "PUSH_DIM is only allowed in inverse and "
                                 "channel expressions");
      uint32_t index;
      if (!reader.readU32(index))
        return false;
      if (index >= dimCount)
        return reader.fail(opAt, "PUSH_DIM dim index out of range");
      token.value = index;
      ++depth;
      break;
    }
    case ExprOp::Add:
    case ExprOp::Sub:
    case ExprOp::Mul:
    case ExprOp::FloorDiv:
    case ExprOp::Mod:
      if (depth < 2)
        return reader.fail(opAt, "expression stack underflow");
      --depth;
      break;
    default:
      return reader.fail(opAt, "unknown expression opcode");
    }
    out.push_back(token);
  }
  if (depth != 1)
    return reader.fail(start, "expression stream must leave exactly one value");
  return true;
}

bool parseElementType(ByteReader &reader, ElementType &out) {
  size_t at = reader.offset();
  uint8_t kind;
  uint32_t bitwidth;
  if (!reader.readU8(kind) || !reader.readU32(bitwidth))
    return false;
  bool valid = false;
  switch (kind) {
  case 0:
    valid = bitwidth == 16 || bitwidth == 32 || bitwidth == 64;
    break;
  case 1:
    valid = bitwidth == 16;
    break;
  case 2:
    valid = bitwidth >= 1 && bitwidth <= 64;
    break;
  case 3:
    valid = bitwidth == 64;
    break;
  default:
    valid = false;
  }
  if (!valid)
    return reader.fail(at, "invalid element type kind/bitwidth");
  out = ElementType{static_cast<ElementTypeKind>(kind), bitwidth};
  return true;
}

bool parseTensorDesc(ByteReader &reader, size_t symbolCount, TensorDesc &out) {
  uint32_t rank;
  if (!reader.readCount(rank, "tensor_desc rank"))
    return false;
  out.extents.clear();
  out.extents.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    ExprStream extent;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, extent))
      return false;
    out.extents.push_back(std::move(extent));
  }

  size_t strideCountAt = reader.offset();
  uint32_t strideCount;
  if (!reader.readCount(strideCount, "tensor_desc stride_count"))
    return false;
  if (strideCount != 0 && strideCount != rank)
    return reader.fail(strideCountAt,
                       "tensor_desc stride_count must be 0 or the rank");
  out.strides.clear();
  out.strides.reserve(strideCount);
  for (uint32_t i = 0; i < strideCount; ++i) {
    ExprStream stride;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, stride))
      return false;
    out.strides.push_back(std::move(stride));
  }

  if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, out.offset))
    return false;

  return parseElementType(reader, out.elementType);
}

/// Sections 3..12 of the v0 layout (also the v1 layout body): src/dst
/// descriptors, perm, axes, pad_fill, divisibility, alignment, contiguity,
/// flags, inverse. `plan.symbols` must already hold the symbol table.
bool parseLayoutBody(ByteReader &reader, RelocationPlan &plan) {
  const size_t symbolCount = plan.symbols.size();

  // 3-4: src, dst descriptors.
  if (!parseTensorDesc(reader, symbolCount, plan.src))
    return false;
  if (!parseTensorDesc(reader, symbolCount, plan.dst))
    return false;

  // 5: perm.
  size_t permAt = reader.offset();
  uint32_t permCount;
  if (!reader.readCount(permCount, "perm count"))
    return false;
  plan.perm.reserve(permCount);
  for (uint32_t i = 0; i < permCount; ++i) {
    uint32_t value;
    if (!reader.readU32(value))
      return false;
    plan.perm.push_back(value);
  }

  // 6: axes.
  uint32_t axisCount;
  if (!reader.readCount(axisCount, "axes count"))
    return false;
  plan.axes.reserve(axisCount);
  for (uint32_t i = 0; i < axisCount; ++i) {
    Axis axis;
    if (!reader.readString(axis.name))
      return false;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.extent))
      return false;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.srcStride))
      return false;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.dstStride))
      return false;
    plan.axes.push_back(std::move(axis));
  }

  // Cross-check: perm must be a size-matching bijection on [0, axes.size()).
  // axes.size() is only known now (axes are encoded after perm), so this
  // check is deferred to here but reported at perm's recorded section
  // offset, like the attribute verifier's equivalent check.
  if (plan.perm.size() != plan.axes.size())
    return reader.fail(permAt, "perm size must equal the axis count");
  {
    std::vector<bool> seen(plan.axes.size(), false);
    for (uint32_t value : plan.perm) {
      if (value >= plan.axes.size() || seen[value])
        return reader.fail(permAt,
                           "perm is not a permutation of [0, axis count)");
      seen[value] = true;
    }
  }

  // 7: pad_fill.
  uint32_t padCount;
  if (!reader.readCount(padCount, "pad_fill count"))
    return false;
  plan.padFill.reserve(padCount);
  for (uint32_t i = 0; i < padCount; ++i) {
    PadFill pad;
    size_t dstAxisAt = reader.offset();
    if (!reader.readU32(pad.dstAxis))
      return false;
    if (pad.dstAxis >= plan.axes.size())
      return reader.fail(dstAxisAt, "pad_fill dst_axis out of range");
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, pad.lo))
      return false;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, pad.hi))
      return false;
    if (!parseElementType(reader, pad.fillType))
      return false;
    if (!reader.readU64(pad.fillBits))
      return false;
    plan.padFill.push_back(std::move(pad));
  }

  // 8: divisibility.
  uint32_t divCount;
  if (!reader.readCount(divCount, "divisibility count"))
    return false;
  plan.divisibility.reserve(divCount);
  for (uint32_t i = 0; i < divCount; ++i) {
    Divisibility entry;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, entry.expr))
      return false;
    if (!reader.readI64(entry.divisor))
      return false;
    plan.divisibility.push_back(std::move(entry));
  }

  // 9: alignment.
  uint32_t alignCount;
  if (!reader.readCount(alignCount, "alignment count"))
    return false;
  plan.alignment.reserve(alignCount);
  for (uint32_t i = 0; i < alignCount; ++i) {
    Alignment entry;
    size_t axisAt = reader.offset();
    if (!reader.readU32(entry.axis))
      return false;
    if (entry.axis >= plan.axes.size())
      return reader.fail(axisAt, "alignment axis out of range");
    if (!reader.readI64(entry.bytes))
      return false;
    plan.alignment.push_back(entry);
  }

  // 10: contiguity.
  size_t contiguityAt = reader.offset();
  uint32_t contiguityCount;
  if (!reader.readCount(contiguityCount, "contiguity count"))
    return false;
  if (contiguityCount != 0 && contiguityCount != plan.axes.size())
    return reader.fail(contiguityAt,
                       "contiguity count must be 0 or the axis count");
  plan.contiguity.reserve(contiguityCount);
  for (uint32_t i = 0; i < contiguityCount; ++i) {
    size_t at = reader.offset();
    uint8_t flag;
    if (!reader.readU8(flag))
      return false;
    if (flag > 1)
      return reader.fail(at, "contiguity flag must be 0 or 1");
    plan.contiguity.push_back(flag);
  }

  // 11: flags.
  {
    size_t noCopyAt = reader.offset();
    uint8_t noCopy;
    if (!reader.readU8(noCopy))
      return false;
    if (noCopy > 1)
      return reader.fail(noCopyAt, "no_copy flag must be 0 or 1");
    size_t runtimePadCheckAt = reader.offset();
    uint8_t runtimePadCheck;
    if (!reader.readU8(runtimePadCheck))
      return false;
    if (runtimePadCheck > 1)
      return reader.fail(runtimePadCheckAt,
                         "runtime_pad_check flag must be 0 or 1");
    plan.noCopy = noCopy != 0;
    plan.runtimePadCheck = runtimePadCheck != 0;
  }

  // 12: inverse.
  {
    size_t numDimsAt = reader.offset();
    uint32_t numDims;
    if (!reader.readU32(numDims))
      return false;
    size_t numResultsAt = reader.offset();
    uint32_t numResults;
    if (!reader.readU32(numResults))
      return false;
    if (numDims != plan.axes.size())
      return reader.fail(numDimsAt,
                         "inverse num_dims must equal the axis count");
    if (numResults != plan.axes.size())
      return reader.fail(numResultsAt,
                         "inverse num_results must equal the axis count");
    plan.inverse.reserve(numResults);
    for (uint32_t i = 0; i < numResults; ++i) {
      ExprStream result;
      if (!parseExpr(reader, ExprContext::Inverse, symbolCount, numDims,
                     result))
        return false;
      plan.inverse.push_back(std::move(result));
    }
  }
  return true;
}

/// Sections 0..2: magic, the expected version, the symbol table.
bool parseHeader(ByteReader &reader, const uint8_t *data, uint32_t expected,
                 std::vector<std::string> &symbols) {
  if (reader.remaining() < 4)
    return reader.failHere("truncated: expected magic");
  if (std::memcmp(data, "RPLN", 4) != 0)
    return reader.fail(0, "bad magic (expected RPLN)");
  if (!reader.skip(4))
    return false;
  size_t versionAt = reader.offset();
  uint32_t version;
  if (!reader.readU32(version))
    return false;
  if (version != expected)
    return reader.fail(versionAt, "unsupported wire format version");
  uint32_t symbolCount;
  if (!reader.readCount(symbolCount, "symbol table count"))
    return false;
  symbols.reserve(symbolCount);
  for (uint32_t i = 0; i < symbolCount; ++i) {
    std::string name;
    if (!reader.readString(name))
      return false;
    symbols.push_back(std::move(name));
  }
  return true;
}

//===----------------------------------------------------------------------===//
// v1: typed plans
//===----------------------------------------------------------------------===//

bool isFloatWidth(ElementType type, uint32_t width) {
  return type.kind == ElementTypeKind::Float && type.bitwidth == width;
}
bool isInt8(ElementType type) {
  return type.kind == ElementTypeKind::Integer && type.bitwidth == 8;
}

bool parseStageType(ByteReader &reader, StageType &out) {
  if (!parseElementType(reader, out.type))
    return false;
  size_t at = reader.offset();
  uint8_t signedness;
  if (!reader.readU8(signedness))
    return false;
  if (signedness > 2)
    return reader.fail(at, "invalid stage type signedness");
  out.signedness = static_cast<Signedness>(signedness);
  return true;
}

bool parseParam(ByteReader &reader, size_t symbolCount, StageParam &out) {
  size_t at = reader.offset();
  uint8_t kind;
  if (!reader.readU8(kind))
    return false;
  if (kind > 2)
    return reader.fail(at, "invalid parameter kind");
  out.kind = static_cast<ParamKind>(kind);
  if (out.kind == ParamKind::None)
    return true;
  if (out.kind == ParamKind::Inline) {
    size_t rankAt = reader.offset();
    uint8_t rank;
    if (!reader.readU8(rank))
      return false;
    if (rank > 1)
      return reader.fail(rankAt, "parameter rank must be 0 or 1");
    out.rank = rank;
    if (!parseElementType(reader, out.elementType))
      return false;
    size_t countAt = reader.offset();
    uint32_t count;
    if (!reader.readCount(count, "inline parameter count"))
      return false;
    if (count == 0)
      return reader.fail(countAt, "inline parameter must hold a value");
    if (rank == 0 && count != 1)
      return reader.fail(countAt,
                         "rank-0 inline parameter must hold exactly one value");
    if (count > reader.remaining() / 8)
      return reader.fail(countAt,
                         "count exceeds remaining bytes: inline parameter");
    out.inlineBits.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      uint64_t bits;
      if (!reader.readU64(bits))
        return false;
      out.inlineBits.push_back(bits);
    }
    return true;
  }
  size_t nameAt = reader.offset();
  if (!reader.readString(out.bindingName))
    return false;
  if (out.bindingName.empty())
    return reader.fail(nameAt, "binding name must not be empty");
  size_t rankAt = reader.offset();
  uint8_t rank;
  if (!reader.readU8(rank))
    return false;
  if (rank > 1)
    return reader.fail(rankAt, "parameter rank must be 0 or 1");
  out.rank = rank;
  for (uint8_t i = 0; i < rank; ++i) {
    ExprStream extent;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, extent))
      return false;
    out.bindingExtents.push_back(std::move(extent));
  }
  return parseElementType(reader, out.elementType);
}

bool parseStage(ByteReader &reader, size_t symbolCount, size_t resultRank,
                ValueStage &out) {
  size_t at = reader.offset();
  uint8_t transform, policy;
  if (!reader.readU8(transform))
    return false;
  if (transform > 2)
    return reader.fail(at, "unknown stage transform");
  out.transform = static_cast<ValueTransformKind>(transform);
  size_t policyAt = reader.offset();
  if (!reader.readU8(policy))
    return false;
  if (policy > 3)
    return reader.fail(policyAt, "unknown numerical policy");
  out.policy = static_cast<NumericPolicyKind>(policy);
  if (!parseStageType(reader, out.input) || !parseStageType(reader, out.output))
    return false;
  size_t rankAt = reader.offset();
  uint32_t rank;
  if (!reader.readCount(rank, "stage shape rank"))
    return false;
  if (rank == 0)
    return reader.fail(rankAt, "stage shape must have rank >= 1");
  out.shape.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    ExprStream extent;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, extent))
      return false;
    out.shape.push_back(std::move(extent));
  }
  if (!parseParam(reader, symbolCount, out.scale) ||
      !parseParam(reader, symbolCount, out.zeroPoint))
    return false;
  if (!reader.readI64(out.axis))
    return false;
  size_t channelAt = reader.offset();
  uint8_t hasChannel;
  if (!reader.readU8(hasChannel))
    return false;
  if (hasChannel > 1)
    return reader.fail(channelAt, "has_channel flag must be 0 or 1");
  out.hasChannel = hasChannel != 0;
  if (out.hasChannel) {
    size_t dimsAt = reader.offset();
    uint32_t numDims;
    if (!reader.readU32(numDims))
      return false;
    if (numDims != resultRank)
      return reader.fail(dimsAt,
                         "channel map num_dims must equal the result rank");
    if (!parseExpr(reader, ExprContext::Channel, symbolCount, numDims,
                   out.channel))
      return false;
  }
  return true;
}

/// A single-constant stream's value, if the stream is exactly PUSH_CONST.
std::optional<int64_t> constantOf(const ExprStream &stream) {
  if (stream.size() == 1 && stream[0].op == ExprOp::PushConst)
    return stream[0].value;
  return std::nullopt;
}

bool sameStream(const ExprStream &lhs, const ExprStream &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i)
    if (lhs[i].op != rhs[i].op || lhs[i].value != rhs[i].value)
      return false;
  return true;
}

bool sameElementType(ElementType lhs, ElementType rhs) {
  return lhs.kind == rhs.kind && lhs.bitwidth == rhs.bitwidth;
}

int64_t signExtendBits(uint64_t bits, uint32_t bitwidth) {
  if (bitwidth >= 64)
    return static_cast<int64_t>(bits);
  const uint64_t mask = (uint64_t{1} << bitwidth) - 1;
  uint64_t value = bits & mask;
  if (value & (uint64_t{1} << (bitwidth - 1)))
    value |= ~mask;
  return static_cast<int64_t>(value);
}

bool isNaNBits(uint64_t bits, ElementType type) {
  if (type.kind != ElementTypeKind::Float)
    return false;
  if (type.bitwidth == 32)
    return ((bits >> 23) & 0xffu) == 0xffu && (bits & 0x7fffffu) != 0;
  if (type.bitwidth == 16)
    return ((bits >> 10) & 0x1fu) == 0x1fu && (bits & 0x3ffu) != 0;
  return false;
}

/// Semantic checks over a fully parsed typed plan (spec "Decoder-enforced
/// invariants"), reported at the recorded section/stage/fill offsets.
bool validateTyped(ByteReader &reader, const TypedRelocationPlan &plan,
                   size_t layoutAt, size_t stagesAt,
                   const std::vector<size_t> &stageAt, size_t fillsAt,
                   const std::vector<size_t> &fillAt) {
  const ElementType sourceType = plan.source.elementType;
  const ElementType resultType = plan.result.elementType;
  if (!sameElementType(plan.layout.src.elementType, sourceType))
    return reader.fail(layoutAt,
                       "layout src element type must equal the source type");
  if (!sameElementType(plan.layout.dst.elementType, resultType))
    return reader.fail(layoutAt,
                       "layout dst element type must equal the result type");
  if (plan.layout.src.extents.size() != plan.source.extents.size())
    return reader.fail(layoutAt, "layout src rank must equal the source rank");
  if (plan.stages.empty())
    return reader.fail(stagesAt, "typed plan needs at least one stage");

  // Type chain.
  if (!sameElementType(plan.stages.front().input.type, sourceType))
    return reader.fail(stageAt[0],
                       "stage 0 input type must equal the source type");
  for (size_t k = 1; k < plan.stages.size(); ++k)
    if (!sameElementType(plan.stages[k].input.type,
                         plan.stages[k - 1].output.type))
      return reader.fail(stageAt[k], "stage input type must equal the "
                                     "previous stage's output type");
  if (!sameElementType(plan.stages.back().output.type, resultType))
    return reader.fail(stageAt.back(),
                       "last stage output type must equal the result type");

  // Per-stage signature and parameters.
  std::vector<const StageParam *> declared; // bindings seen so far
  std::vector<size_t> declaredAt;
  for (size_t k = 0; k < plan.stages.size(); ++k) {
    const ValueStage &stage = plan.stages[k];
    const size_t at = stageAt[k];
    const bool signlessIO = stage.input.signedness == Signedness::Signless &&
                            stage.output.signedness == Signedness::Signless;
    switch (stage.transform) {
    case ValueTransformKind::Cast: {
      const bool narrow = isFloatWidth(stage.input.type, 32) &&
                          isFloatWidth(stage.output.type, 16) &&
                          stage.policy == NumericPolicyKind::IeeeRne;
      const bool widen = isFloatWidth(stage.input.type, 16) &&
                         isFloatWidth(stage.output.type, 32) &&
                         stage.policy == NumericPolicyKind::Exact;
      if (!(narrow || widen) || !signlessIO)
        return reader.fail(at, "invalid cast signature (f32 -> f16 ieee_rne "
                               "or f16 -> f32 exact, signless)");
      if (stage.scale.kind != ParamKind::None ||
          stage.zeroPoint.kind != ParamKind::None || stage.axis != -1 ||
          stage.hasChannel)
        return reader.fail(at, "cast stages carry no parameters");
      continue;
    }
    case ValueTransformKind::Quantize:
      if (!isFloatWidth(stage.input.type, 32) ||
          stage.input.signedness != Signedness::Signless ||
          !isInt8(stage.output.type) ||
          stage.output.signedness != Signedness::Signed ||
          stage.policy != NumericPolicyKind::SymmetricRne)
        return reader.fail(at, "invalid quantize signature (f32 -> signed "
                               "int 8, symmetric_rne)");
      break;
    case ValueTransformKind::Dequantize:
      if (!isInt8(stage.input.type) ||
          stage.input.signedness != Signedness::Signed ||
          !isFloatWidth(stage.output.type, 32) ||
          stage.output.signedness != Signedness::Signless ||
          stage.policy != NumericPolicyKind::Affine)
        return reader.fail(at, "invalid dequantize signature (signed int 8 "
                               "-> f32, affine)");
      break;
    }
    // Quantize / dequantize parameters.
    if (stage.scale.kind == ParamKind::None)
      return reader.fail(at, "quantize/dequantize stages require a scale");
    if (!isFloatWidth(stage.scale.elementType, 32))
      return reader.fail(at, "scale element type must be f32");
    if (stage.zeroPoint.kind != ParamKind::None) {
      if (stage.zeroPoint.elementType.kind != ElementTypeKind::Integer)
        return reader.fail(at, "zero_point element type must be an integer");
      if (stage.zeroPoint.kind == ParamKind::Binding &&
          stage.zeroPoint.elementType.bitwidth != 32)
        return reader.fail(at, "zero_point bindings must declare int 32");
    }
    if (stage.axis == -1) {
      if (stage.scale.rank != 0 ||
          (stage.zeroPoint.kind != ParamKind::None &&
           stage.zeroPoint.rank != 0) ||
          stage.hasChannel)
        return reader.fail(at, "per-tensor stage requires rank-0 parameters "
                               "and no channel map");
    } else {
      if (stage.axis < 0 ||
          static_cast<uint64_t>(stage.axis) >= stage.shape.size())
        return reader.fail(at, "axis out of range for the stage shape");
      if (stage.scale.rank != 1)
        return reader.fail(at, "per-channel stage requires a rank-1 scale");
      if (!stage.hasChannel)
        return reader.fail(at, "per-channel stage requires a channel map");
      std::optional<int64_t> extent = constantOf(stage.shape[stage.axis]);
      for (const StageParam *param : {&stage.scale, &stage.zeroPoint})
        if (param->kind == ParamKind::Inline && param->rank == 1 && extent &&
            static_cast<int64_t>(param->inlineBits.size()) != *extent)
          return reader.fail(at, "inline parameter length must equal the "
                                 "channel extent");
    }
    if (stage.scale.kind == ParamKind::Inline)
      for (uint64_t bits : stage.scale.inlineBits) {
        uint32_t narrow = static_cast<uint32_t>(bits);
        float value;
        std::memcpy(&value, &narrow, sizeof(value));
        if (!std::isfinite(value) || !(value > 0.0f))
          return reader.fail(
              at, "inline scale must be finite and strictly positive");
      }
    if (stage.zeroPoint.kind == ParamKind::Inline)
      for (uint64_t bits : stage.zeroPoint.inlineBits) {
        int64_t value =
            signExtendBits(bits, stage.zeroPoint.elementType.bitwidth);
        if (value < -128 || value > 127)
          return reader.fail(at, "inline zero point out of [-128, 127]");
        if (stage.policy == NumericPolicyKind::SymmetricRne && value != 0)
          return reader.fail(at, "symmetric_rne admits only the constant zero "
                                 "point 0");
      }
    if (stage.policy == NumericPolicyKind::SymmetricRne &&
        stage.zeroPoint.kind == ParamKind::Binding)
      return reader.fail(at, "symmetric_rne rejects a zero-point binding");
    if (stage.scale.kind == ParamKind::Binding &&
        stage.zeroPoint.kind == ParamKind::Binding &&
        stage.scale.bindingName == stage.zeroPoint.bindingName)
      return reader.fail(at,
                         "runtime parameters must use distinct binding names");
    // Binding declarations must agree across stages.
    for (const StageParam *param : {&stage.scale, &stage.zeroPoint}) {
      if (param->kind != ParamKind::Binding)
        continue;
      for (const StageParam *seen : declared) {
        if (seen->bindingName != param->bindingName)
          continue;
        bool same = seen->rank == param->rank &&
                    sameElementType(seen->elementType, param->elementType) &&
                    seen->bindingExtents.size() == param->bindingExtents.size();
        for (size_t e = 0; same && e < seen->bindingExtents.size(); ++e)
          same = sameStream(seen->bindingExtents[e], param->bindingExtents[e]);
        if (!same)
          return reader.fail(at, "conflicting parameter declaration for "
                                 "binding '" +
                                     param->bindingName + "'");
      }
      declared.push_back(param);
      declaredAt.push_back(at);
    }
  }

  // Fills resolve to layout pads, enter at a real stage with that stage's
  // type, and fold to the fused fill.
  std::vector<bool> padHasFill(plan.layout.padFill.size(), false);
  for (size_t i = 0; i < plan.fills.size(); ++i) {
    const TypedFill &fill = plan.fills[i];
    const size_t at = fillAt[i];
    const PadFill *pad = nullptr;
    for (size_t p = 0; p < plan.layout.padFill.size(); ++p)
      if (plan.layout.padFill[p].dstAxis == fill.dstAxis) {
        if (padHasFill[p])
          return reader.fail(at, "duplicate fill for a layout pad");
        padHasFill[p] = true;
        pad = &plan.layout.padFill[p];
      }
    if (!pad)
      return reader.fail(at, "fill dst_axis has no layout pad");
    if (fill.stage > plan.stages.size())
      return reader.fail(at, "fill stage out of range");
    ElementType entry =
        fill.stage == 0 ? sourceType : plan.stages[fill.stage - 1].output.type;
    if (!sameElementType(fill.type, entry))
      return reader.fail(at,
                         "fill type must equal the type entering its stage");
    if (!sameElementType(pad->fillType, resultType))
      return reader.fail(at, "layout pad fill type must equal the result type");
    uint64_t bits = fill.bits;
    ElementType type = fill.type;
    for (size_t k = fill.stage; k < plan.stages.size(); ++k) {
      std::optional<uint64_t> folded =
          typed::foldFill(bits, type, plan.stages[k]);
      if (!folded)
        return reader.fail(at, "fill cannot fold through a per-channel or "
                               "runtime-parameter stage");
      bits = *folded;
      type = plan.stages[k].output.type;
    }
    const bool bothNaN =
        isNaNBits(bits, resultType) && isNaNBits(pad->fillBits, resultType);
    if (bits != pad->fillBits && !bothNaN)
      return reader.fail(at, "fill fold mismatch with the layout's fused fill");
  }
  for (size_t p = 0; p < padHasFill.size(); ++p)
    if (!padHasFill[p])
      return reader.fail(fillsAt, "layout pad without a typed fill");
  return true;
}

} // namespace

std::optional<uint32_t> peekWireVersion(const uint8_t *data, size_t size) {
  if (size < 8 || std::memcmp(data, "RPLN", 4) != 0)
    return std::nullopt;
  return static_cast<uint32_t>(data[4]) | static_cast<uint32_t>(data[5]) << 8 |
         static_cast<uint32_t>(data[6]) << 16 |
         static_cast<uint32_t>(data[7]) << 24;
}

DecodeResult decodePlan(const uint8_t *data, size_t size) {
  ByteReader reader(data, size);
  RelocationPlan plan;
  if (!parseHeader(reader, data, kWireFormatVersion, plan.symbols))
    return reader.error();
  if (!parseLayoutBody(reader, plan))
    return reader.error();
  if (reader.remaining() != 0) {
    reader.failHere("trailing bytes after inverse section");
    return reader.error();
  }
  return plan;
}

TypedDecodeResult decodeTypedPlan(const uint8_t *data, size_t size) {
  ByteReader reader(data, size);
  TypedRelocationPlan plan;
  if (!parseHeader(reader, data, kTypedWireFormatVersion, plan.symbols))
    return reader.error();
  const size_t symbolCount = plan.symbols.size();

  // 3-4: logical source and result descriptors.
  if (!parseTensorDesc(reader, symbolCount, plan.source) ||
      !parseTensorDesc(reader, symbolCount, plan.result))
    return reader.error();

  // 5: the layout body, sharing the symbol table.
  size_t layoutAt = reader.offset();
  plan.layout.symbols = plan.symbols;
  if (!parseLayoutBody(reader, plan.layout))
    return reader.error();

  // 6: stages.
  size_t stagesAt = reader.offset();
  uint32_t stageCount;
  if (!reader.readCount(stageCount, "stage count"))
    return reader.error();
  std::vector<size_t> stageAt;
  plan.stages.reserve(stageCount);
  for (uint32_t i = 0; i < stageCount; ++i) {
    stageAt.push_back(reader.offset());
    ValueStage stage;
    if (!parseStage(reader, symbolCount, plan.result.extents.size(), stage))
      return reader.error();
    plan.stages.push_back(std::move(stage));
  }

  // 7: fills.
  size_t fillsAt = reader.offset();
  uint32_t fillCount;
  if (!reader.readCount(fillCount, "fill count"))
    return reader.error();
  std::vector<size_t> fillAt;
  plan.fills.reserve(fillCount);
  for (uint32_t i = 0; i < fillCount; ++i) {
    fillAt.push_back(reader.offset());
    TypedFill fill;
    if (!reader.readU32(fill.dstAxis) || !reader.readU32(fill.stage) ||
        !parseElementType(reader, fill.type) || !reader.readU64(fill.bits))
      return reader.error();
    plan.fills.push_back(fill);
  }

  if (reader.remaining() != 0) {
    reader.failHere("trailing bytes after fills section");
    return reader.error();
  }
  if (!validateTyped(reader, plan, layoutAt, stagesAt, stageAt, fillsAt,
                     fillAt))
    return reader.error();
  return plan;
}

} // namespace reloc
