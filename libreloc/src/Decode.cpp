//===- Decode.cpp - wire-format v0 decoder --------------------------------===//

#include "reloc/Decode.h"
#include "reloc/Version.h"

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

enum class ExprContext { Plan, Inverse };

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
      if (context != ExprContext::Inverse)
        return reader.fail(opAt,
                           "PUSH_DIM is only allowed in inverse expressions");
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

} // namespace

DecodeResult decodePlan(const uint8_t *data, size_t size) {
  ByteReader reader(data, size);
  RelocationPlan plan;

  // 0-1: magic + version.
  {
    if (reader.remaining() < 4) {
      reader.failHere("truncated: expected magic");
      return reader.error();
    }
    if (std::memcmp(data, "RPLN", 4) != 0) {
      reader.fail(0, "bad magic (expected RPLN)");
      return reader.error();
    }
    if (!reader.skip(4))
      return reader.error();
    size_t versionAt = reader.offset();
    uint32_t version;
    if (!reader.readU32(version))
      return reader.error();
    if (version != kWireFormatVersion) {
      reader.fail(versionAt, "unsupported wire format version");
      return reader.error();
    }
  }

  // 2: symbol table.
  uint32_t symbolCount;
  if (!reader.readCount(symbolCount, "symbol table count"))
    return reader.error();
  plan.symbols.reserve(symbolCount);
  for (uint32_t i = 0; i < symbolCount; ++i) {
    std::string name;
    if (!reader.readString(name))
      return reader.error();
    plan.symbols.push_back(std::move(name));
  }

  // 3-4: src, dst descriptors.
  if (!parseTensorDesc(reader, symbolCount, plan.src))
    return reader.error();
  if (!parseTensorDesc(reader, symbolCount, plan.dst))
    return reader.error();

  // 5: perm.
  size_t permAt = reader.offset();
  uint32_t permCount;
  if (!reader.readCount(permCount, "perm count"))
    return reader.error();
  plan.perm.reserve(permCount);
  for (uint32_t i = 0; i < permCount; ++i) {
    uint32_t value;
    if (!reader.readU32(value))
      return reader.error();
    plan.perm.push_back(value);
  }

  // 6: axes.
  uint32_t axisCount;
  if (!reader.readCount(axisCount, "axes count"))
    return reader.error();
  plan.axes.reserve(axisCount);
  for (uint32_t i = 0; i < axisCount; ++i) {
    Axis axis;
    if (!reader.readString(axis.name))
      return reader.error();
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.extent))
      return reader.error();
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.srcStride))
      return reader.error();
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, axis.dstStride))
      return reader.error();
    plan.axes.push_back(std::move(axis));
  }

  // Cross-check: perm must be a size-matching bijection on [0, axes.size()).
  // axes.size() is only known now (axes are encoded after perm), so this
  // check is deferred to here but reported at perm's recorded section
  // offset, like the attribute verifier's equivalent check.
  if (plan.perm.size() != plan.axes.size()) {
    reader.fail(permAt, "perm size must equal the axis count");
    return reader.error();
  }
  {
    std::vector<bool> seen(plan.axes.size(), false);
    for (uint32_t value : plan.perm) {
      if (value >= plan.axes.size() || seen[value]) {
        reader.fail(permAt, "perm is not a permutation of [0, axis count)");
        return reader.error();
      }
      seen[value] = true;
    }
  }

  // 7: pad_fill.
  uint32_t padCount;
  if (!reader.readCount(padCount, "pad_fill count"))
    return reader.error();
  plan.padFill.reserve(padCount);
  for (uint32_t i = 0; i < padCount; ++i) {
    PadFill pad;
    size_t dstAxisAt = reader.offset();
    if (!reader.readU32(pad.dstAxis))
      return reader.error();
    if (pad.dstAxis >= plan.axes.size()) {
      reader.fail(dstAxisAt, "pad_fill dst_axis out of range");
      return reader.error();
    }
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, pad.lo))
      return reader.error();
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, pad.hi))
      return reader.error();
    if (!parseElementType(reader, pad.fillType))
      return reader.error();
    if (!reader.readU64(pad.fillBits))
      return reader.error();
    plan.padFill.push_back(std::move(pad));
  }

  // 8: divisibility.
  uint32_t divCount;
  if (!reader.readCount(divCount, "divisibility count"))
    return reader.error();
  plan.divisibility.reserve(divCount);
  for (uint32_t i = 0; i < divCount; ++i) {
    Divisibility entry;
    if (!parseExpr(reader, ExprContext::Plan, symbolCount, 0, entry.expr))
      return reader.error();
    if (!reader.readI64(entry.divisor))
      return reader.error();
    plan.divisibility.push_back(std::move(entry));
  }

  // 9: alignment.
  uint32_t alignCount;
  if (!reader.readCount(alignCount, "alignment count"))
    return reader.error();
  plan.alignment.reserve(alignCount);
  for (uint32_t i = 0; i < alignCount; ++i) {
    Alignment entry;
    size_t axisAt = reader.offset();
    if (!reader.readU32(entry.axis))
      return reader.error();
    if (entry.axis >= plan.axes.size()) {
      reader.fail(axisAt, "alignment axis out of range");
      return reader.error();
    }
    if (!reader.readI64(entry.bytes))
      return reader.error();
    plan.alignment.push_back(entry);
  }

  // 10: contiguity.
  size_t contiguityAt = reader.offset();
  uint32_t contiguityCount;
  if (!reader.readCount(contiguityCount, "contiguity count"))
    return reader.error();
  if (contiguityCount != 0 && contiguityCount != plan.axes.size()) {
    reader.fail(contiguityAt, "contiguity count must be 0 or the axis count");
    return reader.error();
  }
  plan.contiguity.reserve(contiguityCount);
  for (uint32_t i = 0; i < contiguityCount; ++i) {
    size_t at = reader.offset();
    uint8_t flag;
    if (!reader.readU8(flag))
      return reader.error();
    if (flag > 1) {
      reader.fail(at, "contiguity flag must be 0 or 1");
      return reader.error();
    }
    plan.contiguity.push_back(flag);
  }

  // 11: flags.
  {
    size_t noCopyAt = reader.offset();
    uint8_t noCopy;
    if (!reader.readU8(noCopy))
      return reader.error();
    if (noCopy > 1) {
      reader.fail(noCopyAt, "no_copy flag must be 0 or 1");
      return reader.error();
    }
    size_t runtimePadCheckAt = reader.offset();
    uint8_t runtimePadCheck;
    if (!reader.readU8(runtimePadCheck))
      return reader.error();
    if (runtimePadCheck > 1) {
      reader.fail(runtimePadCheckAt, "runtime_pad_check flag must be 0 or 1");
      return reader.error();
    }
    plan.noCopy = noCopy != 0;
    plan.runtimePadCheck = runtimePadCheck != 0;
  }

  // 12: inverse.
  {
    size_t numDimsAt = reader.offset();
    uint32_t numDims;
    if (!reader.readU32(numDims))
      return reader.error();
    size_t numResultsAt = reader.offset();
    uint32_t numResults;
    if (!reader.readU32(numResults))
      return reader.error();
    if (numDims != plan.axes.size()) {
      reader.fail(numDimsAt, "inverse num_dims must equal the axis count");
      return reader.error();
    }
    if (numResults != plan.axes.size()) {
      reader.fail(numResultsAt,
                  "inverse num_results must equal the axis count");
      return reader.error();
    }
    plan.inverse.reserve(numResults);
    for (uint32_t i = 0; i < numResults; ++i) {
      ExprStream result;
      if (!parseExpr(reader, ExprContext::Inverse, symbolCount, numDims,
                     result))
        return reader.error();
      plan.inverse.push_back(std::move(result));
    }
  }

  // Trailing bytes.
  if (reader.remaining() != 0) {
    reader.failHere("trailing bytes after inverse section");
    return reader.error();
  }

  return plan;
}

} // namespace reloc
