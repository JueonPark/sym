//===- TypedValue.cpp - runtime reference arithmetic for typed stages -----===//

#include "reloc/TypedValue.h"

#include "reloc/Quant.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace reloc {
namespace typed {

uint32_t byteWidth(ElementType type) {
  if (type.bitwidth == 0 || type.bitwidth % 8 != 0)
    return 0;
  return type.bitwidth / 8;
}

bool sameType(ElementType lhs, ElementType rhs) {
  return lhs.kind == rhs.kind && lhs.bitwidth == rhs.bitwidth;
}

namespace {

bool isF32(ElementType type) {
  return type.kind == ElementTypeKind::Float && type.bitwidth == 32;
}
bool isF16(ElementType type) {
  return type.kind == ElementTypeKind::Float && type.bitwidth == 16;
}
bool isI8(ElementType type) {
  return type.kind == ElementTypeKind::Integer && type.bitwidth == 8;
}

float f32FromBits(uint64_t bits) {
  uint32_t narrow = static_cast<uint32_t>(bits);
  float value;
  std::memcpy(&value, &narrow, sizeof(value));
  return value;
}

uint64_t bitsFromF32(float value) {
  uint32_t narrow;
  std::memcpy(&narrow, &value, sizeof(narrow));
  return narrow;
}

int64_t signExtend(uint64_t bits, uint32_t bitwidth) {
  if (bitwidth >= 64)
    return static_cast<int64_t>(bits);
  const uint64_t mask = (uint64_t{1} << bitwidth) - 1;
  uint64_t value = bits & mask;
  if (value & (uint64_t{1} << (bitwidth - 1)))
    value |= ~mask;
  return static_cast<int64_t>(value);
}

} // namespace

float scaleAt(const StageParam &param, size_t index) {
  assert(param.kind == ParamKind::Inline && index < param.inlineBits.size());
  return f32FromBits(param.inlineBits[index]);
}

int64_t zeroPointAt(const StageParam &param, size_t index) {
  assert(param.kind == ParamKind::Inline && index < param.inlineBits.size());
  return signExtend(param.inlineBits[index], param.elementType.bitwidth);
}

std::optional<uint64_t> foldFill(uint64_t bits, ElementType type,
                                 const ValueStage &stage) {
  if (!sameType(type, stage.input.type))
    return std::nullopt;
  switch (stage.transform) {
  case ValueTransformKind::Cast:
    if (isF32(type) && isF16(stage.output.type))
      return static_cast<uint64_t>(quant::narrowF32F16(f32FromBits(bits)));
    if (isF16(type) && isF32(stage.output.type))
      return bitsFromF32(quant::widenF16F32(static_cast<uint16_t>(bits)));
    return std::nullopt;
  case ValueTransformKind::Quantize: {
    if (!isF32(type) || !isI8(stage.output.type) || stage.axis >= 0 ||
        stage.scale.kind != ParamKind::Inline || stage.scale.rank != 0 ||
        stage.scale.inlineBits.empty())
      return std::nullopt;
    // symmetric_rne (docs/reloc-typed-semantics.md §3.3): the reciprocal is
    // formed once in binary32, then the kernel contract applies.
    const float inv = 1.0f / scaleAt(stage.scale, 0);
    const int8_t q = quant::quantizeOneF32S8(f32FromBits(bits), inv);
    return static_cast<uint64_t>(static_cast<uint8_t>(q));
  }
  case ValueTransformKind::Dequantize: {
    if (!isI8(type) || !isF32(stage.output.type) || stage.axis >= 0 ||
        stage.scale.kind != ParamKind::Inline || stage.scale.rank != 0 ||
        stage.scale.inlineBits.empty())
      return std::nullopt;
    int64_t zeroPoint = 0;
    if (stage.zeroPoint.kind == ParamKind::Inline) {
      if (stage.zeroPoint.rank != 0 || stage.zeroPoint.inlineBits.empty())
        return std::nullopt;
      zeroPoint = zeroPointAt(stage.zeroPoint, 0);
    } else if (stage.zeroPoint.kind != ParamKind::None) {
      return std::nullopt;
    }
    // affine (§3.4): d = q - zp exactly, y = fl32(d * scale).
    const int64_t q = signExtend(bits, 8);
    const float y = static_cast<float>(q - zeroPoint) * scaleAt(stage.scale, 0);
    return bitsFromF32(y);
  }
  }
  return std::nullopt;
}

} // namespace typed
} // namespace reloc
