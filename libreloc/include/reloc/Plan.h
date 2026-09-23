//===- Plan.h - MLIR-free mirror of #reloc.plan -----------------*- C++ -*-===//
//
// RelocationPlan mirrors the wire-format-v0 sections one-to-one
// (docs/reloc-plan-format.md). Expressions are stored as the postfix
// opcode streams from the wire — #C3's stack machine evaluates streams
// directly; there is deliberately no AST.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PLAN_H
#define RELOC_PLAN_H

#include <cstdint>
#include <string>
#include <vector>

namespace reloc {

/// Wire opcodes (spec "Expressions"). Values match the wire encoding.
enum class ExprOp : uint8_t {
  PushSym = 0x00,
  PushConst = 0x01,
  Add = 0x02,
  Sub = 0x03,
  Mul = 0x04,
  FloorDiv = 0x05,
  Mod = 0x06,
  PushDim = 0x07,
};

/// One decoded operation. `value` holds the inline operand: the symbol
/// index for PushSym, the constant for PushConst, the dst-axis coordinate
/// index for PushDim; 0 for the binary operators.
struct ExprToken {
  ExprOp op;
  int64_t value;
};

/// A postfix stream; decode-time validation guarantees it leaves exactly
/// one value on the evaluation stack.
using ExprStream = std::vector<ExprToken>;

/// Wire element-type kinds (spec "Element types").
enum class ElementTypeKind : uint8_t {
  Float = 0,
  BFloat = 1,
  Integer = 2,
  Index = 3,
};

struct ElementType {
  ElementTypeKind kind;
  uint32_t bitwidth;
};

/// Symbolic tensor descriptor (spec "Tensor descriptor"). Empty `strides`
/// means canonical row-major.
struct TensorDesc {
  std::vector<ExprStream> extents;
  std::vector<ExprStream> strides;
  ExprStream offset;
  ElementType elementType;
};

/// One plan axis in destination order.
struct Axis {
  std::string name;
  ExprStream extent;
  ExprStream srcStride;
  ExprStream dstStride;
};

/// Pad-fill spec; `fillBits` is the value's bit pattern zero-extended to
/// 64 bits (spec "Typed value").
struct PadFill {
  uint32_t dstAxis;
  ExprStream lo;
  ExprStream hi;
  ElementType fillType;
  uint64_t fillBits;
};

struct Divisibility {
  ExprStream expr;
  int64_t divisor;
};

struct Alignment {
  uint32_t axis;
  int64_t bytes;
};

/// The decoded plan. Field order mirrors the wire section order.
struct RelocationPlan {
  std::vector<std::string> symbols;
  TensorDesc src;
  TensorDesc dst;
  std::vector<uint32_t> perm;
  std::vector<Axis> axes;
  std::vector<PadFill> padFill;
  std::vector<Divisibility> divisibility;
  std::vector<Alignment> alignment;
  std::vector<uint8_t> contiguity; // empty, or one 0/1 flag per axis
  bool noCopy = false;
  bool runtimePadCheck = false;
  std::vector<ExprStream> inverse; // one stream per dst axis (PushDim only)
};

//===----------------------------------------------------------------------===//
// Wire format v1: typed plans (C3, issue #143; spec "Wire Format v1")
//===----------------------------------------------------------------------===//

/// Stage transform / policy / signedness bytes. Values match the wire.
enum class ValueTransformKind : uint8_t {
  Cast = 0,
  Quantize = 1,
  Dequantize = 2
};
enum class NumericPolicyKind : uint8_t {
  IeeeRne = 0,
  Exact = 1,
  SymmetricRne = 2,
  Affine = 3,
};
enum class Signedness : uint8_t { Signless = 0, Signed = 1, Unsigned = 2 };

/// A stage's element type with its semantic signedness (storage stays
/// signless in the descriptors; quantize output / dequantize input are
/// `Signed` int 8 by operation semantics).
struct StageType {
  ElementType type;
  Signedness signedness = Signedness::Signless;
};

/// A quantization parameter as declared: absent, an inline constant (exact
/// bits), or a runtime binding identified by name.
enum class ParamKind : uint8_t { None = 0, Inline = 1, Binding = 2 };

struct StageParam {
  ParamKind kind = ParamKind::None;
  uint8_t rank = 0;                       // 0 per tensor, 1 per channel
  ElementType elementType{};              // f32 scales, integer zero points
  std::vector<uint64_t> inlineBits;       // Inline: rank-0 => one value
  std::string bindingName;                // Binding
  std::vector<ExprStream> bindingExtents; // Binding: `rank` streams
};

/// One typed value stage (spec section 6).
struct ValueStage {
  ValueTransformKind transform = ValueTransformKind::Cast;
  NumericPolicyKind policy = NumericPolicyKind::IeeeRne;
  StageType input;
  StageType output;
  std::vector<ExprStream> shape; // logical operand shape
  StageParam scale;
  StageParam zeroPoint;
  int64_t axis = -1; // channel axis of `shape`; -1 per tensor
  bool hasChannel = false;
  ExprStream channel; // channel context: PushDim = result coordinate
};

/// A pad fill as the program wrote it, with the stage it entered at.
struct TypedFill {
  uint32_t dstAxis = 0;
  uint32_t stage = 0;
  ElementType type{};
  uint64_t bits = 0;
};

/// The decoded typed plan. `layout.symbols` is the same table as `symbols`.
struct TypedRelocationPlan {
  std::vector<std::string> symbols;
  TensorDesc source;
  TensorDesc result;
  RelocationPlan layout;
  std::vector<ValueStage> stages;
  std::vector<TypedFill> fills;
};

} // namespace reloc

#endif // RELOC_PLAN_H
