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

} // namespace reloc

#endif // RELOC_PLAN_H
