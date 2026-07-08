//===- RelocOps.cpp - Reloc dialect operation implementation --------------===//
//
// Assembly, verification, and builders for the reloc op set. Result shapes
// are fully determined by operands and attributes; a shared compute*Type
// helper per op backs both the builder and the verifier.
//
//===----------------------------------------------------------------------===//

#include "RelocDialect.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::reloc;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Parse the trailing `:` input-type `->` result-type of every reloc op.
static ParseResult parseOpTypes(OpAsmParser &parser, Type &inputType,
                                Type &resultType) {
  return failure(parser.parseColon() || parser.parseType(inputType) ||
                 parser.parseArrow() || parser.parseType(resultType));
}

static void printOpTypes(OpAsmPrinter &printer, Type inputType,
                         Type resultType) {
  printer << " : " << inputType << " -> " << resultType;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

/// Result type of transposing `input` by `perm`: dim k = input dim perm[k].
/// Requires `perm` to be a valid permutation of the input rank.
static sym::SymbolicTensorType
computeTransposedType(sym::SymbolicTensorType input, ArrayRef<int64_t> perm) {
  assert(perm.size() == input.getShape().size() && "perm size must match rank");
  SmallVector<Attribute> shape;
  shape.reserve(perm.size());
  for (int64_t source : perm) {
    assert(source >= 0 &&
           source < static_cast<int64_t>(input.getShape().size()) &&
           "perm entry out of range");
    shape.push_back(input.getShape()[source]);
  }
  return sym::SymbolicTensorType::get(input.getContext(), shape,
                                      input.getElementType());
}

void TransposeOp::build(OpBuilder &builder, OperationState &state, Value input,
                        ArrayRef<int64_t> perm) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  build(builder, state, computeTransposedType(inputType, perm), input,
        builder.getDenseI64ArrayAttr(perm));
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SmallVector<int64_t> perm;
  if (parser.parseOperand(input) || parser.parseKeyword("perm") ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       int64_t value;
                                       if (parser.parseInteger(value))
                                         return failure();
                                       perm.push_back(value);
                                       return success();
                                     }))
    return failure();
  result.addAttribute(getPermAttrName(result.name),
                      parser.getBuilder().getDenseI64ArrayAttr(perm));
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void TransposeOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " perm [";
  llvm::interleaveComma(getPerm(), printer);
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getPermAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult TransposeOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  ArrayRef<int64_t> perm = getPerm();
  size_t rank = inputType.getShape().size();

  if (perm.size() != rank)
    return emitOpError() << "perm size (" << perm.size()
                         << ") must match operand rank (" << rank << ")";
  SmallVector<bool> seen(rank, false);
  for (int64_t value : perm) {
    if (value < 0 || value >= static_cast<int64_t>(rank) || seen[value])
      return emitOpError() << "perm is not a permutation of [0, " << rank
                           << ")";
    seen[value] = true;
  }
  if (resultType.getShape().size() != rank)
    return emitOpError() << "result rank (" << resultType.getShape().size()
                         << ") must match operand rank (" << rank << ")";
  for (size_t k = 0; k < rank; ++k)
    if (!sym::UnificationSolver::areLogicallyEqual(
            resultType.getShape()[k], inputType.getShape()[perm[k]]))
      return emitOpError() << "result dimension " << k
                           << " must equal operand dimension " << perm[k]
                           << ", but got " << resultType.getShape()[k] << " vs "
                           << inputType.getShape()[perm[k]];
  if (resultType.getElementType() != inputType.getElementType())
    return emitOpError() << "result element type must match operand element "
                            "type";
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

/// Product of a symbolic shape's dims, parse-style simplified.
static Attribute shapeProduct(ArrayRef<Attribute> shape, MLIRContext *ctx) {
  Attribute product = sym::ConstantExprAttr::get(ctx, 1);
  for (Attribute dim : shape)
    product = sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Mul,
                                           product, dim);
  return product;
}

void ReshapeOp::build(OpBuilder &builder, OperationState &state, Value input,
                      ArrayRef<Attribute> targetShape) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  auto resultType = sym::SymbolicTensorType::get(
      builder.getContext(), targetShape, inputType.getElementType());
  build(builder, state, resultType, input, builder.getArrayAttr(targetShape));
}

ParseResult ReshapeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  SmallVector<Attribute> targetShape;
  if (parser.parseOperand(input) || parser.parseKeyword("to") ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       Attribute dim = parseSymExpr(parser);
                                       if (!dim)
                                         return failure();
                                       targetShape.push_back(dim);
                                       return success();
                                     }))
    return failure();
  result.addAttribute(getTargetShapeAttrName(result.name),
                      parser.getBuilder().getArrayAttr(targetShape));
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void ReshapeOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " to [";
  llvm::interleaveComma(getTargetShape(), printer,
                        [&](Attribute dim) { printSymExpr(printer, dim); });
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getTargetShapeAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult ReshapeOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  ArrayRef<Attribute> target = getTargetShape().getValue();

  for (auto [index, dim] : llvm::enumerate(target))
    if (!isSymExpr(dim))
      return emitOpError() << "target_shape entry " << index
                           << " must be a sym expression, but got: " << dim;
  if (resultType.getShape().size() != target.size())
    return emitOpError() << "result rank (" << resultType.getShape().size()
                         << ") must match target_shape size (" << target.size()
                         << ")";
  for (size_t k = 0; k < target.size(); ++k)
    if (!sym::UnificationSolver::areLogicallyEqual(resultType.getShape()[k],
                                                   target[k]))
      return emitOpError() << "result dimension " << k
                           << " must equal target_shape entry " << k;
  if (resultType.getElementType() != inputType.getElementType())
    return emitOpError() << "result element type must match operand element "
                            "type";

  // Element-count consistency: reject only provable changes.
  MLIRContext *ctx = getContext();
  Attribute inputCount = shapeProduct(inputType.getShape(), ctx);
  Attribute targetCount = shapeProduct(target, ctx);
  if (proveEqual(inputCount, targetCount) == Proof::Disproven)
    return emitOpError() << "element count provably changes: operand has "
                         << inputCount << " elements, target has "
                         << targetCount;
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

/// The padded dimension: (dim + lo) + hi, parse-style simplified — the same
/// association order the plan verifier uses, so canonical forms match.
static Attribute paddedDim(Attribute dim, Attribute lo, Attribute hi,
                           MLIRContext *ctx) {
  return sym::getSimplifiedBinaryExpr(
      ctx, sym::SymbolicExprOp::Add,
      sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add, dim, lo), hi);
}

void PadOp::build(OpBuilder &builder, OperationState &state, Value input,
                  int64_t axis, Attribute lo, Attribute hi, TypedAttr value) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  SmallVector<Attribute> shape(inputType.getShape());
  assert(axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
         "pad axis out of range");
  shape[axis] = paddedDim(shape[axis], lo, hi, builder.getContext());
  auto resultType = sym::SymbolicTensorType::get(builder.getContext(), shape,
                                                 inputType.getElementType());
  build(builder, state, resultType, input, builder.getI64IntegerAttr(axis), lo,
        hi, value);
}

ParseResult PadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  int64_t axis;
  if (parser.parseOperand(input) || parser.parseKeyword("axis") ||
      parser.parseInteger(axis))
    return failure();
  result.addAttribute(getAxisAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(axis));
  if (parser.parseKeyword("lo"))
    return failure();
  Attribute lo = parseSymExpr(parser);
  if (!lo || parser.parseKeyword("hi"))
    return failure();
  Attribute hi = parseSymExpr(parser);
  if (!hi)
    return failure();
  result.addAttribute(getLoAttrName(result.name), lo);
  result.addAttribute(getHiAttrName(result.name), hi);
  Attribute value;
  if (parser.parseKeyword("value") || parser.parseLParen() ||
      parser.parseAttribute(value) || parser.parseRParen())
    return failure();
  result.addAttribute(getValueAttrName(result.name), value);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void PadOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " axis " << getAxis() << " lo ";
  printSymExpr(printer, getLo());
  printer << " hi ";
  printSymExpr(printer, getHi());
  printer << " value (";
  printer.printAttribute(getValue());
  printer << ")";
  printer.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getAxisAttrName(), getLoAttrName(), getHiAttrName(),
                       getValueAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult PadOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  int64_t axis = getAxis();
  int64_t rank = static_cast<int64_t>(inputType.getShape().size());

  if (axis < 0 || axis >= rank)
    return emitOpError() << "axis (" << axis << ") is out of range for "
                         << "operand rank " << rank;
  if (!isSymExpr(getLo()) || !isSymExpr(getHi()))
    return emitOpError() << "lo and hi must be sym expressions";
  Attribute zero = sym::ConstantExprAttr::get(getContext(), 0);
  if (proveLessEqual(zero, getLo()) == Proof::Disproven)
    return emitOpError() << "lo is provably negative: " << getLo();
  if (proveLessEqual(zero, getHi()) == Proof::Disproven)
    return emitOpError() << "hi is provably negative: " << getHi();
  TypedAttr value = getValue();
  if (value.getType() != inputType.getElementType())
    return emitOpError() << "pad value type (" << value.getType()
                         << ") must match the element type ("
                         << inputType.getElementType() << ")";
  if (resultType.getShape().size() != inputType.getShape().size())
    return emitOpError() << "result rank must match operand rank";
  for (int64_t k = 0; k < rank; ++k) {
    Attribute expected = k == axis ? paddedDim(inputType.getShape()[k], getLo(),
                                               getHi(), getContext())
                                   : inputType.getShape()[k];
    if (!sym::UnificationSolver::areLogicallyEqual(resultType.getShape()[k],
                                                   expected)) {
      if (k == axis)
        return emitOpError() << "result dimension " << k
                             << " must equal operand dimension + lo + hi";
      return emitOpError() << "result dimension " << k
                           << " must equal operand dimension " << k;
    }
  }
  if (resultType.getElementType() != inputType.getElementType())
    return emitOpError() << "result element type must match operand element "
                            "type";
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RelocOps.cpp.inc"
