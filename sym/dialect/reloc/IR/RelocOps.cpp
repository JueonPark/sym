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
// PlanResultOp
//===----------------------------------------------------------------------===//

ParseResult PlanResultOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  PlanAttr plan;
  if (parser.parseOperand(input) || parser.parseKeyword("plan") ||
      parser.parseLParen() || parser.parseAttribute(plan) ||
      parser.parseRParen())
    return failure();
  result.addAttribute(getPlanAttrName(result.name), plan);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void PlanResultOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " plan(";
  printer.printAttribute(getPlan());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getPlanAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult PlanResultOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  PlanAttr plan = getPlan();
  ArrayRef<Attribute> srcExtents = plan.getSrc().getExtents();
  ArrayRef<Attribute> dstExtents = plan.getDst().getExtents();

  if (inputType.getShape().size() != srcExtents.size())
    return emitOpError() << "input rank (" << inputType.getShape().size()
                         << ") must match the plan src rank ("
                         << srcExtents.size() << ")";
  if (inputType.getElementType() != plan.getSrc().getElementType())
    return emitOpError() << "input element type (" << inputType.getElementType()
                         << ") must match the plan src element type ("
                         << plan.getSrc().getElementType() << ")";
  if (resultType.getElementType() != plan.getDst().getElementType())
    return emitOpError() << "result element type ("
                         << resultType.getElementType()
                         << ") must match the plan dst element type ("
                         << plan.getDst().getElementType() << ")";

  // Disproven-only extent checks (mirrors the plan verifier's philosophy:
  // reject only provable mismatches; symbolic Unknowns pass).
  for (size_t k = 0; k < srcExtents.size(); ++k)
    if (proveEqual(inputType.getShape()[k], srcExtents[k]) == Proof::Disproven)
      return emitOpError() << "input dimension " << k
                           << " provably disagrees with the plan src extent";

  if (resultType.getShape().size() != dstExtents.size()) {
    // Canonicalized plans collapse dst axes (#B5): accept a rank change
    // when the element counts do not provably disagree.
    if (proveEqual(shapeProduct(resultType.getShape(), getContext()),
                   shapeProduct(dstExtents, getContext())) == Proof::Disproven)
      return emitOpError()
             << "result element count provably disagrees with the plan dst";
  } else {
    for (size_t k = 0; k < dstExtents.size(); ++k)
      if (proveEqual(resultType.getShape()[k], dstExtents[k]) ==
          Proof::Disproven)
        return emitOpError() << "result dimension " << k
                             << " provably disagrees with the plan dst extent";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Typed value transforms (C1, issue #141)
//===----------------------------------------------------------------------===//
//
// reloc.cast / reloc.quantize / reloc.dequantize preserve the logical shape
// and change only the element type. The numerical meaning of each policy is
// docs/reloc-typed-semantics.md; the verifiers below enforce static legality
// only, through the helpers in RelocUtils shared with the typed-plan stage
// attribute (C2). Value guards on runtime parameters and undecidable symbolic
// channel lengths are bind-time obligations (C3): a Proof::Unknown never
// rejects.

/// Parse `policy <keyword>` into `attrName`.
static ParseResult parsePolicy(OpAsmParser &parser, OperationState &result,
                               StringAttr attrName) {
  if (parser.parseKeyword("policy"))
    return failure();
  llvm::SMLoc loc = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<NumericPolicy> policy = symbolizeNumericPolicy(keyword);
  if (!policy)
    return parser.emitError(loc)
           << "unknown numerical policy '" << keyword << "'";
  result.addAttribute(attrName,
                      NumericPolicyAttr::get(parser.getContext(), *policy));
  return success();
}

/// Parse `( <attribute> )`.
static ParseResult parseParenAttr(OpAsmParser &parser, Attribute &attr) {
  return failure(parser.parseLParen() || parser.parseAttribute(attr) ||
                 parser.parseRParen());
}

/// Value transforms keep the logical shape: same rank, logically equal dims.
static LogicalResult verifyPreservedShape(Operation *op,
                                          sym::SymbolicTensorType input,
                                          sym::SymbolicTensorType result) {
  size_t rank = input.getShape().size();
  if (result.getShape().size() != rank)
    return op->emitOpError() << "result rank (" << result.getShape().size()
                             << ") must match operand rank (" << rank << ")";
  for (size_t k = 0; k < rank; ++k)
    if (!sym::UnificationSolver::areLogicallyEqual(result.getShape()[k],
                                                   input.getShape()[k]))
      return op->emitOpError()
             << "result dimension " << k << " must equal operand dimension "
             << k << " (value transforms preserve the logical shape)";
  return success();
}

template <typename OpType>
static ParseResult parseQuantLike(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();
  if (succeeded(parser.parseOptionalKeyword("axis"))) {
    int64_t axis;
    if (parser.parseInteger(axis))
      return failure();
    result.addAttribute(OpType::getAxisAttrName(result.name),
                        parser.getBuilder().getI64IntegerAttr(axis));
  }
  Attribute scale;
  if (parser.parseKeyword("scale") || parseParenAttr(parser, scale))
    return failure();
  result.addAttribute(OpType::getScaleAttrName(result.name), scale);
  if (succeeded(parser.parseOptionalKeyword("zero_point"))) {
    Attribute zeroPoint;
    if (parseParenAttr(parser, zeroPoint))
      return failure();
    result.addAttribute(OpType::getZeroPointAttrName(result.name), zeroPoint);
  }
  if (parsePolicy(parser, result, OpType::getPolicyAttrName(result.name)) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

template <typename OpType>
static void printQuantLike(OpType op, OpAsmPrinter &printer) {
  printer << " " << op.getInput();
  if (std::optional<int64_t> axis = op.getAxis())
    printer << " axis " << *axis;
  printer << " scale(";
  printer.printAttribute(op.getScale());
  printer << ")";
  if (Attribute zeroPoint = op.getZeroPointAttr()) {
    printer << " zero_point(";
    printer.printAttribute(zeroPoint);
    printer << ")";
  }
  printer << " policy " << stringifyNumericPolicy(op.getPolicy());
  printer.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getScaleAttrName(), op.getZeroPointAttrName(),
                       op.getAxisAttrName(), op.getPolicyAttrName()});
  printOpTypes(printer, op.getInput().getType(), op.getResult().getType());
}

/// Shared quantize/dequantize verifier: signature, shape, parameters.
template <typename OpType>
static LogicalResult verifyQuantLike(OpType op, ValueTransform transform) {
  auto inputType = cast<sym::SymbolicTensorType>(op.getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(op.getResult().getType());
  auto emitError = [&]() { return op.emitOpError(); };
  if (failed(verifyValueTransformSignature(emitError, transform, op.getPolicy(),
                                           inputType.getElementType(),
                                           resultType.getElementType())))
    return failure();
  if (failed(verifyPreservedShape(op, inputType, resultType)))
    return failure();
  return verifyQuantizationParameters(emitError, inputType.getShape(),
                                      op.getScale(), op.getZeroPointAttr(),
                                      op.getAxis(), op.getPolicy());
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Type elementType, NumericPolicy policy) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  build(builder, state,
        sym::SymbolicTensorType::get(builder.getContext(), inputType.getShape(),
                                     elementType),
        input, NumericPolicyAttr::get(builder.getContext(), policy));
}

ParseResult CastOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input) ||
      parsePolicy(parser, result, getPolicyAttrName(result.name)) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void CastOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " policy "
          << stringifyNumericPolicy(getPolicy());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getPolicyAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult CastOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  if (failed(verifyValueTransformSignature(
          [&]() { return emitOpError(); }, ValueTransform::Cast, getPolicy(),
          inputType.getElementType(), resultType.getElementType())))
    return failure();
  return verifyPreservedShape(*this, inputType, resultType);
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

void QuantizeOp::build(OpBuilder &builder, OperationState &state, Value input,
                       Attribute scale, Attribute zeroPoint,
                       std::optional<int64_t> axis, NumericPolicy policy) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  auto resultType = sym::SymbolicTensorType::get(
      builder.getContext(), inputType.getShape(), builder.getIntegerType(8));
  build(builder, state, resultType, input, scale, zeroPoint,
        axis ? builder.getI64IntegerAttr(*axis) : IntegerAttr(),
        NumericPolicyAttr::get(builder.getContext(), policy));
}

ParseResult QuantizeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseQuantLike<QuantizeOp>(parser, result);
}

void QuantizeOp::print(OpAsmPrinter &printer) {
  printQuantLike(*this, printer);
}

LogicalResult QuantizeOp::verify() {
  return verifyQuantLike(*this, ValueTransform::Quantize);
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

void DequantizeOp::build(OpBuilder &builder, OperationState &state, Value input,
                         Attribute scale, Attribute zeroPoint,
                         std::optional<int64_t> axis, NumericPolicy policy) {
  auto inputType = cast<sym::SymbolicTensorType>(input.getType());
  auto resultType = sym::SymbolicTensorType::get(
      builder.getContext(), inputType.getShape(), builder.getF32Type());
  build(builder, state, resultType, input, scale, zeroPoint,
        axis ? builder.getI64IntegerAttr(*axis) : IntegerAttr(),
        NumericPolicyAttr::get(builder.getContext(), policy));
}

ParseResult DequantizeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseQuantLike<DequantizeOp>(parser, result);
}

void DequantizeOp::print(OpAsmPrinter &printer) {
  printQuantLike(*this, printer);
}

LogicalResult DequantizeOp::verify() {
  return verifyQuantLike(*this, ValueTransform::Dequantize);
}

//===----------------------------------------------------------------------===//
// TypedPlanResultOp (C2, issue #142)
//===----------------------------------------------------------------------===//

ParseResult TypedPlanResultOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  TypedPlanAttr plan;
  if (parser.parseOperand(input) || parser.parseKeyword("plan") ||
      parser.parseLParen() || parser.parseAttribute(plan) ||
      parser.parseRParen())
    return failure();
  result.addAttribute(getPlanAttrName(result.name), plan);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Type inputType, resultType;
  if (parseOpTypes(parser, inputType, resultType) ||
      parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void TypedPlanResultOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " plan(";
  printer.printAttribute(getPlan());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{getPlanAttrName()});
  printOpTypes(printer, getInput().getType(), getResult().getType());
}

LogicalResult TypedPlanResultOp::verify() {
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  TypedPlanAttr plan = getPlan();
  ArrayRef<Attribute> sourceExtents = plan.getSource().getExtents();
  ArrayRef<Attribute> resultExtents = plan.getResult().getExtents();

  if (inputType.getShape().size() != sourceExtents.size())
    return emitOpError() << "input rank (" << inputType.getShape().size()
                         << ") must match the typed plan source rank ("
                         << sourceExtents.size() << ")";
  if (inputType.getElementType() != plan.getSource().getElementType())
    return emitOpError() << "input element type (" << inputType.getElementType()
                         << ") must match the typed plan source element type ("
                         << plan.getSource().getElementType() << ")";
  if (resultType.getElementType() != plan.getResult().getElementType())
    return emitOpError() << "result element type ("
                         << resultType.getElementType()
                         << ") must match the typed plan result element type ("
                         << plan.getResult().getElementType() << ")";
  // The typed plan's result descriptor keeps the logical rank (channel maps
  // are written over it), so ranks match exactly; extents are checked as
  // disproven-only, like plan_result.
  if (resultType.getShape().size() != resultExtents.size())
    return emitOpError() << "result rank (" << resultType.getShape().size()
                         << ") must match the typed plan result rank ("
                         << resultExtents.size() << ")";
  for (size_t k = 0; k < sourceExtents.size(); ++k)
    if (proveEqual(inputType.getShape()[k], sourceExtents[k]) ==
        Proof::Disproven)
      return emitOpError() << "input dimension " << k
                           << " provably disagrees with the typed plan source "
                              "extent";
  for (size_t k = 0; k < resultExtents.size(); ++k)
    if (proveEqual(resultType.getShape()[k], resultExtents[k]) ==
        Proof::Disproven)
      return emitOpError() << "result dimension " << k
                           << " provably disagrees with the typed plan result "
                              "extent";
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RelocOps.cpp.inc"
