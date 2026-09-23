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
// only. Value guards on runtime parameters and undecidable symbolic channel
// lengths are bind-time obligations (C3): a Proof::Unknown never rejects.

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

/// Constant extents print as plain integers in diagnostics; anything else
/// prints as the attribute.
static std::string describeExtent(Attribute extent) {
  std::string text;
  llvm::raw_string_ostream os(text);
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(extent))
    os << constant.getValue();
  else
    os << extent;
  return text;
}

namespace {
enum class ParamRole { Scale, ZeroPoint };
} // namespace

static StringRef roleName(ParamRole role) {
  return role == ParamRole::Scale ? "scale" : "zero_point";
}

/// One quantization parameter: a dense constant or a #reloc.binding, rank 0
/// (per tensor) or rank 1 (per channel, length == the channel axis extent
/// unless undecidable), role-specific element type, valid constant values.
static LogicalResult verifyQuantParam(Operation *op, ParamRole role,
                                      Attribute attr,
                                      sym::SymbolicTensorType input,
                                      std::optional<int64_t> axis) {
  StringRef name = roleName(role);
  MLIRContext *ctx = op->getContext();
  int64_t rank = 0;
  Attribute channels; // rank-1 length as a sym expression
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    ShapedType type = dense.getType();
    rank = type.getRank();
    if (rank > 1)
      return op->emitOpError() << "parameters must have rank 0 or 1, but "
                               << name << " has rank " << rank;
    Type element = type.getElementType();
    if (role == ParamRole::Scale) {
      if (!element.isF32())
        return op->emitOpError()
               << "scale constants must have element type f32, but got "
               << element;
      for (auto [index, value] : llvm::enumerate(dense.getValues<APFloat>()))
        if (!value.isFinite() || value.isNegative() || value.isZero())
          return op->emitOpError()
                 << "scale must be finite and strictly positive, but element "
                 << index << " is " << FloatAttr::get(element, value);
    } else {
      if (!element.isSignlessInteger())
        return op->emitOpError() << "zero_point constants must have a "
                                    "signless integer element type, but got "
                                 << element;
      for (auto [index, value] : llvm::enumerate(dense.getValues<APInt>())) {
        int64_t zeroPoint = value.getSExtValue();
        if (zeroPoint < -128 || zeroPoint > 127)
          return op->emitOpError()
                 << "zero point must lie in [-128, 127], but element " << index
                 << " is " << zeroPoint;
      }
    }
    if (rank == 1)
      channels = sym::ConstantExprAttr::get(ctx, type.getDimSize(0));
  } else if (auto binding = dyn_cast<ParamBindingAttr>(attr)) {
    rank = static_cast<int64_t>(binding.getExtents().size());
    Type element = binding.getElementType();
    if (role == ParamRole::Scale && !element.isF32())
      return op->emitOpError()
             << "scale bindings must declare element type f32, but got "
             << element;
    if (role == ParamRole::ZeroPoint && !element.isSignlessInteger(32))
      return op->emitOpError()
             << "zero_point bindings must declare element type i32, but got "
             << element;
    if (rank == 1)
      channels = binding.getExtents()[0];
  } else {
    return op->emitOpError()
           << name << " must be a dense constant or a #reloc.binding, but got "
           << attr;
  }

  if (!axis) {
    if (rank != 0)
      return op->emitOpError() << "per-tensor form (no axis) requires rank-0 "
                                  "parameters, but "
                               << name << " has rank " << rank;
    return success();
  }
  if (role == ParamRole::Scale && rank != 1)
    return op->emitOpError()
           << "per-channel form requires a rank-1 scale, but scale has rank "
           << rank;
  if (rank == 1) {
    Attribute extent = input.getShape()[*axis];
    if (proveEqual(channels, extent) == Proof::Disproven)
      return op->emitOpError()
             << name << (isa<DenseElementsAttr>(attr) ? " has " : " declares ")
             << describeExtent(channels) << " channel entries, but axis "
             << *axis << " has extent " << describeExtent(extent);
  }
  return success();
}

/// Shared quantize/dequantize checks: axis range, both parameters, distinct
/// binding names.
static LogicalResult verifyQuantParams(Operation *op,
                                       sym::SymbolicTensorType input,
                                       Attribute scale, Attribute zeroPoint,
                                       std::optional<int64_t> axis) {
  int64_t rank = static_cast<int64_t>(input.getShape().size());
  if (axis && (*axis < 0 || *axis >= rank))
    return op->emitOpError() << "axis (" << *axis
                             << ") is out of range for operand rank " << rank;
  if (failed(verifyQuantParam(op, ParamRole::Scale, scale, input, axis)))
    return failure();
  if (zeroPoint && failed(verifyQuantParam(op, ParamRole::ZeroPoint, zeroPoint,
                                           input, axis)))
    return failure();
  auto scaleBinding = dyn_cast<ParamBindingAttr>(scale);
  auto zeroPointBinding = dyn_cast_or_null<ParamBindingAttr>(zeroPoint);
  if (scaleBinding && zeroPointBinding &&
      scaleBinding.getName() == zeroPointBinding.getName())
    return op->emitOpError() << "runtime parameters must use distinct binding "
                                "names, but scale and zero_point both bind \""
                             << scaleBinding.getName() << "\"";
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
  Type from = inputType.getElementType(), to = resultType.getElementType();
  std::optional<NumericPolicy> required;
  StringRef pair;
  if (from.isF32() && to.isF16()) {
    required = NumericPolicy::IeeeRne;
    pair = "f32 -> f16";
  } else if (from.isF16() && to.isF32()) {
    required = NumericPolicy::Exact;
    pair = "f16 -> f32";
  }
  if (!required)
    return emitOpError() << "cast from " << from << " to " << to
                         << " is not a supported typed conversion (f32 -> "
                            "f16, f16 -> f32)";
  if (failed(verifyPreservedShape(*this, inputType, resultType)))
    return failure();
  if (getPolicy() != *required)
    return emitOpError() << "policy '" << stringifyNumericPolicy(getPolicy())
                         << "' is not defined for the " << pair
                         << " cast (use '" << stringifyNumericPolicy(*required)
                         << "')";
  return success();
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
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  Type from = inputType.getElementType(), to = resultType.getElementType();
  if (!from.isF32() || !to.isSignlessInteger(8))
    return emitOpError() << "quantize expects an f32 operand and a signless i8 "
                            "result (int8 signedness is declared by the "
                            "operation, not by the storage type), but got "
                         << from << " -> " << to;
  if (failed(verifyPreservedShape(*this, inputType, resultType)))
    return failure();
  if (getPolicy() != NumericPolicy::SymmetricRne)
    return emitOpError() << "policy '" << stringifyNumericPolicy(getPolicy())
                         << "' is not defined for reloc.quantize (use "
                            "'symmetric_rne')";
  if (failed(verifyQuantParams(*this, inputType, getScale(), getZeroPointAttr(),
                               getAxis())))
    return failure();
  // symmetric_rne has no zero point: only the constant 0 is admitted, so the
  // runtime never needs a bind-time zero-point guard for this policy.
  if (Attribute zeroPoint = getZeroPointAttr()) {
    if (isa<ParamBindingAttr>(zeroPoint))
      return emitOpError() << "policy symmetric_rne admits only the constant "
                              "zero point 0, but zero_point is a runtime "
                              "binding";
    for (auto [index, value] :
         llvm::enumerate(cast<DenseElementsAttr>(zeroPoint).getValues<APInt>()))
      if (!value.isZero())
        return emitOpError() << "policy symmetric_rne admits only the constant "
                                "zero point 0, but element "
                             << index << " is " << value.getSExtValue();
  }
  return success();
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
  auto inputType = cast<sym::SymbolicTensorType>(getInput().getType());
  auto resultType = cast<sym::SymbolicTensorType>(getResult().getType());
  Type from = inputType.getElementType(), to = resultType.getElementType();
  if (!from.isSignlessInteger(8) || !to.isF32())
    return emitOpError() << "dequantize expects a signless i8 operand and an "
                            "f32 result, but got "
                         << from << " -> " << to;
  if (failed(verifyPreservedShape(*this, inputType, resultType)))
    return failure();
  if (getPolicy() != NumericPolicy::Affine)
    return emitOpError() << "policy '" << stringifyNumericPolicy(getPolicy())
                         << "' is not defined for reloc.dequantize (use "
                            "'affine')";
  return verifyQuantParams(*this, inputType, getScale(), getZeroPointAttr(),
                           getAxis());
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RelocOps.cpp.inc"
