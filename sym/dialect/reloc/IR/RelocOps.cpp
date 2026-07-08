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

/// Elementwise logical equality of two symbolic shapes of equal rank.
static bool logicallyEqualShapes(ArrayRef<Attribute> lhs,
                                 ArrayRef<Attribute> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (!sym::UnificationSolver::areLogicallyEqual(left, right))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

/// Result type of transposing `input` by `perm`: dim k = input dim perm[k].
/// Requires `perm` to be a valid permutation of the input rank.
static sym::SymbolicTensorType
computeTransposedType(sym::SymbolicTensorType input, ArrayRef<int64_t> perm) {
  SmallVector<Attribute> shape;
  shape.reserve(perm.size());
  for (int64_t source : perm)
    shape.push_back(input.getShape()[source]);
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
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "RelocOps.cpp.inc"
