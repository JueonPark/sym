//===- RelocAttributes.cpp - Reloc dialect attribute implementation -------===//
//
// This file implements the Reloc dialect attributes. Assembly-format
// conventions are documented in docs/reloc-design.md.
//
//===----------------------------------------------------------------------===//

#include "RelocDialect.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::reloc;

//===----------------------------------------------------------------------===//
// Shared parse helpers
//===----------------------------------------------------------------------===//

/// Parse `[expr, expr, ...]` (possibly empty) into `out`.
static ParseResult parseExprList(AsmParser &parser,
                                 SmallVectorImpl<Attribute> &out) {
  return parser.parseCommaSeparatedList(
      AsmParser::Delimiter::Square, [&]() -> ParseResult {
        Attribute expr = parseSymExpr(parser);
        if (!expr)
          return failure();
        out.push_back(expr);
        return success();
      });
}

/// Print `[expr, expr, ...]`.
static void printExprList(AsmPrinter &printer, ArrayRef<Attribute> exprs) {
  printer << "[";
  llvm::interleaveComma(exprs, printer,
                        [&](Attribute expr) { printSymExpr(printer, expr); });
  printer << "]";
}

//===----------------------------------------------------------------------===//
// TensorDescAttr
//===----------------------------------------------------------------------===//

/// Parse a tensor descriptor body:
///   [extents], elemType (, strides = [exprs])? (, offset = expr)?
/// Shared with PlanAttr's parser (which prefixes it with `tensor<`).
static TensorDescAttr parseTensorDescBody(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  SmallVector<Attribute> extents;
  if (parseExprList(parser, extents) || parser.parseComma())
    return {};

  Type elementType;
  if (parser.parseType(elementType))
    return {};

  SmallVector<Attribute> strides;
  Attribute offset = sym::ConstantExprAttr::get(ctx, 0);
  if (succeeded(parser.parseOptionalComma())) {
    bool parsedStrides = false;
    if (succeeded(parser.parseOptionalKeyword("strides"))) {
      if (parser.parseEqual() || parseExprList(parser, strides))
        return {};
      parsedStrides = true;
    }
    bool expectOffset =
        !parsedStrides || succeeded(parser.parseOptionalComma());
    if (expectOffset) {
      if (parser.parseKeyword("offset") || parser.parseEqual())
        return {};
      offset = parseSymExpr(parser);
      if (!offset)
        return {};
    }
  }

  return TensorDescAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      extents, strides, offset, elementType);
}

/// Print a tensor descriptor body, omitting default strides/offset.
static void printTensorDescBody(AsmPrinter &printer, TensorDescAttr desc) {
  printExprList(printer, desc.getExtents());
  printer << ", " << desc.getElementType();
  if (!desc.getStrides().empty()) {
    printer << ", strides = ";
    printExprList(printer, desc.getStrides());
  }
  if (!sym::UnificationSolver::isConstantValue(desc.getOffset(), 0)) {
    printer << ", offset = ";
    printSymExpr(printer, desc.getOffset());
  }
}

Attribute TensorDescAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  TensorDescAttr desc = parseTensorDescBody(parser);
  if (!desc || parser.parseGreater())
    return {};
  return desc;
}

void TensorDescAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printTensorDescBody(printer, *this);
  printer << ">";
}

LogicalResult
TensorDescAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> extents, ArrayRef<Attribute> strides,
                       Attribute offset, Type elementType) {
  for (Attribute extent : extents)
    if (!isSymExpr(extent))
      return emitError() << "extent must be a sym expression (symbol, "
                            "constant, or binary), but got: "
                         << extent;
  for (Attribute stride : strides)
    if (!isSymExpr(stride))
      return emitError() << "stride must be a sym expression (symbol, "
                            "constant, or binary), but got: "
                         << stride;
  if (!isSymExpr(offset))
    return emitError() << "offset must be a sym expression (symbol, "
                          "constant, or binary), but got: "
                       << offset;
  if (!strides.empty() && strides.size() != extents.size())
    return emitError() << "strides size (" << strides.size()
                       << ") must match extents size (" << extents.size()
                       << ") or be empty";
  if (!elementType.isIntOrIndexOrFloat() && !isa<ComplexType>(elementType))
    return emitError() << "element type must be a valid tensor element type, "
                          "but got: "
                       << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// RelocDialect attribute registration
//===----------------------------------------------------------------------===//
//
// Defined here (rather than in RelocDialect.cpp's initialize()) because
// addAttributes<...>() requires the generated attribute storage classes to
// be complete in the same translation unit; this file is the only one that
// includes RelocAttributes.cpp.inc under GET_ATTRDEF_CLASSES.

void RelocDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RelocAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "RelocAttributes.cpp.inc"
