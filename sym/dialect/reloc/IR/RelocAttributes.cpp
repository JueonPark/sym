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
// AxisInfoAttr
//===----------------------------------------------------------------------===//

/// Parse an expression field of the form `, <keyword> = <expr>`.
static ParseResult parseExprField(AsmParser &parser, StringRef keyword,
                                  Attribute &out) {
  if (parser.parseComma() || parser.parseKeyword(keyword) ||
      parser.parseEqual())
    return failure();
  out = parseSymExpr(parser);
  return out ? success() : failure();
}

/// Parse an axis-info body:
///   {name = "n0", extent = e, src_stride = e, dst_stride = e}
/// Shared with PlanAttr's parser.
static AxisInfoAttr parseAxisInfoBody(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  std::string name;
  if (parser.parseLBrace() || parser.parseKeyword("name") ||
      parser.parseEqual() || parser.parseString(&name))
    return {};
  Attribute extent, srcStride, dstStride;
  if (parseExprField(parser, "extent", extent) ||
      parseExprField(parser, "src_stride", srcStride) ||
      parseExprField(parser, "dst_stride", dstStride) || parser.parseRBrace())
    return {};
  return AxisInfoAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      name, extent, srcStride, dstStride);
}

static void printAxisInfoBody(AsmPrinter &printer, AxisInfoAttr axis) {
  printer << "{name = \"" << axis.getName() << "\", extent = ";
  printSymExpr(printer, axis.getExtent());
  printer << ", src_stride = ";
  printSymExpr(printer, axis.getSrcStride());
  printer << ", dst_stride = ";
  printSymExpr(printer, axis.getDstStride());
  printer << "}";
}

Attribute AxisInfoAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  AxisInfoAttr axis = parseAxisInfoBody(parser);
  if (!axis || parser.parseGreater())
    return {};
  return axis;
}

void AxisInfoAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printAxisInfoBody(printer, *this);
  printer << ">";
}

LogicalResult
AxisInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                     StringRef name, Attribute extent, Attribute srcStride,
                     Attribute dstStride) {
  if (!isSymExpr(extent) || !isSymExpr(srcStride) || !isSymExpr(dstStride))
    return emitError() << "extent, src_stride, and dst_stride must be sym "
                          "expressions (symbol, constant, or binary)";
  return success();
}

//===----------------------------------------------------------------------===//
// PadFillAttr
//===----------------------------------------------------------------------===//

/// Parse a pad-fill body:
///   {dst_axis = 1, lo = e, hi = e, value = 0.0 : f32}
/// Shared with PlanAttr's parser.
static PadFillAttr parsePadFillBody(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int64_t dstAxis;
  if (parser.parseLBrace() || parser.parseKeyword("dst_axis") ||
      parser.parseEqual() || parser.parseInteger(dstAxis))
    return {};
  Attribute lo, hi;
  if (parseExprField(parser, "lo", lo) || parseExprField(parser, "hi", hi))
    return {};
  TypedAttr value;
  if (parser.parseComma() || parser.parseKeyword("value") ||
      parser.parseEqual() || parser.parseAttribute(value) ||
      parser.parseRBrace())
    return {};
  return PadFillAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      dstAxis, lo, hi, value);
}

static void printPadFillBody(AsmPrinter &printer, PadFillAttr pad) {
  printer << "{dst_axis = " << pad.getDstAxis() << ", lo = ";
  printSymExpr(printer, pad.getLo());
  printer << ", hi = ";
  printSymExpr(printer, pad.getHi());
  printer << ", value = ";
  printer.printAttribute(pad.getValue());
  printer << "}";
}

Attribute PadFillAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};
  PadFillAttr pad = parsePadFillBody(parser);
  if (!pad || parser.parseGreater())
    return {};
  return pad;
}

void PadFillAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printPadFillBody(printer, *this);
  printer << ">";
}

LogicalResult
PadFillAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                    int64_t dstAxis, Attribute lo, Attribute hi,
                    TypedAttr value) {
  if (dstAxis < 0)
    return emitError() << "dst_axis must be non-negative, but got: " << dstAxis;
  if (!isSymExpr(lo) || !isSymExpr(hi))
    return emitError() << "lo and hi must be sym expressions (symbol, "
                          "constant, or binary)";
  if (!value)
    return emitError() << "value must be a typed attribute";
  return success();
}

//===----------------------------------------------------------------------===//
// DivisibilityAttr
//===----------------------------------------------------------------------===//

Attribute DivisibilityAttr::parse(AsmParser &parser, Type type) {
  MLIRContext *ctx = parser.getContext();
  if (parser.parseLess())
    return {};
  Attribute expr = parseSymExpr(parser);
  if (!expr)
    return {};
  int64_t divisor;
  if (parser.parseComma() || parser.parseInteger(divisor) ||
      parser.parseGreater())
    return {};
  return DivisibilityAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      expr, divisor);
}

void DivisibilityAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printSymExpr(printer, getExpr());
  printer << ", " << getDivisor() << ">";
}

LogicalResult
DivisibilityAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         Attribute expr, int64_t divisor) {
  if (!isSymExpr(expr))
    return emitError() << "expr must be a sym expression (symbol, constant, "
                          "or binary), but got: "
                       << expr;
  if (divisor <= 0)
    return emitError() << "divisor must be positive, but got: " << divisor;
  return success();
}

//===----------------------------------------------------------------------===//
// AlignmentAttr
//===----------------------------------------------------------------------===//

Attribute AlignmentAttr::parse(AsmParser &parser, Type type) {
  MLIRContext *ctx = parser.getContext();
  int64_t axis, bytes;
  if (parser.parseLess() || parser.parseInteger(axis) || parser.parseComma() ||
      parser.parseInteger(bytes) || parser.parseGreater())
    return {};
  return AlignmentAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
      axis, bytes);
}

void AlignmentAttr::print(AsmPrinter &printer) const {
  printer << "<" << getAxis() << ", " << getBytes() << ">";
}

LogicalResult
AlignmentAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      int64_t axis, int64_t bytes) {
  if (axis < 0)
    return emitError() << "axis must be non-negative, but got: " << axis;
  if (bytes <= 0)
    return emitError() << "bytes must be positive, but got: " << bytes;
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
