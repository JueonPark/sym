//===- RelocAttributes.cpp - Reloc dialect attribute implementation -------===//
//
// This file implements the Reloc dialect attributes.
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
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        [&]() -> ParseResult {
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
  if (!elementType.isIntOrIndexOrFloat() &&
      !mlir::isa<ComplexType>(elementType))
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

LogicalResult AxisInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   StringRef name, Attribute extent,
                                   Attribute srcStride, Attribute dstStride) {
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

LogicalResult PadFillAttr::verify(function_ref<InFlightDiagnostic()> emitError,
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
// PlanAttr
//===----------------------------------------------------------------------===//

/// Parse `<keyword> = tensor<BODY>` into a descriptor.
static TensorDescAttr parseDescField(AsmParser &parser, StringRef keyword) {
  if (parser.parseKeyword(keyword) || parser.parseEqual() ||
      parser.parseKeyword("tensor") || parser.parseLess())
    return {};
  TensorDescAttr desc = parseTensorDescBody(parser);
  if (!desc || parser.parseGreater())
    return {};
  return desc;
}

/// Parse the optional constraints block:
///   constraints = {divisible(e, k), align(a, b), contiguous = [bools],
///                  no_copy = bool}
/// Items are comma-separated and may appear in any order; divisible/align
/// may repeat.
static ParseResult parseConstraintsBlock(
    AsmParser &parser, SmallVectorImpl<DivisibilityAttr> &divisibility,
    SmallVectorImpl<AlignmentAttr> &alignment,
    SmallVectorImpl<bool> &contiguity, bool &noCopy, bool &runtimePadCheck) {
  MLIRContext *ctx = parser.getContext();
  if (parser.parseLBrace())
    return failure();
  if (succeeded(parser.parseOptionalRBrace()))
    return success();

  auto parseBool = [&](bool &out) -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword("true"))) {
      out = true;
      return success();
    }
    if (succeeded(parser.parseOptionalKeyword("false"))) {
      out = false;
      return success();
    }
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'true' or 'false'");
  };

  do {
    if (succeeded(parser.parseOptionalKeyword("divisible"))) {
      if (parser.parseLParen())
        return failure();
      Attribute expr = parseSymExpr(parser);
      if (!expr)
        return failure();
      int64_t divisor;
      if (parser.parseComma() || parser.parseInteger(divisor) ||
          parser.parseRParen())
        return failure();
      auto attr = DivisibilityAttr::getChecked(
          [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
          expr, divisor);
      if (!attr)
        return failure();
      divisibility.push_back(attr);
    } else if (succeeded(parser.parseOptionalKeyword("align"))) {
      int64_t axis, bytes;
      if (parser.parseLParen() || parser.parseInteger(axis) ||
          parser.parseComma() || parser.parseInteger(bytes) ||
          parser.parseRParen())
        return failure();
      auto attr = AlignmentAttr::getChecked(
          [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx,
          axis, bytes);
      if (!attr)
        return failure();
      alignment.push_back(attr);
    } else if (succeeded(parser.parseOptionalKeyword("contiguous"))) {
      if (parser.parseEqual() ||
          parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         [&]() -> ParseResult {
                                           bool flag;
                                           if (parseBool(flag))
                                             return failure();
                                           contiguity.push_back(flag);
                                           return success();
                                         }))
        return failure();
    } else if (succeeded(parser.parseOptionalKeyword("no_copy"))) {
      if (parser.parseEqual() || parseBool(noCopy))
        return failure();
    } else if (succeeded(parser.parseOptionalKeyword("runtime_pad_check"))) {
      runtimePadCheck = true;
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected 'divisible', 'align', 'contiguous', "
                              "'no_copy', or 'runtime_pad_check' in "
                              "constraints");
    }
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRBrace();
}

Attribute PlanAttr::parse(AsmParser &parser, Type type) {
  MLIRContext *ctx = parser.getContext();
  if (parser.parseLess())
    return {};

  // src = tensor<...>, dst = tensor<...>,
  TensorDescAttr src = parseDescField(parser, "src");
  if (!src || parser.parseComma())
    return {};
  TensorDescAttr dst = parseDescField(parser, "dst");
  if (!dst || parser.parseComma())
    return {};

  // perm = [ints],
  SmallVector<int64_t> perm;
  if (parser.parseKeyword("perm") || parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       int64_t v;
                                       if (parser.parseInteger(v))
                                         return failure();
                                       perm.push_back(v);
                                       return success();
                                     }) ||
      parser.parseComma())
    return {};

  // axes = [{...}, ...],
  SmallVector<AxisInfoAttr> axes;
  if (parser.parseKeyword("axes") || parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       AxisInfoAttr axis =
                                           parseAxisInfoBody(parser);
                                       if (!axis)
                                         return failure();
                                       axes.push_back(axis);
                                       return success();
                                     }) ||
      parser.parseComma())
    return {};

  // Optional: pad_fill = [{...}, ...],
  SmallVector<PadFillAttr> padFill;
  if (succeeded(parser.parseOptionalKeyword("pad_fill"))) {
    if (parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       [&]() -> ParseResult {
                                         PadFillAttr pad =
                                             parsePadFillBody(parser);
                                         if (!pad)
                                           return failure();
                                         padFill.push_back(pad);
                                         return success();
                                       }) ||
        parser.parseComma())
      return {};
  }

  // Optional: constraints = {...},
  SmallVector<DivisibilityAttr> divisibility;
  SmallVector<AlignmentAttr> alignment;
  SmallVector<bool> contiguity;
  bool noCopy = false;
  bool runtimePadCheck = false;
  if (succeeded(parser.parseOptionalKeyword("constraints"))) {
    if (parser.parseEqual() ||
        parseConstraintsBlock(parser, divisibility, alignment, contiguity,
                              noCopy, runtimePadCheck) ||
        parser.parseComma())
      return {};
  }

  // inverse = affine_map<...>
  AffineMapAttr inverse;
  if (parser.parseKeyword("inverse") || parser.parseEqual() ||
      parser.parseAttribute(inverse) || parser.parseGreater())
    return {};

  // Degradation path: an undecidable pad range marks the plan as
  // requiring a runtime check instead of rejecting.
  for (PadFillAttr pad : padFill)
    if (provePadRange(pad, axes, dst) == Proof::Unknown)
      runtimePadCheck = true;

  return PlanAttr::getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); }, ctx, src,
      dst, DenseI64ArrayAttr::get(ctx, perm), axes, padFill, divisibility,
      alignment, DenseBoolArrayAttr::get(ctx, contiguity), noCopy,
      runtimePadCheck, inverse);
}

void PlanAttr::print(AsmPrinter &printer) const {
  printer << "<src = tensor<";
  printTensorDescBody(printer, getSrc());
  printer << ">, dst = tensor<";
  printTensorDescBody(printer, getDst());
  printer << ">, perm = [";
  llvm::interleaveComma(getPerm().asArrayRef(), printer);
  printer << "], axes = [";
  llvm::interleaveComma(getAxes(), printer, [&](AxisInfoAttr axis) {
    printAxisInfoBody(printer, axis);
  });
  printer << "]";

  if (!getPadFill().empty()) {
    printer << ", pad_fill = [";
    llvm::interleaveComma(getPadFill(), printer, [&](PadFillAttr pad) {
      printPadFillBody(printer, pad);
    });
    printer << "]";
  }

  bool hasConstraints = !getDivisibility().empty() || !getAlignment().empty() ||
                        !getContiguity().empty() || getNoCopy() ||
                        getRuntimePadCheck();
  if (hasConstraints) {
    printer << ", constraints = {";
    bool first = true;
    auto comma = [&]() {
      if (!first)
        printer << ", ";
      first = false;
    };
    for (DivisibilityAttr div : getDivisibility()) {
      comma();
      printer << "divisible(";
      printSymExpr(printer, div.getExpr());
      printer << ", " << div.getDivisor() << ")";
    }
    for (AlignmentAttr align : getAlignment()) {
      comma();
      printer << "align(" << align.getAxis() << ", " << align.getBytes() << ")";
    }
    if (!getContiguity().empty()) {
      comma();
      printer << "contiguous = [";
      llvm::interleaveComma(
          getContiguity().asArrayRef(), printer,
          [&](bool flag) { printer << (flag ? "true" : "false"); });
      printer << "]";
    }
    comma();
    printer << "no_copy = " << (getNoCopy() ? "true" : "false");
    if (getRuntimePadCheck()) {
      comma();
      printer << "runtime_pad_check";
    }
    printer << "}";
  }

  printer << ", inverse = ";
  printer.printAttribute(getInverse());
  printer << ">";
}

LogicalResult PlanAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, TensorDescAttr src,
    TensorDescAttr dst, DenseI64ArrayAttr perm, ArrayRef<AxisInfoAttr> axes,
    ArrayRef<PadFillAttr> padFill, ArrayRef<DivisibilityAttr> divisibility,
    ArrayRef<AlignmentAttr> alignment, DenseBoolArrayAttr contiguity,
    bool noCopy, bool runtimePadCheck, AffineMapAttr inverse) {
  if (!src || !dst)
    return emitError() << "src and dst descriptors are required";
  if (!perm)
    return emitError() << "perm is required";
  if (perm.size() != static_cast<int64_t>(axes.size()))
    return emitError() << "perm size (" << perm.size()
                       << ") must match number of axes (" << axes.size() << ")";
  if (contiguity && !contiguity.empty() &&
      contiguity.size() != static_cast<int64_t>(axes.size()))
    return emitError() << "contiguity size (" << contiguity.size()
                       << ") must match number of axes (" << axes.size()
                       << ") or be empty";
  if (!inverse)
    return emitError() << "inverse map is required";

  // --- A3: pad widths (tensor.pad convention: lo/hi are leading/trailing
  // pad counts; extent + lo + hi must equal the dst extent) ---
  for (PadFillAttr pad : padFill) {
    int64_t axis = pad.getDstAxis();
    if (axis >= static_cast<int64_t>(axes.size()) ||
        axis >= static_cast<int64_t>(dst.getExtents().size()))
      return emitError() << "pad dst_axis (" << axis << ") is out of range for "
                         << axes.size() << " axes";
    MLIRContext *ctx = pad.getContext();
    Attribute zero = sym::ConstantExprAttr::get(ctx, 0);
    if (proveLessEqual(zero, pad.getLo()) == Proof::Disproven)
      return emitError() << "pad lo for dst_axis " << axis
                         << " is provably negative: " << pad.getLo();
    if (proveLessEqual(zero, pad.getHi()) == Proof::Disproven)
      return emitError() << "pad hi for dst_axis " << axis
                         << " is provably negative: " << pad.getHi();
    Attribute sum = sym::getSimplifiedBinaryExpr(
        ctx, sym::SymbolicExprOp::Add,
        sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add,
                                     axes[axis].getExtent(), pad.getLo()),
        pad.getHi());
    Proof relation = proveEqual(sum, dst.getExtents()[axis]);
    if (relation == Proof::Disproven)
      return emitError() << "pad on dst_axis " << axis
                         << " is inconsistent: extent + lo + hi must equal "
                            "the dst extent, but "
                         << sum << " != " << dst.getExtents()[axis];
    if (relation == Proof::Unknown && !runtimePadCheck)
      return emitError() << "pad range for dst_axis " << axis
                         << " is not provable; set runtime_pad_check";
  }

  // --- A3: perm must be a bijection on [0, axes.size()) ---
  {
    SmallVector<bool> seen(axes.size(), false);
    for (int64_t value : perm.asArrayRef()) {
      if (value < 0 || value >= static_cast<int64_t>(axes.size()) ||
          seen[value])
        return emitError() << "perm is not a permutation of [0, " << axes.size()
                           << ")";
      seen[value] = true;
    }
  }

  // --- A3: inverse is square on the axes space ---
  AffineMap inverseMap = inverse.getValue();
  if (inverseMap.getNumDims() != axes.size() ||
      inverseMap.getNumResults() != axes.size())
    return emitError() << "inverse map must be square on the axes space: "
                       << "expected " << axes.size() << " dims and "
                       << axes.size() << " results, got "
                       << inverseMap.getNumDims() << " and "
                       << inverseMap.getNumResults();

  // --- A3: pure-permutation plans must have a matching inverse.
  // Floordiv-style inverses are not permutations; for those only the
  // arity checks above apply (full bijectivity checking for such plans
  // needs P1b's composition algebra). ---
  if (!axes.empty() && inverseMap.isPermutation()) {
    AffineMap forward =
        AffineMap::getPermutationMap(perm.asArrayRef(), inverse.getContext());
    if (inverseMap != inversePermutation(forward))
      return emitError() << "inverse permutation does not invert perm";
  }

  // --- A3: direct axis order — dst rank and per-axis dst_stride ---
  if (dst.getExtents().size() != axes.size())
    return emitError() << "dst rank (" << dst.getExtents().size()
                       << ") must match number of axes (" << axes.size() << ")";
  {
    SmallVector<Attribute> dstStrides;
    if (!dst.getStrides().empty())
      dstStrides.assign(dst.getStrides().begin(), dst.getStrides().end());
    else
      dstStrides = canonicalRowMajorStrides(dst.getExtents(), dst.getContext());
    for (size_t k = 0; k < axes.size(); ++k)
      if (proveEqual(axes[k].getDstStride(), dstStrides[k]) == Proof::Disproven)
        return emitError() << "axis " << k
                           << " dst_stride provably disagrees with the dst "
                              "descriptor: "
                           << axes[k].getDstStride() << " vs " << dstStrides[k];
  }

  // --- A3: contiguity[k] asserts unit source stride ---
  if (contiguity && !contiguity.empty()) {
    Attribute one = sym::ConstantExprAttr::get(dst.getContext(), 1);
    for (int64_t k = 0; k < contiguity.size(); ++k)
      if (contiguity.asArrayRef()[k] &&
          proveEqual(axes[k].getSrcStride(), one) == Proof::Disproven)
        return emitError() << "contiguity[" << k
                           << "] asserts unit src_stride, but axis " << k
                           << " has src_stride " << axes[k].getSrcStride();
  }

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
