//===- RelocUtils.cpp - Reloc dialect utilities ---------------------------===//
//
// This file implements utilities for the Reloc dialect.
//
//===----------------------------------------------------------------------===//

#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::reloc;

//===----------------------------------------------------------------------===//
// Compact expression syntax
//===----------------------------------------------------------------------===//

bool mlir::reloc::isSymExpr(Attribute attr) {
  return isa_and_nonnull<sym::SymbolExprAttr, sym::ConstantExprAttr,
                         sym::BinaryExprAttr>(attr);
}

static Attribute parseExprImpl(AsmParser &parser);

/// factor := INTEGER | IDENT | STRING | '(' expr ')'
static Attribute parseFactor(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  // Integer literal (negative literals included).
  int64_t value;
  OptionalParseResult intResult = parser.parseOptionalInteger(value);
  if (intResult.has_value()) {
    if (failed(*intResult))
      return {};
    return sym::ConstantExprAttr::get(ctx, value);
  }

  // Quoted string symbol, e.g. "my dim".
  std::string quoted;
  if (succeeded(parser.parseOptionalString(&quoted)))
    return sym::SymbolExprAttr::get(ctx, quoted);

  // Parenthesized subexpression.
  if (succeeded(parser.parseOptionalLParen())) {
    Attribute inner = parseExprImpl(parser);
    if (!inner || parser.parseRParen())
      return {};
    return inner;
  }

  // Bare identifier symbol.
  StringRef ident;
  if (succeeded(parser.parseOptionalKeyword(&ident))) {
    if (ident == "floordiv" || ident == "div" || ident == "mod") {
      parser.emitError(parser.getCurrentLocation())
          << "unexpected operator keyword '" << ident
          << "'; expected expression operand";
      return {};
    }
    return sym::SymbolExprAttr::get(ctx, ident);
  }

  parser.emitError(parser.getCurrentLocation(),
                   "expected integer, symbol, or '(' in expression");
  return {};
}

/// term := factor (('*' | 'floordiv' | 'div' | 'mod') factor)*
static Attribute parseTerm(AsmParser &parser) {
  Attribute lhs = parseFactor(parser);
  if (!lhs)
    return {};
  while (true) {
    sym::SymbolicExprOp opcode;
    if (succeeded(parser.parseOptionalStar()))
      opcode = sym::SymbolicExprOp::Mul;
    else if (succeeded(parser.parseOptionalKeyword("floordiv")) ||
             succeeded(parser.parseOptionalKeyword("div")))
      opcode = sym::SymbolicExprOp::Div;
    else if (succeeded(parser.parseOptionalKeyword("mod")))
      opcode = sym::SymbolicExprOp::Mod;
    else
      return lhs;
    Attribute rhs = parseFactor(parser);
    if (!rhs)
      return {};
    lhs = sym::getSimplifiedBinaryExpr(parser.getContext(), opcode, lhs, rhs);
  }
}

/// expr := term (('+' | '-') term)*
static Attribute parseExprImpl(AsmParser &parser) {
  Attribute lhs = parseTerm(parser);
  if (!lhs)
    return {};
  while (true) {
    sym::SymbolicExprOp opcode;
    if (succeeded(parser.parseOptionalPlus()))
      opcode = sym::SymbolicExprOp::Add;
    else if (succeeded(parser.parseOptionalMinus()))
      opcode = sym::SymbolicExprOp::Sub;
    else
      return lhs;
    Attribute rhs = parseTerm(parser);
    if (!rhs)
      return {};
    lhs = sym::getSimplifiedBinaryExpr(parser.getContext(), opcode, lhs, rhs);
  }
}

Attribute mlir::reloc::parseSymExpr(AsmParser &parser) {
  return parseExprImpl(parser);
}

static int precedence(sym::SymbolicExprOp op) {
  switch (op) {
  case sym::SymbolicExprOp::Add:
  case sym::SymbolicExprOp::Sub:
    return 1;
  case sym::SymbolicExprOp::Mul:
  case sym::SymbolicExprOp::Div:
  case sym::SymbolicExprOp::Mod:
    return 2;
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

static StringRef opKeyword(sym::SymbolicExprOp op) {
  switch (op) {
  case sym::SymbolicExprOp::Add:
    return "+";
  case sym::SymbolicExprOp::Sub:
    return "-";
  case sym::SymbolicExprOp::Mul:
    return "*";
  case sym::SymbolicExprOp::Div:
    return "floordiv";
  case sym::SymbolicExprOp::Mod:
    return "mod";
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

/// True if `name` can print as a bare identifier (and is not an operator
/// keyword).
static bool isBareIdent(StringRef name) {
  if (name.empty() || name == "floordiv" || name == "div" || name == "mod")
    return false;
  if (!llvm::isAlpha(name.front()) && name.front() != '_')
    return false;
  return llvm::all_of(
      name, [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

static void printExprImpl(AsmPrinter &printer, Attribute expr, int parentPrec,
                          bool isRhs) {
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(expr)) {
    printer << constant.getValue();
    return;
  }
  if (auto symbol = dyn_cast<sym::SymbolExprAttr>(expr)) {
    if (isBareIdent(symbol.getName()))
      printer << symbol.getName();
    else
      printer << "\"" << symbol.getName() << "\"";
    return;
  }
  auto binary = cast<sym::BinaryExprAttr>(expr);
  int prec = precedence(binary.getOpcode());
  bool needParens = prec < parentPrec || (isRhs && prec == parentPrec);
  if (needParens)
    printer << "(";
  printExprImpl(printer, binary.getLhs(), prec, /*isRhs=*/false);
  printer << " " << opKeyword(binary.getOpcode()) << " ";
  printExprImpl(printer, binary.getRhs(), prec, /*isRhs=*/true);
  if (needParens)
    printer << ")";
}

void mlir::reloc::printSymExpr(AsmPrinter &printer, Attribute expr) {
  printExprImpl(printer, expr, /*parentPrec=*/0, /*isRhs=*/false);
}

//===----------------------------------------------------------------------===//
// sym <-> affine expression bridge
//===----------------------------------------------------------------------===//

FailureOr<AffineExpr>
mlir::reloc::symToAffine(Attribute expr,
                         SmallVectorImpl<StringRef> &symbolNames,
                         MLIRContext *ctx) {
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(expr))
    return getAffineConstantExpr(constant.getValue(), ctx);

  if (auto symbol = dyn_cast<sym::SymbolExprAttr>(expr)) {
    StringRef name = symbol.getName();
    for (auto [index, existing] : llvm::enumerate(symbolNames))
      if (existing == name)
        return getAffineSymbolExpr(index, ctx);
    symbolNames.push_back(name);
    return getAffineSymbolExpr(symbolNames.size() - 1, ctx);
  }

  if (auto binary = dyn_cast<sym::BinaryExprAttr>(expr)) {
    FailureOr<AffineExpr> lhs = symToAffine(binary.getLhs(), symbolNames, ctx);
    if (failed(lhs))
      return failure();
    FailureOr<AffineExpr> rhs = symToAffine(binary.getRhs(), symbolNames, ctx);
    if (failed(rhs))
      return failure();
    switch (binary.getOpcode()) {
    case sym::SymbolicExprOp::Add:
      return *lhs + *rhs;
    case sym::SymbolicExprOp::Sub:
      return *lhs - *rhs; // affine encodes as lhs + rhs * -1
    case sym::SymbolicExprOp::Mul:
      return *lhs * *rhs;
    case sym::SymbolicExprOp::Div:
      return lhs->floorDiv(*rhs); // Div is floordiv (docs/reloc-design.md)
    case sym::SymbolicExprOp::Mod:
      return *lhs % *rhs;
    }
    llvm_unreachable("unknown SymbolicExprOp");
  }

  return failure();
}

Attribute mlir::reloc::affineToSym(AffineExpr expr,
                                   ArrayRef<StringRef> symbolNames,
                                   MLIRContext *ctx) {
  using sym::SymbolicExprOp;

  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return sym::ConstantExprAttr::get(ctx, constant.getValue());

  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    if (symbol.getPosition() >= symbolNames.size())
      return {};
    return sym::SymbolExprAttr::get(ctx, symbolNames[symbol.getPosition()]);
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return {}; // AffineDimExpr: no sym counterpart in A2.

  // Rebuild subtraction from affine's encodings so `a - b` survives the
  // round trip: `x + (y * -1)` -> `x - y`, and `x + (-c)` -> `x - c`.
  if (binary.getKind() == AffineExprKind::Add) {
    AffineExpr rhs = binary.getRHS();
    if (auto rhsBinary = dyn_cast<AffineBinaryOpExpr>(rhs))
      if (rhsBinary.getKind() == AffineExprKind::Mul)
        if (auto factor = dyn_cast<AffineConstantExpr>(rhsBinary.getRHS()))
          if (factor.getValue() == -1) {
            Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
            Attribute sub = affineToSym(rhsBinary.getLHS(), symbolNames, ctx);
            if (!lhs || !sub)
              return {};
            return sym::getSimplifiedBinaryExpr(ctx, SymbolicExprOp::Sub, lhs,
                                                sub);
          }
    if (auto constant = dyn_cast<AffineConstantExpr>(rhs))
      if (constant.getValue() < 0) {
        Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
        if (!lhs)
          return {};
        return sym::getSimplifiedBinaryExpr(
            ctx, SymbolicExprOp::Sub, lhs,
            sym::ConstantExprAttr::get(ctx, -constant.getValue()));
      }
  }

  SymbolicExprOp opcode;
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    opcode = SymbolicExprOp::Add;
    break;
  case AffineExprKind::Mul:
    opcode = SymbolicExprOp::Mul;
    break;
  case AffineExprKind::Mod:
    opcode = SymbolicExprOp::Mod;
    break;
  case AffineExprKind::FloorDiv:
    opcode = SymbolicExprOp::Div;
    break;
  default:
    return {}; // CeilDiv: no sym counterpart in A2.
  }

  Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
  Attribute rhs = affineToSym(binary.getRHS(), symbolNames, ctx);
  if (!lhs || !rhs)
    return {};
  return sym::getSimplifiedBinaryExpr(ctx, opcode, lhs, rhs);
}
