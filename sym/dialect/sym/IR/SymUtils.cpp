//===- SymUtils.cpp - Sym dialect utilities -------------------------------===//
//
// This file implements utility classes for the Sym dialect.
//
//===----------------------------------------------------------------------===//

#include "SymUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::sym;

//===----------------------------------------------------------------------===//
// UnificationSolver
//===----------------------------------------------------------------------===//

UnificationSolver::UnificationSolver(MLIRContext *context, Location loc)
    : context(context), loc(loc.getContext() ? loc : UnknownLoc::get(context)) {
}

bool UnificationSolver::areLogicallyEqual(Attribute a, Attribute b) {
  if (a == b)
    return true;

  // Check ConstantExprAttr equality by value
  if (auto constA = dyn_cast<ConstantExprAttr>(a)) {
    if (auto constB = dyn_cast<ConstantExprAttr>(b)) {
      return constA.getValue() == constB.getValue();
    }
    return false;
  }

  // Check SymbolExprAttr equality by name
  if (auto symA = dyn_cast<SymbolExprAttr>(a)) {
    if (auto symB = dyn_cast<SymbolExprAttr>(b)) {
      return symA.getName() == symB.getName();
    }
    return false;
  }

  // Check BinaryExprAttr - must have same opcode and operands
  // For commutative ops (Add, Mul), also check swapped operands
  if (auto binA = dyn_cast<BinaryExprAttr>(a)) {
    if (auto binB = dyn_cast<BinaryExprAttr>(b)) {
      if (binA.getOpcode() != binB.getOpcode())
        return false;

      // Direct match
      if (areLogicallyEqual(binA.getLhs(), binB.getLhs()) &&
          areLogicallyEqual(binA.getRhs(), binB.getRhs()))
        return true;

      // For commutative ops, check swapped operands
      if (binA.getOpcode() == SymbolicExprOp::Add ||
          binA.getOpcode() == SymbolicExprOp::Mul) {
        return areLogicallyEqual(binA.getLhs(), binB.getRhs()) &&
               areLogicallyEqual(binA.getRhs(), binB.getLhs());
      }
    }
    return false;
  }

  return false;
}

bool UnificationSolver::isConstantValue(Attribute attr, int64_t value) {
  if (auto constAttr = dyn_cast<ConstantExprAttr>(attr))
    return constAttr.getValue() == value;
  return false;
}

bool UnificationSolver::isConstantOne(Attribute attr) {
  return isConstantValue(attr, 1);
}

InFlightDiagnostic UnificationSolver::emitError(const Twine &message) {
  return mlir::emitError(loc, message);
}

std::string UnificationSolver::formatAttr(Attribute attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.print(os);
  return result;
}

FailureOr<SmallVector<Attribute>>
UnificationSolver::unify(ArrayRef<Attribute> shape1,
                         ArrayRef<Attribute> shape2) {
  size_t rank1 = shape1.size();
  size_t rank2 = shape2.size();
  size_t maxRank = std::max(rank1, rank2);

  SmallVector<Attribute> result;
  result.reserve(maxRank);

  // Create the constant 1 attribute for padding
  Attribute one = ConstantExprAttr::get(context, 1);

  // Iterate from the rightmost dimension to the left
  for (size_t i = 0; i < maxRank; ++i) {
    // Compute indices from the right (0 = rightmost)
    // For shape1: if i >= rank1, use 1; else use shape1[rank1 - 1 - i]
    // For shape2: if i >= rank2, use 1; else use shape2[rank2 - 1 - i]

    Attribute d1 = (i < rank1) ? shape1[rank1 - 1 - i] : one;
    Attribute d2 = (i < rank2) ? shape2[rank2 - 1 - i] : one;

    // Apply broadcasting rules
    if (areLogicallyEqual(d1, d2)) {
      // d1 == d2: result is d1
      result.push_back(d1);
    } else if (isConstantOne(d1)) {
      // d1 == 1: result is d2
      result.push_back(d2);
    } else if (isConstantOne(d2)) {
      // d2 == 1: result is d1
      result.push_back(d1);
    } else {
      // Incompatible dimensions
      size_t dimIndex = maxRank - 1 - i;
      return emitError("incompatible dimensions at index ")
             << dimIndex << ": " << formatAttr(d1) << " vs " << formatAttr(d2);
    }
  }

  // Reverse result since we built it from right to left
  std::reverse(result.begin(), result.end());
  return result;
}

//===----------------------------------------------------------------------===//
// Compact expression syntax
//===----------------------------------------------------------------------===//

bool mlir::sym::isSymExpr(Attribute attr) {
  return isa_and_nonnull<SymbolExprAttr, ConstantExprAttr, BinaryExprAttr>(
      attr);
}

static Attribute parseExprImpl(AsmParser &parser);

/// factor := INTEGER | IDENT | STRING | full sym-expression attribute
///         | '(' expr ')'
static Attribute parseFactor(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  // Integer literal (negative literals included).
  int64_t value;
  OptionalParseResult intResult = parser.parseOptionalInteger(value);
  if (intResult.has_value()) {
    if (failed(*intResult))
      return {};
    return ConstantExprAttr::get(ctx, value);
  }

  // Quoted string symbol, e.g. "my dim".
  std::string quoted;
  if (succeeded(parser.parseOptionalString(&quoted)))
    return SymbolExprAttr::get(ctx, quoted);

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
    return SymbolExprAttr::get(ctx, ident);
  }

  // Full sym-expression attribute (#sym.binary<...>, #sym.symbol<...>,
  // #sym.constant<...>): the legacy dim spelling of the !sym.tensor
  // grammar, admitted at any factor position.
  Attribute attr;
  OptionalParseResult attrResult = parser.parseOptionalAttribute(attr);
  if (attrResult.has_value()) {
    if (failed(*attrResult))
      return {};
    if (!isSymExpr(attr)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected a sym expression attribute");
      return {};
    }
    return attr;
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
    SymbolicExprOp opcode;
    if (succeeded(parser.parseOptionalStar()))
      opcode = SymbolicExprOp::Mul;
    else if (succeeded(parser.parseOptionalKeyword("floordiv")) ||
             succeeded(parser.parseOptionalKeyword("div")))
      opcode = SymbolicExprOp::Div;
    else if (succeeded(parser.parseOptionalKeyword("mod")))
      opcode = SymbolicExprOp::Mod;
    else
      return lhs;
    Attribute rhs = parseFactor(parser);
    if (!rhs)
      return {};
    lhs = getSimplifiedBinaryExpr(parser.getContext(), opcode, lhs, rhs);
  }
}

/// expr := term (('+' | '-') term)*
static Attribute parseExprImpl(AsmParser &parser) {
  Attribute lhs = parseTerm(parser);
  if (!lhs)
    return {};
  while (true) {
    SymbolicExprOp opcode;
    if (succeeded(parser.parseOptionalPlus()))
      opcode = SymbolicExprOp::Add;
    else if (succeeded(parser.parseOptionalMinus()))
      opcode = SymbolicExprOp::Sub;
    else
      return lhs;
    Attribute rhs = parseTerm(parser);
    if (!rhs)
      return {};
    lhs = getSimplifiedBinaryExpr(parser.getContext(), opcode, lhs, rhs);
  }
}

Attribute mlir::sym::parseSymExpr(AsmParser &parser) {
  return parseExprImpl(parser);
}

static int precedence(SymbolicExprOp op) {
  switch (op) {
  case SymbolicExprOp::Add:
  case SymbolicExprOp::Sub:
    return 1;
  case SymbolicExprOp::Mul:
  case SymbolicExprOp::Div:
  case SymbolicExprOp::Mod:
    return 2;
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

static StringRef opKeyword(SymbolicExprOp op) {
  switch (op) {
  case SymbolicExprOp::Add:
    return "+";
  case SymbolicExprOp::Sub:
    return "-";
  case SymbolicExprOp::Mul:
    return "*";
  case SymbolicExprOp::Div:
    return "floordiv";
  case SymbolicExprOp::Mod:
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
  return llvm::all_of(name,
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

static void printExprImpl(AsmPrinter &printer, Attribute expr, int parentPrec,
                          bool isRhs) {
  if (auto constant = dyn_cast<ConstantExprAttr>(expr)) {
    printer << constant.getValue();
    return;
  }
  if (auto symbol = dyn_cast<SymbolExprAttr>(expr)) {
    if (isBareIdent(symbol.getName()))
      printer << symbol.getName();
    else
      printer << "\"" << symbol.getName() << "\"";
    return;
  }
  auto binary = cast<BinaryExprAttr>(expr);
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

void mlir::sym::printSymExpr(AsmPrinter &printer, Attribute expr) {
  printExprImpl(printer, expr, /*parentPrec=*/0, /*isRhs=*/false);
}
