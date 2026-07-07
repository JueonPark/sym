#include "SymDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::sym;

// Include generated dialect definitions
#include "SymDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Sym Dialect Initialization
//===----------------------------------------------------------------------===//

void SymDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SymTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "SymAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "SymOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SymbolicExprOp Enum Helpers
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::sym::stringifySymbolicExprOp(SymbolicExprOp op) {
  switch (op) {
  case SymbolicExprOp::Add:
    return "add";
  case SymbolicExprOp::Sub:
    return "sub";
  case SymbolicExprOp::Mul:
    return "mul";
  case SymbolicExprOp::Div:
    return "div";
  case SymbolicExprOp::Mod:
    return "mod";
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

std::optional<SymbolicExprOp>
mlir::sym::symbolizeSymbolicExprOp(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<SymbolicExprOp>>(str)
      .Case("add", SymbolicExprOp::Add)
      .Case("sub", SymbolicExprOp::Sub)
      .Case("mul", SymbolicExprOp::Mul)
      .Case("div", SymbolicExprOp::Div)
      .Case("mod", SymbolicExprOp::Mod)
      .Default(std::nullopt);
}

// Operator symbol helpers for infix notation
llvm::StringRef getOperatorSymbol(SymbolicExprOp op) {
  switch (op) {
  case SymbolicExprOp::Add:
    return "+";
  case SymbolicExprOp::Sub:
    return "-";
  case SymbolicExprOp::Mul:
    return "*";
  case SymbolicExprOp::Div:
    return "div";
  case SymbolicExprOp::Mod:
    return "mod";
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

std::optional<SymbolicExprOp> symbolizeFromOperator(AsmParser &parser) {
  // Try to parse operator symbols: +, -, *, /, %
  if (succeeded(parser.parseOptionalPlus()))
    return SymbolicExprOp::Add;
  if (succeeded(parser.parseOptionalMinus()))
    return SymbolicExprOp::Sub;
  if (succeeded(parser.parseOptionalStar()))
    return SymbolicExprOp::Mul;
  // For division (/) - parse using keyword "div"
  if (succeeded(parser.parseOptionalKeyword("div")))
    return SymbolicExprOp::Div;
  // For modulo (%) - parse using keyword "mod"
  if (succeeded(parser.parseOptionalKeyword("mod")))
    return SymbolicExprOp::Mod;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Expression Simplification Helpers
//===----------------------------------------------------------------------===//

/// Check if two expression attributes are logically equal.
/// This handles commutativity for Add and Mul operations.
static bool areLogicallyEqual(Attribute a, Attribute b) {
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

/// Helper to check if an attribute is a constant with a specific value.
static bool isConstantValue(Attribute attr, int64_t value) {
  if (auto constAttr = dyn_cast<ConstantExprAttr>(attr))
    return constAttr.getValue() == value;
  return false;
}

/// Simplify a binary expression. Returns the simplified attribute, which may be
/// a ConstantExprAttr, SymbolExprAttr, or BinaryExprAttr.
/// This function performs:
/// - Constant folding: 3 + 4 -> 7
/// - Identity removal: x + 0 -> x, x * 1 -> x, x - 0 -> x, x / 1 -> x
/// - Zero propagation: x * 0 -> 0, 0 / x -> 0, 0 % x -> 0
/// - Self-cancellation: x - x -> 0
/// - Associativity for constants: (x + 1) + 2 -> x + 3, (x * 2) * 3 -> x * 6
static Attribute simplifyBinaryExpr(MLIRContext *context, SymbolicExprOp opcode,
                                    Attribute lhs, Attribute rhs) {
  auto constLhs = dyn_cast<ConstantExprAttr>(lhs);
  auto constRhs = dyn_cast<ConstantExprAttr>(rhs);

  // 1. Constant folding: both operands are constants
  if (constLhs && constRhs) {
    int64_t l = constLhs.getValue();
    int64_t r = constRhs.getValue();
    int64_t result;
    switch (opcode) {
    case SymbolicExprOp::Add:
      result = l + r;
      break;
    case SymbolicExprOp::Sub:
      result = l - r;
      break;
    case SymbolicExprOp::Mul:
      result = l * r;
      break;
    case SymbolicExprOp::Div:
      if (r == 0)
        return nullptr; // Division by zero - don't simplify
      // Floor division (see SymbolicExprOp semantics note in SymDialect.h).
      result = llvm::divideFloorSigned(l, r);
      break;
    case SymbolicExprOp::Mod:
      if (r == 0)
        return nullptr; // Mod by zero - don't simplify
      // Floor modulo: l - floor(l/r) * r has the sign of the divisor.
      result = l - llvm::divideFloorSigned(l, r) * r;
      break;
    }
    return ConstantExprAttr::get(context, result);
  }

  // 2. Identity and zero propagation rules
  switch (opcode) {
  case SymbolicExprOp::Add:
    // x + 0 -> x
    if (isConstantValue(rhs, 0))
      return lhs;
    // 0 + x -> x
    if (isConstantValue(lhs, 0))
      return rhs;
    break;

  case SymbolicExprOp::Sub:
    // x - 0 -> x
    if (isConstantValue(rhs, 0))
      return lhs;
    // x - x -> 0
    if (areLogicallyEqual(lhs, rhs))
      return ConstantExprAttr::get(context, 0);
    break;

  case SymbolicExprOp::Mul:
    // x * 1 -> x
    if (isConstantValue(rhs, 1))
      return lhs;
    // 1 * x -> x
    if (isConstantValue(lhs, 1))
      return rhs;
    // x * 0 -> 0
    if (isConstantValue(rhs, 0))
      return ConstantExprAttr::get(context, 0);
    // 0 * x -> 0
    if (isConstantValue(lhs, 0))
      return ConstantExprAttr::get(context, 0);
    break;

  case SymbolicExprOp::Div:
    // x / 1 -> x
    if (isConstantValue(rhs, 1))
      return lhs;
    // 0 / x -> 0 (assuming x != 0)
    if (isConstantValue(lhs, 0))
      return ConstantExprAttr::get(context, 0);
    break;

  case SymbolicExprOp::Mod:
    // x % 1 -> 0
    if (isConstantValue(rhs, 1))
      return ConstantExprAttr::get(context, 0);
    // 0 % x -> 0
    if (isConstantValue(lhs, 0))
      return ConstantExprAttr::get(context, 0);
    break;
  }

  // 3. Associativity: (x op c1) op c2 -> x op (c1 op c2) for Add/Mul
  //    Also: (x + c1) - c2 -> x + (c1 - c2) and (x - c1) + c2 -> x + (c2 - c1)
  if (auto binLhs = dyn_cast<BinaryExprAttr>(lhs)) {
    if (constRhs) {
      auto innerConstRhs = dyn_cast<ConstantExprAttr>(binLhs.getRhs());
      if (innerConstRhs) {
        Attribute innerLhs = binLhs.getLhs();
        int64_t c1 = innerConstRhs.getValue();
        int64_t c2 = constRhs.getValue();

        // (x + c1) + c2 -> x + (c1 + c2)
        if (opcode == SymbolicExprOp::Add &&
            binLhs.getOpcode() == SymbolicExprOp::Add) {
          Attribute newConst = ConstantExprAttr::get(context, c1 + c2);
          if (Attribute simplified = simplifyBinaryExpr(
                  context, SymbolicExprOp::Add, innerLhs, newConst))
            return simplified;
          return BinaryExprAttr::get(context, SymbolicExprOp::Add, innerLhs,
                                     newConst);
        }

        // (x + c1) - c2 -> x + (c1 - c2)
        if (opcode == SymbolicExprOp::Sub &&
            binLhs.getOpcode() == SymbolicExprOp::Add) {
          Attribute newConst = ConstantExprAttr::get(context, c1 - c2);
          if (Attribute simplified = simplifyBinaryExpr(
                  context, SymbolicExprOp::Add, innerLhs, newConst))
            return simplified;
          return BinaryExprAttr::get(context, SymbolicExprOp::Add, innerLhs,
                                     newConst);
        }

        // (x - c1) + c2 -> x + (c2 - c1)
        if (opcode == SymbolicExprOp::Add &&
            binLhs.getOpcode() == SymbolicExprOp::Sub) {
          Attribute newConst = ConstantExprAttr::get(context, c2 - c1);
          if (Attribute simplified = simplifyBinaryExpr(
                  context, SymbolicExprOp::Add, innerLhs, newConst))
            return simplified;
          return BinaryExprAttr::get(context, SymbolicExprOp::Add, innerLhs,
                                     newConst);
        }

        // (x - c1) - c2 -> x - (c1 + c2)
        if (opcode == SymbolicExprOp::Sub &&
            binLhs.getOpcode() == SymbolicExprOp::Sub) {
          Attribute newConst = ConstantExprAttr::get(context, c1 + c2);
          if (Attribute simplified = simplifyBinaryExpr(
                  context, SymbolicExprOp::Sub, innerLhs, newConst))
            return simplified;
          return BinaryExprAttr::get(context, SymbolicExprOp::Sub, innerLhs,
                                     newConst);
        }

        // (x * c1) * c2 -> x * (c1 * c2)
        if (opcode == SymbolicExprOp::Mul &&
            binLhs.getOpcode() == SymbolicExprOp::Mul) {
          Attribute newConst = ConstantExprAttr::get(context, c1 * c2);
          if (Attribute simplified = simplifyBinaryExpr(
                  context, SymbolicExprOp::Mul, innerLhs, newConst))
            return simplified;
          return BinaryExprAttr::get(context, SymbolicExprOp::Mul, innerLhs,
                                     newConst);
        }
      }
    }
  }

  // No simplification possible - return nullptr to signal caller should create
  // the BinaryExprAttr as-is
  return nullptr;
}

/// Public API: Get a simplified BinaryExprAttr. This is the main entry point
/// for creating binary expressions with automatic simplification.
Attribute mlir::sym::getSimplifiedBinaryExpr(MLIRContext *context,
                                             SymbolicExprOp opcode,
                                             Attribute lhs, Attribute rhs) {
  if (Attribute simplified = simplifyBinaryExpr(context, opcode, lhs, rhs))
    return simplified;
  // No simplification - create the BinaryExprAttr directly
  return BinaryExprAttr::get(context, opcode, lhs, rhs);
}

// --- Attributes Implementation ---

// 1. Verify BinaryExprAttr
LogicalResult
BinaryExprAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       SymbolicExprOp opcode, Attribute lhs, Attribute rhs) {

  // Ensure LHS is one of our known Expression Attributes
  bool validLHS = mlir::isa<SymbolExprAttr>(lhs) ||
                  mlir::isa<ConstantExprAttr>(lhs) ||
                  mlir::isa<BinaryExprAttr>(lhs);

  bool validRHS = mlir::isa<SymbolExprAttr>(rhs) ||
                  mlir::isa<ConstantExprAttr>(rhs) ||
                  mlir::isa<BinaryExprAttr>(rhs);

  if (!validLHS || !validRHS) {
    return emitError()
           << "BinaryExpr operands must be Symbol, Constant, or BinaryExpr";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicTensorType Verification
//===----------------------------------------------------------------------===//

LogicalResult
SymbolicTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           ArrayRef<Attribute> shape, Type elementType) {
  // Verify each shape dimension is a valid symbolic expression attribute
  for (size_t idx = 0; idx < shape.size(); ++idx) {
    Attribute attr = shape[idx];
    bool isValidShapeAttr = mlir::isa<SymbolExprAttr>(attr) ||
                            mlir::isa<ConstantExprAttr>(attr) ||
                            mlir::isa<BinaryExprAttr>(attr);

    if (!isValidShapeAttr) {
      return emitError() << "shape dimension " << idx
                         << " must be SymbolExprAttr, ConstantExprAttr, or "
                            "BinaryExprAttr, but got: "
                         << attr;
    }
  }

  // Verify elementType is a valid tensor element type
  if (!elementType.isIntOrIndexOrFloat() &&
      !mlir::isa<ComplexType>(elementType)) {
    return emitError() << "element type must be a valid tensor element type, "
                          "but got: "
                       << elementType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BinaryExprAttr Custom Assembly Format
//===----------------------------------------------------------------------===//

Attribute BinaryExprAttr::parse(AsmParser &parser, Type type) {
  // Parse format: <lhs op rhs>
  // e.g., #sym.binary<#sym.symbol<"a"> + #sym.constant<1>>
  // Operators: + (add), - (sub), * (mul), // (div), % (mod)

  if (parser.parseLess())
    return {};

  // Parse lhs attribute
  Attribute lhs;
  if (parser.parseAttribute(lhs))
    return {};

  // Parse the operator symbol
  auto opcode = symbolizeFromOperator(parser);
  if (!opcode) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected operator: +, -, *, //, or %");
    return {};
  }

  // Parse rhs attribute
  Attribute rhs;
  if (parser.parseAttribute(rhs))
    return {};

  if (parser.parseGreater())
    return {};

  // Use getSimplifiedBinaryExpr to automatically simplify the expression
  return getSimplifiedBinaryExpr(parser.getContext(), *opcode, lhs, rhs);
}

void BinaryExprAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getLhs());
  printer << " " << getOperatorSymbol(getOpcode()) << " ";
  printer.printAttribute(getRhs());
  printer << ">";
}

// --- Type Parser (Custom Assembly) ---

// We want to support syntax: !sym.tensor<[32, "a" + 1], f32>
// This requires a Recursive Descent Parser in C++ to build the Attribute Tree.

Attribute parseExpression(AsmParser &parser) {
  // 1. Try Parse Integer (Constant)
  int64_t val;
  OptionalParseResult intResult = parser.parseOptionalInteger(val);
  if (intResult.has_value() && succeeded(*intResult)) {
    return ConstantExprAttr::get(parser.getContext(), val);
  }

  // 2. Try Parse String (Symbol)
  std::string symName;
  if (succeeded(parser.parseOptionalString(&symName)) && !symName.empty()) {
    return SymbolExprAttr::get(parser.getContext(), symName);
  }

  // 3. Try Parse Parentheses for Binary Ops "(lhs + rhs)"
  // NOTE: Full infix parsing (precedence) is complex.
  // MLIR often prefers prefix notation in attributes: #sym.op<add, ...>
  // If you want infix "a + b", you need to write a shunting-yard parser here.

  // Fallback: Use standard Attribute parser
  Attribute attr;
  if (parser.parseAttribute(attr))
    return {};
  return attr;
}

Type SymbolicTensorType::parse(AsmParser &parser) {
  // Parse wrapper <[ ... ]>
  if (parser.parseLess() || parser.parseLSquare())
    return Type();

  SmallVector<Attribute, 4> shape;

  // Parse Comma Separated List of Expressions
  auto parseElem = [&]() -> ParseResult {
    Attribute attr = parseExpression(parser);
    if (!attr)
      return failure();
    shape.push_back(attr);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseElem))
    return Type();

  if (parser.parseRSquare() || parser.parseComma())
    return Type();

  Type elemType;
  if (parser.parseType(elemType))
    return Type();
  if (parser.parseGreater())
    return Type();

  // Use getChecked to properly emit diagnostics on verification failure
  return getChecked(
      [&]() { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), shape, elemType);
}

// --- Type Printer ---

void SymbolicTensorType::print(AsmPrinter &printer) const {
  printer << "<[";
  for (auto it : llvm::enumerate(getShape())) {
    Attribute attr = it.value();

    // Pretty Print logic
    TypeSwitch<Attribute>(attr)
        .Case<ConstantExprAttr>([&](auto a) { printer << a.getValue(); })
        .Case<SymbolExprAttr>(
            [&](auto a) { printer << "\"" << a.getName() << "\""; })
        .Case<BinaryExprAttr>([&](auto a) {
          // Print as infix: (lhs op rhs)
          printer << "(";
          printer.printAttribute(a.getLhs());
          printer << " " << getOperatorSymbol(a.getOpcode()) << " ";
          printer.printAttribute(a.getRhs());
          printer << ")";
        })
        .Default([&](Attribute a) { printer << a; }); // Fallback

    if (it.index() != getShape().size() - 1)
      printer << ", ";
  }
  printer << "], " << getElementType() << ">";
}

// boilerplate
#define GET_ATTRDEF_CLASSES
#include "SymAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "SymTypes.cpp.inc"
#define GET_OP_CLASSES
#include "SymOps.cpp.inc"