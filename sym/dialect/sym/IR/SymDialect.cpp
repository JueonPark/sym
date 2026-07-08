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
  registerAttributes();
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
#define GET_TYPEDEF_CLASSES
#include "SymTypes.cpp.inc"
#define GET_OP_CLASSES
#include "SymOps.cpp.inc"