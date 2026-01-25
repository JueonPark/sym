#include "SymDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

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

// --- Attributes Implementation ---

//===----------------------------------------------------------------------===//
// SymbolExprAttr Custom Assembly Format
//===----------------------------------------------------------------------===//

Attribute SymbolExprAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  std::string name;
  if (parser.parseString(&name))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), name);
}

void SymbolExprAttr::print(AsmPrinter &printer) const {
  printer << "<\"" << getName() << "\">";
}

//===----------------------------------------------------------------------===//
// ConstantExprAttr Custom Assembly Format
//===----------------------------------------------------------------------===//

Attribute ConstantExprAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  int64_t value;
  if (parser.parseInteger(value))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), value);
}

void ConstantExprAttr::print(AsmPrinter &printer) const {
  printer << "<" << getValue() << ">";
}

// 1. Verify BinaryExprAttr
LogicalResult
BinaryExprAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       SymbolicExprOp opcode, Attribute lhs, Attribute rhs) {

  // Ensure LHS is one of our known Expression Attributes
  bool validLHS = isa<SymbolExprAttr>(lhs) || isa<ConstantExprAttr>(lhs) ||
                  isa<BinaryExprAttr>(lhs);

  bool validRHS = isa<SymbolExprAttr>(rhs) || isa<ConstantExprAttr>(rhs) ||
                  isa<BinaryExprAttr>(rhs);

  if (!validLHS || !validRHS) {
    return emitError()
           << "BinaryExpr operands must be Symbol, Constant, or BinaryExpr";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BinaryExprAttr Custom Assembly Format
//===----------------------------------------------------------------------===//

Attribute BinaryExprAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse the opcode as a keyword
  StringRef opcodeStr;
  if (parser.parseKeyword(&opcodeStr))
    return {};

  auto opcode = symbolizeSymbolicExprOp(opcodeStr);
  if (!opcode) {
    parser.emitError(parser.getCurrentLocation(), "unknown opcode: ")
        << opcodeStr;
    return {};
  }

  if (parser.parseComma())
    return {};

  // Parse LHS attribute
  Attribute lhs;
  if (parser.parseAttribute(lhs))
    return {};

  if (parser.parseComma())
    return {};

  // Parse RHS attribute
  Attribute rhs;
  if (parser.parseAttribute(rhs))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), *opcode, lhs, rhs);
}

void BinaryExprAttr::print(AsmPrinter &printer) const {
  printer << "<" << stringifySymbolicExprOp(getOpcode()) << ", ";
  printer.printAttribute(getLhs());
  printer << ", ";
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

  return get(parser.getContext(), shape, elemType);
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
          // Print as prefix: add(lhs, rhs)
          printer << stringifySymbolicExprOp(a.getOpcode()) << "(";
          printer.printAttribute(a.getLhs());
          printer << ", ";
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