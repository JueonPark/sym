// RelocDialect.td - Main dialect definition for Reloc
#ifndef RELOC_DIALECT_TD
#define RELOC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Reloc Dialect Definition
//===----------------------------------------------------------------------===//

def Reloc_Dialect : Dialect {
  let name = "reloc";
  let cppNamespace = "::mlir::reloc";
  let summary = "A dialect for symbolic relocation / layout-transform plans";
  let description = [{
    The Reloc dialect provides a serializable, symbolic representation of
    folded layout-transform chains (RelocationPlan IR) and the intermediate
    op set that a downstream folding pass consumes. This is the dialect shell;
    attributes, ops, verifiers, and serialization are added in later
    sub-issues.
  }];

  // Use generated attribute printer/parser with custom hooks
  let useDefaultAttributePrinterParser = 1;

  // Reloc attributes are built directly from sym::SymbolExprAttr /
  // ConstantExprAttr / BinaryExprAttr via the C++ API (RelocUtils.cpp), not
  // through the textual `#sym....` syntax, so the SymDialect is never
  // force-loaded by the parser encountering a sym-prefixed attribute. List it
  // as a dependent dialect so loading Reloc also loads Sym.
  let dependentDialects = ["sym::SymDialect"];

  // Attribute registration is defined out-of-line in RelocAttributes.cpp,
  // the only TU where the generated attribute storage classes are complete
  // (see RelocAttributes.cpp's `GET_ATTRDEF_CLASSES` include). Calling
  // addAttributes<...>() requires a complete storage type in the same
  // translation unit, so this cannot live in RelocDialect.cpp's initialize().
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

// Base class for Reloc dialect attributes
class Reloc_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Reloc_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // RELOC_DIALECT_TD
