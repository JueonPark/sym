// SymDialect.td - Main dialect definition for Sym
#ifndef SYM_DIALECT_TD
#define SYM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Sym Dialect Definition
//===----------------------------------------------------------------------===//

def Sym_Dialect : Dialect {
  let name = "sym";
  let cppNamespace = "::mlir::sym";
  let summary = "A symbolic, algebraic type inference dialect";
  let description = [{
    The Sym dialect provides infrastructure for symbolic and algebraic
    type inference in MLIR. It supports symbolic expressions, tensor
    types with symbolic dimensions, and operations for manipulating
    symbolic values.
  }];

  // Use generated type/attribute printer/parser with custom hooks
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

// Base class for Sym dialect types
class Sym_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Sym_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Base class for Sym dialect attributes  
class Sym_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Sym_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // SYM_DIALECT_TD
