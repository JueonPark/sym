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
}

#endif // RELOC_DIALECT_TD
