//===- RelocPasses.td - Reloc dialect passes -------------------------------===//
//
// This file defines passes for the Reloc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PASSES_TD
#define RELOC_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TestRelocUtilsPass
//===----------------------------------------------------------------------===//

def TestRelocUtilsPass : Pass<"test-reloc-utils", "::mlir::ModuleOp"> {
  let summary = "Test pass exercising reloc utility functions";
  let description = [{
    Testing-only pass. Walks operations and emits remarks:
    - ops with an `expr` attribute: sym -> affine -> sym bridge round-trip
      verdict and the affine rendering;
    - ops with a `plan` attribute and a `pair` DenseI64ArrayAttr: the
      isContiguousCompatible(axes[pair[0]], axes[pair[1]]) result;
    - ops with only a `plan` attribute: the isPureView result.
  }];

  let dependentDialects = [
    "sym::SymDialect",
    "reloc::RelocDialect"
  ];
}

#endif // RELOC_PASSES_TD
