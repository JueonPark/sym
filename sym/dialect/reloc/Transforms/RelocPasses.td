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

//===----------------------------------------------------------------------===//
// TestRelocTransferPass
//===----------------------------------------------------------------------===//

def TestRelocTransferPass : Pass<"test-reloc-transfer", "::mlir::ModuleOp"> {
  let summary = "Test pass folding straight-line reloc chains via transfer functions";
  let description = [{
    Testing-only pass. For every foldable chain op (reloc.transpose,
    reloc.reshape, reloc.pad) whose result has no reloc user (a chain tail),
    walks back through the straight-line chain to its root, seeds a
    PlanBuilder from the root's type, applies the transfer functions in
    order, and emits the finalized #reloc.plan as a remark on the tail. A
    transfer function that bails emits `fold bail: <op name>` on the
    failing op instead.
  }];

  let dependentDialects = [
    "sym::SymDialect",
    "reloc::RelocDialect"
  ];
}

//===----------------------------------------------------------------------===//
// RelocFoldPass
//===----------------------------------------------------------------------===//

def RelocFoldPass : Pass<"reloc-fold", "::mlir::ModuleOp"> {
  let summary = "Fold straight-line reloc.* chains into reloc.plan_result ops";
  let description = [{
    Identifies maximal straight-line chains of reloc.transpose /
    reloc.reshape / reloc.pad ops, folds each front-to-back through the
    P1b transfer functions, and materializes the result as a
    reloc.plan_result op carrying the folded #reloc.plan. Folding is
    all-or-nothing per chain: when a transfer function bails, a non-reloc
    op interrupts the chain, or an intermediate value has multiple uses,
    every op in the chain is marked with a `reloc.fallback` unit attribute
    and left intact (per-op fallback). A chain containing an already-marked
    op (previous bail or manual opt-out) is itself marked in full rather
    than being left alone, which preserves the invariant that after the
    pass every foldable reloc op is either folded into a reloc.plan_result
    or carries reloc.fallback; re-marking is idempotent, so the pass is
    still a no-op on its own output. Every produced plan passes the
    #reloc.plan verifier; a verification failure is a pass error, never a
    silent skip.
  }];

  let dependentDialects = [
    "sym::SymDialect",
    "reloc::RelocDialect"
  ];
}

#endif // RELOC_PASSES_TD
