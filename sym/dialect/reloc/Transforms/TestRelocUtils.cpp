//===- TestRelocUtils.cpp - Test pass for reloc utilities ------------------===//
//
// Testing-only pass that exposes RelocUtils (bridge, predicates) to lit via
// remarks. See RelocPasses.td for the op-attribute protocol.
//
//===----------------------------------------------------------------------===//

#include "RelocDialect.h"
#include "RelocPasses.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace reloc {

#define GEN_PASS_DEF_TESTRELOCUTILSPASS
#include "RelocPasses.h.inc"

namespace {

/// Render an AffineExpr to a string for remark output.
static std::string renderAffine(AffineExpr expr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << expr;
  return text;
}

struct TestRelocUtilsPass
    : public impl::TestRelocUtilsPassBase<TestRelocUtilsPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    getOperation().walk([&](Operation *op) {
      if (Attribute expr = op->getAttr("expr"))
        testBridge(op, expr, context);
      // Task 7 extends this walk with plan-predicate remarks.
    });
  }

  void testBridge(Operation *op, Attribute expr, MLIRContext *context) {
    SmallVector<StringRef> symbolNames;
    FailureOr<AffineExpr> affine = symToAffine(expr, symbolNames, context);
    if (failed(affine)) {
      op->emitRemark() << "bridge: not a sym expression";
      return;
    }
    Attribute back = affineToSym(*affine, symbolNames, context);
    bool identical =
        back && sym::UnificationSolver::areLogicallyEqual(expr, back);
    op->emitRemark() << "bridge round-trip "
                     << (identical ? "ok" : "MISMATCH")
                     << ": affine = " << renderAffine(*affine);
  }
};

} // namespace

void registerRelocPasses() { registerPasses(); }

} // namespace reloc
} // namespace mlir
