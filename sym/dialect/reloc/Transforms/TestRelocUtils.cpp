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
      if (auto plan = op->getAttrOfType<PlanAttr>("plan")) {
        if (auto pair = op->getAttrOfType<DenseI64ArrayAttr>("pair"))
          testContiguous(op, plan, pair);
        else
          op->emitRemark()
              << "isPureView = " << (isPureView(plan) ? "true" : "false");
      }
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

  void testContiguous(Operation *op, PlanAttr plan, DenseI64ArrayAttr pair) {
    ArrayRef<int64_t> indices = pair.asArrayRef();
    ArrayRef<AxisInfoAttr> axes = plan.getAxes();
    if (indices.size() != 2 || indices[0] < 0 || indices[1] < 0 ||
        indices[0] >= static_cast<int64_t>(axes.size()) ||
        indices[1] >= static_cast<int64_t>(axes.size())) {
      op->emitRemark() << "isContiguousCompatible: pair index out of range";
      return;
    }
    bool compatible =
        isContiguousCompatible(axes[indices[0]], axes[indices[1]]);
    op->emitRemark() << "isContiguousCompatible(" << indices[0] << ", "
                     << indices[1] << ") = "
                     << (compatible ? "true" : "false");
  }
};

} // namespace

void registerRelocPasses() { registerPasses(); }

} // namespace reloc
} // namespace mlir
