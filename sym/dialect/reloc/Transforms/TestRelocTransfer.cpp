//===- TestRelocTransfer.cpp - Test pass for P1b transfer functions -------===//
//
// Testing-only pass that folds straight-line reloc op chains through the
// P1b transfer functions and reports the finalized #reloc.plan as a remark
// on each chain tail. See RelocPasses.td for the protocol.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocDialect.h"
#include "RelocPasses.h"
#include "SymDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace reloc {

#define GEN_PASS_DEF_TESTRELOCTRANSFERPASS
#include "RelocPasses.h.inc"

namespace {

/// True if any user of `value` is a reloc op (the chain continues past it).
static bool hasRelocUser(Value value) {
  for (Operation *user : value.getUsers())
    if (isFoldableChainOp(user))
      return true;
  return false;
}

struct TestRelocTransferPass
    : public impl::TestRelocTransferPassBase<TestRelocTransferPass> {
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (!isFoldableChainOp(op) || hasRelocUser(op->getResult(0)))
        return;
      // Walk back through the straight-line foldable chain to its root.
      SmallVector<Operation *> chain;
      Operation *def = op;
      while (def && isFoldableChainOp(def)) {
        chain.push_back(def);
        def = def->getOperand(0).getDefiningOp();
      }
      std::reverse(chain.begin(), chain.end());
      PlanBuilder builder(cast<sym::SymbolicTensorType>(
          chain.front()->getOperand(0).getType()));
      for (Operation *link : chain) {
        if (failed(foldChainOp(builder, link))) {
          link->emitRemark() << "fold bail: " << link->getName();
          return;
        }
      }
      if (PlanAttr plan = builder.finalize(op->getLoc()))
        op->emitRemark() << "folded plan: " << plan;
    });
  }
};

} // namespace
} // namespace reloc
} // namespace mlir
