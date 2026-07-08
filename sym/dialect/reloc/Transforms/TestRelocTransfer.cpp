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
    if (isa<TransposeOp, ReshapeOp, PadOp>(user))
      return true;
  return false;
}

struct TestRelocTransferPass
    : public impl::TestRelocTransferPassBase<TestRelocTransferPass> {
  void runOnOperation() override {
    getOperation().walk([&](TransposeOp tail) {
      if (hasRelocUser(tail.getResult()))
        return;
      // Walk back through the straight-line transpose chain to its root.
      SmallVector<TransposeOp> chain;
      Operation *def = tail;
      while (auto op = dyn_cast_or_null<TransposeOp>(def)) {
        chain.push_back(op);
        def = op.getInput().getDefiningOp();
      }
      std::reverse(chain.begin(), chain.end());
      PlanBuilder builder(
          cast<sym::SymbolicTensorType>(chain.front().getInput().getType()));
      for (TransposeOp op : chain)
        foldTranspose(builder, op.getPerm());
      if (PlanAttr plan = builder.finalize(tail.getLoc()))
        tail->emitRemark() << "folded plan: " << plan;
    });
  }
};

} // namespace
} // namespace reloc
} // namespace mlir
