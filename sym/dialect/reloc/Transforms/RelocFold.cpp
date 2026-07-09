//===- RelocFold.cpp - Fold reloc chains into single plans ----------------===//
//
// The P1b fold pass (#B4): walks maximal straight-line reloc.* chains,
// folds them through the transfer functions, and materializes each result
// as a reloc.plan_result op. Bail semantics are all-or-nothing per chain
// (issue #15 design decision 2): any bail marks the whole chain with the
// reloc.fallback unit attribute and leaves the IR intact.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocDialect.h"
#include "RelocPasses.h"
#include "SymDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace reloc {

#define GEN_PASS_DEF_RELOCFOLDPASS
#include "RelocPasses.h.inc"

namespace {

/// Discardable marker for chains left to per-op fallback. Also serves as a
/// manual opt-out: marked ops are never folded.
constexpr StringLiteral kFallbackAttrName = "reloc.fallback";

struct RelocFoldPass : public impl::RelocFoldPassBase<RelocFoldPass> {
  void runOnOperation() override {
    // Collect tails first: rewriting invalidates walk iteration.
    SmallVector<Operation *> tails;
    getOperation().walk([&](Operation *op) {
      if (!isFoldableChainOp(op))
        return;
      for (Operation *user : op->getResult(0).getUsers())
        if (isFoldableChainOp(user))
          return;
      tails.push_back(op);
    });
    for (Operation *tail : tails)
      if (failed(processChain(tail)))
        return signalPassFailure();
  }

  /// Mark every chain op for per-op fallback; the chain stays intact.
  LogicalResult markFallback(ArrayRef<Operation *> chain) {
    for (Operation *op : chain)
      op->setAttr(kFallbackAttrName, UnitAttr::get(&getContext()));
    return success();
  }

  LogicalResult processChain(Operation *tail) {
    // Walk back the straight line to the chain root.
    SmallVector<Operation *> chain;
    Operation *def = tail;
    while (def && isFoldableChainOp(def)) {
      chain.push_back(def);
      def = def->getOperand(0).getDefiningOp();
    }
    std::reverse(chain.begin(), chain.end());

    // Already marked (previous bail or manual opt-out): leave alone. This
    // also makes the pass idempotent on bailed chains.
    for (Operation *op : chain)
      if (op->hasAttr(kFallbackAttrName))
        return success();

    // Structural bail (a): a non-tail member's value escapes the chain;
    // folding it away would need partial folding (and erasing it would be
    // invalid). All-or-nothing: mark and keep.
    for (Operation *op : chain)
      if (op != tail && !op->getResult(0).hasOneUse())
        return markFallback(chain);

    // Fold front-to-back; any transfer-function bail falls the whole
    // chain back (all-or-nothing).
    Value root = chain.front()->getOperand(0);
    PlanBuilder builder(cast<sym::SymbolicTensorType>(root.getType()));
    for (Operation *op : chain)
      if (failed(foldChainOp(builder, op)))
        return markFallback(chain);

    // In-pass verification: finalize builds the plan through
    // PlanAttr::getChecked. A null plan means the P1a verifier rejected
    // our own fold output - a pass error, never a silent skip. (The
    // transfer functions only construct verifier-clean plans, so this is
    // defensive.)
    PlanAttr plan = builder.finalize(tail->getLoc());
    if (!plan)
      return failure(); // getChecked already emitted the diagnostic

    OpBuilder rewriter(tail);
    auto materialized = rewriter.create<PlanResultOp>(
        tail->getLoc(), tail->getResult(0).getType(), root, plan);
    tail->getResult(0).replaceAllUsesWith(materialized.getResult());
    // Erase tail-first: each predecessor's single use dies with its
    // successor.
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
      (*it)->erase();
    return success();
  }
};

} // namespace
} // namespace reloc
} // namespace mlir
