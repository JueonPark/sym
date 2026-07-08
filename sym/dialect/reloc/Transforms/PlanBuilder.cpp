//===- PlanBuilder.cpp - In-progress reloc plan state ---------------------===//
//
// This file implements the P1b folding state and its transfer functions.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::reloc;

PlanBuilder::PlanBuilder(sym::SymbolicTensorType input)
    : ctx(input.getContext()) {
  ArrayRef<Attribute> shape = input.getShape();
  assert(!shape.empty() && "rank-0 tensors have no relocation plan");
  Attribute zero = sym::ConstantExprAttr::get(ctx, 0);
  src = TensorDescAttr::get(ctx, shape, /*strides=*/ArrayRef<Attribute>(),
                            zero, input.getElementType());
  SmallVector<Attribute> srcStrides = canonicalRowMajorStrides(shape, ctx);
  for (auto [k, extent] : llvm::enumerate(shape)) {
    axes.push_back({("d" + Twine(k)).str(), extent, srcStrides[k]});
    perm.push_back(static_cast<int64_t>(k));
  }
}

PlanAttr PlanBuilder::finalize(Location loc) const {
  return finalizePlanBuilder(*this, loc);
}
