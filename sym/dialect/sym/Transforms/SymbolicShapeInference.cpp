//===- SymbolicShapeInference.cpp - Symbolic shape inference pass ---------===//
//
// This file implements the symbolic shape inference pass.
//
//===----------------------------------------------------------------------===//

#include "SymDialect.h"
#include "SymInterfaces.h"
#include "SymPasses.h"
#include "SymUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace sym {

#define GEN_PASS_DEF_SYMBOLICSHAPEINFERENCEPASS
#include "SymPasses.h.inc"

namespace {

/// Extract symbolic shape from a value's type.
/// Returns an empty vector if the type doesn't have symbolic shape info.
static SmallVector<Attribute> getSymbolicShape(Value value) {
  Type type = value.getType();

  // If it's already a SymbolicTensorType, extract its shape
  if (auto symTensor = dyn_cast<SymbolicTensorType>(type)) {
    return SmallVector<Attribute>(symTensor.getShape());
  }

  // If it's a ranked tensor with static shape, convert to ConstantExprAttr
  if (auto rankedTensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<Attribute> shape;
    for (int64_t dim : rankedTensor.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        // Dynamic dimension - we don't have symbolic info for it yet
        return {};
      }
      shape.push_back(ConstantExprAttr::get(value.getContext(), dim));
    }
    return shape;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SymbolicShapeInferencePass
//===----------------------------------------------------------------------===//

struct SymbolicShapeInferencePass
    : public impl::SymbolicShapeInferencePassBase<SymbolicShapeInferencePass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Walk all operations in the function
    funcOp.walk([&](Operation *op) {
      // Check if the operation implements our interface
      auto shapeInterface = dyn_cast<SymbolicShapeOpInterface>(op);
      if (!shapeInterface)
        return;

      // Collect symbolic shapes for all operands
      SmallVector<SmallVector<Attribute>> operandShapes;
      for (Value operand : op->getOperands()) {
        operandShapes.push_back(getSymbolicShape(operand));
      }

      // Check if we have symbolic info for at least one operand
      bool hasSymbolicInfo =
          llvm::any_of(operandShapes, [](const auto &s) { return !s.empty(); });

      if (!hasSymbolicInfo)
        return;

      // Convert to ArrayRef for the interface call
      SmallVector<ArrayRef<Attribute>> operandShapeRefs;
      for (const auto &shape : operandShapes) {
        operandShapeRefs.push_back(shape);
      }

      // Infer symbolic shapes for results
      SmallVector<SmallVector<Attribute>> inferredShapes;
      if (failed(shapeInterface.inferSymbolicShapes(context, operandShapeRefs,
                                                    inferredShapes))) {
        // Inference failed - leave the operation unchanged
        return;
      }

      // Update result types
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (i >= inferredShapes.size() || inferredShapes[i].empty())
          continue;

        Type elementType = shapeInterface.getResultElementType(i);
        if (!elementType)
          continue;

        Type newType =
            SymbolicTensorType::get(context, inferredShapes[i], elementType);
        op->getResult(i).setType(newType);
      }
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerSymPasses() { registerPasses(); }

} // namespace sym
} // namespace mlir
