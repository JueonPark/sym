//===- SymExtensions.cpp - External model implementations -----------------===//
//
// This file implements external models to attach SymbolicShapeOpInterface
// to standard MLIR operations (arith dialect).
//
//===----------------------------------------------------------------------===//

#include "SymDialect.h"
#include "SymInterfaces.h"
#include "SymUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::sym;

namespace {

//===----------------------------------------------------------------------===//
// Arith Dialect External Models
//===----------------------------------------------------------------------===//

/// Helper to implement broadcasting for element-wise binary operations.
/// This is shared by AddIOp, AddFOp, SubIOp, SubFOp, MulIOp, MulFOp, etc.
template <typename OpTy>
struct ElementwiseBinaryOpInterface
    : public SymbolicShapeOpInterface::ExternalModel<
          ElementwiseBinaryOpInterface<OpTy>, OpTy> {

  LogicalResult inferSymbolicShapes(
      Operation *op, MLIRContext *context,
      ArrayRef<ArrayRef<Attribute>> operandShapes,
      SmallVectorImpl<SmallVector<Attribute>> &inferredShapes) const {
    // Element-wise binary ops have exactly 2 operands and 1 result
    if (operandShapes.size() != 2)
      return failure();

    ArrayRef<Attribute> lhsShape = operandShapes[0];
    ArrayRef<Attribute> rhsShape = operandShapes[1];

    // If either operand has no symbolic shape, we can't infer
    if (lhsShape.empty() && rhsShape.empty())
      return failure();

    // If one operand has no symbolic shape, use the other
    if (lhsShape.empty()) {
      inferredShapes.push_back(SmallVector<Attribute>(rhsShape));
      return success();
    }
    if (rhsShape.empty()) {
      inferredShapes.push_back(SmallVector<Attribute>(lhsShape));
      return success();
    }

    // Both have symbolic shapes - use UnificationSolver for broadcasting
    UnificationSolver solver(context, op->getLoc());
    auto result = solver.unify(lhsShape, rhsShape);
    if (failed(result))
      return failure();

    inferredShapes.push_back(std::move(*result));
    return success();
  }

  Type getResultElementType(Operation *op, unsigned resultIndex) const {
    // Return element type from first tensor operand
    for (Value operand : op->getOperands()) {
      if (auto tensorTy = dyn_cast<TensorType>(operand.getType()))
        return tensorTy.getElementType();
      if (auto symTensorTy = dyn_cast<SymbolicTensorType>(operand.getType()))
        return symTensorTy.getElementType();
    }
    return {};
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::sym::registerSymbolicShapeOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    // Integer arithmetic
    arith::AddIOp::attachInterface<ElementwiseBinaryOpInterface<arith::AddIOp>>(
        *ctx);
    arith::SubIOp::attachInterface<ElementwiseBinaryOpInterface<arith::SubIOp>>(
        *ctx);
    arith::MulIOp::attachInterface<ElementwiseBinaryOpInterface<arith::MulIOp>>(
        *ctx);
    arith::DivSIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::DivSIOp>>(*ctx);
    arith::DivUIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::DivUIOp>>(*ctx);
    arith::RemSIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::RemSIOp>>(*ctx);
    arith::RemUIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::RemUIOp>>(*ctx);

    // Floating-point arithmetic
    arith::AddFOp::attachInterface<ElementwiseBinaryOpInterface<arith::AddFOp>>(
        *ctx);
    arith::SubFOp::attachInterface<ElementwiseBinaryOpInterface<arith::SubFOp>>(
        *ctx);
    arith::MulFOp::attachInterface<ElementwiseBinaryOpInterface<arith::MulFOp>>(
        *ctx);
    arith::DivFOp::attachInterface<ElementwiseBinaryOpInterface<arith::DivFOp>>(
        *ctx);
    arith::RemFOp::attachInterface<ElementwiseBinaryOpInterface<arith::RemFOp>>(
        *ctx);

    // Comparison operations (result is i1 tensor, but same shape as inputs)
    arith::CmpIOp::attachInterface<ElementwiseBinaryOpInterface<arith::CmpIOp>>(
        *ctx);
    arith::CmpFOp::attachInterface<ElementwiseBinaryOpInterface<arith::CmpFOp>>(
        *ctx);

    // Bitwise operations
    arith::AndIOp::attachInterface<ElementwiseBinaryOpInterface<arith::AndIOp>>(
        *ctx);
    arith::OrIOp::attachInterface<ElementwiseBinaryOpInterface<arith::OrIOp>>(
        *ctx);
    arith::XOrIOp::attachInterface<ElementwiseBinaryOpInterface<arith::XOrIOp>>(
        *ctx);

    // Min/Max operations
    arith::MaxSIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MaxSIOp>>(*ctx);
    arith::MaxUIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MaxUIOp>>(*ctx);
    arith::MinSIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MinSIOp>>(*ctx);
    arith::MinUIOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MinUIOp>>(*ctx);
    arith::MaximumFOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MaximumFOp>>(*ctx);
    arith::MinimumFOp::attachInterface<
        ElementwiseBinaryOpInterface<arith::MinimumFOp>>(*ctx);
  });
}
