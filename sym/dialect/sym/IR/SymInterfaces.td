//===- SymInterfaces.td - Sym dialect interfaces --------------------------===//
//
// This file defines interfaces for symbolic shape inference.
//
//===----------------------------------------------------------------------===//

#ifndef SYM_INTERFACES_TD
#define SYM_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SymbolicShapeOpInterface
//===----------------------------------------------------------------------===//

def SymbolicShapeOpInterface : OpInterface<"SymbolicShapeOpInterface"> {
  let description = [{
    Interface for operations that can propagate symbolic shape information.
    
    Operations implementing this interface can describe how they compute
    their output shapes from their input shapes symbolically. This enables
    shape inference to work with symbolic dimensions (e.g., "batch", "seq_len")
    rather than just concrete integer dimensions.
    
    The interface is used by the SymbolicShapeInferencePass to propagate
    symbolic shape information through the IR.
  }];

  let cppNamespace = "::mlir::sym";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer the symbolic shape of the operation's results.
        
        Given the symbolic shapes of the operands (as arrays of Attribute,
        where each Attribute is a SymbolExprAttr, ConstantExprAttr, or
        BinaryExprAttr), compute the symbolic shape of each result.
        
        The inferredShapes vector should be populated with one SmallVector<Attribute>
        per result, representing the symbolic dimensions of that result.
        
        Returns success if inference succeeded, failure otherwise.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"inferSymbolicShapes",
      /*args=*/(ins
        "::mlir::MLIRContext *":$context,
        "::llvm::ArrayRef<::llvm::ArrayRef<::mlir::Attribute>>":$operandShapes,
        "::llvm::SmallVectorImpl<::llvm::SmallVector<::mlir::Attribute>> &":$inferredShapes
      )
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the element type for the operation's results.
        
        For element-wise operations, this typically returns the element type
        of the first operand. For type-changing operations, this may differ.
        
        Returns the element type to use for the result tensor.
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getResultElementType",
      /*args=*/(ins
        "unsigned":$resultIndex
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: return the element type of the first tensor operand
        for (auto operand : $_op->getOperands()) {
          if (auto tensorTy = ::mlir::dyn_cast<::mlir::TensorType>(operand.getType()))
            return tensorTy.getElementType();
          if (auto symTensorTy = ::mlir::dyn_cast<::mlir::sym::SymbolicTensorType>(operand.getType()))
            return symTensorTy.getElementType();
        }
        return {};
      }]
    >
  ];
}

#endif // SYM_INTERFACES_TD
