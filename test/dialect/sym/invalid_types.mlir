// RUN: sym-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// Test that invalid element types are rejected by SymbolicTensorType verification

func.func @invalid_element_type() {
  %0 = "test.use_type"() : () -> !sym.tensor<["a"], memref<2xf32>>
  // expected-error @below {{element type must be a valid tensor element type}}
  return
}

// -----

// Operator keywords are not valid expression operands.
// expected-error @below {{unexpected operator keyword 'floordiv'; expected expression operand}}
func.func private @bad_dim() -> !sym.tensor<[floordiv, 4], f32>
