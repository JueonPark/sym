// RUN: sym-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// Static legality of the typed value transforms (C1, issue #141). Every
// rejection below is a verifier or parser diagnostic; dynamic obligations
// (runtime parameter values, symbolic channel lengths) are deliberately not
// rejected here and are handed to C3 (docs/reloc-typed-semantics.md).

//===----------------------------------------------------------------------===//
// reloc.cast: type pairs, policy, shape
//===----------------------------------------------------------------------===//

func.func @cast_same_type(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{cast from 'f32' to 'f32' is not a supported typed conversion (f32 -> f16, f16 -> f32)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @cast_unsupported_pair(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{cast from 'f32' to 'f64' is not a supported typed conversion (f32 -> f16, f16 -> f32)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], f64>
  return
}

// -----

// A cast never changes integer storage: quantization is a distinct operation.
func.func @cast_to_int8(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{cast from 'f32' to 'i8' is not a supported typed conversion (f32 -> f16, f16 -> f32)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @cast_wrong_policy_narrow(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{policy 'exact' is not defined for the f32 -> f16 cast (use 'ieee_rne')}}
  %0 = reloc.cast %t policy exact : !sym.tensor<[4], f32> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @cast_wrong_policy_widen(%t: !sym.tensor<[4], f16>) {
  // expected-error @below {{policy 'ieee_rne' is not defined for the f16 -> f32 cast (use 'exact')}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4], f16> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @cast_quantize_policy(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{policy 'symmetric_rne' is not defined for the f32 -> f16 cast (use 'ieee_rne')}}
  %0 = reloc.cast %t policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @cast_unknown_policy(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{unknown numerical policy 'nearest_up'}}
  %0 = reloc.cast %t policy nearest_up : !sym.tensor<[4], f32> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @cast_rank_change(%t: !sym.tensor<[4, 6], f32>) {
  // expected-error @below {{result rank (1) must match operand rank (2)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[24], f16>
  return
}

// -----

func.func @cast_shape_change(%t: !sym.tensor<[4, 6], f32>) {
  // expected-error @below {{result dimension 1 must equal operand dimension 1 (value transforms preserve the logical shape)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 7], f16>
  return
}

// -----

func.func @cast_symbolic_shape_change(%t: !sym.tensor<["N", 6], f32>) {
  // expected-error @below {{result dimension 0 must equal operand dimension 0 (value transforms preserve the logical shape)}}
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<["N", 6], f32> -> !sym.tensor<["M", 6], f16>
  return
}

//===----------------------------------------------------------------------===//
// reloc.quantize: type pairs, policy, missing parameters
//===----------------------------------------------------------------------===//

// -----

func.func @quantize_f16_input(%t: !sym.tensor<[4], f16>) {
  // expected-error @below {{quantize expects an f32 operand and a signless i8 result (int8 signedness is declared by the operation, not by the storage type), but got 'f16' -> 'i8'}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f16> -> !sym.tensor<[4], i8>
  return
}

// -----

// Signed/unsigned integer element types are rejected: storage is signless i8
// and the signed interpretation is part of the operation semantics.
func.func @quantize_si8_result(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{quantize expects an f32 operand and a signless i8 result (int8 signedness is declared by the operation, not by the storage type), but got 'f32' -> 'si8'}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], si8>
  return
}

// -----

func.func @quantize_wrong_policy(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{policy 'affine' is not defined for reloc.quantize (use 'symmetric_rne')}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_missing_scale(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{requires attribute 'scale'}}
  %0 = "reloc.quantize"(%t) {policy = #reloc.policy<symmetric_rne>} : (!sym.tensor<[4], f32>) -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_missing_policy(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{requires attribute 'policy'}}
  %0 = "reloc.quantize"(%t) {scale = dense<0.5> : tensor<f32>} : (!sym.tensor<[4], f32>) -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_shape_change(%t: !sym.tensor<[4, 6], f32>) {
  // expected-error @below {{result dimension 0 must equal operand dimension 0 (value transforms preserve the logical shape)}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 6], i8>
  return
}

//===----------------------------------------------------------------------===//
// Constant parameters: kinds, element types, ranks, values
//===----------------------------------------------------------------------===//

// -----

func.func @quantize_scale_wrong_kind(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale must be a dense constant or a #reloc.binding, but got "0.5"}}
  %0 = "reloc.quantize"(%t) {scale = "0.5", policy = #reloc.policy<symmetric_rne>} : (!sym.tensor<[4], f32>) -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_scale_f64(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale constants must have element type f32, but got 'f64'}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f64>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_scale_zero(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale must be finite and strictly positive, but element 0 is 0.000000e+00}}
  %0 = reloc.quantize %t scale(dense<0.0> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_scale_negative(%t: !sym.tensor<[2, 3], f32>) {
  // expected-error @below {{scale must be finite and strictly positive, but element 1 is -2.500000e-01}}
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, -0.25, 1.0]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  return
}

// -----

func.func @quantize_scale_nan(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale must be finite and strictly positive, but element 0 is 0x7FC00000}}
  %0 = reloc.quantize %t scale(dense<0x7FC00000> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_scale_inf(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale must be finite and strictly positive, but element 0 is 0x7F800000}}
  %0 = reloc.quantize %t scale(dense<0x7F800000> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_scale_rank_two(%t: !sym.tensor<[2, 3], f32>) {
  // expected-error @below {{parameters must have rank 0 or 1, but scale has rank 2}}
  %0 = reloc.quantize %t axis 1 scale(dense<0.5> : tensor<1x3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  return
}

// -----

func.func @quantize_zero_point_float(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{zero_point constants must have a signless integer element type, but got 'f32'}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) zero_point(dense<0.0> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @dequantize_zero_point_above_range(%q: !sym.tensor<[4], i8>) {
  // expected-error @below {{zero point must lie in [-128, 127], but element 0 is 128}}
  %0 = reloc.dequantize %q scale(dense<0.5> : tensor<f32>) zero_point(dense<128> : tensor<i32>) policy affine : !sym.tensor<[4], i8> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @dequantize_zero_point_below_range(%q: !sym.tensor<[2, 3], i8>) {
  // expected-error @below {{zero point must lie in [-128, 127], but element 2 is -129}}
  %0 = reloc.dequantize %q axis 1 scale(dense<[1.0, 1.0, 1.0]> : tensor<3xf32>) zero_point(dense<[0, 127, -129]> : tensor<3xi32>) policy affine : !sym.tensor<[2, 3], i8> -> !sym.tensor<[2, 3], f32>
  return
}

//===----------------------------------------------------------------------===//
// Per-tensor versus per-channel forms and the channel axis
//===----------------------------------------------------------------------===//

// -----

func.func @quantize_per_tensor_rank_one_scale(%t: !sym.tensor<[2, 3], f32>) {
  // expected-error @below {{per-tensor form (no axis) requires rank-0 parameters, but scale has rank 1}}
  %0 = reloc.quantize %t scale(dense<[0.5, 0.5, 0.5]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  return
}

// -----

func.func @quantize_per_channel_rank_zero_scale(%t: !sym.tensor<[2, 3], f32>) {
  // expected-error @below {{per-channel form requires a rank-1 scale, but scale has rank 0}}
  %0 = reloc.quantize %t axis 1 scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  return
}

// -----

func.func @quantize_axis_out_of_range(%t: !sym.tensor<[2, 3, 4], f32>) {
  // expected-error @below {{axis (3) is out of range for operand rank 3}}
  %0 = reloc.quantize %t axis 3 scale(dense<[0.5, 0.5]> : tensor<2xf32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  return
}

// -----

// Negative axes are not Python-style aliases: the channel axis is an
// explicit non-negative index into the operand's logical shape.
func.func @quantize_negative_axis(%t: !sym.tensor<[2, 3, 4], f32>) {
  // expected-error @below {{axis (-1) is out of range for operand rank 3}}
  %0 = reloc.quantize %t axis -1 scale(dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  return
}

// -----

func.func @quantize_channel_length_mismatch(%t: !sym.tensor<[2, 3, 4], f32>) {
  // expected-error @below {{scale has 4 channel entries, but axis 1 has extent 3}}
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  return
}

// -----

func.func @dequantize_zero_point_length_mismatch(%q: !sym.tensor<[2, 3, 4], i8>) {
  // expected-error @below {{zero_point has 2 channel entries, but axis 1 has extent 3}}
  %0 = reloc.dequantize %q axis 1 scale(dense<[0.5, 0.5, 0.5]> : tensor<3xf32>) zero_point(dense<[0, 1]> : tensor<2xi32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>
  return
}

// -----

// A binding's declared channel length that provably disagrees with the axis
// extent is a static error; a symbolic disagreement is C3's guard.
func.func @quantize_binding_length_mismatch(%t: !sym.tensor<[2, 3, 4], f32>) {
  // expected-error @below {{scale declares 4 channel entries, but axis 1 has extent 3}}
  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [4], f32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  return
}

//===----------------------------------------------------------------------===//
// Zero points under symmetric_rne
//===----------------------------------------------------------------------===//

// -----

func.func @quantize_nonzero_zero_point(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{policy symmetric_rne admits only the constant zero point 0, but element 0 is 1}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) zero_point(dense<1> : tensor<i32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @quantize_per_channel_nonzero_zero_point(%t: !sym.tensor<[2, 3], f32>) {
  // expected-error @below {{policy symmetric_rne admits only the constant zero point 0, but element 2 is -128}}
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 0.5, 0.5]> : tensor<3xf32>) zero_point(dense<[0, 0, -128]> : tensor<3xi32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  return
}

// -----

func.func @quantize_runtime_zero_point(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{policy symmetric_rne admits only the constant zero point 0, but zero_point is a runtime binding}}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) zero_point(#reloc.binding<"zp" : [], i32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

//===----------------------------------------------------------------------===//
// Runtime bindings: declarations and conflicts
//===----------------------------------------------------------------------===//

// -----

func.func @quantize_binding_wrong_dtype(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{scale bindings must declare element type f32, but got 'i32'}}
  %0 = reloc.quantize %t scale(#reloc.binding<"s" : [], i32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return
}

// -----

func.func @dequantize_zero_point_binding_wrong_dtype(%q: !sym.tensor<[4], i8>) {
  // expected-error @below {{zero_point bindings must declare element type i32, but got 'f32'}}
  %0 = reloc.dequantize %q scale(dense<0.5> : tensor<f32>) zero_point(#reloc.binding<"zp" : [], f32>) policy affine : !sym.tensor<[4], i8> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @dequantize_duplicate_binding_names(%q: !sym.tensor<[4], i8>) {
  // expected-error @below {{runtime parameters must use distinct binding names, but scale and zero_point both bind "p"}}
  %0 = reloc.dequantize %q scale(#reloc.binding<"p" : [], f32>) zero_point(#reloc.binding<"p" : [], i32>) policy affine : !sym.tensor<[4], i8> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @binding_empty_name() {
  // expected-error @below {{binding name must not be empty}}
  "test.use_attr"() {b = #reloc.binding<"" : [], f32>} : () -> ()
  return
}

// -----

func.func @binding_rank_two() {
  // expected-error @below {{binding extents must have rank 0 or 1, but got rank 2}}
  "test.use_attr"() {b = #reloc.binding<"s" : [2, 3], f32>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// reloc.dequantize: type pairs and policy
//===----------------------------------------------------------------------===//

// -----

func.func @dequantize_float_input(%t: !sym.tensor<[4], f32>) {
  // expected-error @below {{dequantize expects a signless i8 operand and an f32 result, but got 'f32' -> 'f32'}}
  %0 = reloc.dequantize %t scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<[4], f32> -> !sym.tensor<[4], f32>
  return
}

// -----

func.func @dequantize_f16_result(%q: !sym.tensor<[4], i8>) {
  // expected-error @below {{dequantize expects a signless i8 operand and an f32 result, but got 'i8' -> 'f16'}}
  %0 = reloc.dequantize %q scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<[4], i8> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @dequantize_wrong_policy(%q: !sym.tensor<[4], i8>) {
  // expected-error @below {{policy 'symmetric_rne' is not defined for reloc.dequantize (use 'affine')}}
  %0 = reloc.dequantize %q scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], i8> -> !sym.tensor<[4], f32>
  return
}
