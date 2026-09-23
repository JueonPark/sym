// RUN: sym-opt %s | sym-opt | FileCheck %s

// Typed value transforms (C1, issue #141): reloc.cast / reloc.quantize /
// reloc.dequantize round-trip through the printer with exact parameter bits,
// channel axis, binding declarations and numerical policy preserved. The
// semantics behind every policy name are docs/reloc-typed-semantics.md.

//===----------------------------------------------------------------------===//
// reloc.cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cast_narrow
func.func @cast_narrow(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[4, 6], f16> {
  // CHECK: reloc.cast %{{.*}} policy ieee_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], f16>
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], f16>
  return %0 : !sym.tensor<[4, 6], f16>
}

// Symbolic extents are preserved verbatim by the shape-preserving transform.
// CHECK-LABEL: func.func @cast_widen_symbolic
func.func @cast_widen_symbolic(%t: !sym.tensor<["N", 64], f16>) -> !sym.tensor<["N", 64], f32> {
  // CHECK: reloc.cast %{{.*}} policy exact : !sym.tensor<["N", 64], f16> -> !sym.tensor<["N", 64], f32>
  %0 = reloc.cast %t policy exact : !sym.tensor<["N", 64], f16> -> !sym.tensor<["N", 64], f32>
  return %0 : !sym.tensor<["N", 64], f32>
}

// Generic syntax spells the policy as a #reloc.policy attribute and prints
// back in the custom form.
// CHECK-LABEL: func.func @cast_generic
func.func @cast_generic(%t: !sym.tensor<[8], f32>) -> !sym.tensor<[8], f16> {
  // CHECK: reloc.cast %{{.*}} policy ieee_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], f16>
  %0 = "reloc.cast"(%t) {policy = #reloc.policy<ieee_rne>} : (!sym.tensor<[8], f32>) -> !sym.tensor<[8], f16>
  return %0 : !sym.tensor<[8], f16>
}

//===----------------------------------------------------------------------===//
// reloc.quantize
//===----------------------------------------------------------------------===//

// Per-tensor form: no axis, rank-0 parameters, zero point defaults to 0.
// Non-round f32 bits survive the round trip exactly: 0x3EAAAAAB prints as
// the shortest decimal that reproduces those bits (0.333333343) and the
// second sym-opt pass in the RUN line re-parses it to the same value.
// CHECK-LABEL: func.func @quantize_per_tensor
func.func @quantize_per_tensor(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[4, 6], i8> {
  // CHECK: reloc.quantize %{{.*}} scale(dense<0.333333343> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], i8>
  %0 = reloc.quantize %t scale(dense<0x3EAAAAAB> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], i8>
  return %0 : !sym.tensor<[4, 6], i8>
}

// Explicit per-tensor zero point (the only value symmetric_rne admits).
// CHECK-LABEL: func.func @quantize_explicit_zero_point
func.func @quantize_explicit_zero_point(%t: !sym.tensor<[8], f32>) -> !sym.tensor<[8], i8> {
  // CHECK: reloc.quantize %{{.*}} scale(dense<5.000000e-01> : tensor<f32>) zero_point(dense<0> : tensor<i32>) policy symmetric_rne
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) zero_point(dense<0> : tensor<i32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>
  return %0 : !sym.tensor<[8], i8>
}

// Per-channel form on the issue's witness shape: axis 1 of [2, 3, 4] carries
// three distinct scales; the channel axis is relative to this operand.
// CHECK-LABEL: func.func @quantize_per_channel
func.func @quantize_per_channel(%t: !sym.tensor<[2, 3, 4], f32>) -> !sym.tensor<[2, 3, 4], i8> {
  // CHECK: reloc.quantize %{{.*}} axis 1 scale(dense<[5.000000e-01, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) zero_point(dense<0> : tensor<3xi32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 1.0, 2.0]> : tensor<3xf32>) zero_point(dense<0> : tensor<3xi32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  return %0 : !sym.tensor<[2, 3, 4], i8>
}

// Runtime parameter: a named binding declares dtype and extents; the channel
// length is the symbolic extent itself, so the length guard is C3's.
// CHECK-LABEL: func.func @quantize_runtime_scale
func.func @quantize_runtime_scale(%t: !sym.tensor<["N", 64], f32>) -> !sym.tensor<["N", 64], i8> {
  // CHECK: reloc.quantize %{{.*}} axis 0 scale(#reloc.binding<"w_scale" : [N], f32>) policy symmetric_rne : !sym.tensor<["N", 64], f32> -> !sym.tensor<["N", 64], i8>
  %0 = reloc.quantize %t axis 0 scale(#reloc.binding<"w_scale" : [N], f32>) policy symmetric_rne : !sym.tensor<["N", 64], f32> -> !sym.tensor<["N", 64], i8>
  return %0 : !sym.tensor<["N", 64], i8>
}

// A constant channel count against a symbolic extent is undecidable at
// verification time and is accepted; C3 receives the length guard.
// CHECK-LABEL: func.func @quantize_symbolic_channel_extent
func.func @quantize_symbolic_channel_extent(%t: !sym.tensor<["C", 64], f32>) -> !sym.tensor<["C", 64], i8> {
  // CHECK: reloc.quantize %{{.*}} axis 0 scale(dense<[2.500000e-01, 5.000000e-01]> : tensor<2xf32>) policy symmetric_rne
  %0 = reloc.quantize %t axis 0 scale(dense<[0.25, 0.5]> : tensor<2xf32>) policy symmetric_rne : !sym.tensor<["C", 64], f32> -> !sym.tensor<["C", 64], i8>
  return %0 : !sym.tensor<["C", 64], i8>
}

// Generic syntax with every attribute spelled out.
// CHECK-LABEL: func.func @quantize_generic
func.func @quantize_generic(%t: !sym.tensor<[3, 5], f32>) -> !sym.tensor<[3, 5], i8> {
  // CHECK: reloc.quantize %{{.*}} axis 0 scale(#reloc.binding<"s" : [3], f32>) zero_point(dense<0> : tensor<i8>) policy symmetric_rne
  %0 = "reloc.quantize"(%t) {axis = 0 : i64, scale = #reloc.binding<"s" : [3], f32>, zero_point = dense<0> : tensor<i8>, policy = #reloc.policy<symmetric_rne>} : (!sym.tensor<[3, 5], f32>) -> !sym.tensor<[3, 5], i8>
  return %0 : !sym.tensor<[3, 5], i8>
}

//===----------------------------------------------------------------------===//
// reloc.dequantize
//===----------------------------------------------------------------------===//

// Per-channel affine dequantization with the full zero-point range.
// CHECK-LABEL: func.func @dequantize_per_channel
func.func @dequantize_per_channel(%q: !sym.tensor<[2, 3, 4], i8>) -> !sym.tensor<[2, 3, 4], f32> {
  // CHECK: reloc.dequantize %{{.*}} axis 1 scale(dense<[5.000000e-01, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) zero_point(dense<[-128, 0, 127]> : tensor<3xi32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>
  %0 = reloc.dequantize %q axis 1 scale(dense<[0.5, 1.0, 2.0]> : tensor<3xf32>) zero_point(dense<[-128, 0, 127]> : tensor<3xi32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>
  return %0 : !sym.tensor<[2, 3, 4], f32>
}

// Per-channel form with a rank-0 (broadcast) zero point.
// CHECK-LABEL: func.func @dequantize_broadcast_zero_point
func.func @dequantize_broadcast_zero_point(%q: !sym.tensor<[2, 3, 4], i8>) -> !sym.tensor<[2, 3, 4], f32> {
  // CHECK: reloc.dequantize %{{.*}} axis 2 scale(dense<[1.000000e+00, 2.000000e+00, 4.000000e+00, 8.000000e+00]> : tensor<4xf32>) zero_point(dense<-3> : tensor<i32>) policy affine
  %0 = reloc.dequantize %q axis 2 scale(dense<[1.0, 2.0, 4.0, 8.0]> : tensor<4xf32>) zero_point(dense<-3> : tensor<i32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>
  return %0 : !sym.tensor<[2, 3, 4], f32>
}

// Both parameters bound at runtime, per tensor, with distinct names.
// CHECK-LABEL: func.func @dequantize_runtime_parameters
func.func @dequantize_runtime_parameters(%q: !sym.tensor<["N"], i8>) -> !sym.tensor<["N"], f32> {
  // CHECK: reloc.dequantize %{{.*}} scale(#reloc.binding<"s" : [], f32>) zero_point(#reloc.binding<"zp" : [], i32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
  %0 = reloc.dequantize %q scale(#reloc.binding<"s" : [], f32>) zero_point(#reloc.binding<"zp" : [], i32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
  return %0 : !sym.tensor<["N"], f32>
}

//===----------------------------------------------------------------------===//
// Mixed chains: value transforms interleave with layout operations, and each
// stage keeps its own element type.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @layout_then_quantize
func.func @layout_then_quantize(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], i8> {
  // CHECK: reloc.transpose
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  // CHECK: reloc.quantize %{{.*}} axis 0 scale(#reloc.binding<"row_scale" : [6], f32>) policy symmetric_rne
  %1 = reloc.quantize %0 axis 0 scale(#reloc.binding<"row_scale" : [6], f32>) policy symmetric_rne : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], i8>
  return %1 : !sym.tensor<[6, 4], i8>
}

// CHECK-LABEL: func.func @cast_then_pad
func.func @cast_then_pad(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f16> {
  // CHECK: reloc.cast %{{.*}} policy ieee_rne
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], f16>
  // The fill value belongs to the stage's element type (f16 after the cast).
  // CHECK: reloc.pad %{{.*}} axis 0 lo 1 hi 1 value (0.000000e+00 : f16)
  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (0.0 : f16) : !sym.tensor<[6], f16> -> !sym.tensor<[8], f16>
  return %1 : !sym.tensor<[8], f16>
}

// CHECK-LABEL: func.func @quantize_dequantize_is_not_cancelled
func.func @quantize_dequantize_is_not_cancelled(%t: !sym.tensor<[8], f32>) -> !sym.tensor<[8], f32> {
  // CHECK: reloc.quantize
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>
  // CHECK: reloc.dequantize
  %1 = reloc.dequantize %0 scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<[8], i8> -> !sym.tensor<[8], f32>
  return %1 : !sym.tensor<[8], f32>
}
