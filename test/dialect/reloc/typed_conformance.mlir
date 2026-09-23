// RUN: sym-opt --reloc-fold --mlir-print-local-scope %s | FileCheck %s

// Compiler-side witnesses for the typed conformance corpus (C4, issue #144;
// libreloc/test/corpus/typed). The fixtures' chains must fold to the
// structure the runtime and the NumPy oracle assume: stage ORDER is kept
// (no lossy pair is cancelled), the fill enters at the stage the pad sat at,
// and a per-channel stage's channel map follows the result coordinates
// after the layout moved its axis.

// CHECK-LABEL: func.func @quant_dequant_padded
// The pad after both stages: fills = [{... stage = 2 ...}] with the f32 bits
// of 1.5, and two stages in program order (quantize before dequantize).
// CHECK: reloc.typed_plan_result
// CHECK-SAME: pad_fill = [{dst_axis = 0, lo = 1, hi = 2, value = 1.500000e+00 : f32}]
// CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [N], scale = dense<5.000000e-01> : tensor<f32>>, #reloc.stage<dequantize affine : i8 -> f32, shape = [N], scale = dense<5.000000e-01> : tensor<f32>>]
// CHECK-SAME: fills = [{dst_axis = 0, stage = 2, value = 1.500000e+00 : f32}]
func.func @quant_dequant_padded(%t: !sym.tensor<["N"], f32>) -> !sym.tensor<[(3 + "N")], f32> {
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<["N"], f32> -> !sym.tensor<["N"], i8>
  %1 = reloc.dequantize %0 scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
  %2 = reloc.pad %1 axis 0 lo 1 hi 2 value (1.5 : f32) : !sym.tensor<["N"], f32> -> !sym.tensor<[(3 + "N")], f32>
  return %2 : !sym.tensor<[(3 + "N")], f32>
}

// CHECK-LABEL: func.func @pad_then_quantize
// The f32 fill 1.0 entered at stage 0 and was fused to code 2 in the layout.
// CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}]
// CHECK-SAME: fills = [{dst_axis = 0, stage = 0, value = 1.000000e+00 : f32}]
func.func @pad_then_quantize(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %1 = reloc.quantize %0 scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>
  return %1 : !sym.tensor<[8], i8>
}

// CHECK-LABEL: func.func @quantize_then_pad
// The s8 fill 1 entered at stage 1 and stays 1.
// CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}]
// CHECK-SAME: fills = [{dst_axis = 0, stage = 1, value = 1 : i8}]
func.func @quantize_then_pad(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], i8>
  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (1 : i8) : !sym.tensor<[6], i8> -> !sym.tensor<[8], i8>
  return %1 : !sym.tensor<[8], i8>
}

// CHECK-LABEL: func.func @dequant_channel_runtime_transpose
// Runtime per-channel scales and zero points on operand axis 1; after the
// transpose the channel is result coordinate d0.
// CHECK: stages = [#reloc.stage<dequantize affine : i8 -> f32, shape = [2, 3, 4], scale = #reloc.binding<"scales" : [3], f32>, zero_point = #reloc.binding<"zero_points" : [3], i32>, axis = 1, channel = (d0, d1, d2) -> (d0)>]
func.func @dequant_channel_runtime_transpose(%q: !sym.tensor<[2, 3, 4], i8>) -> !sym.tensor<[3, 2, 4], f32> {
  %0 = reloc.dequantize %q axis 1 scale(#reloc.binding<"scales" : [3], f32>) zero_point(#reloc.binding<"zero_points" : [3], i32>) policy affine : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[2, 3, 4], f32>
  %1 = reloc.transpose %0 perm [1, 0, 2] : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[3, 2, 4], f32>
  return %1 : !sym.tensor<[3, 2, 4], f32>
}

// CHECK-LABEL: func.func @dequant_inline_zp_cast
// Two lossy stages stay two stages: dequantize then narrow.
// CHECK: stages = [#reloc.stage<dequantize affine : i8 -> f32, shape = [8], scale = dense<2.500000e-01> : tensor<f32>, zero_point = dense<-3> : tensor<i32>>, #reloc.stage<cast ieee_rne : f32 -> f16, shape = [8]>]
func.func @dequant_inline_zp_cast(%q: !sym.tensor<[8], i8>) -> !sym.tensor<[8], f16> {
  %0 = reloc.dequantize %q scale(dense<0.25> : tensor<f32>) zero_point(dense<-3> : tensor<i32>) policy affine : !sym.tensor<[8], i8> -> !sym.tensor<[8], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], f16>
  return %1 : !sym.tensor<[8], f16>
}

// CHECK-LABEL: func.func @widen_reshape
// The exact widening cast precedes the reshape; the reshape is an index map
// on the f32 result.
// CHECK: result = tensor<[6, 4], f32>
// CHECK-SAME: stages = [#reloc.stage<cast exact : f16 -> f32, shape = [2, 12]>]
func.func @widen_reshape(%h: !sym.tensor<[2, 12], f16>) -> !sym.tensor<[6, 4], f32> {
  %0 = reloc.cast %h policy exact : !sym.tensor<[2, 12], f16> -> !sym.tensor<[2, 12], f32>
  %1 = reloc.reshape %0 to [6, 4] : !sym.tensor<[2, 12], f32> -> !sym.tensor<[6, 4], f32>
  return %1 : !sym.tensor<[6, 4], f32>
}
