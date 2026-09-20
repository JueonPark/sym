// RUN: python3 %S/export_check.py sym-reloc-export %s
func.func @split_transpose(%t: !sym.tensor<["s0"], f32>) -> !sym.tensor<[64, (s0 floordiv 64)], f32> {
  %0 = reloc.reshape %t to [s0 floordiv 64, 64] : !sym.tensor<["s0"], f32> -> !sym.tensor<[(s0 floordiv 64), 64], f32>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[(s0 floordiv 64), 64], f32> -> !sym.tensor<[64, (s0 floordiv 64)], f32>
  return %1 : !sym.tensor<[64, (s0 floordiv 64)], f32>
}
