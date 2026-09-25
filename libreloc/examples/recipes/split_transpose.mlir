// R4 (#148) example recipe: one symbolic artifact rebound at several sizes.
// f32 [s0] -> reshape [s0 floordiv 64, 64] -> transpose -> [64, s0 floordiv 64];
// s0 must be a positive multiple of 64 (a divisibility constraint).
func.func @split_transpose(%t: !sym.tensor<["s0"], f32>) -> !sym.tensor<[64, (s0 floordiv 64)], f32> {
  %0 = reloc.reshape %t to [s0 floordiv 64, 64] : !sym.tensor<["s0"], f32> -> !sym.tensor<[(s0 floordiv 64), 64], f32>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[(s0 floordiv 64), 64], f32> -> !sym.tensor<[64, (s0 floordiv 64)], f32>
  return %1 : !sym.tensor<[64, (s0 floordiv 64)], f32>
}
