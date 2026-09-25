// R4 (#148) example recipe: a non-self-inverse permutation followed by a
// constant pad with a non-zero fill (1.5). f32 [2, 3, 4] -> permute
// [1, 2, 0] -> [3, 4, 2] -> pad axis 0 lo 1 hi 2 -> [6, 4, 2].
func.func @transpose_pad(%x: !sym.tensor<[2, 3, 4], f32>) -> !sym.tensor<[6, 4, 2], f32> {
  %0 = reloc.transpose %x perm [1, 2, 0] : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[3, 4, 2], f32>
  %1 = reloc.pad %0 axis 0 lo 1 hi 2 value (1.5 : f32) : !sym.tensor<[3, 4, 2], f32> -> !sym.tensor<[6, 4, 2], f32>
  return %1 : !sym.tensor<[6, 4, 2], f32>
}
