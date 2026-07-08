// RUN: sym-opt --test-reloc-transfer --verify-diagnostics %s

// Bail conditions produce a fold-failure remark on the failing op and no
// plan — never a wrong plan (issue #15 design decision 1).

// Merging transposed (non-contiguous) axes is not provable: after
// transpose the outer axis has src stride 1, but contiguity would need
// outer.src_stride == inner.src_stride * inner.extent = 6 * 4.
func.func @noncontiguous_merge_bails(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], f32> {
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  // expected-remark @below {{fold bail: reloc.reshape}}
  %1 = reloc.reshape %0 to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  return %1 : !sym.tensor<[24], f32>
}
