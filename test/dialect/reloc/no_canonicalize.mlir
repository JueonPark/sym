// RUN: sym-opt --allow-unregistered-dialect --reloc-fold %s | FileCheck %s --check-prefix=CANON
// RUN: sym-opt --allow-unregistered-dialect --reloc-fold=canonicalize=false %s | FileCheck %s --check-prefix=RAW
// RUN: python3 %S/no_canonicalize_check.py sym-reloc-export %s

// The attribution mode of the fold pass (canonicalize=false) keeps the raw
// transfer-function output: the identity permutation stays two axes with
// no_copy = false, where the canonicalizer merges them into one pure-view
// axis. Both describe the same transfer.

// CANON-LABEL: func.func @identity_view
// CANON: reloc.plan_result
// CANON-SAME: axes = [{name = "d0", extent = 1024, src_stride = 1, dst_stride = 1}]
// CANON-SAME: no_copy = true

// RAW-LABEL: func.func @identity_view
// RAW: reloc.plan_result
// RAW-SAME: extent = 8, src_stride = 128, dst_stride = 128
// RAW-SAME: extent = 128, src_stride = 1, dst_stride = 1
// RAW-NOT: no_copy = true
func.func @identity_view(%t: !sym.tensor<[8, 128], f32>) -> !sym.tensor<[8, 128], f32> {
  %0 = reloc.transpose %t perm [0, 1] : !sym.tensor<[8, 128], f32> -> !sym.tensor<[8, 128], f32>
  return %0 : !sym.tensor<[8, 128], f32>
}
