// RUN: python3 %S/advice_export_check.py sym-reloc-export %s

// Cost advice (interface 2). This file is the layout-only case: a K/V-style
// permutation whose advice admits the pipelined and grouped paths. The check
// script also exports a typed host-stage chain, a dequantize-first identity
// chain and an indexed row gather (each in a file of its own) and checks the
// advice shape of every kind, plus --no-advice.
func.func @kv_permute(%t: !sym.tensor<[16, "S", 64], f32>) -> !sym.tensor<["S", 16, 64], f32> {
  %0 = reloc.transpose %t perm [1, 0, 2] : !sym.tensor<[16, "S", 64], f32> -> !sym.tensor<["S", 16, 64], f32>
  return %0 : !sym.tensor<["S", 16, 64], f32>
}
