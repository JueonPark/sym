// RUN: python3 %S/typed_export_check.py sym-reloc-export %s

// C3 (issue #143): the typed artifact interface. With --typed this chain
// exports as wire format v1 with a schema-2 manifest; without the flag it is
// the same `typed_unsupported` rejection the layout-only interface has always
// published. The check script also proves that layout-only chains export
// byte-identically with and without the flag.
func.func @quantize_channel_transpose(%t: !sym.tensor<["B", 3], f32>) -> !sym.tensor<[3, "B"], i8> {
  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [3], f32>) policy symmetric_rne : !sym.tensor<["B", 3], f32> -> !sym.tensor<["B", 3], i8>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<["B", 3], i8> -> !sym.tensor<[3, "B"], i8>
  return %1 : !sym.tensor<[3, "B"], i8>
}
