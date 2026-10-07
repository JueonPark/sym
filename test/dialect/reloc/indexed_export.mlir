// RUN: python3 %S/indexed_export_check.py sym-reloc-export %s
func.func @select_cast(%x: !sym.tensor<["N", "D"], f32>, %i: !sym.tensor<["M"], i64>) -> !sym.tensor<["M", "D"], f16> {
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<["N", "D"], f32>, !sym.tensor<["M"], i64> -> !sym.tensor<["M", "D"], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<["M", "D"], f32> -> !sym.tensor<["M", "D"], f16>
  return %1 : !sym.tensor<["M", "D"], f16>
}
