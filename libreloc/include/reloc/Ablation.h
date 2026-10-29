//===- Ablation.h - Runtime knobs that disable one optimization -*- C++ -*-===//
//
// Measurement-only switches for the attribution experiments in
// bench/beat_torch: each knob turns one native optimization off so that its
// share of a model-level result can be measured in the same process as the
// full path. Knobs change performance only, never results (every disabled
// path is the bit-identical reference the optimization replaced). They are
// read at launch time with a relaxed atomic load and default to off;
// RELOC_ABLATE="scalar_dequant,..." in the environment enables knobs at
// startup and pyreloc.set_ablation() toggles them at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_ABLATION_H
#define RELOC_ABLATION_H

#include <optional>
#include <string_view>

namespace reloc {
namespace ablation {

enum class Knob {
  /// dequantS8F32 launches the one-element-per-thread kernel instead of the
  /// char4/float4 quad kernel.
  ScalarDequant = 0,
  /// Device stages run dequantize -> cast f32->f16 as two kernels with an
  /// f32 intermediate instead of the fused dequantS8F16 kernel.
  UnfusedDequantF16 = 1,
  Count
};

bool enabled(Knob knob);
void set(Knob knob, bool on);
std::optional<Knob> parse(std::string_view name);
const char *name(Knob knob);

} // namespace ablation
} // namespace reloc

#endif // RELOC_ABLATION_H
