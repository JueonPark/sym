//===- workloads.h - the R1 workload matrix (T1-T5 + anchor) ----*- C++ -*-===//
//
// Each workload names its plan, its Method-A per-chunk CPU transform
// (R0.1 kernels / gatherChunk) and its Method-B post-transfer GPU stage
// (R0.2 kernels). r = output bytes / input bytes; the final artifact of
// both methods is identical (dtypeOut in the plan's dst layout) and
// bit-exact comparable, per the R0.1/R0.2 CPU==GPU quantize contract.
// R2's dequant/unpack receive variants slot in as new RecvStage values.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_WORKLOADS_H
#define BENCH_RTRACK_WORKLOADS_H

#include "rtrack/plans.h"

#include <string>
#include <vector>

namespace bench {
namespace rtrack {

enum class DtypeOut { F32, F16, S8 };

inline const char *dtypeName(DtypeOut d) {
  switch (d) {
  case DtypeOut::F32:
    return "f32";
  case DtypeOut::F16:
    return "f16";
  case DtypeOut::S8:
    return "s8";
  }
  return "?";
}

inline int dtypeBytes(DtypeOut d) {
  switch (d) {
  case DtypeOut::F32:
    return 4;
  case DtypeOut::F16:
    return 2;
  case DtypeOut::S8:
    return 1;
  }
  return 0;
}

/// Payload dtype on the PCIe wire (Method A's staged form). The R2 r-sweep
/// (issue #83) fixes the final artifact at f32 and varies only this.
enum class Wire { F32, F16, S8, S4 };

inline const char *wireName(Wire w) {
  switch (w) {
  case Wire::F32:
    return "f32";
  case Wire::F16:
    return "f16";
  case Wire::S8:
    return "s8";
  case Wire::S4:
    return "s4";
  }
  return "?";
}

/// Staged bytes for `elems` elements of this wire dtype. S4 packs two
/// elements per byte; every rtrack plan's inner-row product is even for
/// N % 64 == 0, so the division is exact.
inline int64_t wireBytes(Wire w, int64_t elems) {
  switch (w) {
  case Wire::F32:
    return elems * 4;
  case Wire::F16:
    return elems * 2;
  case Wire::S8:
    return elems;
  case Wire::S4:
    return elems / 2;
  }
  return 0;
}

inline double wireRatio(Wire w) {
  return static_cast<double>(wireBytes(w, 8)) / 32.0;
}

/// Method A's per-chunk CPU transform into pinned staging. The *S4 and
/// GatherF16 stages are two-pass (gather/quant into a heap scratch, then
/// pack/convert into staging) -- a real Method-A cost, measured as-is; a
/// fused kernel is a noted future optimization, not silently assumed.
enum class CpuStage {
  GatherF32,
  GatherQuant,
  QuantPack,
  ConvertF16,
  CopyF32,       // rsweep r=1.0 contiguous: plain staging copy
  GatherF16,     // rsweep r=0.5 strided: gather f32 pass + convert pass
  GatherQuantS4, // rsweep r=0.125 strided: gather+quant pass + pack pass
  QuantPackS4    // rsweep r=0.125 contiguous: quant pass + pack pass
};

/// Method B's post-transfer GPU kernel sequence. None = the DMA target IS
/// the final artifact (rsweep T3 family: full-f32 transfer, no transform).
enum class GpuStage { Relocate, RelocateQuant, Quantize, ConvertF16, None };

/// Method A's post-DMA GPU receive stage (R2 fixed-f32 frame): decompress
/// the wire payload back to f32 in the dst layout, in-stream, per chunk.
enum class RecvStage { None, ConvertF16F32, DequantS8, UnpackDequantS4 };

struct Workload {
  const char *id;        // CLI name
  const char *transform; // CSV transform column (family name for rsweep)
  const char *variant;   // CSV variant column: "matrix" | "rsweep"
  DtypeOut dtypeOut;     // final artifact dtype
  Wire wire;             // Method A's on-wire payload dtype
  double r;              // wire bytes / input bytes
  reloc::BoundPlan (*makePlan)(int64_t n);
  CpuStage cpuStage;
  GpuStage gpuStage;   // Method B's transform
  RecvStage recvStage; // Method A's receive stage
  bool methodB;        // false: A-only row (B is r-independent; each
                       // family's R100 row measures it once)
};

/// NOTE (quant workloads): the quantization channel is the plan's
/// coalesced OUTER axis -- the gatherQuantizeF32S8 contract -- so T4's
/// scales are per BATCH image, not per C channel (a per-C NHWC quantize
/// is not expressible with the R0 kernel set). Read T4 as "strided
/// relocation + fused int8 with batch-granular scales" in R1 analysis.

inline const std::vector<Workload> &allWorkloads() {
  static const std::vector<Workload> ws = {
      // R1 matrix (issue #82), unchanged semantics.
      {"T1", "transpose", "matrix", DtypeOut::F32, Wire::F32, 1.0,
       &transposePlan, CpuStage::GatherF32, GpuStage::Relocate, RecvStage::None,
       true},
      {"T1b", "blocked_transpose", "matrix", DtypeOut::F32, Wire::F32, 1.0,
       &blockedTransposePlan, CpuStage::GatherF32, GpuStage::Relocate,
       RecvStage::None, true},
      {"T2", "transpose_quant", "matrix", DtypeOut::S8, Wire::S8, 0.25,
       &transposePlan, CpuStage::GatherQuant, GpuStage::RelocateQuant,
       RecvStage::None, true},
      {"T3", "quant", "matrix", DtypeOut::S8, Wire::S8, 0.25, &identityPlan,
       CpuStage::QuantPack, GpuStage::Quantize, RecvStage::None, true},
      {"T4", "nchw_nhwc_quant", "matrix", DtypeOut::S8, Wire::S8, 0.25,
       &nchwToNhwcPlan, CpuStage::GatherQuant, GpuStage::RelocateQuant,
       RecvStage::None, true},
      {"T5", "convert_f16", "matrix", DtypeOut::F16, Wire::F16, 0.5,
       &identityPlan, CpuStage::ConvertF16, GpuStage::ConvertF16,
       RecvStage::None, true},
      // R2 r-sweep (issue #83): fixed-f32 artifact, r = transfer
      // compression only. B is r-independent; only R100 rows run it.
      {"T1bR100", "blocked_transpose", "rsweep", DtypeOut::F32, Wire::F32, 1.0,
       &blockedTransposePlan, CpuStage::GatherF32, GpuStage::Relocate,
       RecvStage::None, true},
      {"T1bR050", "blocked_transpose", "rsweep", DtypeOut::F32, Wire::F16, 0.5,
       &blockedTransposePlan, CpuStage::GatherF16, GpuStage::Relocate,
       RecvStage::ConvertF16F32, false},
      {"T1bR025", "blocked_transpose", "rsweep", DtypeOut::F32, Wire::S8, 0.25,
       &blockedTransposePlan, CpuStage::GatherQuant, GpuStage::Relocate,
       RecvStage::DequantS8, false},
      {"T1bR0125", "blocked_transpose", "rsweep", DtypeOut::F32, Wire::S4,
       0.125, &blockedTransposePlan, CpuStage::GatherQuantS4,
       GpuStage::Relocate, RecvStage::UnpackDequantS4, false},
      {"T3R100", "quant", "rsweep", DtypeOut::F32, Wire::F32, 1.0,
       &identityPlan, CpuStage::CopyF32, GpuStage::None, RecvStage::None, true},
      {"T3R050", "quant", "rsweep", DtypeOut::F32, Wire::F16, 0.5,
       &identityPlan, CpuStage::ConvertF16, GpuStage::None,
       RecvStage::ConvertF16F32, false},
      {"T3R025", "quant", "rsweep", DtypeOut::F32, Wire::S8, 0.25,
       &identityPlan, CpuStage::QuantPack, GpuStage::None, RecvStage::DequantS8,
       false},
      {"T3R0125", "quant", "rsweep", DtypeOut::F32, Wire::S4, 0.125,
       &identityPlan, CpuStage::QuantPackS4, GpuStage::None,
       RecvStage::UnpackDequantS4, false},
      {"T2R100", "transpose_quant", "rsweep", DtypeOut::F32, Wire::F32, 1.0,
       &transposePlan, CpuStage::GatherF32, GpuStage::Relocate, RecvStage::None,
       true},
      {"T2R050", "transpose_quant", "rsweep", DtypeOut::F32, Wire::F16, 0.5,
       &transposePlan, CpuStage::GatherF16, GpuStage::Relocate,
       RecvStage::ConvertF16F32, false},
      {"T2R025", "transpose_quant", "rsweep", DtypeOut::F32, Wire::S8, 0.25,
       &transposePlan, CpuStage::GatherQuant, GpuStage::Relocate,
       RecvStage::DequantS8, false},
      {"T2R0125", "transpose_quant", "rsweep", DtypeOut::F32, Wire::S4, 0.125,
       &transposePlan, CpuStage::GatherQuantS4, GpuStage::Relocate,
       RecvStage::UnpackDequantS4, false},
      {"T4R100", "nchw_nhwc_quant", "rsweep", DtypeOut::F32, Wire::F32, 1.0,
       &nchwToNhwcPlan, CpuStage::GatherF32, GpuStage::Relocate,
       RecvStage::None, true},
      {"T4R050", "nchw_nhwc_quant", "rsweep", DtypeOut::F32, Wire::F16, 0.5,
       &nchwToNhwcPlan, CpuStage::GatherF16, GpuStage::Relocate,
       RecvStage::ConvertF16F32, false},
      {"T4R025", "nchw_nhwc_quant", "rsweep", DtypeOut::F32, Wire::S8, 0.25,
       &nchwToNhwcPlan, CpuStage::GatherQuant, GpuStage::Relocate,
       RecvStage::DequantS8, false},
      {"T4R0125", "nchw_nhwc_quant", "rsweep", DtypeOut::F32, Wire::S4, 0.125,
       &nchwToNhwcPlan, CpuStage::GatherQuantS4, GpuStage::Relocate,
       RecvStage::UnpackDequantS4, false},
  };
  return ws;
}

inline const Workload *findWorkload(const std::string &id) {
  for (const Workload &w : allWorkloads())
    if (id == w.id)
      return &w;
  return nullptr;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_WORKLOADS_H
