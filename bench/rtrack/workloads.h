//===- workloads.h - the R1 workload matrix (T1-T5 + anchor) ----*- C++ -*-===//
//
// Each workload names its plan, its Method-A per-chunk CPU transform
// (R0.1 kernels / gatherChunk) and its Method-B post-transfer GPU stage
// (R0.2 kernels). r = output bytes / input bytes; the final artifact of
// both methods is identical (dtypeOut in the plan's dst layout) and
// bit-exact comparable, per the R0.1/R0.2 CPU==GPU quantize contract.
// R2's dequant/unpack receive variants slot in as new GpuStage values.
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

/// Method A's per-chunk CPU transform into pinned staging.
enum class CpuStage { GatherF32, GatherQuant, QuantPack, ConvertF16 };

/// Method B's post-transfer GPU kernel sequence.
enum class GpuStage { Relocate, RelocateQuant, Quantize, ConvertF16 };

struct Workload {
  const char *id;        // CLI name
  const char *transform; // CSV transform column
  DtypeOut dtypeOut;
  double r; // output bytes / input bytes
  reloc::BoundPlan (*makePlan)(int64_t n);
  CpuStage cpuStage;
  GpuStage gpuStage;
};

inline const std::vector<Workload> &allWorkloads() {
  static const std::vector<Workload> ws = {
      {"T1", "transpose", DtypeOut::F32, 1.0, &transposePlan,
       CpuStage::GatherF32, GpuStage::Relocate},
      {"T1b", "blocked_transpose", DtypeOut::F32, 1.0, &blockedTransposePlan,
       CpuStage::GatherF32, GpuStage::Relocate},
      {"T2", "transpose_quant", DtypeOut::S8, 0.25, &transposePlan,
       CpuStage::GatherQuant, GpuStage::RelocateQuant},
      {"T3", "quant", DtypeOut::S8, 0.25, &identityPlan, CpuStage::QuantPack,
       GpuStage::Quantize},
      {"T4", "nchw_nhwc_quant", DtypeOut::S8, 0.25, &nchwToNhwcPlan,
       CpuStage::GatherQuant, GpuStage::RelocateQuant},
      {"T5", "convert_f16", DtypeOut::F16, 0.5, &identityPlan,
       CpuStage::ConvertF16, GpuStage::ConvertF16},
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
