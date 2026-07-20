//===- csv.h - rtrack CSV rows (issue #76 output format) --------*- C++ -*-===//
//
// One row per (workload, method, chunk) config. Session metadata (machine
// calibration, versions, environment controls) travels as '#'-prefixed
// header comment lines written by run_rtrack.py, not here. Stage columns
// are medians over the 30 timed iterations. No quoting: commas inside
// string fields are rewritten to ';' (an assert would vanish under the
// -DNDEBUG benchmarking build, and one comma would shift every downstream
// column).
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_CSV_H
#define BENCH_RTRACK_CSV_H

#include "protocol.h"
#include "rtrack/rstats.h"

#include <cstdint>
#include <string>

namespace bench {
namespace rtrack {

struct CsvRow {
  std::string machine, gpu, method, transform, dtypeOut;
  int64_t n = 0;
  double r = 0;
  unsigned threads = 1;
  int64_t chunkReqBytes = 0;
  int64_t stagingBytes = 0;
  int64_t nChunks = 0;
  RStats wall, gpuPipe, cpuStage, h2d, gpuKernel;
  double effectiveInputGbps = 0;
  bool verified = false;
};

inline std::string csvHeaderLine() {
  return "machine,gpu,method,transform,N,dtype_out,r,threads,chunk_req_mib,"
         "staging_bytes,n_chunks,median_ms,min_ms,p95_ms,iqr_over_median_pct,"
         "unstable,effective_input_GBps,gpu_pipeline_ms,cpu_stage_ms,h2d_ms,"
         "gpu_kernel_ms,verified";
}

inline std::string csvField(std::string s) {
  for (char &c : s)
    if (c == ',')
      c = ';';
  return s;
}

inline std::string csvRowLine(const CsvRow &r) {
  auto num = [](double v) { return bench::jsonNumber(v); };
  std::string out;
  out += csvField(r.machine) + ',' + csvField(r.gpu) + ',' +
         csvField(r.method) + ',' + csvField(r.transform) + ',';
  out +=
      std::to_string(r.n) + ',' + csvField(r.dtypeOut) + ',' + num(r.r) + ',';
  out += std::to_string(r.threads) + ',';
  out += num(static_cast<double>(r.chunkReqBytes) / (1 << 20)) + ',';
  out += std::to_string(r.stagingBytes) + ',' + std::to_string(r.nChunks);
  out += ',' + num(r.wall.median) + ',' + num(r.wall.min) + ',' +
         num(r.wall.p95) + ',' + num(r.wall.iqrOverMedianPct) + ',' +
         (r.wall.unstable ? "1" : "0");
  out += ',' + num(r.effectiveInputGbps);
  out += ',' + num(r.gpuPipe.median) + ',' + num(r.cpuStage.median) + ',' +
         num(r.h2d.median) + ',' + num(r.gpuKernel.median);
  out += r.verified ? ",1" : ",0";
  return out;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_CSV_H
