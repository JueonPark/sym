//===- e2e_overlap.cu - R7 / Regime-5 end-to-end overlap (issue #88) ------===//
//
// L-layer weight-loading loop with per-layer GEMM compute concurrent to the
// load. Method A transforms on the host (zero GPU kernels); b_pipelined
// ships raw f32 and runs its receive kernel ONCE PER LAYER (not per chunk:
// no sub-plan slicing; cross-layer overlap of recv/H2D/compute is what the
// regime probes, and BP quant's best chunk was monolithic anyway — G3
// checks the per-layer cost still matches BP within 10%). a_prefold DMAs a
// pre-transformed pinned wire image (V4 semantics). compute_only /
// compute_bare / load-only anchors calibrate the gates.
//
// POST-FREEZE ADDENDUM (2026-09-02 eval freeze): every row carries
// post_freeze=1; frozen CSVs are never touched. Gates: gates.py --exp r7.
//
//===----------------------------------------------------------------------===//

#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Quant.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaKernels.h"
#endif
#include "rtrack/chunking.h"
#include "rtrack/plans.h"
#include "rtrack/rstats.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                 \
                   cudaGetErrorString(err_));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,   \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

using bench::rtrack::kIters;
using bench::rtrack::kWarmup;
using bench::rtrack::nowMs;
using bench::rtrack::RStats;
using bench::rtrack::summarizeSamples;

struct Args {
  std::string machine;
  std::string family = "quant"; // quant | blocked_transpose
  std::string methods = "all";  // comma list or "all"
  int64_t n = 8192;
  int64_t layers = 16;
  int64_t gemmBatch = 1024;
  std::string chunksMib;        // comma list; default per family+method
  std::string targetCs = "0.5,1,2,4";
  std::string csvPath;
  bool verify = false;
  bool smoke = false;
};

std::vector<double> splitDoubles(const std::string &s) {
  std::vector<double> out;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos)
      q = s.size();
    out.push_back(std::atof(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return out;
}

Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--machine")
      a.machine = next();
    else if (k == "--family")
      a.family = next();
    else if (k == "--methods")
      a.methods = next();
    else if (k == "--n")
      a.n = std::atoll(next().c_str());
    else if (k == "--layers")
      a.layers = std::atoll(next().c_str());
    else if (k == "--gemm-batch")
      a.gemmBatch = std::atoll(next().c_str());
    else if (k == "--chunk-mib")
      a.chunksMib = next();
    else if (k == "--target-c")
      a.targetCs = next();
    else if (k == "--csv")
      a.csvPath = next();
    else if (k == "--verify")
      a.verify = true;
    else if (k == "--smoke")
      a.smoke = true;
    else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

// One cuBLAS engine per process: fixed stream, fixed 32 MiB workspace,
// CUBLAS_GEMM_DEFAULT — determinism across methods within an invocation
// comes from identical config + identical inputs.
struct GemmEngine {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  void *workspace = nullptr;
  static constexpr size_t kWorkspaceBytes = 32u << 20;

  void init() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaMalloc(&workspace, kWorkspaceBytes));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, kWorkspaceBytes));
  }
  void destroy() {
    if (handle)
      cublasDestroy(handle);
    if (workspace)
      cudaFree(workspace);
    if (stream)
      cudaStreamDestroy(stream);
  }
  // y(f32, n x b) = W(f32, n x n) * x(f32, n x b), column-major.
  void sgemm(const float *w, const float *x, float *y, int64_t n,
             int64_t b) const {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)b,
                             (int)n, &alpha, w, (int)n, x, (int)n, &beta, y,
                             (int)n));
  }
  // y(s32, n x b) = W(s8, n x n) * x(s8, n x b): int8 GemmEx, 32I accumulate.
  // sm_75 DP4A path. Dim/lda restrictions (%4) hold for n=8192, b=1024.
  void igemm(const int8_t *w, const int8_t *x, int32_t *y, int64_t n,
             int64_t b) const {
    const int32_t alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)b,
                              (int)n, &alpha, w, CUDA_R_8I, (int)n, x,
                              CUDA_R_8I, (int)n, &beta, y, CUDA_R_32I, (int)n,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  }
};

// --smoke: validate both GEMM paths on this arch and print per-call medians
// (5+30 protocol). The int8 path is spec risk 1 — if cublasGemmEx rejects
// the OP_N/OP_N int8 config on this box, this mode is where we learn it.
int runSmoke(int64_t n, int64_t bAct) {
  GemmEngine g;
  g.init();
  float *wF = nullptr, *xF = nullptr, *yF = nullptr;
  int8_t *wI = nullptr, *xI = nullptr;
  int32_t *yI = nullptr;
  CUDA_CHECK(cudaMalloc(&wF, sizeof(float) * n * n));
  CUDA_CHECK(cudaMalloc(&xF, sizeof(float) * n * bAct));
  CUDA_CHECK(cudaMalloc(&yF, sizeof(float) * n * bAct));
  CUDA_CHECK(cudaMalloc(&wI, (size_t)(n * n)));
  CUDA_CHECK(cudaMalloc(&xI, (size_t)(n * bAct)));
  CUDA_CHECK(cudaMalloc(&yI, sizeof(int32_t) * n * bAct));
  CUDA_CHECK(cudaMemset(wF, 0, sizeof(float) * n * n));
  CUDA_CHECK(cudaMemset(xF, 0, sizeof(float) * n * bAct));
  CUDA_CHECK(cudaMemset(wI, 1, (size_t)(n * n)));
  CUDA_CHECK(cudaMemset(xI, 1, (size_t)(n * bAct)));

  auto timeGemm = [&](const char *name, auto &&call) {
    std::vector<double> ms;
    for (int it = 0; it < kWarmup + kIters; ++it) {
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      const double t0 = nowMs();
      call();
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
      const double t1 = nowMs();
      if (it >= kWarmup)
        ms.push_back(t1 - t0);
    }
    const RStats s = summarizeSamples(ms);
    std::printf("smoke,%s,ok=1,t_gemm_ms=%.4f,min=%.4f,p95=%.4f\n", name,
                s.median, s.min, s.p95);
  };
  timeGemm("sgemm", [&] { g.sgemm(wF, xF, yF, n, bAct); });
  timeGemm("igemm", [&] { g.igemm(wI, xI, yI, n, bAct); });

  cudaFree(wF); cudaFree(xF); cudaFree(yF);
  cudaFree(wI); cudaFree(xI); cudaFree(yI);
  g.destroy();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Args a = parseArgs(argc, argv);
  if (a.smoke)
    return runSmoke(a.n, a.gemmBatch);
  std::fprintf(stderr, "e2e_overlap: only --smoke is implemented so far\n");
  return 2;
}
