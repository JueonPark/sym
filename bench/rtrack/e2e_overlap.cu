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
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaKernels.h"
#endif
#include "rtrack/chunking.h"
#include "rtrack/plans.h"
#include "rtrack/rstats.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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
  std::string chunksMib; // comma list; default per family+method
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

// House machine names are "<cpu>-<gpu>" (e.g. "epyc7351-2080ti",
// "7800x3d-4070tis"); the gpu column is everything after the FIRST '-'.
// A machine string with no '-' falls back to the whole string, or
// "unknown" if empty.
std::string gpuFromMachine(const std::string &machine) {
  const size_t dash = machine.find('-');
  if (dash == std::string::npos)
    return machine.empty() ? "unknown" : machine;
  return machine.substr(dash + 1);
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

// Family = plan + host transform + device format + GEMM type. One host
// source W (256 MiB) reused for all L layers: per-layer data identity does
// not change transfer or compute cost, and verify compares one layer's
// wire image bit-exactly.
struct Family {
  std::string name; // "quant" | "blocked_transpose"
  double r;         // wire ratio
  reloc::BoundPlan bound;
  bool s8; // device format / GEMM type selector
};

struct Fixture {
  Family fam;
  int64_t n, layers, bAct;
  // Host side.
  std::vector<float> hostW;     // pageable f32 source, n*n
  std::vector<float> invScales; // n (quant): host & GPU quantize agree
  std::vector<int8_t> refWire;  // host reference wire image (s8 or f32
                                // bytes) — verify target & prefold source
  float *pinnedRaw = nullptr;   // pinned f32 W (B methods' DMA source)
  int8_t *pinnedWire = nullptr; // pinned transformed wire (a_prefold)
  void *staging[2] = {nullptr, nullptr}; // Method A per-chunk staging
  int64_t stagingBytes = 0;
  // Device side.
  void *dW[2] = {nullptr, nullptr};    // ping-pong weight buffers (final fmt)
  float *dRaw[2] = {nullptr, nullptr}; // b_*: raw f32 landing per slot
  float *dInv = nullptr;               // quant: invScales for GPU quantize
  int8_t *dXi = nullptr;               // int8 activations, n*bAct
  float *dXf = nullptr;                // f32 activations
  void *dY = nullptr;                  // s32 or f32, n*bAct
  int64_t wireBytes = 0;               // bytes DMA'd per layer (A/prefold)
  int64_t rawBytes = 0;                // n*n*4 (B methods per layer)
};

Fixture makeFixture(const std::string &family, int64_t n, int64_t layers,
                    int64_t bAct) {
  Fixture f;
  f.n = n;
  f.layers = layers;
  f.bAct = bAct;
  if (family == "quant") {
    f.fam = {"quant", 0.25, bench::rtrack::identityPlan(n), true};
  } else if (family == "blocked_transpose") {
    f.fam = {"blocked_transpose", 1.0, bench::rtrack::blockedTransposePlan(n),
             false};
  } else {
    std::fprintf(stderr, "unknown family %s\n", family.c_str());
    std::exit(2);
  }
  f.rawBytes = n * n * 4;
  f.wireBytes = f.fam.s8 ? n * n : f.rawBytes;

  // Deterministic host weights (fixed seed, same convention as rtrack).
  f.hostW.resize((size_t)(n * n));
  std::mt19937_64 rng(0x52375237ULL);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : f.hostW)
    v = dist(rng);

  // Honest per-channel scales (channel = outer row), exactly rtrack's rule.
  if (f.fam.s8) {
    f.invScales.assign((size_t)n, 1.0f);
    for (int64_t c = 0; c < n; ++c) {
      float maxAbs = 0;
      for (int64_t i = 0; i < n; ++i)
        maxAbs = std::max(maxAbs, std::fabs(f.hostW[(size_t)(c * n + i)]));
      f.invScales[(size_t)c] = maxAbs > 0 ? 127.0f / maxAbs : 1.0f;
    }
  }

  // Reference wire image (scalar variant: the verify oracle).
  f.refWire.resize((size_t)f.wireBytes);
  if (f.fam.s8) {
    reloc::quant::quantizePackF32S8(f.hostW.data(), f.refWire.data(), n, n,
                                    f.invScales.data(),
                                    reloc::quant::Variant::Scalar);
  } else {
    reloc::gatherChunk(f.fam.bound, f.hostW.data(), f.refWire.data(), 0,
                       f.fam.bound.extents[0]);
  }

  // Pinned buffers.
  CUDA_CHECK(cudaHostAlloc((void **)&f.pinnedRaw, (size_t)f.rawBytes,
                           cudaHostAllocDefault));
  std::memcpy(f.pinnedRaw, f.hostW.data(), (size_t)f.rawBytes);
  CUDA_CHECK(cudaHostAlloc((void **)&f.pinnedWire, (size_t)f.wireBytes,
                           cudaHostAllocDefault));
  std::memcpy(f.pinnedWire, f.refWire.data(), (size_t)f.wireBytes);

  // Device buffers.
  for (int s = 0; s < 2; ++s) {
    CUDA_CHECK(cudaMalloc(&f.dW[s], (size_t)f.wireBytes));
    CUDA_CHECK(cudaMalloc((void **)&f.dRaw[s], (size_t)f.rawBytes));
  }
  if (f.fam.s8) {
    CUDA_CHECK(cudaMalloc((void **)&f.dInv, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(f.dInv, f.invScales.data(), sizeof(float) * n,
                          cudaMemcpyHostToDevice));
    // Static int8 activations: symmetric scale over the whole x, RNE.
    std::vector<float> xf((size_t)(n * bAct));
    std::mt19937_64 rng2(0xAC71AC71ULL);
    for (auto &v : xf)
      v = dist(rng2);
    float maxAbs = 0;
    for (float v : xf)
      maxAbs = std::max(maxAbs, std::fabs(v));
    const float xInv = maxAbs > 0 ? 127.0f / maxAbs : 1.0f;
    std::vector<int8_t> xi(xf.size());
    for (size_t i = 0; i < xf.size(); ++i)
      xi[i] = (int8_t)std::lrintf(
          std::min(127.0f, std::max(-127.0f, xf[i] * xInv)));
    CUDA_CHECK(cudaMalloc((void **)&f.dXi, xi.size()));
    CUDA_CHECK(cudaMemcpy(f.dXi, xi.data(), xi.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&f.dY, sizeof(int32_t) * n * bAct));
  } else {
    std::vector<float> xf((size_t)(n * bAct));
    std::mt19937_64 rng2(0xAC71AC71ULL);
    for (auto &v : xf)
      v = dist(rng2);
    CUDA_CHECK(cudaMalloc((void **)&f.dXf, sizeof(float) * n * bAct));
    CUDA_CHECK(cudaMemcpy(f.dXf, xf.data(), sizeof(float) * n * bAct,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&f.dY, sizeof(float) * n * bAct));
  }
  return f;
}

void destroyFixture(Fixture &f) {
  for (int s = 0; s < 2; ++s) {
    cudaFree(f.dW[s]);
    cudaFree(f.dRaw[s]);
    if (f.staging[s])
      cudaFreeHost(f.staging[s]);
  }
  if (f.dInv)
    cudaFree(f.dInv);
  if (f.dXi)
    cudaFree(f.dXi);
  if (f.dXf)
    cudaFree(f.dXf);
  if (f.dY)
    cudaFree(f.dY);
  cudaFreeHost(f.pinnedRaw);
  cudaFreeHost(f.pinnedWire);
}

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

  cudaFree(wF);
  cudaFree(xF);
  cudaFree(yF);
  cudaFree(wI);
  cudaFree(xI);
  cudaFree(yI);
  g.destroy();
  return 0;
}

// Streams + per-layer events for one measured pass. copyStream carries all
// H2D; recvStream carries b's per-layer receive kernel; GemmEngine.stream
// carries compute. loadDone[slot] gates compute; bufFree[slot] gates the
// slot's next overwrite (recorded on the compute stream).
struct PipeCtx {
  cudaStream_t copyStream = nullptr, recvStream = nullptr;
  std::vector<cudaEvent_t> loadDone, bufFree, h2dTail;
  void init(int64_t layers) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&recvStream, cudaStreamNonBlocking));
    loadDone.resize((size_t)layers);
    bufFree.resize((size_t)layers);
    h2dTail.resize((size_t)layers);
    for (int64_t k = 0; k < layers; ++k) {
      CUDA_CHECK(cudaEventCreateWithFlags(&loadDone[(size_t)k],
                                          cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&bufFree[(size_t)k],
                                          cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&h2dTail[(size_t)k],
                                          cudaEventDisableTiming));
    }
  }
  void destroy() {
    for (auto &e : loadDone)
      cudaEventDestroy(e);
    for (auto &e : bufFree)
      cudaEventDestroy(e);
    for (auto &e : h2dTail)
      cudaEventDestroy(e);
    cudaStreamDestroy(copyStream);
    cudaStreamDestroy(recvStream);
  }
};

enum class Loader { A, BPipelined, APrefold, None };

struct PassCfg {
  Loader loader = Loader::A;
  bool serial = false; // sync between load and compute, layer by layer
  bool compute = true; // false = load-only anchor
  int64_t repeats = 1; // GEMMs per layer
  int64_t chunkBytes = 4 << 20;
};

// Method A / a_prefold host-side load of one layer into slot. For A the
// host transform writes chunks into pinned staging (double buffered) and
// DMAs each; for prefold the pre-transformed pinned wire DMAs directly.
// Returns after ISSUING all copies on copyStream (host has synced only on
// staging reuse, exactly the rtrack Method-A discipline).
void issueHostLoad(Fixture &f, PipeCtx &px, const PassCfg &cfg, int slot,
                   reloc::GatherPool &pool,
                   std::vector<cudaEvent_t> &chunkEvents) {
  char *dst = (char *)f.dW[slot];
  if (cfg.loader == Loader::APrefold) {
    const auto ck = bench::rtrack::planByteChunks(f.wireBytes, cfg.chunkBytes);
    int64_t off = 0;
    for (int64_t c = 0; c < ck.nChunks; ++c) {
      const int64_t bytes = std::min(ck.bytesPerChunk, f.wireBytes - off);
      CUDA_CHECK(cudaMemcpyAsync(dst + off, (char *)f.pinnedWire + off,
                                 (size_t)bytes, cudaMemcpyHostToDevice,
                                 px.copyStream));
      off += bytes;
    }
    return;
  }
  // Loader::A — chunk on dst outer rows; staging holds OUTPUT bytes.
  const int64_t rows = f.fam.bound.extents[0];
  const int64_t outRowBytes = f.wireBytes / rows;
  const auto ck =
      bench::rtrack::planRowChunks(rows, outRowBytes, cfg.chunkBytes);
  const int64_t rowSrcBytes = f.rawBytes / rows;
  const int64_t minRows =
      std::max<int64_t>(1, (int64_t)reloc::kMinGatherBytesPerWorker /
                               std::max<int64_t>(1, rowSrcBytes));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int buf = (int)(c & 1);
    // Unconditional: chunkEvents/staging persist across layers within a
    // pass, so staging[buf]'s prior DMA (this layer's earlier chunk OR the
    // previous layer's tail chunk) must be observed complete before the
    // host overwrites it, not just chunks c>=2 within this layer.
    CUDA_CHECK(cudaEventSynchronize(chunkEvents[(size_t)(c & 1)]));
    const int64_t rb = c * ck.rowsPerChunk;
    const int64_t re = std::min(rows, rb + ck.rowsPerChunk);
    char *stage = (char *)f.staging[buf];
    if (f.fam.s8) {
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::quantizePackF32S8(
            f.hostW.data() + sb * f.n, (int8_t *)stage + (sb - rb) * f.n,
            se - sb, f.n, f.invScales.data() + sb, reloc::quant::Variant::Auto);
      });
    } else {
      void *rebased = stage - rb * outRowBytes;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::gatherChunk(f.fam.bound, f.hostW.data(), rebased, sb, se);
      });
    }
    CUDA_CHECK(cudaMemcpyAsync(dst + rb * outRowBytes, stage,
                               (size_t)((re - rb) * outRowBytes),
                               cudaMemcpyHostToDevice, px.copyStream));
    CUDA_CHECK(cudaEventRecord(chunkEvents[(size_t)buf], px.copyStream));
  }
}

// b_pipelined: chunked raw H2D into dRaw[slot] on copyStream; ONE receive
// kernel per layer on recvStream after the layer's H2D tail (see file
// header for why per-layer, not per-chunk).
void issueBLoad(Fixture &f, PipeCtx &px, const PassCfg &cfg, int slot,
                int64_t layer) {
  const auto ck = bench::rtrack::planByteChunks(f.rawBytes, cfg.chunkBytes);
  char *dst = (char *)f.dRaw[slot];
  int64_t off = 0;
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int64_t bytes = std::min(ck.bytesPerChunk, f.rawBytes - off);
    CUDA_CHECK(cudaMemcpyAsync(dst + off, (char *)f.pinnedRaw + off,
                               (size_t)bytes, cudaMemcpyHostToDevice,
                               px.copyStream));
    off += bytes;
  }
  CUDA_CHECK(cudaEventRecord(px.h2dTail[(size_t)layer], px.copyStream));
  CUDA_CHECK(cudaStreamWaitEvent(px.recvStream, px.h2dTail[(size_t)layer], 0));
  if (f.fam.s8) {
    reloc::cuda::quantizeF32S8(f.dRaw[slot], (int8_t *)f.dW[slot], f.n, f.n,
                               f.dInv, px.recvStream);
  } else {
    reloc::cuda::relocateF32(f.fam.bound, f.dRaw[slot], (float *)f.dW[slot],
                             px.recvStream);
  }
}

// One full L-layer pass. Returns host wall ms (nowMs bracket around issue +
// final device sync).
double runPass(Fixture &f, PipeCtx &px, GemmEngine &g, const PassCfg &cfg,
               reloc::GatherPool &pool) {
  std::vector<cudaEvent_t> chunkEvents(2);
  CUDA_CHECK(cudaEventCreateWithFlags(&chunkEvents[0], cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&chunkEvents[1], cudaEventDisableTiming));
  // Pre-record both on copyStream (empty stream -> instantly complete) so
  // issueHostLoad's unconditional per-chunk sync has something valid to
  // wait on even for the very first chunk written into each staging slot.
  CUDA_CHECK(cudaEventRecord(chunkEvents[0], px.copyStream));
  CUDA_CHECK(cudaEventRecord(chunkEvents[1], px.copyStream));
  const double t0 = nowMs();
  for (int64_t k = 0; k < f.layers; ++k) {
    const int slot = (int)(k & 1);
    // The slot must be free of layer k-2's compute before overwrite.
    if (cfg.loader != Loader::None && cfg.compute && k >= 2)
      CUDA_CHECK(
          cudaStreamWaitEvent(px.copyStream, px.bufFree[(size_t)(k - 2)], 0));
    // Load-only anchors skip the bufFree chain (no compute), but
    // b_pipelined's dRaw[slot] is still reused every 2 layers: order the
    // H2D overwrite behind layer k-2's recv kernel (loadDone, recorded on
    // recvStream) so it doesn't race the reader.
    if (cfg.loader == Loader::BPipelined && !cfg.compute && k >= 2)
      CUDA_CHECK(
          cudaStreamWaitEvent(px.copyStream, px.loadDone[(size_t)(k - 2)], 0));
    // Issue the load.
    cudaStream_t loadTail = px.copyStream;
    if (cfg.loader == Loader::A || cfg.loader == Loader::APrefold) {
      issueHostLoad(f, px, cfg, slot, pool, chunkEvents);
    } else if (cfg.loader == Loader::BPipelined) {
      issueBLoad(f, px, cfg, slot, k);
      loadTail = px.recvStream;
    }
    if (cfg.loader != Loader::None)
      CUDA_CHECK(cudaEventRecord(px.loadDone[(size_t)k], loadTail));
    if (cfg.serial)
      CUDA_CHECK(cudaDeviceSynchronize());
    // Issue the compute.
    if (cfg.compute) {
      if (cfg.loader != Loader::None)
        CUDA_CHECK(cudaStreamWaitEvent(g.stream, px.loadDone[(size_t)k], 0));
      for (int64_t rpt = 0; rpt < cfg.repeats; ++rpt) {
        if (f.fam.s8)
          g.igemm((const int8_t *)f.dW[slot], f.dXi, (int32_t *)f.dY, f.n,
                  f.bAct);
        else
          g.sgemm((const float *)f.dW[slot], f.dXf, (float *)f.dY, f.n, f.bAct);
      }
      CUDA_CHECK(cudaEventRecord(px.bufFree[(size_t)k], g.stream));
      if (cfg.serial)
        CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const double t1 = nowMs();
  cudaEventDestroy(chunkEvents[0]);
  cudaEventDestroy(chunkEvents[1]);
  return t1 - t0;
}

RStats timePass(Fixture &f, PipeCtx &px, GemmEngine &g, const PassCfg &cfg,
                reloc::GatherPool &pool) {
  std::vector<double> ms;
  for (int it = 0; it < kWarmup + kIters; ++it) {
    const double w = runPass(f, px, g, cfg, pool);
    if (it >= kWarmup)
      ms.push_back(w);
  }
  return summarizeSamples(ms);
}

// compute_bare: plain GEMM loop, no pipeline scaffolding, no per-layer
// events — the G3a reference. Weights = the reference wire, preloaded.
RStats timeComputeBare(Fixture &f, GemmEngine &g, int64_t repeats) {
  CUDA_CHECK(cudaMemcpy(f.dW[0], f.refWire.data(), (size_t)f.wireBytes,
                        cudaMemcpyHostToDevice));
  // Hygiene: make sure the pageable-copy DMA above is fully drained before
  // the first warmup GEMM runs (untimed warmup only; no data changes).
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ms;
  for (int it = 0; it < kWarmup + kIters; ++it) {
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
    const double t0 = nowMs();
    for (int64_t k = 0; k < f.layers; ++k)
      for (int64_t rpt = 0; rpt < repeats; ++rpt) {
        if (f.fam.s8)
          g.igemm((const int8_t *)f.dW[0], f.dXi, (int32_t *)f.dY, f.n, f.bAct);
        else
          g.sgemm((const float *)f.dW[0], f.dXf, (float *)f.dY, f.n, f.bAct);
      }
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
    const double t1 = nowMs();
    if (it >= kWarmup)
      ms.push_back(t1 - t0);
  }
  return summarizeSamples(ms);
}

// Bit-exact verify: last-written weight slot vs the host reference wire;
// and y vs the stored reference y (set by the first verified config of the
// family — compute_bare's y, same GEMM config on identical weights).
bool verifyConfig(Fixture &f, int lastSlot, std::vector<char> &refY) {
  std::vector<char> w((size_t)f.wireBytes);
  CUDA_CHECK(cudaMemcpy(w.data(), f.dW[lastSlot], (size_t)f.wireBytes,
                        cudaMemcpyDeviceToHost));
  if (std::memcmp(w.data(), f.refWire.data(), (size_t)f.wireBytes) != 0)
    return false;
  const size_t yBytes = (size_t)(f.n * f.bAct * 4);
  std::vector<char> y(yBytes);
  CUDA_CHECK(cudaMemcpy(y.data(), f.dY, yBytes, cudaMemcpyDeviceToHost));
  if (refY.empty()) {
    refY = y;
    return true;
  }
  return std::memcmp(y.data(), refY.data(), yBytes) == 0;
}

std::string num(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}

void writeRow(std::ofstream &csv, const Args &a, const Fixture &f,
              const std::string &method, int64_t chunkBytes, int64_t nChunks,
              int64_t repeats, double measuredC, const RStats &wall,
              double loadOnly, double computeOnly, bool verified) {
  const double delta = (loadOnly > 0 || computeOnly > 0)
                           ? wall.median - std::max(loadOnly, computeOnly)
                           : 0.0;
  const double exposed = computeOnly > 0 ? wall.median - computeOnly : 0.0;
  csv << a.machine << ',' << gpuFromMachine(a.machine) << ',' << method << ','
      << f.fam.name << ',' << f.n << ',' << num(f.fam.r) << ",8,"
      << num((double)chunkBytes / (1 << 20)) << ',' << nChunks << ','
      << f.layers << ',' << f.bAct << ',' << repeats << ',' << num(measuredC)
      << ',' << num(wall.median) << ',' << num(wall.min) << ',' << num(wall.p95)
      << ',' << num(wall.iqrOverMedianPct) << ',' << (wall.unstable ? 1 : 0)
      << ',' << num(loadOnly) << ',' << num(computeOnly) << ',' << num(delta)
      << ',' << num(exposed) << ',' << (verified ? 1 : 0) << ",1\n";
  csv.flush();
}

} // namespace

int main(int argc, char **argv) {
  Args a = parseArgs(argc, argv);
  if (a.smoke)
    return runSmoke(a.n, a.gemmBatch);
  if (a.machine.empty() || a.csvPath.empty()) {
    std::fprintf(stderr, "--machine and --csv are required\n");
    return 2;
  }
  Fixture f = makeFixture(a.family, a.n, a.layers, a.gemmBatch);
  GemmEngine g;
  g.init();
  PipeCtx px;
  px.init(a.layers);
  reloc::GatherPool pool(8);

  // Staging for Method A: 2 x the largest requested chunk (output bytes).
  const std::vector<double> chunksMib =
      a.chunksMib.empty() ? (f.fam.s8 ? std::vector<double>{4, 16}
                                      : std::vector<double>{16, 64})
                          : splitDoubles(a.chunksMib);
  const std::vector<double> bChunksMib = a.chunksMib.empty()
                                             ? std::vector<double>{256, 16}
                                             : splitDoubles(a.chunksMib);
  int64_t maxStage = 0;
  for (double c : chunksMib) {
    const int64_t rows = f.fam.bound.extents[0];
    const auto ck = bench::rtrack::planRowChunks(rows, f.wireBytes / rows,
                                                 (int64_t)(c * (1 << 20)));
    maxStage = std::max(maxStage, ck.stagingBytes);
  }
  f.stagingBytes = maxStage;
  for (int s = 0; s < 2; ++s)
    CUDA_CHECK(
        cudaHostAlloc(&f.staging[s], (size_t)maxStage, cudaHostAllocDefault));

  std::ofstream csv(a.csvPath, std::ios::app);
  csv << "machine,gpu,method,transform,N,r,threads,chunk_req_mib,n_chunks,"
         "layers,gemm_batch,repeats,measured_C,median_ms,min_ms,p95_ms,"
         "iqr_over_median_pct,unstable,load_only_ms,compute_only_ms,"
         "delta_ms,exposed_load_ms,verified,post_freeze\n";

  // 1. Estimate repeats per C target from 3-rep medians (deterministic
  //    procedure; measured_C is what gates key on, so R is only a knob).
  auto quick = [&](const PassCfg &cfg) {
    std::vector<double> ms;
    for (int it = 0; it < 3; ++it)
      ms.push_back(runPass(f, px, g, cfg, pool));
    std::sort(ms.begin(), ms.end());
    return ms[1];
  };
  PassCfg gemCfg;
  gemCfg.loader = Loader::None;
  gemCfg.repeats = 1;
  const double tGemmLayer = quick(gemCfg) / (double)a.layers;
  PassCfg la;
  la.loader = Loader::A;
  la.compute = false;
  la.chunkBytes = (int64_t)(chunksMib[0] * (1 << 20));
  const double tLoadALayer = quick(la) / (double)a.layers;
  std::vector<int64_t> repeatsGrid;
  for (double c : splitDoubles(a.targetCs)) {
    const int64_t r = std::max<int64_t>(
        1, (int64_t)std::llround(c * tLoadALayer / tGemmLayer));
    if (repeatsGrid.empty() || repeatsGrid.back() != r)
      repeatsGrid.push_back(r);
  }

  // 2. Anchors per repeats value: compute_only (pipeline scaffolding, no
  //    loads) and compute_bare (G3a reference).
  std::vector<char> refY;
  struct Anchor {
    RStats only, bare;
  };
  std::vector<Anchor> anchors(repeatsGrid.size());
  for (size_t i = 0; i < repeatsGrid.size(); ++i) {
    PassCfg co;
    co.loader = Loader::None;
    co.repeats = repeatsGrid[i];
    // Preload the reference wire so compute_only computes on real weights.
    CUDA_CHECK(cudaMemcpy(f.dW[0], f.refWire.data(), (size_t)f.wireBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f.dW[1], f.refWire.data(), (size_t)f.wireBytes,
                          cudaMemcpyHostToDevice));
    // Hygiene: drain the pageable-copy DMA before the first warmup GEMM
    // (untimed warmup only; no data changes).
    CUDA_CHECK(cudaDeviceSynchronize());
    anchors[i].only = timePass(f, px, g, co, pool);
    anchors[i].bare = timeComputeBare(f, g, repeatsGrid[i]);
    bool v = !a.verify || verifyConfig(f, 0, refY);
    writeRow(csv, a, f, "compute_only", 0, 0, repeatsGrid[i], 0,
             anchors[i].only, 0, anchors[i].only.median, v);
    writeRow(csv, a, f, "compute_bare", 0, 0, repeatsGrid[i], 0,
             anchors[i].bare, 0, anchors[i].bare.median, v);
  }

  // 3. Loader matrix.
  struct M {
    const char *name;
    Loader l;
    bool serial;
  };
  const std::vector<M> all = {{"a", Loader::A, false},
                              {"b_pipelined", Loader::BPipelined, false},
                              {"a_prefold", Loader::APrefold, false},
                              {"a_serial", Loader::A, true},
                              {"b_serial", Loader::BPipelined, true}};
  for (const M &m : all) {
    if (a.methods != "all" && a.methods.find(m.name) == std::string::npos)
      continue;
    const std::vector<double> &grid = (m.l == Loader::BPipelined) ? bChunksMib
                                      : (m.l == Loader::APrefold)
                                          ? std::vector<double>{64}
                                          : chunksMib;
    for (double cMib : grid) {
      PassCfg cfg;
      cfg.loader = m.l;
      cfg.serial = m.serial;
      cfg.chunkBytes = (int64_t)(cMib * (1 << 20));
      // load-only anchor for this (method, chunk).
      PassCfg lo = cfg;
      lo.compute = false;
      lo.serial = false;
      const RStats loadOnly = timePass(f, px, g, lo, pool);
      const int64_t nCh =
          (m.l == Loader::BPipelined)
              ? bench::rtrack::planByteChunks(f.rawBytes, cfg.chunkBytes)
                    .nChunks
          : (m.l == Loader::APrefold)
              ? bench::rtrack::planByteChunks(f.wireBytes, cfg.chunkBytes)
                    .nChunks
              : bench::rtrack::planRowChunks(
                    f.fam.bound.extents[0],
                    f.wireBytes / f.fam.bound.extents[0], cfg.chunkBytes)
                    .nChunks;
      writeRow(csv, a, f, std::string(m.name) + "_load_only", cfg.chunkBytes,
               nCh, 0, 0, loadOnly, loadOnly.median, 0, true);
      for (size_t i = 0; i < repeatsGrid.size(); ++i) {
        cfg.repeats = repeatsGrid[i];
        const RStats wall = timePass(f, px, g, cfg, pool);
        const double mC =
            loadOnly.median > 0 ? anchors[i].only.median / loadOnly.median : 0;
        bool v = !a.verify || verifyConfig(f, (int)((f.layers - 1) & 1), refY);
        writeRow(csv, a, f, m.name, cfg.chunkBytes, nCh, repeatsGrid[i], mC,
                 wall, loadOnly.median, anchors[i].only.median, v);
        if (a.verify && !v) {
          std::fprintf(stderr, "VERIFY FAIL: %s chunk=%g repeats=%" PRId64 "\n",
                       m.name, cMib, repeatsGrid[i]);
          return 1;
        }
      }
    }
  }
  destroyFixture(f);
  px.destroy();
  g.destroy();
  std::printf("done: %s\n", a.csvPath.c_str());
  return 0;
}
