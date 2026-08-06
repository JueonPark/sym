//===- rtrack_bench.cu - R0.3 pipeline & measurement harness (#76) --------===//
//
// Method A vs Method B end-to-end (transform + transfer) latency for one
// tensor. Source: pageable host DRAM. Staging: 2 pinned buffers of chunk
// size (double-buffered, event-gated). Destination: final layout in GPU
// global memory.
//   A: per-chunk CPU transform (R0.1 kernels / gatherChunk, parallelized
//      over a GatherPool) into pinned staging -> cudaMemcpyAsync of r*S
//      bytes total, optionally followed by an in-stream GPU receive kernel
//      (R2's dequant/unpack: convert_f16_f32, dequant_s8_f32, unpack_s4_s8
//      + dequant_s8_f32) that decompresses the wire payload back to f32 in
//      the final layout.
//   B: per-chunk pageable->pinned memcpy (parallelized over the same pool,
//      so both methods get the same thread budget) -> cudaMemcpyAsync of S
//      bytes -> R0.2 GPU transform kernels into the final layout.
//   B_fair (issue #95): same as B but the source is resident in pinned
//      memory, so the per-chunk staging memcpy is gone -- the DMA reads the
//      source directly. This is the admissible baseline (host_stage_ms=0);
//      B is kept as the "b_staged" comparison, not silently replaced.
// Timing: full pipeline via CUDA events (start recorded before the first
// stage, stop after the last enqueued op, then cudaStreamSynchronize),
// fenced steady_clock wall time, and per-stage sums (CPU transform ms,
// summed per-chunk H2D event ms, GPU kernel event ms). Protocol: 5 warmup
// + 30 timed; median/min/p95; IQR/median > 5% flags the row.
// Every (workload, method, chunk) config is verified bit-exact against a
// CPU reference before it is timed; plans are hand-authored
// (rtrack/plans.h) and oracle-checked in RtrackTest, NOT decoded from the
// frozen golden blob (issue #63).
//
//===----------------------------------------------------------------------===//

#include "rtrack/chunking.h"
#include "rtrack/csv.h"
#include "rtrack/rstats.h"
#include "rtrack/workloads.h"

#include "reloc/CudaKernels.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,         \
                   __LINE__, cudaGetErrorString(err_), #x);                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

using namespace bench::rtrack;

// GPU side of T5. Bench-local (the poc_transpose precedent): issue #75's
// kernel set has no f32->f16 convert. __float2half_rn is round-to-nearest-
// even, matching the CPU convertF32F16 contract for non-NaN inputs (the
// generated data has no NaNs).
__global__ void convertF32F16Kernel(const float *src, __half *dst,
                                    int64_t count) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < count)
    dst[i] = __float2half_rn(src[i]);
}

void launchConvertF32F16(const float *dSrc, void *dDst, int64_t count,
                         cudaStream_t stream) {
  const int block = 256;
  const int64_t grid = (count + block - 1) / block;
  convertF32F16Kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(
      dSrc, static_cast<__half *>(dDst), count);
  CUDA_CHECK(cudaGetLastError());
}

// --plan-wire: read a corpus-format plan (wire-format v0, e.g. a committed
// blocked_transpose_sym.bin from the compiler fold path) and produce the
// BoundPlan the rest of the pipeline consumes -- the same decodePlan/bind
// contract exercised by libreloc/test/{Decode,Bind}Test.cpp and the r-track
// bench/gather_bw.cpp precedent, just wired into the rtrack harness so a
// COMPILER-EMITTED plan can be measured end to end instead of only the
// hand-authored plans.h table.
std::vector<uint8_t> readFileBytes(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "error: --plan-wire: cannot open %s\n", path);
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> bytes(size > 0 ? static_cast<size_t>(size) : 0);
  if (size > 0 &&
      std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    std::fprintf(stderr, "error: --plan-wire: short read on %s\n", path);
    std::fclose(f);
    std::exit(1);
  }
  std::fclose(f);
  return bytes;
}

reloc::BoundPlan decodeAndBindWire(const char *path, int64_t n) {
  std::vector<uint8_t> bytes = readFileBytes(path);
  reloc::DecodeResult decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan) {
    const reloc::DecodeError &e = std::get<reloc::DecodeError>(decoded);
    std::fprintf(stderr,
                 "error: --plan-wire: decode failed at byte offset %zu: %s\n",
                 e.offset, e.message.c_str());
    std::exit(1);
  }
  reloc::BindResult bound = reloc::bind(*plan, {{"N", n}});
  auto *b = std::get_if<reloc::BoundPlan>(&bound);
  if (!b) {
    const reloc::BindError &e = std::get<reloc::BindError>(bound);
    std::fprintf(stderr, "error: --plan-wire: bind failed for N=%lld: %s\n",
                 static_cast<long long>(n), e.message.c_str());
    std::exit(1);
  }
  return *b;
}

// Synthetic T1b-shaped workload entry for the decoded+bound --plan-wire
// plan. makePlan is never called: buildFixture takes the wire plan via its
// boundOverride parameter instead (see run()). Not part of allWorkloads()
// (it has no static plan builder), so it is special-cased in the
// --transform parser rather than looked up by findWorkload.
const Workload &wireWorkload() {
  static const Workload w = {
      "TW",       "blocked_transpose_wire",
      "matrix",   DtypeOut::F32,
      Wire::F32,  1.0,
      nullptr,    CpuStage::GatherF32,
      GpuStage::Relocate, RecvStage::None,
      true};
  return w;
}

const char *variantName(reloc::quant::Variant v) {
  switch (v) {
  case reloc::quant::Variant::Auto:
    return "auto";
  case reloc::quant::Variant::Scalar:
    return "scalar";
  case reloc::quant::Variant::AVX2:
    return "avx2";
  case reloc::quant::Variant::AVX512:
    return "avx512";
  case reloc::quant::Variant::AVX512Pf:
    return "avx512pf";
  }
  return "?";
}

// Method A (CPU transform + DMA of r*S bytes) vs the three Method-B
// baselines. BStaged is the sym#63/#76 baseline: pageable source re-staged
// through a pinned buffer per chunk before the DMA. BFair (issue #95)
// removes that staging memcpy -- the source is resident in pinned memory
// and the DMA reads it directly -- so B is measured as a competent
// pure-relocation baseline would run it. Both emit the R0.2 receive kernels
// after transfer. BPipelined (issue #114) rides BFair's DMA path but issues
// the transform kernel per chunk, in-stream, instead of once after the full
// transfer -- Method A's loop shape on B's buffer model.
enum class Method { A, BStaged, BFair, BPipelined };

// CSV method tag. BStaged stays "b" so the R1/figure1/gates consumers that
// key off "a"/"b" are unchanged; BFair and BPipelined are new tags they
// ignore.
const char *methodTag(Method m) {
  switch (m) {
  case Method::A:
    return "a";
  case Method::BStaged:
    return "b";
  case Method::BFair:
    return "b_fair";
  case Method::BPipelined:
    // Reserved: cm4_registered_predictions.json's placement_map keys the
    // Overlapped placement off this tag.
    return "b_pipelined";
  }
  return "?";
}

bool hasMethod(const std::vector<Method> &ms, Method m) {
  return std::find(ms.begin(), ms.end(), m) != ms.end();
}

struct Options {
  std::vector<const Workload *> workloads;
  std::vector<Method> methods = {Method::A, Method::BStaged};
  int64_t n = 8192;
  std::vector<int64_t> chunkBytes = {4ll << 20, 16ll << 20, 64ll << 20,
                                     256ll << 20};
  unsigned threads = 1;
  reloc::quant::Variant variant = reloc::quant::Variant::Auto;
  int warmup = kWarmup;
  int iters = kIters;
  std::string machine;
  const char *csvPath = "-";
  bool csvHeader = false;
  bool verify = true;
  // Compiler-emitted plan row (title-gap closure): a corpus wire-format
  // file, decoded + bound at --n and substituted for the TW workload's
  // (nonexistent) makePlan. Only valid together with --transform TW.
  const char *planWire = nullptr;
};

std::string defaultMachine() {
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0)
    return "unknown";
  return host;
}

bool needsQuant(const Workload &w) {
  return w.wire == Wire::S8 || w.wire == Wire::S4;
}

// Per-(workload, N) fixture: pageable source, per-channel scales, CPU
// reference of the final artifact, device buffers.
struct Fixture {
  const Workload *w = nullptr;
  reloc::BoundPlan bound;
  std::vector<float> hostSrc;   // pageable; N^2 elements
  std::vector<float> invScales; // extents[0] entries (quant workloads)
  std::vector<float> scales;    // dequant scales = 1/invScales (recv rows)
  std::vector<uint8_t> ref;     // Method B's expected artifact, outBytes
  std::vector<uint8_t> refA;    // Method A's expected artifact (lossy
                                // roundtrip for compressed wires)
  int64_t totalElems = 0, inBytes = 0, outBytes = 0;
  int64_t rows = 0, rowOutBytes = 0, channels = 0, channelSize = 0;
  // Host buffers:
  float *pinnedSrc = nullptr; // B_fair: source resident in pinned memory, so
                              // the DMA reads it directly (no staging memcpy)
  // Device buffers:
  void *dOut = nullptr;     // final artifact (A's DMA target, B's kernel dst)
  float *dLin = nullptr;    // B: linear fp32 source copy
  float *dTmp = nullptr;    // B RelocateQuant: relocated fp32 before quantize
  float *dInv = nullptr;    // per-channel invScales (B quantize kernels)
  void *dRecv = nullptr;    // A: compressed DMA target (wire != F32)
  int8_t *dS8 = nullptr;    // A: S4 receive intermediate
  float *dScales = nullptr; // per-channel dequant scales (A receive)

  ~Fixture() {
    if (pinnedSrc)
      cudaFreeHost(pinnedSrc);
    cudaFree(dOut);
    cudaFree(dLin);
    cudaFree(dTmp);
    cudaFree(dInv);
    cudaFree(dRecv);
    cudaFree(dS8);
    cudaFree(dScales);
  }
};

void buildFixture(Fixture &f, const Workload &w, int64_t n, bool needB,
                  bool needBFair, bool verify,
                  const reloc::BoundPlan *boundOverride = nullptr) {
  f.w = &w;
  f.bound = boundOverride ? *boundOverride : w.makePlan(n);
  f.totalElems = n * n;
  f.inBytes = f.totalElems * 4;
  f.outBytes = f.totalElems * dtypeBytes(w.dtypeOut);
  f.rows = f.bound.extents[0];
  f.rowOutBytes = wireBytes(w.wire, f.bound.dstStrides[0]);
  f.channels = f.bound.extents[0];
  f.channelSize = f.totalElems / f.channels;

  // Fail fast on recipes the chunked staging rebase, the parallel
  // per-outer-row gather, and the per-channel quantize cannot handle.
  // These are library-assert territory, but release benchmarking builds
  // compile with -DNDEBUG, so check unconditionally (issue #63's lesson:
  // a silently wrong benchmark is worse than none).
  int64_t innerExtent = 1;
  for (size_t k = 1; k < f.bound.extents.size(); ++k)
    innerExtent *= f.bound.extents[k];
  if (!f.bound.padRegions.empty() || f.bound.dstStrides.back() != 1 ||
      f.bound.dstStrides[0] != innerExtent) {
    std::fprintf(stderr,
                 "error: %s: plan is not packed-dst/pad-free; the rtrack "
                 "pipeline cannot measure it\n",
                 w.id);
    std::exit(1);
  }
  if ((w.cpuStage == CpuStage::QuantPack ||
       w.cpuStage == CpuStage::ConvertF16 || w.cpuStage == CpuStage::CopyF32 ||
       w.cpuStage == CpuStage::QuantPackS4) &&
      f.bound.srcStrides != f.bound.dstStrides) {
    std::fprintf(stderr,
                 "error: %s: contiguous CPU stage requires an identity "
                 "plan (the reference computation assumes it too)\n",
                 w.id);
    std::exit(1);
  }

  f.hostSrc.resize(static_cast<size_t>(f.totalElems));
  for (int64_t i = 0; i < f.totalElems; ++i)
    f.hostSrc[static_cast<size_t>(i)] =
        (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;

  if (needsQuant(w)) {
    // Honest per-channel scales: channel = the plan's coalesced outer
    // axis; maxAbs over each channel's strided source footprint.
    const float qmax = w.wire == Wire::S4 ? 7.0f : 127.0f;
    f.invScales.assign(static_cast<size_t>(f.channels), 0.0f);
    const size_t rank = f.bound.extents.size();
    for (int64_t c = 0; c < f.channels; ++c) {
      float maxAbs = 0.0f;
      // Odometer over the inner axes with an incrementally maintained
      // source offset (O(1) per element, not O(rank)).
      std::vector<int64_t> idx(rank, 0);
      int64_t so = c * f.bound.srcStrides[0];
      bool done = false;
      while (!done) {
        maxAbs =
            std::max(maxAbs, std::fabs(f.hostSrc[static_cast<size_t>(so)]));
        size_t k = rank;
        for (;;) {
          if (k <= 1) {
            done = true;
            break;
          }
          --k;
          if (++idx[k] < f.bound.extents[k]) {
            so += f.bound.srcStrides[k];
            break;
          }
          idx[k] = 0;
          so -= (f.bound.extents[k] - 1) * f.bound.srcStrides[k];
        }
      }
      f.invScales[static_cast<size_t>(c)] = maxAbs > 0 ? qmax / maxAbs : 1.0f;
    }
    if (w.recvStage == RecvStage::DequantS8 ||
        w.recvStage == RecvStage::UnpackDequantS4) {
      f.scales.resize(f.invScales.size());
      for (size_t c = 0; c < f.scales.size(); ++c)
        f.scales[c] = 1.0f / f.invScales[c];
    }
  }

  // CPU reference of the final artifact (scalar kernels; the plan itself
  // is oracle-verified in RtrackTest). Skipped under --no-verify: at large
  // N the serial scalar reference costs seconds per workload and nothing
  // reads it when the gate is off.
  if (verify) {
    f.ref.resize(static_cast<size_t>(f.outBytes));
    switch (w.dtypeOut) {
    case DtypeOut::F32:
      reloc::executeH2D(f.bound, f.hostSrc.data(), f.ref.data());
      break;
    case DtypeOut::S8:
      reloc::quant::gatherQuantizeF32S8(
          f.bound, f.hostSrc.data(), reinterpret_cast<int8_t *>(f.ref.data()),
          f.invScales.data(), 0, f.channels, reloc::quant::Variant::Scalar);
      break;
    case DtypeOut::F16:
      // Valid only because ConvertF16 workloads are identity-plan (checked
      // above); a relocating f16 workload needs a plan-aware reference.
      reloc::quant::convertF32F16(f.hostSrc.data(),
                                  reinterpret_cast<uint16_t *>(f.ref.data()),
                                  f.totalElems, reloc::quant::Variant::Scalar);
      break;
    }

    // Method A's expected artifact. Identical to f.ref except when the
    // wire is compressed (the receive path dequantizes what the CPU
    // quantized -- lossy by construction, exact by contract: the same
    // float scales and the same RNE quantize on both sides).
    switch (w.recvStage) {
    case RecvStage::None:
      f.refA = f.ref;
      break;
    case RecvStage::ConvertF16F32: {
      std::vector<float> g(static_cast<size_t>(f.totalElems));
      reloc::executeH2D(f.bound, f.hostSrc.data(), g.data());
      std::vector<uint16_t> h(g.size());
      reloc::quant::convertF32F16(g.data(), h.data(), f.totalElems,
                                  reloc::quant::Variant::Scalar);
      f.refA.resize(static_cast<size_t>(f.outBytes));
      float *out = reinterpret_cast<float *>(f.refA.data());
      for (int64_t i = 0; i < f.totalElems; ++i) {
        __half v;
        std::memcpy(&v, &h[static_cast<size_t>(i)], sizeof(v));
        out[i] = __half2float(v);
      }
      break;
    }
    case RecvStage::DequantS8:
    case RecvStage::UnpackDequantS4: {
      std::vector<int8_t> q(static_cast<size_t>(f.totalElems));
      reloc::quant::gatherQuantizeF32S8(f.bound, f.hostSrc.data(), q.data(),
                                        f.invScales.data(), 0, f.channels,
                                        reloc::quant::Variant::Scalar);
      f.refA.resize(static_cast<size_t>(f.outBytes));
      float *out = reinterpret_cast<float *>(f.refA.data());
      for (int64_t i = 0; i < f.totalElems; ++i) {
        int v = q[static_cast<size_t>(i)];
        if (w.recvStage == RecvStage::UnpackDequantS4)
          v = v < -8 ? -8 : (v > 7 ? 7 : v); // packS8S4 nibble saturation
        out[i] = static_cast<float>(v) *
                 f.scales[static_cast<size_t>(i / f.channelSize)];
      }
      break;
    }
    }
  }

  CUDA_CHECK(cudaMalloc(&f.dOut, static_cast<size_t>(f.outBytes)));
  if ((needB || needBFair) && w.gpuStage != GpuStage::None) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.inBytes)));
    f.dLin = static_cast<float *>(p);
    if (w.gpuStage == GpuStage::RelocateQuant) {
      CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.inBytes)));
      f.dTmp = static_cast<float *>(p);
    }
  }
  if (w.recvStage != RecvStage::None)
    CUDA_CHECK(cudaMalloc(
        &f.dRecv, static_cast<size_t>(wireBytes(w.wire, f.totalElems))));
  if (w.recvStage == RecvStage::UnpackDequantS4) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.totalElems)));
    f.dS8 = static_cast<int8_t *>(p);
  }
  if (needBFair) {
    // Pinned copy of the source, materialized once outside the timed loop.
    void *p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, static_cast<size_t>(f.inBytes),
                             cudaHostAllocDefault));
    f.pinnedSrc = static_cast<float *>(p);
    std::memcpy(f.pinnedSrc, f.hostSrc.data(), static_cast<size_t>(f.inBytes));
  }
  if (needsQuant(w)) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.channels) * 4));
    f.dInv = static_cast<float *>(p);
    CUDA_CHECK(cudaMemcpy(f.dInv, f.invScales.data(),
                          static_cast<size_t>(f.channels) * 4,
                          cudaMemcpyHostToDevice));
    if (!f.scales.empty()) {
      void *ps = nullptr;
      CUDA_CHECK(cudaMalloc(&ps, static_cast<size_t>(f.channels) * 4));
      f.dScales = static_cast<float *>(ps);
      CUDA_CHECK(cudaMemcpy(f.dScales, f.scales.data(),
                            static_cast<size_t>(f.channels) * 4,
                            cudaMemcpyHostToDevice));
    }
  }
}

struct StageTimes {
  double wall = 0, gpu = 0, cpu = 0, h2d = 0, kern = 0, recv = 0;
};

// Shared pipeline scaffolding: one non-blocking stream, 2 pinned staging
// buffers, per-chunk H2D event pairs (the end event doubles as the
// double-buffer reuse gate), pipeline start/stop + kernel events.
struct Pipeline {
  cudaStream_t stream = nullptr;
  void *staging[2] = {nullptr, nullptr};
  int64_t nChunks = 0;
  std::vector<cudaEvent_t> h2dBeg, h2dEnd;
  std::vector<cudaEvent_t> recvBeg, recvEnd; // Method A receive stages only
  std::vector<cudaEvent_t> kernBeg, kernEnd; // BPipelined per-chunk kernel leg
  cudaEvent_t evStart = nullptr, evStop = nullptr, kBeg = nullptr,
              kEnd = nullptr;

  // allocStaging=false for B_fair/BPipelined: their source is already
  // pinned, so there is no staging buffer to double-buffer through.
  // withKern=true only for BPipelined on a workload with a kernel leg.
  Pipeline(int64_t stagingBytes, int64_t chunks, bool withRecv,
           bool allocStaging = true, bool withKern = false)
      : nChunks(chunks) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (allocStaging)
      for (void *&p : staging) {
        if (cudaHostAlloc(&p, static_cast<size_t>(stagingBytes),
                          cudaHostAllocDefault) != cudaSuccess) {
          std::fprintf(stderr,
                       "error: pinned staging alloc of %lld bytes failed\n",
                       static_cast<long long>(stagingBytes));
          std::exit(1);
        }
      }
    h2dBeg.resize(static_cast<size_t>(chunks));
    h2dEnd.resize(static_cast<size_t>(chunks));
    for (int64_t c = 0; c < chunks; ++c) {
      CUDA_CHECK(cudaEventCreate(&h2dBeg[static_cast<size_t>(c)]));
      CUDA_CHECK(cudaEventCreate(&h2dEnd[static_cast<size_t>(c)]));
    }
    if (withRecv) {
      recvBeg.resize(static_cast<size_t>(chunks));
      recvEnd.resize(static_cast<size_t>(chunks));
      for (int64_t c = 0; c < chunks; ++c) {
        CUDA_CHECK(cudaEventCreate(&recvBeg[static_cast<size_t>(c)]));
        CUDA_CHECK(cudaEventCreate(&recvEnd[static_cast<size_t>(c)]));
      }
    }
    if (withKern) {
      kernBeg.resize(static_cast<size_t>(chunks));
      kernEnd.resize(static_cast<size_t>(chunks));
      for (int64_t c = 0; c < chunks; ++c) {
        CUDA_CHECK(cudaEventCreate(&kernBeg[static_cast<size_t>(c)]));
        CUDA_CHECK(cudaEventCreate(&kernEnd[static_cast<size_t>(c)]));
      }
    }
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventCreate(&kBeg));
    CUDA_CHECK(cudaEventCreate(&kEnd));
  }

  ~Pipeline() {
    for (cudaEvent_t e : h2dBeg)
      cudaEventDestroy(e);
    for (cudaEvent_t e : h2dEnd)
      cudaEventDestroy(e);
    for (cudaEvent_t e : recvBeg)
      cudaEventDestroy(e);
    for (cudaEvent_t e : recvEnd)
      cudaEventDestroy(e);
    for (cudaEvent_t e : kernBeg)
      cudaEventDestroy(e);
    for (cudaEvent_t e : kernEnd)
      cudaEventDestroy(e);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaEventDestroy(kBeg);
    cudaEventDestroy(kEnd);
    for (void *p : staging)
      cudaFreeHost(p);
    cudaStreamDestroy(stream);
  }

  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  double sumH2dMs() const {
    double total = 0;
    for (int64_t c = 0; c < nChunks; ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, h2dBeg[static_cast<size_t>(c)],
                                      h2dEnd[static_cast<size_t>(c)]));
      total += ms;
    }
    return total;
  }

  double sumRecvMs() const {
    double total = 0;
    for (size_t c = 0; c < recvBeg.size(); ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, recvBeg[c], recvEnd[c]));
      total += ms;
    }
    return total;
  }

  double sumKernMs() const {
    double total = 0;
    for (size_t c = 0; c < kernBeg.size(); ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, kernBeg[c], kernEnd[c]));
      total += ms;
    }
    return total;
  }
};

// Heap scratch for the two-pass CPU stages (gather/quant pass 1, then
// convert/pack pass 2 into pinned staging). Sized once per config in
// runConfig; never DMA'd, so pageable is fine.
struct AScratch {
  std::vector<float> f32; // GatherF16: rowsPerChunk * rowElems
  std::vector<int8_t> s8; // *S4: rowsPerChunk * rowElems
};

// Method A: per-chunk CPU transform into staging, then DMA straight into
// the final artifact. The rebase pointer trick follows Execute.h's
// gatherChunk contract ("dstBase is the address at which dst element
// offset 0 would land -- rebase it for a staging buffer").
StageTimes runMethodA(const Fixture &f, const RowChunks &ck, Pipeline &pl,
                      reloc::GatherPool &pool, reloc::quant::Variant variant,
                      AScratch &scratch) {
  const Workload &w = *f.w;
  // Per-worker floor in SOURCE bytes (the library wrappers' convention,
  // Quant.cpp): an s8-output row stages 1 byte/element but still reads 4,
  // so flooring on staged output bytes would cap parallelism 4x too early
  // for the quant workloads at small chunks.
  const int64_t rowSrcBytes = f.bound.dstStrides[0] * 4;
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowSrcBytes));
  const bool recv = w.recvStage != RecvStage::None;
  char *dmaBase =
      recv ? static_cast<char *>(f.dRecv) : static_cast<char *>(f.dOut);

  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int buf = static_cast<int>(c & 1);
    if (c >= 2)
      CUDA_CHECK(cudaEventSynchronize(pl.h2dEnd[static_cast<size_t>(c - 2)]));
    const int64_t rb = c * ck.rowsPerChunk;
    const int64_t re = std::min(f.rows, rb + ck.rowsPerChunk);
    char *stage = static_cast<char *>(pl.staging[buf]);
    const double t0 = nowMs();
    switch (w.cpuStage) {
    case CpuStage::GatherF32: {
      void *rebased = stage - rb * ck.rowBytes;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::gatherChunk(f.bound, f.hostSrc.data(), rebased, sb, se);
      });
      break;
    }
    case CpuStage::GatherQuant: {
      // int8 image of the dst layout: element offsets == byte offsets.
      int8_t *rebased =
          reinterpret_cast<int8_t *>(stage) - rb * f.bound.dstStrides[0];
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::gatherQuantizeF32S8(f.bound, f.hostSrc.data(), rebased,
                                          f.invScales.data(), sb, se, variant);
      });
      break;
    }
    case CpuStage::QuantPack:
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::quantizePackF32S8(
            f.hostSrc.data() + sb * f.channelSize,
            reinterpret_cast<int8_t *>(stage) + (sb - rb) * f.channelSize,
            se - sb, f.channelSize, f.invScales.data() + sb, variant);
      });
      break;
    case CpuStage::ConvertF16:
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::convertF32F16(f.hostSrc.data() + sb * f.channelSize,
                                    reinterpret_cast<uint16_t *>(stage) +
                                        (sb - rb) * f.channelSize,
                                    (se - sb) * f.channelSize, variant);
      });
      break;
    case CpuStage::CopyF32: {
      // rsweep T3 r=1.0: plain staging copy of contiguous source rows
      // (identity plan: dst row == src row, rowBytes = 4 * rowElems).
      const char *srcBytes = reinterpret_cast<const char *>(f.hostSrc.data());
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        std::memcpy(stage + (sb - rb) * ck.rowBytes,
                    srcBytes + sb * ck.rowBytes,
                    static_cast<size_t>((se - sb) * ck.rowBytes));
      });
      break;
    }
    case CpuStage::GatherF16: {
      // Pass 1: strided gather to f32 scratch; pass 2: contiguous f16
      // convert into staging. Both passes inside the worker keep its rows
      // cache-hot between passes.
      const int64_t rowElems = f.channelSize;
      char *rebased =
          reinterpret_cast<char *>(scratch.f32.data()) - rb * rowElems * 4;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::gatherChunk(f.bound, f.hostSrc.data(), rebased, sb, se);
        reloc::quant::convertF32F16(scratch.f32.data() + (sb - rb) * rowElems,
                                    reinterpret_cast<uint16_t *>(stage) +
                                        (sb - rb) * rowElems,
                                    (se - sb) * rowElems, variant);
      });
      break;
    }
    case CpuStage::GatherQuantS4: {
      const int64_t rowElems = f.channelSize;
      int8_t *rebased = scratch.s8.data() - rb * rowElems;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::gatherQuantizeF32S8(f.bound, f.hostSrc.data(), rebased,
                                          f.invScales.data(), sb, se, variant);
        reloc::quant::packS8S4(scratch.s8.data() + (sb - rb) * rowElems,
                               reinterpret_cast<uint8_t *>(stage) +
                                   (sb - rb) * rowElems / 2,
                               (se - sb) * rowElems / 2, variant);
      });
      break;
    }
    case CpuStage::QuantPackS4: {
      const int64_t cs = f.channelSize;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        int8_t *tmp = scratch.s8.data() + (sb - rb) * cs;
        reloc::quant::quantizePackF32S8(f.hostSrc.data() + sb * cs, tmp,
                                        se - sb, cs, f.invScales.data() + sb,
                                        variant);
        reloc::quant::packS8S4(
            tmp, reinterpret_cast<uint8_t *>(stage) + (sb - rb) * cs / 2,
            (se - sb) * cs / 2, variant);
      });
      break;
    }
    }
    t.cpu += nowMs() - t0;
    const int64_t dstOff = rb * ck.rowBytes;
    const int64_t bytes = (re - rb) * ck.rowBytes;
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(dmaBase + dstOff, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
    // Safety: staging reuse stays gated on h2dEnd[c-2] above; the receive
    // kernels below read dRecv/dS8, never staging. Chunk regions in dRecv
    // are disjoint across c, and everything here is stream-ordered.
    if (recv) {
      const int64_t eOff = rb * f.channelSize;
      const int64_t elems = (re - rb) * f.channelSize;
      CUDA_CHECK(
          cudaEventRecord(pl.recvBeg[static_cast<size_t>(c)], pl.stream));
      switch (w.recvStage) {
      case RecvStage::ConvertF16F32:
        reloc::cuda::convertF16F32(
            reinterpret_cast<const uint16_t *>(f.dRecv) + eOff,
            static_cast<float *>(f.dOut) + eOff, elems, pl.stream);
        break;
      case RecvStage::DequantS8:
        reloc::cuda::dequantS8F32(static_cast<const int8_t *>(f.dRecv) + eOff,
                                  static_cast<float *>(f.dOut) + eOff, re - rb,
                                  f.channelSize, f.dScales + rb, pl.stream);
        break;
      case RecvStage::UnpackDequantS4:
        reloc::cuda::unpackS4S8(static_cast<const uint8_t *>(f.dRecv) +
                                    eOff / 2,
                                f.dS8 + eOff, elems / 2, pl.stream);
        reloc::cuda::dequantS8F32(f.dS8 + eOff,
                                  static_cast<float *>(f.dOut) + eOff, re - rb,
                                  f.channelSize, f.dScales + rb, pl.stream);
        break;
      case RecvStage::None:
        break;
      }
      CUDA_CHECK(
          cudaEventRecord(pl.recvEnd[static_cast<size_t>(c)], pl.stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  if (recv)
    t.recv = pl.sumRecvMs();
  t.h2d = pl.sumH2dMs();
  return t;
}

// The R0.2 receive kernels both Method-B paths run over the raw fp32 tensor
// in dLin after it lands, transforming it into the final layout in dOut.
void enqueueReceiveKernels(const Fixture &f, cudaStream_t stream) {
  const Workload &w = *f.w;
  switch (w.gpuStage) {
  case GpuStage::Relocate:
    reloc::cuda::relocateF32(f.bound, f.dLin, static_cast<float *>(f.dOut),
                             stream);
    break;
  case GpuStage::RelocateQuant:
    reloc::cuda::relocateF32(f.bound, f.dLin, f.dTmp, stream);
    reloc::cuda::quantizeF32S8(f.dTmp, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, stream);
    break;
  case GpuStage::Quantize:
    reloc::cuda::quantizeF32S8(f.dLin, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, stream);
    break;
  case GpuStage::ConvertF16:
    launchConvertF32F16(f.dLin, f.dOut, f.totalElems, stream);
    break;
  case GpuStage::None:
    break; // full-f32 DMA IS the artifact (rsweep T3 family)
  }
}

// Both Method-B paths DMA into dLin for the receive kernels above -- except
// GpuStage::None rows, where the raw f32 transfer is already the artifact
// and the DMA lands directly in dOut (dLin is not allocated).
char *methodBDmaDst(const Fixture &f) {
  return f.w->gpuStage == GpuStage::None ? static_cast<char *>(f.dOut)
                                         : reinterpret_cast<char *>(f.dLin);
}

[[noreturn]] void bpipeMisaligned(const char *id, int64_t off) {
  std::fprintf(stderr,
               "error: b_pipelined chunk misaligned for %s at byte %lld -- "
               "refusing to approximate (issue #114 chunkability audit)\n",
               id, static_cast<long long>(off));
  std::exit(1);
}

// b_pipelined's per-chunk kernel (issue #114): the chunk is a contiguous
// slab [byteOff, byteOff+bytes) of the linear fp32 source that just
// landed in dLin; launch the workload's transform over exactly that
// slab. Slicing per the chunkability audit recorded on issue #114:
// identity plans use pointer/channel offsets; blocked/nchw plans slice
// whole 64-row groups / images (a rectangular sub-plan); transposePlan
// slabs are dst column bands (legal relocateF32 input, but no longer
// isTranspose2D-shaped, so the SMEM tile path falls back to the naive
// kernel -- the audit's recorded perf-class caveat). T2 (RelocateQuant
// on transposePlan) never reaches here: run() skips it as N/A.
void launchBPipeChunkKernel(const Fixture &f, int64_t byteOff, int64_t bytes,
                            cudaStream_t stream) {
  const Workload &w = *f.w;
  const int64_t elemOff = byteOff / 4;
  const int64_t elems = bytes / 4;
  switch (w.gpuStage) {
  case GpuStage::None:
    return; // DMA-only row (T3R100): no kernel leg by construction
  case GpuStage::ConvertF16:
    launchConvertF32F16(f.dLin + elemOff,
                        static_cast<uint16_t *>(f.dOut) + elemOff, elems,
                        stream);
    return;
  case GpuStage::Quantize: {
    if (elemOff % f.channelSize != 0 || elems % f.channelSize != 0)
      bpipeMisaligned(w.id, byteOff);
    const int64_t c0 = elemOff / f.channelSize;
    reloc::cuda::quantizeF32S8(f.dLin + elemOff,
                               static_cast<int8_t *>(f.dOut) + elemOff,
                               elems / f.channelSize, f.channelSize,
                               f.dInv + c0, stream);
    return;
  }
  case GpuStage::Relocate:
  case GpuStage::RelocateQuant: {
    reloc::BoundPlan sub = f.bound;
    const float *src = f.dLin + elemOff;
    float *relocDst = nullptr;
    int64_t chanBegin = 0, chanCount = 0;
    const size_t rank = f.bound.extents.size();
    if (rank == 3) {
      // blockedTransposePlan {64, m, n}: slab = whole 64-src-row groups
      // (group = 64*n elems); slice axis 1, shift dst by j0*n.
      const int64_t n = f.bound.extents[2];
      const int64_t group = 64 * n;
      if (elemOff % group != 0 || elems % group != 0)
        bpipeMisaligned(w.id, byteOff);
      const int64_t j0 = elemOff / group;
      sub.extents[1] = elems / group;
      relocDst = static_cast<float *>(f.dOut) + j0 * n;
    } else if (rank == 4) {
      // nchwToNhwcPlan {b, H, W, C}: slab = whole images (64*n elems);
      // slice axis 0 -- the dst channel axis, so the quantize leg
      // chunks with dInv + b0.
      const int64_t image = f.channelSize; // = 64*n
      if (elemOff % image != 0 || elems % image != 0)
        bpipeMisaligned(w.id, byteOff);
      chanBegin = elemOff / image;
      chanCount = elems / image;
      sub.extents[0] = chanCount;
      relocDst = (w.gpuStage == GpuStage::RelocateQuant ? f.dTmp
                                                        : static_cast<float *>(
                                                              f.dOut)) +
                 elemOff;
    } else {
      // transposePlan {n, n}, srcStrides {1, n}: slab = axis-1 band
      // (dst column band j0..j0+nj). RelocateQuant on this shape is T2,
      // which run() already skipped as N/A -- guard defensively.
      if (w.gpuStage == GpuStage::RelocateQuant) {
        std::fprintf(stderr,
                     "error: %s: RelocateQuant on a rank-2 transpose plan "
                     "is N/A for b_pipelined (issue #114 audit)\n",
                     w.id);
        std::exit(1);
      }
      const int64_t n = f.bound.extents[0];
      if (elemOff % n != 0 || elems % n != 0)
        bpipeMisaligned(w.id, byteOff);
      const int64_t j0 = elemOff / n;
      sub.extents[1] = elems / n;
      relocDst = static_cast<float *>(f.dOut) + j0;
    }
    if (w.gpuStage == GpuStage::Relocate) {
      reloc::cuda::relocateF32(sub, src, relocDst, stream);
      return;
    }
    // RelocateQuant (rank-4 / T4 family only): relocate the image slab
    // into dTmp, then per-channel quantize exactly that channel range.
    reloc::cuda::relocateF32(sub, src, relocDst, stream);
    reloc::cuda::quantizeF32S8(f.dTmp + chanBegin * f.channelSize,
                               static_cast<int8_t *>(f.dOut) +
                                   chanBegin * f.channelSize,
                               chanCount, f.channelSize, f.dInv + chanBegin,
                               stream);
    return;
  }
  }
}

// Method B (staged): per-chunk pageable->pinned memcpy + DMA of the raw fp32
// tensor, then the R0.2 transform kernels into the final layout (after the
// full transfer, matching the sym#63 baseline; the kernel cost shows up in
// gpu_kernel_ms). The staging memcpy is the host-side overhead issue #95's
// admissibility bar exists to expose; see runMethodBFair for the version
// without it.
StageTimes runMethodB(const Fixture &f, const ByteChunks &ck, Pipeline &pl,
                      reloc::GatherPool &pool) {
  const char *src = reinterpret_cast<const char *>(f.hostSrc.data());

  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int buf = static_cast<int>(c & 1);
    if (c >= 2)
      CUDA_CHECK(cudaEventSynchronize(pl.h2dEnd[static_cast<size_t>(c - 2)]));
    const int64_t off = c * ck.bytesPerChunk;
    const int64_t bytes = std::min(ck.bytesPerChunk, f.inBytes - off);
    char *stage = static_cast<char *>(pl.staging[buf]);
    const double t0 = nowMs();
    // Same thread budget as Method A's transform: split the staging copy.
    pool.parallelFor(0, bytes, 1 << 20, [&](int64_t bb, int64_t be) {
      std::memcpy(stage + bb, src + off + bb, static_cast<size_t>(be - bb));
    });
    t.cpu += nowMs() - t0;
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(methodBDmaDst(f) + off, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.kBeg, pl.stream));
  enqueueReceiveKernels(f, pl.stream);
  CUDA_CHECK(cudaEventRecord(pl.kEnd, pl.stream));
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.kBeg, pl.kEnd));
  t.kern = ms;
  return t;
}

// Method B (fair) -- issue #95's admissible baseline. The source is resident
// in pinned memory (f.pinnedSrc), so each chunk's DMA reads it directly:
// there is NO pageable->pinned staging copy on the host critical path, and
// host_stage_ms (t.cpu) is therefore 0. Everything else matches runMethodB
// (chunked cudaMemcpyAsync of the full fp32 tensor, then the same receive
// kernels). This is what a competent pure-relocation baseline looks like;
// on an r=1.0 / m<ratio workload it should land at the pinned link rate.
StageTimes runMethodBFair(const Fixture &f, const ByteChunks &ck,
                          Pipeline &pl) {
  const char *src = reinterpret_cast<const char *>(f.pinnedSrc);

  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int64_t off = c * ck.bytesPerChunk;
    const int64_t bytes = std::min(ck.bytesPerChunk, f.inBytes - off);
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(methodBDmaDst(f) + off, src + off,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.kBeg, pl.stream));
  enqueueReceiveKernels(f, pl.stream);
  CUDA_CHECK(cudaEventRecord(pl.kEnd, pl.stream));
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.kBeg, pl.kEnd));
  t.kern = ms;
  t.cpu = 0.0; // no host staging copy -- that is the whole point
  return t;
}

// Method B (pipelined) -- issue #114's overlap-fair baseline: b_fair's
// DMA path (pinned source, no staging, no reuse gate) with the per-chunk
// transform kernel issued in-stream after each chunk's copy -- Method
// A's loop shape on B's buffer model. Chunk c's kernel is ordered after
// chunk c's copy by the stream; dLin chunk regions are disjoint across
// c. Per-family slicing: see launchBPipeChunkKernel and the audit on
// issue #114.
StageTimes runMethodBPipelined(const Fixture &f, const ByteChunks &ck,
                               Pipeline &pl) {
  const char *src = reinterpret_cast<const char *>(f.pinnedSrc);
  const bool hasKern = f.w->gpuStage != GpuStage::None;
  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int64_t off = c * ck.bytesPerChunk;
    const int64_t bytes = std::min(ck.bytesPerChunk, f.inBytes - off);
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(methodBDmaDst(f) + off, src + off,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
    if (hasKern) {
      CUDA_CHECK(
          cudaEventRecord(pl.kernBeg[static_cast<size_t>(c)], pl.stream));
      launchBPipeChunkKernel(f, off, bytes, pl.stream);
      CUDA_CHECK(
          cudaEventRecord(pl.kernEnd[static_cast<size_t>(c)], pl.stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  t.kern = pl.sumKernMs(); // 0 when hasKern is false (empty vectors)
  t.cpu = 0.0;             // pinned source: no host staging copy
  return t;
}

// Run one (workload, method, chunk) config: verify gate, 5+30 protocol,
// emit a CSV row. Returns false on a verify failure.
bool runConfig(const Fixture &f, Method method, int64_t chunkReqBytes,
               const Options &opt, reloc::GatherPool &pool,
               const std::string &gpuName, std::FILE *csv) {
  const Workload &w = *f.w;
  const bool methodA = method == Method::A;
  RowChunks rck{};
  ByteChunks bck{};
  int64_t stagingBytes = 0, nChunks = 0;
  AScratch scratch;
  if (methodA) {
    rck = planRowChunks(f.rows, f.rowOutBytes, chunkReqBytes);
    stagingBytes = rck.stagingBytes;
    nChunks = rck.nChunks;
    if (w.cpuStage == CpuStage::GatherF16)
      scratch.f32.resize(static_cast<size_t>(rck.rowsPerChunk * f.channelSize));
    if (w.cpuStage == CpuStage::GatherQuantS4 ||
        w.cpuStage == CpuStage::QuantPackS4)
      scratch.s8.resize(static_cast<size_t>(rck.rowsPerChunk * f.channelSize));
  } else {
    bck = planByteChunks(f.inBytes, chunkReqBytes);
    stagingBytes = bck.bytesPerChunk;
    nChunks = bck.nChunks;
  }
  // B_fair/BPipelined's source is already pinned; they need no staging
  // buffers. withKern is set only for BPipelined on a workload with a
  // kernel leg (GpuStage::None rows, T3R100, have none).
  Pipeline pl(stagingBytes, nChunks,
              /*withRecv=*/methodA && w.recvStage != RecvStage::None,
              /*allocStaging=*/method != Method::BFair &&
                  method != Method::BPipelined,
              /*withKern=*/method == Method::BPipelined &&
                  w.gpuStage != GpuStage::None);

  auto iterate = [&]() -> StageTimes {
    switch (method) {
    case Method::A:
      return runMethodA(f, rck, pl, pool, opt.variant, scratch);
    case Method::BStaged:
      return runMethodB(f, bck, pl, pool);
    case Method::BFair:
      return runMethodBFair(f, bck, pl);
    case Method::BPipelined:
      return runMethodBPipelined(f, bck, pl);
    }
    return {};
  };

  if (opt.verify) {
    CUDA_CHECK(cudaMemset(f.dOut, 0xAB, static_cast<size_t>(f.outBytes)));
    CUDA_CHECK(cudaDeviceSynchronize());
    (void)iterate();
    std::vector<uint8_t> got(static_cast<size_t>(f.outBytes));
    CUDA_CHECK(
        cudaMemcpy(got.data(), f.dOut, got.size(), cudaMemcpyDeviceToHost));
    const std::vector<uint8_t> &want = methodA ? f.refA : f.ref;
    if (std::memcmp(got.data(), want.data(), got.size()) != 0) {
      std::fprintf(stderr,
                   "VERIFY FAILED: %s method %s chunk %lld MiB != CPU "
                   "reference\n",
                   w.id, methodTag(method),
                   static_cast<long long>(chunkReqBytes >> 20));
      return false;
    }
  }

  std::vector<double> wall, gpu, cpu, h2d, kern, recv;
  for (int i = 0; i < opt.warmup; ++i)
    (void)iterate();
  for (int i = 0; i < opt.iters; ++i) {
    StageTimes t = iterate();
    wall.push_back(t.wall);
    gpu.push_back(t.gpu);
    cpu.push_back(t.cpu);
    h2d.push_back(t.h2d);
    kern.push_back(t.kern);
    recv.push_back(t.recv);
  }

  CsvRow row;
  row.machine = opt.machine;
  row.gpu = gpuName;
  row.method = methodTag(method);
  row.transform = w.transform;
  row.dtypeOut = dtypeName(w.dtypeOut);
  row.n = opt.n;
  row.r = w.r;
  row.threads = static_cast<unsigned>(pool.threadCount());
  row.chunkReqBytes = chunkReqBytes;
  row.stagingBytes = stagingBytes;
  row.nChunks = nChunks;
  row.wall = summarizeSamples(wall);
  row.gpuPipe = summarizeSamples(gpu);
  row.cpuStage = summarizeSamples(cpu);
  row.h2d = summarizeSamples(h2d);
  row.gpuKernel = summarizeSamples(kern);
  row.gpuRecv = summarizeSamples(recv);
  row.variant = w.variant;
  row.wire = wireName(w.wire);
  row.effectiveInputGbps =
      row.wall.median > 0
          ? static_cast<double>(f.inBytes) / (row.wall.median * 1e-3) / 1e9
          : 0.0;
  row.verified = opt.verify;
  std::fprintf(csv, "%s\n", csvRowLine(row).c_str());
  std::fflush(csv);
  std::fprintf(stderr,
               "rtrack: %-4s %-6s chunk=%4lldMiB T=%d wall %8.2f ms "
               "(%5.2f GB/s in, iqr %4.1f%%%s) cpu %7.2f h2d %7.2f kern "
               "%6.2f recv %6.2f%s\n",
               w.id, methodTag(method),
               static_cast<long long>(chunkReqBytes >> 20), pool.threadCount(),
               row.wall.median, row.effectiveInputGbps,
               row.wall.iqrOverMedianPct, row.wall.unstable ? " UNSTABLE" : "",
               row.cpuStage.median, row.h2d.median, row.gpuKernel.median,
               row.gpuRecv.median, opt.verify ? " [verified]" : "");
  return true;
}

int run(const Options &opt) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const std::string gpuName = prop.name;

  std::FILE *csv = stdout;
  if (std::strcmp(opt.csvPath, "-") != 0) {
    csv = std::fopen(opt.csvPath, "a");
    if (!csv) {
      std::fprintf(stderr, "error: cannot open %s\n", opt.csvPath);
      return 1;
    }
  }
  if (opt.csvHeader) {
    // Appending a header into the middle of an existing file would poison
    // every downstream parser; only emit into an empty target.
    if (csv == stdout || std::ftell(csv) == 0)
      std::fprintf(csv, "%s\n", csvHeaderLine().c_str());
    else
      std::fprintf(stderr,
                   "rtrack: --csv-header suppressed (%s is non-empty)\n",
                   opt.csvPath);
  }

  reloc::GatherPool pool(opt.threads);
  const bool needB = hasMethod(opt.methods, Method::BStaged);
  const bool needBFair = hasMethod(opt.methods, Method::BFair);
  const bool needBPipe = hasMethod(opt.methods, Method::BPipelined);
  int rc = 0;
  for (const Workload *w : opt.workloads) {
    Fixture f;
    reloc::BoundPlan wireBound;
    const reloc::BoundPlan *boundOverride = nullptr;
    if (opt.planWire) {
      wireBound = decodeAndBindWire(opt.planWire, opt.n);
      boundOverride = &wireBound;
    }
    // BPipelined rides the BFair predicate: same pinned source, same
    // dLin/dTmp.
    buildFixture(f, *w, opt.n, needB && w->methodB,
                 (needBFair || needBPipe) && w->methodB, opt.verify,
                 boundOverride);
    // Chunk requests past the artifact size all clamp to the same 1-chunk
    // plan; measuring the identical config again would only hand
    // figure1's best-chunk argmin duplicate samples. Dedup is per method:
    // the two Method-B paths share a chunk plan but are distinct configs.
    std::map<Method, std::set<std::pair<int64_t, int64_t>>> seen;
    for (int64_t chunk : opt.chunkBytes) {
      for (Method m : opt.methods) {
        // A-only rows (methodB=false): B is r-independent, so each rsweep
        // family measures both B paths once, on its R100 row.
        if (m != Method::A && !w->methodB)
          continue;
        // T2's quantize leg cannot run per chunk (column bands are not
        // channel-contiguous): N/A per the chunkability audit recorded
        // on issue #114 -- a loud skip, never a silent omission.
        if (m == Method::BPipelined && std::strcmp(w->id, "T2") == 0) {
          std::fprintf(stderr,
                       "rtrack: T2   b_pipelined N/A (chunkability audit, "
                       "issue #114); skipped\n");
          continue;
        }
        std::pair<int64_t, int64_t> key;
        if (m == Method::A) {
          RowChunks rck = planRowChunks(f.rows, f.rowOutBytes, chunk);
          key = {rck.stagingBytes, rck.nChunks};
        } else {
          ByteChunks bck = planByteChunks(f.inBytes, chunk);
          key = {bck.bytesPerChunk, bck.nChunks};
        }
        if (!seen[m].insert(key).second) {
          std::fprintf(stderr,
                       "rtrack: %-4s %-6s chunk=%4lldMiB duplicates an earlier "
                       "sweep point; skipped\n",
                       w->id, methodTag(m),
                       static_cast<long long>(chunk >> 20));
          continue;
        }
        if (!runConfig(f, m, chunk, opt, pool, gpuName, csv)) {
          rc = 1;
          break;
        }
      }
      if (rc)
        break;
    }
    if (rc)
      break;
  }
  pool.close();
  if (csv != stdout)
    std::fclose(csv);
  return rc;
}

bool parseVariant(const std::string &s, reloc::quant::Variant &out) {
  for (auto v : {reloc::quant::Variant::Auto, reloc::quant::Variant::Scalar,
                 reloc::quant::Variant::AVX2, reloc::quant::Variant::AVX512,
                 reloc::quant::Variant::AVX512Pf})
    if (s == variantName(v)) {
      out = v;
      return true;
    }
  return false;
}

std::vector<std::string> splitCommas(const std::string &s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t next = s.find(',', pos);
    if (next == std::string::npos)
      next = s.size();
    if (next > pos)
      out.push_back(s.substr(pos, next - pos));
    pos = next + 1;
  }
  return out;
}

int usage() {
  std::fprintf(
      stderr,
      "usage: bench-rtrack [--transform all|matrix|rsweep|T1,T1b,...,T3R0125]\n"
      "  [--method both|all|a|b|bfair|bpipe] [--n N] [--chunk-mib "
      "4,16,64,256]\n"
      "  [--threads T] [--variant auto|scalar|avx2|avx512|avx512pf]\n"
      "  [--warmup W] [--iters I] [--machine NAME] [--csv PATH|-]\n"
      "  [--csv-header] [--no-verify] [--plan-wire PATH]\n"
      "  N must be divisible by 64. Rows append to the CSV target.\n"
      "  --plan-wire PATH decodes+binds a corpus wire-format plan (--n's N)\n"
      "  in place of a hand-authored plans.h builder; valid only together\n"
      "  with --transform TW (and no other transform in the same run).\n");
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  std::string transformArg = "all";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--transform")
      transformArg = next();
    else if (a == "--method") {
      std::string m = next();
      if (m == "a")
        opt.methods = {Method::A};
      else if (m == "b")
        opt.methods = {Method::BStaged};
      else if (m == "bfair")
        opt.methods = {Method::BFair};
      else if (m == "bpipe")
        opt.methods = {Method::BPipelined};
      else if (m == "both")
        opt.methods = {Method::A, Method::BStaged};
      else if (m == "all")
        opt.methods = {Method::A, Method::BStaged, Method::BFair,
                       Method::BPipelined};
      else
        return usage();
    } else if (a == "--n")
      opt.n = std::atoll(next());
    else if (a == "--chunk-mib") {
      opt.chunkBytes.clear();
      for (const std::string &c : splitCommas(next())) {
        int64_t mib = std::atoll(c.c_str());
        if (mib < 1)
          return usage();
        opt.chunkBytes.push_back(mib << 20);
      }
      if (opt.chunkBytes.empty())
        return usage();
    } else if (a == "--threads")
      opt.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--variant") {
      if (!parseVariant(next(), opt.variant))
        return usage();
    } else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--machine")
      opt.machine = next();
    else if (a == "--csv")
      opt.csvPath = next();
    else if (a == "--csv-header")
      opt.csvHeader = true;
    else if (a == "--no-verify")
      opt.verify = false;
    else if (a == "--plan-wire")
      opt.planWire = next();
    else
      return usage();
  }
  if (opt.n <= 0 || opt.n % 64 != 0) {
    std::fprintf(stderr, "error: N must be positive and divisible by 64\n");
    return 2;
  }
  if (opt.warmup < 0 || opt.iters < 1)
    return usage();
  if (opt.machine.empty())
    opt.machine = defaultMachine();

  if (transformArg == "all" || transformArg == "matrix" ||
      transformArg == "rsweep") {
    const bool wantRsweep = transformArg == "rsweep";
    for (const Workload &w : allWorkloads())
      if ((std::strcmp(w.variant, "rsweep") == 0) == wantRsweep)
        opt.workloads.push_back(&w);
  } else {
    for (const std::string &id : splitCommas(transformArg)) {
      // TW is the synthetic --plan-wire workload: not in allWorkloads()
      // (no static plans.h builder), so it is special-cased here rather
      // than looked up by findWorkload.
      if (id == "TW") {
        opt.workloads.push_back(&wireWorkload());
        continue;
      }
      const Workload *w = findWorkload(id);
      if (!w) {
        std::fprintf(stderr, "error: unknown transform %s\n", id.c_str());
        return 2;
      }
      opt.workloads.push_back(w);
    }
  }
  if (opt.workloads.empty())
    return usage();

  // --plan-wire <-> --transform TW is a bijective requirement: TW has no
  // makePlan (buildFixture would dereference a null function pointer), and
  // --plan-wire silently interacting with any other transform in the same
  // sweep is exactly what issue #97's brief calls out to avoid.
  const bool hasTW =
      std::any_of(opt.workloads.begin(), opt.workloads.end(),
                  [](const Workload *w) { return std::strcmp(w->id, "TW") == 0; });
  if (opt.planWire && !(hasTW && opt.workloads.size() == 1)) {
    std::fprintf(stderr,
                 "error: --plan-wire is only valid with --transform TW "
                 "(and no other transform)\n");
    return usage();
  }
  if (hasTW && !opt.planWire) {
    std::fprintf(stderr, "error: --transform TW requires --plan-wire PATH\n");
    return usage();
  }

  // Validate the requested SIMD variant against each workload's CPU kernel
  // (the quant_bw convention: fail fast, before any setup).
  if (hasMethod(opt.methods, Method::A)) {
    for (const Workload *w : opt.workloads) {
      std::vector<reloc::quant::Kernel> ks;
      switch (w->cpuStage) {
      case CpuStage::GatherQuant:
        ks = {reloc::quant::Kernel::GatherQuantize};
        break;
      case CpuStage::QuantPack:
        ks = {reloc::quant::Kernel::QuantizePack};
        break;
      case CpuStage::ConvertF16:
      case CpuStage::GatherF16:
        ks = {reloc::quant::Kernel::ConvertF32F16};
        break;
      case CpuStage::GatherQuantS4:
        ks = {reloc::quant::Kernel::GatherQuantize,
              reloc::quant::Kernel::PackS8S4};
        break;
      case CpuStage::QuantPackS4:
        ks = {reloc::quant::Kernel::QuantizePack,
              reloc::quant::Kernel::PackS8S4};
        break;
      case CpuStage::GatherF32:
      case CpuStage::CopyF32:
        continue; // no SIMD variants
      }
      for (reloc::quant::Kernel k : ks)
        if (!reloc::quant::kernelHasVariant(k, opt.variant) ||
            !reloc::quant::cpuSupports(opt.variant)) {
          std::fprintf(stderr,
                       "error: variant %s not available for %s on this host\n",
                       variantName(opt.variant), w->id);
          return 3;
        }
    }
  }
  return run(opt);
}
