//===- rtrack_bench.cu - R0.3 pipeline & measurement harness (#76) --------===//
//
// Method A vs Method B end-to-end (transform + transfer) latency for one
// tensor. Source: pageable host DRAM. Staging: 2 pinned buffers of chunk
// size (double-buffered, event-gated). Destination: final layout in GPU
// global memory.
//   A: per-chunk CPU transform (R0.1 kernels / gatherChunk, parallelized
//      over a GatherPool) into pinned staging -> cudaMemcpyAsync of r*S
//      bytes total. (R2's dequant/unpack receive kernels slot in later.)
//   B: per-chunk pageable->pinned memcpy (parallelized over the same pool,
//      so both methods get the same thread budget) -> cudaMemcpyAsync of S
//      bytes -> R0.2 GPU transform kernels into the final layout.
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
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err_), #x);                   \
      std::exit(1);                                                           \
    }                                                                         \
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

struct Options {
  std::vector<const Workload *> workloads;
  bool runA = true, runB = true;
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
};

std::string defaultMachine() {
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0)
    return "unknown";
  return host;
}

bool needsQuant(const Workload &w) {
  return w.cpuStage == CpuStage::GatherQuant ||
         w.cpuStage == CpuStage::QuantPack;
}

// Per-(workload, N) fixture: pageable source, per-channel scales, CPU
// reference of the final artifact, device buffers.
struct Fixture {
  const Workload *w = nullptr;
  reloc::BoundPlan bound;
  std::vector<float> hostSrc;   // pageable; N^2 elements
  std::vector<float> invScales; // extents[0] entries (quant workloads)
  std::vector<uint8_t> ref;     // expected final artifact, outBytes
  int64_t totalElems = 0, inBytes = 0, outBytes = 0;
  int64_t rows = 0, rowOutBytes = 0, channels = 0, channelSize = 0;
  // Device buffers:
  void *dOut = nullptr;  // final artifact (A's DMA target, B's kernel dst)
  float *dLin = nullptr; // B: linear fp32 source copy
  float *dTmp = nullptr; // B RelocateQuant: relocated fp32 before quantize
  float *dInv = nullptr; // per-channel invScales

  ~Fixture() {
    cudaFree(dOut);
    cudaFree(dLin);
    cudaFree(dTmp);
    cudaFree(dInv);
  }
};

void buildFixture(Fixture &f, const Workload &w, int64_t n, bool methodB,
                  bool verify) {
  f.w = &w;
  f.bound = w.makePlan(n);
  f.totalElems = n * n;
  f.inBytes = f.totalElems * 4;
  f.outBytes = f.totalElems * dtypeBytes(w.dtypeOut);
  f.rows = f.bound.extents[0];
  f.rowOutBytes = f.bound.dstStrides[0] * dtypeBytes(w.dtypeOut);
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
       w.cpuStage == CpuStage::ConvertF16) &&
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
      f.invScales[static_cast<size_t>(c)] = maxAbs > 0 ? 127.0f / maxAbs : 1.0f;
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
  }

  CUDA_CHECK(cudaMalloc(&f.dOut, static_cast<size_t>(f.outBytes)));
  if (methodB) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.inBytes)));
    f.dLin = static_cast<float *>(p);
    if (w.gpuStage == GpuStage::RelocateQuant) {
      CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.inBytes)));
      f.dTmp = static_cast<float *>(p);
    }
  }
  if (needsQuant(w)) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(f.channels) * 4));
    f.dInv = static_cast<float *>(p);
    CUDA_CHECK(cudaMemcpy(f.dInv, f.invScales.data(),
                          static_cast<size_t>(f.channels) * 4,
                          cudaMemcpyHostToDevice));
  }
}

struct StageTimes {
  double wall = 0, gpu = 0, cpu = 0, h2d = 0, kern = 0;
};

// Shared pipeline scaffolding: one non-blocking stream, 2 pinned staging
// buffers, per-chunk H2D event pairs (the end event doubles as the
// double-buffer reuse gate), pipeline start/stop + kernel events.
struct Pipeline {
  cudaStream_t stream = nullptr;
  void *staging[2] = {nullptr, nullptr};
  int64_t nChunks = 0;
  std::vector<cudaEvent_t> h2dBeg, h2dEnd;
  cudaEvent_t evStart = nullptr, evStop = nullptr, kBeg = nullptr,
              kEnd = nullptr;

  Pipeline(int64_t stagingBytes, int64_t chunks) : nChunks(chunks) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
};

// Method A: per-chunk CPU transform into staging, then DMA straight into
// the final artifact. The rebase pointer trick follows Execute.h's
// gatherChunk contract ("dstBase is the address at which dst element
// offset 0 would land -- rebase it for a staging buffer").
StageTimes runMethodA(const Fixture &f, const RowChunks &ck, Pipeline &pl,
                      reloc::GatherPool &pool, reloc::quant::Variant variant) {
  const Workload &w = *f.w;
  // Per-worker floor in SOURCE bytes (the library wrappers' convention,
  // Quant.cpp): an s8-output row stages 1 byte/element but still reads 4,
  // so flooring on staged output bytes would cap parallelism 4x too early
  // for the quant workloads at small chunks.
  const int64_t rowSrcBytes = f.bound.dstStrides[0] * 4;
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowSrcBytes));

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
        reloc::quant::convertF32F16(
            f.hostSrc.data() + sb * f.channelSize,
            reinterpret_cast<uint16_t *>(stage) + (sb - rb) * f.channelSize,
            (se - sb) * f.channelSize, variant);
      });
      break;
    }
    t.cpu += nowMs() - t0;
    const int64_t dstOff = rb * ck.rowBytes;
    const int64_t bytes = (re - rb) * ck.rowBytes;
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char *>(f.dOut) + dstOff, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  return t;
}

// Method B: per-chunk pageable->pinned memcpy + DMA of the raw fp32
// tensor, then the R0.2 transform kernels into the final layout (after the
// full transfer, matching the sym#63 baseline; the kernel cost shows up in
// gpu_kernel_ms).
StageTimes runMethodB(const Fixture &f, const ByteChunks &ck, Pipeline &pl,
                      reloc::GatherPool &pool) {
  const Workload &w = *f.w;
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
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(f.dLin) + off, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.kBeg, pl.stream));
  switch (w.gpuStage) {
  case GpuStage::Relocate:
    reloc::cuda::relocateF32(f.bound, f.dLin, static_cast<float *>(f.dOut),
                             pl.stream);
    break;
  case GpuStage::RelocateQuant:
    reloc::cuda::relocateF32(f.bound, f.dLin, f.dTmp, pl.stream);
    reloc::cuda::quantizeF32S8(f.dTmp, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, pl.stream);
    break;
  case GpuStage::Quantize:
    reloc::cuda::quantizeF32S8(f.dLin, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, pl.stream);
    break;
  case GpuStage::ConvertF16:
    launchConvertF32F16(f.dLin, f.dOut, f.totalElems, pl.stream);
    break;
  }
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

// Run one (workload, method, chunk) config: verify gate, 5+30 protocol,
// emit a CSV row. Returns false on a verify failure.
bool runConfig(const Fixture &f, bool methodA, int64_t chunkReqBytes,
               const Options &opt, reloc::GatherPool &pool,
               const std::string &gpuName, std::FILE *csv) {
  const Workload &w = *f.w;
  RowChunks rck{};
  ByteChunks bck{};
  int64_t stagingBytes = 0, nChunks = 0;
  if (methodA) {
    rck = planRowChunks(f.rows, f.rowOutBytes, chunkReqBytes);
    stagingBytes = rck.stagingBytes;
    nChunks = rck.nChunks;
  } else {
    bck = planByteChunks(f.inBytes, chunkReqBytes);
    stagingBytes = bck.bytesPerChunk;
    nChunks = bck.nChunks;
  }
  Pipeline pl(stagingBytes, nChunks);

  auto iterate = [&]() -> StageTimes {
    return methodA ? runMethodA(f, rck, pl, pool, opt.variant)
                   : runMethodB(f, bck, pl, pool);
  };

  if (opt.verify) {
    CUDA_CHECK(cudaMemset(f.dOut, 0xAB, static_cast<size_t>(f.outBytes)));
    CUDA_CHECK(cudaDeviceSynchronize());
    (void)iterate();
    std::vector<uint8_t> got(static_cast<size_t>(f.outBytes));
    CUDA_CHECK(cudaMemcpy(got.data(), f.dOut, got.size(),
                          cudaMemcpyDeviceToHost));
    if (std::memcmp(got.data(), f.ref.data(), got.size()) != 0) {
      std::fprintf(stderr,
                   "VERIFY FAILED: %s method %c chunk %lld MiB != CPU "
                   "reference\n",
                   w.id, methodA ? 'a' : 'b',
                   static_cast<long long>(chunkReqBytes >> 20));
      return false;
    }
  }

  std::vector<double> wall, gpu, cpu, h2d, kern;
  for (int i = 0; i < opt.warmup; ++i)
    (void)iterate();
  for (int i = 0; i < opt.iters; ++i) {
    StageTimes t = iterate();
    wall.push_back(t.wall);
    gpu.push_back(t.gpu);
    cpu.push_back(t.cpu);
    h2d.push_back(t.h2d);
    kern.push_back(t.kern);
  }

  CsvRow row;
  row.machine = opt.machine;
  row.gpu = gpuName;
  row.method = methodA ? "a" : "b";
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
  row.effectiveInputGbps =
      row.wall.median > 0
          ? static_cast<double>(f.inBytes) / (row.wall.median * 1e-3) / 1e9
          : 0.0;
  row.verified = opt.verify;
  std::fprintf(csv, "%s\n", csvRowLine(row).c_str());
  std::fflush(csv);
  std::fprintf(stderr,
               "rtrack: %-4s %c chunk=%4lldMiB T=%d wall %8.2f ms "
               "(%5.2f GB/s in, iqr %4.1f%%%s) cpu %7.2f h2d %7.2f kern "
               "%6.2f%s\n",
               w.id, methodA ? 'a' : 'b',
               static_cast<long long>(chunkReqBytes >> 20), pool.threadCount(),
               row.wall.median, row.effectiveInputGbps,
               row.wall.iqrOverMedianPct, row.wall.unstable ? " UNSTABLE" : "",
               row.cpuStage.median, row.h2d.median, row.gpuKernel.median,
               opt.verify ? " [verified]" : "");
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
  int rc = 0;
  for (const Workload *w : opt.workloads) {
    Fixture f;
    buildFixture(f, *w, opt.n, opt.runB, opt.verify);
    // Chunk requests past the artifact size all clamp to the same 1-chunk
    // plan; measuring the identical config again would only hand
    // figure1's best-chunk argmin duplicate samples.
    std::set<std::pair<int64_t, int64_t>> seenA, seenB;
    for (int64_t chunk : opt.chunkBytes) {
      if (opt.runA) {
        RowChunks rck = planRowChunks(f.rows, f.rowOutBytes, chunk);
        if (!seenA.insert({rck.stagingBytes, rck.nChunks}).second)
          std::fprintf(stderr,
                       "rtrack: %-4s a chunk=%4lldMiB duplicates an earlier "
                       "sweep point; skipped\n",
                       w->id, static_cast<long long>(chunk >> 20));
        else if (!runConfig(f, /*methodA=*/true, chunk, opt, pool, gpuName,
                            csv))
          rc = 1;
      }
      if (rc)
        break;
      if (opt.runB) {
        ByteChunks bck = planByteChunks(f.inBytes, chunk);
        if (!seenB.insert({bck.bytesPerChunk, bck.nChunks}).second)
          std::fprintf(stderr,
                       "rtrack: %-4s b chunk=%4lldMiB duplicates an earlier "
                       "sweep point; skipped\n",
                       w->id, static_cast<long long>(chunk >> 20));
        else if (!runConfig(f, /*methodA=*/false, chunk, opt, pool, gpuName,
                            csv))
          rc = 1;
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
  for (auto v :
       {reloc::quant::Variant::Auto, reloc::quant::Variant::Scalar,
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
      "usage: bench-rtrack [--transform all|T1,T1b,T2,T3,T4,T5]\n"
      "  [--method both|a|b] [--n N] [--chunk-mib 4,16,64,256]\n"
      "  [--threads T] [--variant auto|scalar|avx2|avx512|avx512pf]\n"
      "  [--warmup W] [--iters I] [--machine NAME] [--csv PATH|-]\n"
      "  [--csv-header] [--no-verify]\n"
      "  N must be divisible by 64. Rows append to the CSV target.\n");
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
        opt.runB = false;
      else if (m == "b")
        opt.runA = false;
      else if (m != "both")
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

  if (transformArg == "all") {
    for (const Workload &w : allWorkloads())
      opt.workloads.push_back(&w);
  } else {
    for (const std::string &id : splitCommas(transformArg)) {
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

  // Validate the requested SIMD variant against each workload's CPU kernel
  // (the quant_bw convention: fail fast, before any setup).
  if (opt.runA) {
    for (const Workload *w : opt.workloads) {
      reloc::quant::Kernel k;
      switch (w->cpuStage) {
      case CpuStage::GatherQuant:
        k = reloc::quant::Kernel::GatherQuantize;
        break;
      case CpuStage::QuantPack:
        k = reloc::quant::Kernel::QuantizePack;
        break;
      case CpuStage::ConvertF16:
        k = reloc::quant::Kernel::ConvertF32F16;
        break;
      case CpuStage::GatherF32:
        continue; // gatherChunk has no SIMD variants
      }
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
