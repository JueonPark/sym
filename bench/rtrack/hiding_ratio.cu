//===- hiding_ratio.cu - R4/EXP-4 Turing hiding-ratio kernels (#85) -------===//
//
// Measures the GPU-side transform kernels in isolation on Turing to fill
// the literature gap (no published 2080 Ti transpose GB/s) and to feed the
// hiding-ratio model validation: a Method-B transform hides under the PCIe
// transfer iff its HBM traffic multiplier m < ratio = HBM_BW / PCIe_BW.
// This tool emits raw per-kernel times + effective bandwidths; the model
// arithmetic (ratio, m, hide verdict, pipeline-overlap cross-check) lives
// in bench/rtrack/hiding_model.py, the gates.py split.
//
// Kernels: copy_f32 (the HBM ceiling), relocate_naive_f32, an UNPADDED
// SMEM transpose (bench-local, to expose the bank-conflict delta the
// library's padded relocateF32 avoids), the padded library transpose,
// scatter_random_f32 over an index-entropy sweep (within-block random
// permutations of decreasing locality), and three Method-A receive
// kernels (CM1, issue #109): convert_f16_f32, dequant_s8_f32, and
// unpack_dequant_s4 (the r=0.125 unpack+dequant chain). Every kernel is
// verified against a CPU/oracle reference before it is timed (the repo
// rule).
//
//===----------------------------------------------------------------------===//

#include "rtrack/plans.h"
#include "rtrack/rstats.h"

#include "reloc/CudaKernels.h"
#include "reloc/Execute.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
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

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

// UNPADDED SMEM transpose: identical to the library's tiled kernel but with
// tile[32][32] instead of [32][33]. The missing pad column makes every SMEM
// column access hit one bank -> 32-way conflicts; the measured gap vs the
// padded library kernel is the bank-conflict cost the paper quantifies.
__global__ void transposeUnpaddedKernel(const float *in, float *out,
                                        int64_t inRows, int64_t inCols) {
  __shared__ float tile[kTile][kTile];
  int64_t x = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.x;
  int64_t y = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.y;
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inCols && y + j < inRows)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * inCols + x];
  __syncthreads();
  x = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.x;
  y = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.y;
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inRows && y + j < inCols)
      out[(y + j) * inRows + x] = tile[threadIdx.x][threadIdx.y + j];
}

void launchTransposeUnpadded(const float *dIn, float *dOut, int64_t n,
                             cudaStream_t stream) {
  // transposePlan(n): dst[r][c] = src[c*n + r]; view src as (n x n) `in`.
  dim3 block(kTile, kBlockRows);
  dim3 grid(static_cast<unsigned>((n + kTile - 1) / kTile),
            static_cast<unsigned>((n + kTile - 1) / kTile));
  transposeUnpaddedKernel<<<grid, block, 0, stream>>>(dIn, dOut, n, n);
  CUDA_CHECK(cudaGetLastError());
}

// --- host references ------------------------------------------------------

std::vector<float> makeSrc(int64_t total) {
  std::vector<float> v(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i)
    v[static_cast<size_t>(i)] =
        (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  return v;
}

// Within-block random permutation of [0, total): partition into contiguous
// blocks of `blk` and shuffle each block internally (blocks stay in place).
// blk == 1 -> identity (coalesced); blk == total -> full random. The write
// displacement is bounded by blk, so blk is the locality/entropy knob.
std::vector<int64_t> makeIndexPerm(int64_t total, int64_t blk, uint64_t seed) {
  std::vector<int64_t> idx(static_cast<size_t>(total));
  std::iota(idx.begin(), idx.end(), int64_t{0});
  if (blk <= 1)
    return idx;
  std::mt19937_64 rng(seed);
  for (int64_t base = 0; base < total; base += blk) {
    int64_t hi = std::min(total, base + blk);
    for (int64_t i = hi - 1; i > base; --i) {
      std::uniform_int_distribution<int64_t> d(base, i);
      std::swap(idx[static_cast<size_t>(i)], idx[static_cast<size_t>(d(rng))]);
    }
  }
  return idx;
}

struct DeviceBuf {
  void *p = nullptr;
  explicit DeviceBuf(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuf() { cudaFree(p); }
  DeviceBuf(const DeviceBuf &) = delete;
  DeviceBuf &operator=(const DeviceBuf &) = delete;
  template <typename T> T *as() const { return static_cast<T *>(p); }
};

struct Timing {
  RStats ms;
  double gbps = 0; // effective HBM: (readBytes + writeBytes) / time
};

// Time `launch` with CUDA events: warmup, then iters timed; median via the
// R0.3 protocol. trafficBytes = HBM bytes moved (read + write) per call.
template <typename Launch>
Timing timeKernel(Launch &&launch, int64_t trafficBytes, int warmup,
                  int iters, cudaStream_t stream) {
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(beg, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    samples.push_back(ms);
  }
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  Timing t;
  t.ms = summarizeSamples(samples);
  t.gbps = t.ms.median > 0
               ? static_cast<double>(trafficBytes) / (t.ms.median * 1e-3) / 1e9
               : 0.0;
  return t;
}

std::string timingJson(const std::string &name, int64_t traffic,
                       const Timing &t, const std::string &extra = "") {
  return "    \"" + name + "\": {\"traffic_bytes\": " +
         std::to_string(traffic) +
         ", \"median_ms\": " + bench::jsonNumber(t.ms.median) +
         ", \"min_ms\": " + bench::jsonNumber(t.ms.min) +
         ", \"p95_ms\": " + bench::jsonNumber(t.ms.p95) +
         ", \"iqr_over_median_pct\": " +
         bench::jsonNumber(t.ms.iqrOverMedianPct) +
         ", \"gb_per_s\": " + bench::jsonNumber(t.gbps) + extra + "}";
}

struct Options {
  std::vector<int64_t> ns = {8192, 16384};
  std::vector<int64_t> entropyBlk; // resolved per-N if empty
  int warmup = kWarmup;
  int iters = kIters;
  const char *jsonPath = "-";
};

// Exact host IEEE binary16 -> binary32 (finite inputs only; the bench
// forces finite bit patterns below, so the inf/NaN branch is untaken).
float f16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign; // signed zero
    } else { // subnormal: renormalize into f32
      int shift = 0;
      while (!(man & 0x400u)) {
        man <<= 1;
        ++shift;
      }
      man &= 0x3ffu;
      bits = sign | ((113u - static_cast<uint32_t>(shift)) << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (man << 13); // inf/NaN (untaken)
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

int runN(int64_t n, const Options &opt, cudaStream_t stream,
         std::string &body) {
  const int64_t total = n * n;
  const int64_t S = total * 4;    // tensor bytes
  const int64_t traffic = 2 * S;  // read + write, the copy/transpose basis
  std::vector<float> hSrc = makeSrc(total);

  DeviceBuf dSrc(static_cast<size_t>(S)), dDst(static_cast<size_t>(S));
  CUDA_CHECK(cudaMemcpyAsync(dSrc.p, hSrc.data(), static_cast<size_t>(S),
                             cudaMemcpyHostToDevice, stream));

  // All setup runs on `stream` (the compute stream) via the Async variants:
  // a NULL-stream memset/memcpy does NOT order against a non-blocking
  // stream, so mixing them races the setup with the kernel. download()
  // syncs the stream first, so its plain memcpy reads a settled buffer.
  auto download = [&](const DeviceBuf &d) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> h(static_cast<size_t>(total));
    CUDA_CHECK(cudaMemcpy(h.data(), d.p, static_cast<size_t>(S),
                          cudaMemcpyDeviceToHost));
    return h;
  };
  auto clearDst = [&] {
    CUDA_CHECK(cudaMemsetAsync(dDst.p, 0xCD, static_cast<size_t>(S), stream));
  };

  std::string kernels;
  auto emit = [&](const std::string &j) {
    if (!kernels.empty())
      kernels += ",\n";
    kernels += j;
  };

  // --- copy_f32: the HBM ceiling -----------------------------------------
  clearDst();
  reloc::cuda::copyF32(dSrc.as<float>(), dDst.as<float>(), total, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(download(dDst).data(), hSrc.data(),
                  static_cast<size_t>(S)) != 0) {
    std::fprintf(stderr, "VERIFY FAILED: copy_f32 N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }
  emit(timingJson(
      "copy_f32", traffic,
      timeKernel([&] { reloc::cuda::copyF32(dSrc.as<float>(),
                                            dDst.as<float>(), total, stream); },
                 traffic, opt.warmup, opt.iters, stream)));

  // --- transpose references (naive, unpadded SMEM, padded SMEM) ----------
  reloc::BoundPlan tp = transposePlan(n);
  std::vector<float> oracle(static_cast<size_t>(total));
  reloc::executeH2D(tp, hSrc.data(), oracle.data());

  struct TCase {
    const char *name;
    void (*launch)(const reloc::BoundPlan &, const float *, float *, void *);
  };
  // naive + padded go through the library; unpadded is bench-local (handled
  // separately below because its signature differs).
  clearDst();
  reloc::cuda::relocateNaiveF32(tp, dSrc.as<float>(), dDst.as<float>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(download(dDst).data(), oracle.data(),
                  static_cast<size_t>(S)) != 0) {
    std::fprintf(stderr, "VERIFY FAILED: relocate_naive N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }
  emit(timingJson("relocate_naive_f32", traffic,
                  timeKernel(
                      [&] {
                        reloc::cuda::relocateNaiveF32(tp, dSrc.as<float>(),
                                                      dDst.as<float>(), stream);
                      },
                      traffic, opt.warmup, opt.iters, stream)));

  clearDst();
  launchTransposeUnpadded(dSrc.as<float>(), dDst.as<float>(), n, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(download(dDst).data(), oracle.data(),
                  static_cast<size_t>(S)) != 0) {
    std::fprintf(stderr, "VERIFY FAILED: transpose_smem_unpadded N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }
  emit(timingJson("transpose_smem_unpadded", traffic,
                  timeKernel(
                      [&] {
                        launchTransposeUnpadded(dSrc.as<float>(),
                                                dDst.as<float>(), n, stream);
                      },
                      traffic, opt.warmup, opt.iters, stream)));

  clearDst();
  reloc::cuda::relocateF32(tp, dSrc.as<float>(), dDst.as<float>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(download(dDst).data(), oracle.data(),
                  static_cast<size_t>(S)) != 0) {
    std::fprintf(stderr, "VERIFY FAILED: transpose_smem_padded N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }
  emit(timingJson("transpose_smem_padded", traffic,
                  timeKernel(
                      [&] {
                        reloc::cuda::relocateF32(tp, dSrc.as<float>(),
                                                 dDst.as<float>(), stream);
                      },
                      traffic, opt.warmup, opt.iters, stream)));

  // --- Method-A receive kernels (issue #109/CM1) --------------------------
  // R4-style: isolated kernel BW on a read+write traffic basis, later
  // divided into the same run's copy_f32 ceiling by make_calibration.py.
  {
    // convert_f16_f32: read 2B + write 4B per element = 1.5*S traffic.
    const int64_t halfBytes = total * 2;
    DeviceBuf dHalf(static_cast<size_t>(halfBytes));
    std::vector<uint16_t> hHalf(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      uint16_t h = static_cast<uint16_t>((i * 2654435761ull) & 0xffff);
      if ((h & 0x7c00) == 0x7c00)
        h = static_cast<uint16_t>(h ^ 0x0400); // force finite
      hHalf[static_cast<size_t>(i)] = h;
    }
    CUDA_CHECK(cudaMemcpyAsync(dHalf.p, hHalf.data(),
                               static_cast<size_t>(halfBytes),
                               cudaMemcpyHostToDevice, stream));
    std::vector<float> ref(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i)
      ref[static_cast<size_t>(i)] = f16ToF32(hHalf[static_cast<size_t>(i)]);
    clearDst();
    reloc::cuda::convertF16F32(dHalf.as<uint16_t>(), dDst.as<float>(), total,
                               stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: convert_f16_f32 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trConv = halfBytes + S;
    emit(timingJson("convert_f16_f32", trConv,
                    timeKernel(
                        [&] {
                          reloc::cuda::convertF16F32(dHalf.as<uint16_t>(),
                                                     dDst.as<float>(), total,
                                                     stream);
                        },
                        trConv, opt.warmup, opt.iters, stream)));
  }
  {
    // dequant_s8_f32: read 1B + write 4B per element = 1.25*S traffic
    // (per-channel scales excluded by definition: n floats, negligible).
    DeviceBuf dS8(static_cast<size_t>(total));
    DeviceBuf dScales(static_cast<size_t>(n) * 4);
    std::vector<int8_t> hS8(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i)
      hS8[static_cast<size_t>(i)] = static_cast<int8_t>((i * 131) & 0xff);
    std::vector<float> hScales(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c)
      hScales[static_cast<size_t>(c)] = 0.25f * static_cast<float>((c & 7) + 1);
    CUDA_CHECK(cudaMemcpyAsync(dS8.p, hS8.data(), static_cast<size_t>(total),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dScales.p, hScales.data(),
                               static_cast<size_t>(n) * 4,
                               cudaMemcpyHostToDevice, stream));
    std::vector<float> ref(static_cast<size_t>(total));
    for (int64_t c = 0; c < n; ++c)
      for (int64_t j = 0; j < n; ++j)
        ref[static_cast<size_t>(c * n + j)] =
            static_cast<float>(hS8[static_cast<size_t>(c * n + j)]) *
            hScales[static_cast<size_t>(c)];
    clearDst();
    reloc::cuda::dequantS8F32(dS8.as<int8_t>(), dDst.as<float>(), n, n,
                              dScales.as<float>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: dequant_s8_f32 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trDeq = total + S;
    emit(timingJson("dequant_s8_f32", trDeq,
                    timeKernel(
                        [&] {
                          reloc::cuda::dequantS8F32(dS8.as<int8_t>(),
                                                    dDst.as<float>(), n, n,
                                                    dScales.as<float>(),
                                                    stream);
                        },
                        trDeq, opt.warmup, opt.iters, stream)));

    // unpack_dequant_s4: the r=0.125 receive CHAIN (unpackS4S8 then
    // dequantS8F32, two launches -- how rtrack_bench.cu:716-722 runs it).
    // Traffic: 0.5B read + 1B write (unpack) + 1B read + 4B write
    // (dequant) per element = 1.625*S.
    const int64_t pairs = total / 2;
    DeviceBuf dPacked(static_cast<size_t>(pairs));
    DeviceBuf dS8mid(static_cast<size_t>(total));
    std::vector<uint8_t> hPacked(static_cast<size_t>(pairs));
    for (int64_t i = 0; i < pairs; ++i)
      hPacked[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 37) & 0xff);
    CUDA_CHECK(cudaMemcpyAsync(dPacked.p, hPacked.data(),
                               static_cast<size_t>(pairs),
                               cudaMemcpyHostToDevice, stream));
    for (int64_t i = 0; i < pairs; ++i) {
      const uint8_t b = hPacked[static_cast<size_t>(i)];
      const int8_t lo =
          static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4);
      const int8_t hi = static_cast<int8_t>(static_cast<int8_t>(b) >> 4);
      const int64_t e0 = 2 * i, e1 = 2 * i + 1;
      ref[static_cast<size_t>(e0)] =
          static_cast<float>(lo) * hScales[static_cast<size_t>(e0 / n)];
      ref[static_cast<size_t>(e1)] =
          static_cast<float>(hi) * hScales[static_cast<size_t>(e1 / n)];
    }
    clearDst();
    reloc::cuda::unpackS4S8(dPacked.as<uint8_t>(), dS8mid.as<int8_t>(), pairs,
                            stream);
    reloc::cuda::dequantS8F32(dS8mid.as<int8_t>(), dDst.as<float>(), n, n,
                              dScales.as<float>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: unpack_dequant_s4 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trS4 = pairs + total + total + S;
    emit(timingJson("unpack_dequant_s4", trS4,
                    timeKernel(
                        [&] {
                          reloc::cuda::unpackS4S8(dPacked.as<uint8_t>(),
                                                  dS8mid.as<int8_t>(), pairs,
                                                  stream);
                          reloc::cuda::dequantS8F32(dS8mid.as<int8_t>(),
                                                    dDst.as<float>(), n, n,
                                                    dScales.as<float>(),
                                                    stream);
                        },
                        trS4, opt.warmup, opt.iters, stream)));
  }

  // --- scatter_random_f32 over the entropy sweep -------------------------
  std::vector<int64_t> blks = opt.entropyBlk;
  if (blks.empty())
    blks = {1, 1024, 1 << 20, total}; // identity .. full random
  DeviceBuf dIdx(static_cast<size_t>(total) * 8);
  for (int64_t blk : blks) {
    if (blk < 1 || blk > total)
      continue;
    std::vector<int64_t> idx = makeIndexPerm(total, blk, 0x5eed + blk);
    CUDA_CHECK(cudaMemcpyAsync(dIdx.p, idx.data(),
                               static_cast<size_t>(total) * 8,
                               cudaMemcpyHostToDevice, stream));
    clearDst();
    reloc::cuda::scatterRandomF32(dSrc.as<float>(), dIdx.as<int64_t>(),
                                  dDst.as<float>(), total, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> got = download(dDst);
    bool ok = true;
    for (int64_t i = 0; i < total && ok; ++i)
      if (got[static_cast<size_t>(idx[static_cast<size_t>(i)])] !=
          hSrc[static_cast<size_t>(i)])
        ok = false;
    if (!ok) {
      std::fprintf(stderr, "VERIFY FAILED: scatter blk=%lld N=%lld\n",
                   static_cast<long long>(blk), static_cast<long long>(n));
      return 1;
    }
    const std::string extra =
        ", \"block\": " + std::to_string(blk) +
        ", \"block_bytes\": " + std::to_string(blk * 4);
    emit(timingJson(
        "scatter_random_f32_blk" + std::to_string(blk), traffic,
        timeKernel(
            [&] {
              reloc::cuda::scatterRandomF32(dSrc.as<float>(),
                                            dIdx.as<int64_t>(),
                                            dDst.as<float>(), total, stream);
            },
            traffic, opt.warmup, opt.iters, stream),
        extra));
  }

  if (!body.empty())
    body += ",\n";
  body += "  \"" + std::to_string(n) + "\": {\n" + kernels + "\n  }";
  return 0;
}

int run(const Options &opt) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // Theoretical HBM peak from the memory clock + bus width (both in prop).
  const double hbmPeakGbps =
      2.0 * (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8) / 1e9;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::string body;
  for (int64_t n : opt.ns)
    if (runN(n, opt, stream, body) != 0)
      return 1;

  std::string doc =
      "{\n  \"config\": {\"benchmark\": \"hiding_ratio\", \"gpu\": \"" +
      std::string(prop.name) +
      "\", \"hbm_peak_gb_per_s\": " + bench::jsonNumber(hbmPeakGbps) +
      ", \"warmup\": " + std::to_string(opt.warmup) +
      ", \"iters\": " + std::to_string(opt.iters) +
      "},\n  \"by_n\": {\n" + body + "\n  }\n}\n";
  cudaStreamDestroy(stream);
  if (std::strcmp(opt.jsonPath, "-") == 0) {
    std::fputs(doc.c_str(), stdout);
  } else {
    std::FILE *f = std::fopen(opt.jsonPath, "w");
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", opt.jsonPath);
      return 1;
    }
    std::fputs(doc.c_str(), f);
    std::fclose(f);
  }
  std::fprintf(stderr, "hiding_ratio: %s HBM peak %.1f GB/s, done\n",
               prop.name, hbmPeakGbps);
  return 0;
}

std::vector<int64_t> parseList(const std::string &s) {
  std::vector<int64_t> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t next = s.find(',', pos);
    if (next == std::string::npos)
      next = s.size();
    if (next > pos)
      out.push_back(std::atoll(s.substr(pos, next - pos).c_str()));
    pos = next + 1;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--n")
      opt.ns = parseList(next());
    else if (a == "--entropy-blk")
      opt.entropyBlk = parseList(next());
    else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--json")
      opt.jsonPath = next();
    else {
      std::fprintf(stderr,
                   "usage: bench-hiding-ratio [--n 8192,16384] "
                   "[--entropy-blk 1,1024,...] [--warmup W] [--iters I] "
                   "[--json PATH|-]\n");
      return 2;
    }
  }
  for (int64_t n : opt.ns)
    if (n <= 0 || n % 32 != 0) {
      std::fprintf(stderr, "error: N must be positive and divisible by 32\n");
      return 2;
    }
  if (opt.warmup < 0 || opt.iters < 1) {
    std::fprintf(stderr, "error: bad warmup/iters\n");
    return 2;
  }
  return run(opt);
}
