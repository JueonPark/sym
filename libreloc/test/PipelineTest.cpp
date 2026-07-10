//===- PipelineTest.cpp - Strategy-4 pipeline oracle tests ----------------===//
//
// The pipeline over HostBackend must be byte-identical to the reference
// executors (executeH2D for H2D). We sweep buffer counts {1,2,4} x chunk sizes
// {tiny (many chunks), heuristic (single/few chunks)} x a set of plans, plus
// an interleaving stress case (small buffers, many chunks, many streams) that
// is meaningful under TSan.
//
//===----------------------------------------------------------------------===//

#include "reloc/Pipeline.h"
#include "reloc/Bind.h"
#include "reloc/ChunkSchedule.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/PinnedBufferPool.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

using reloc::BoundPlan;
using reloc::HostBackend;
using reloc::PadRegion;

int64_t product(const std::vector<int64_t> &v) {
  return std::accumulate(v.begin(), v.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

std::vector<uint8_t> iotaBytes(int64_t elements, uint32_t elementSize) {
  std::vector<uint8_t> buf(static_cast<size_t>(elements) * elementSize);
  for (int64_t e = 0; e < elements; ++e)
    for (uint32_t b = 0; b < elementSize; ++b)
      buf[e * elementSize + b] =
          static_cast<uint8_t>((e * 131 + b * 17) & 0xff);
  return buf;
}

BoundPlan makeBound(std::vector<int64_t> extents,
                    std::vector<int64_t> srcStrides,
                    std::vector<int64_t> dstStrides, uint32_t elementSize,
                    std::vector<PadRegion> pads = {}) {
  BoundPlan b;
  b.extents = std::move(extents);
  b.srcStrides = std::move(srcStrides);
  b.dstStrides = std::move(dstStrides);
  b.elementSize = elementSize;
  b.padRegions = std::move(pads);
  int64_t total = elementSize;
  std::vector<int64_t> padded = b.extents;
  for (const PadRegion &p : b.padRegions)
    padded[p.axis] += p.lo + p.hi;
  for (int64_t e : padded)
    total *= e;
  b.totalBytes = total;
  b.L = (!b.extents.empty() && b.srcStrides.back() == 1 &&
         b.dstStrides.back() == 1)
            ? b.extents.back()
            : 1;
  return b;
}

// Run executeH2DPipelined over HostBackend and assert byte-exact vs executeH2D.
// gatherThreads: 0 = hardware concurrency, 1 = the inline regression path.
void expectPipelineExactH2D(const BoundPlan &b, int nBuffers, int nStreams,
                            size_t chunkOverride, unsigned gatherThreads = 1) {
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(nStreams);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
  reloc::executeH2DPipelined(b, src.data(), device.data(), backend, nBuffers,
                             chunkOverride, gatherThreads);
  EXPECT_EQ(device, reference)
      << "nBuffers=" << nBuffers << " nStreams=" << nStreams
      << " override=" << chunkOverride << " gatherThreads=" << gatherThreads;
}

const std::vector<BoundPlan> &plans() {
  static const std::vector<BoundPlan> kPlans = [] {
    std::vector<BoundPlan> v;
    v.push_back(makeBound({32, 64}, {1, 32}, {64, 1}, 4));   // transpose
    v.push_back(makeBound({8, 128}, {128, 1}, {128, 1}, 4)); // contiguous
    v.push_back(makeBound({32, 64}, {1, 32}, {64, 1}, 1));   // int8 transpose
    PadRegion pad{0, 1, 1};
    pad.fillBits = 0x11223344;
    v.push_back(makeBound({6, 4}, {4, 1}, {4, 1}, 4, {pad})); // outer-padded
    return v;
  }();
  return kPlans;
}

TEST(Pipeline, H2DByteExactMatrix) {
  for (const BoundPlan &b : plans())
    for (int nBuffers : {1, 2, 4})
      for (int nStreams : {1, 2})
        for (size_t override : {size_t(0), size_t(16), size_t(64)})
          for (unsigned threads : {1u, 2u, 0u}) // 0 == hardware concurrency
            expectPipelineExactH2D(b, nBuffers, nStreams, override, threads);
}

TEST(Pipeline, H2DInterleavingStress) {
  // Larger outer axis, smallest buffers, tiny chunks, several streams: forces
  // deep interleaving of gather/copy/event. Meaningful under TSan.
  BoundPlan b = makeBound({256, 16}, {16, 1}, {16, 1}, 4);
  expectPipelineExactH2D(b, /*nBuffers=*/2, /*nStreams=*/4, /*override=*/64);
}

TEST(Pipeline, H2DSingleBufferSerializes) {
  // 1 buffer = fully serialized; must still complete and be exact.
  BoundPlan b = makeBound({64, 16}, {16, 1}, {16, 1}, 4);
  expectPipelineExactH2D(b, /*nBuffers=*/1, /*nStreams=*/2, /*override=*/64);
}

TEST(Pipeline, H2DParallelGatherEngagesWorkers) {
  // 4096 rows x 4 KiB = 16 MiB. A 2 MiB chunk override gives 512-row chunks,
  // and the 1 MiB/worker byte floor yields 2 workers per chunk -- unlike the
  // tiny matrix plans (which collapse to the inline path via the floor), this
  // actually exercises concurrent gatherChunk sub-ranges. Meaningful under
  // TSan.
  BoundPlan b = makeBound({4096, 1024}, {1024, 1}, {1024, 1}, 4);
  for (unsigned threads : {2u, 0u})
    expectPipelineExactH2D(b, /*nBuffers=*/2, /*nStreams=*/2,
                           /*override=*/size_t(2) << 20, threads);
}

TEST(Pipeline, H2DNonDisjointDstStaysExactWithThreads) {
  // Column-major dst: outer rows interleave in dst byte space, planChunks
  // serializes to one whole-tensor chunk, and the gather guard must keep
  // that chunk single-threaded (partitioning it would race). Exactness with
  // threads requested is the assertion.
  BoundPlan b = makeBound({16, 4}, {4, 1}, {1, 16}, 4);
  expectPipelineExactH2D(b, /*nBuffers=*/2, /*nStreams=*/2, /*override=*/0,
                         /*gatherThreads=*/8);
}

TEST(Pipeline, CallerOwnedGatherPoolReusedAcrossCalls) {
  // One GatherPool across several pipeline calls: byte-exact every time,
  // close() observable afterwards (the pybind context object relies on
  // exactly this reuse pattern).
  BoundPlan b = makeBound({4096, 1024}, {1024, 1}, {1024, 1}, 4);
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(2);
  reloc::GatherPool gather(4);
  for (int call = 0; call < 3; ++call) {
    std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
    reloc::executeH2DPipelined(b, src.data(), device.data(), backend,
                               /*nBuffers=*/2, /*override=*/size_t(2) << 20,
                               gather);
    EXPECT_EQ(device, reference) << "call " << call;
  }
  gather.close();
  EXPECT_TRUE(gather.closed());
}

// Run executeD2HPipelined over HostBackend and assert it reconstructs src
// byte-exact (== executeD2H). `dst` is a known-good dst-layout buffer.
void expectPipelineExactD2H(const BoundPlan &b, int nBuffers, int nStreams,
                            size_t chunkOverride) {
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), device.data()); // build the dst layout

  HostBackend backend(nStreams);
  std::vector<uint8_t> back(static_cast<size_t>(srcElems) * b.elementSize,
                            0xCD);
  reloc::executeD2HPipelined(b, device.data(), back.data(), backend, nBuffers,
                             chunkOverride);
  EXPECT_EQ(back, src) << "nBuffers=" << nBuffers << " nStreams=" << nStreams
                       << " override=" << chunkOverride;
}

TEST(Pipeline, D2HByteExactMatrix) {
  for (const BoundPlan &b : plans())
    for (int nBuffers : {1, 2, 4})
      for (int nStreams : {1, 2})
        for (size_t override : {size_t(0), size_t(16), size_t(64)})
          expectPipelineExactD2H(b, nBuffers, nStreams, override);
}

TEST(Pipeline, RoundTripH2DThenD2H) {
  // D2H(H2D(x)) == x, both through the pipeline, across buffer counts.
  BoundPlan b = makeBound({64, 16}, {16, 1}, {16, 1}, 4);
  int64_t elems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(elems, b.elementSize);
  for (int nBuffers : {1, 2, 4}) {
    HostBackend backend(2);
    std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xAB);
    reloc::executeH2DPipelined(b, src.data(), device.data(), backend, nBuffers,
                               /*override=*/64);
    std::vector<uint8_t> back(static_cast<size_t>(elems) * b.elementSize, 0xCD);
    reloc::executeD2HPipelined(b, device.data(), back.data(), backend, nBuffers,
                               /*override=*/64);
    EXPECT_EQ(back, src) << "nBuffers=" << nBuffers;
  }
}

TEST(Pipeline, CallerOwnedPoolReusedAcrossCalls) {
  // The pool-reuse overload must be byte-identical to the pool-per-call
  // entry point, and the same pool must be safely reusable across
  // consecutive calls (drain leaves no pending events behind).
  BoundPlan b = makeBound({64, 16}, {16, 1}, {16, 1}, 4);
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(2);
  reloc::ChunkSchedule sched = reloc::planChunks(b, /*nBuffers=*/2,
                                                 /*override=*/64);
  reloc::PinnedBufferPool pool(backend, 2, sched.maxChunkBytes);
  for (int call = 0; call < 3; ++call) {
    std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
    reloc::executeH2DPipelined(b, src.data(), device.data(), backend, pool,
                               /*override=*/64);
    EXPECT_EQ(device, reference) << "call " << call;
  }
}

} // namespace
