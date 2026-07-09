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
#include "reloc/Execute.h"
#include "reloc/Bind.h"
#include "reloc/HostBackend.h"
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
      buf[e * elementSize + b] = static_cast<uint8_t>((e * 131 + b * 17) & 0xff);
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
void expectPipelineExactH2D(const BoundPlan &b, int nBuffers, int nStreams,
                            size_t chunkOverride) {
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(nStreams);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
  reloc::executeH2DPipelined(b, src.data(), device.data(), backend, nBuffers,
                             chunkOverride);
  EXPECT_EQ(device, reference)
      << "nBuffers=" << nBuffers << " nStreams=" << nStreams
      << " override=" << chunkOverride;
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
          expectPipelineExactH2D(b, nBuffers, nStreams, override);
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

} // namespace
