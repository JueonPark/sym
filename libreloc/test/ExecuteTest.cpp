//===- ExecuteTest.cpp - CPU executor oracle tests ------------------------===//
//
// Byte-exact index-table oracles (P1b PlanBuilderTest style): build a
// BoundPlan, run an executor over an iota source buffer, and compare the
// destination bytes against a NumPy-semantics reference. dtype is just
// elementSize in {4, 2, 1}; the copy is a dtype-agnostic bit move.
//
//===----------------------------------------------------------------------===//

#include "reloc/Execute.h"
#include "reloc/Bind.h"
#include "reloc/CopyRun.h"
#include "reloc/Decode.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

using reloc::BoundPlan;
using reloc::PadRegion;
using reloc::ViewDescriptor;

// A byte buffer whose element i starts with a distinct pattern, so a
// misplaced element is caught byte-for-byte.
std::vector<uint8_t> iotaBytes(int64_t elements, uint32_t elementSize) {
  std::vector<uint8_t> buf(static_cast<size_t>(elements) * elementSize);
  for (int64_t e = 0; e < elements; ++e)
    for (uint32_t b = 0; b < elementSize; ++b)
      // low byte encodes the element index, higher bytes a per-byte salt
      buf[e * elementSize + b] =
          static_cast<uint8_t>((e * 131 + b * 17) & 0xff);
  return buf;
}

// Total elements over a set of extents.
int64_t product(const std::vector<int64_t> &v) {
  return std::accumulate(v.begin(), v.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

// Reference H2D: enumerate the VALID index space, place each source
// element at dst offset Σ(i_k+lo_k)·dstStride_k, and fill everything else
// with the pad pattern. Returns the expected dst byte buffer.
std::vector<uint8_t> referenceH2D(const BoundPlan &bound,
                                  const std::vector<uint8_t> &src) {
  const uint32_t es = bound.elementSize;
  const size_t r = bound.extents.size();
  // Padded (physical) dst extents = valid extent + lo + hi per axis.
  std::vector<int64_t> lo(r, 0), padded = bound.extents;
  for (const PadRegion &p : bound.padRegions) {
    lo[p.axis] = p.lo;
    padded[p.axis] = bound.extents[p.axis] + p.lo + p.hi;
  }
  int64_t dstElems = 1;
  for (int64_t e : padded)
    dstElems *= e;
  std::vector<uint8_t> dst(static_cast<size_t>(dstElems) * es);
  // Pad fill: splat the single fill pattern across all of dst first.
  if (!bound.padRegions.empty()) {
    uint64_t bits = bound.padRegions.front().fillBits;
    for (int64_t e = 0; e < dstElems; ++e)
      std::memcpy(&dst[e * es], &bits, es); // little-endian low es bytes
  }
  // Valid region.
  std::vector<int64_t> idx(r, 0);
  int64_t validElems = product(bound.extents);
  for (int64_t n = 0; n < validElems; ++n) {
    int64_t srcOff = 0, dstOff = 0;
    for (size_t k = 0; k < r; ++k) {
      srcOff += idx[k] * bound.srcStrides[k];
      dstOff += (idx[k] + lo[k]) * bound.dstStrides[k];
    }
    std::memcpy(&dst[dstOff * es], &src[srcOff * es], es);
    for (int64_t k = static_cast<int64_t>(r) - 1; k >= 0; --k) {
      if (++idx[k] < bound.extents[k])
        break;
      idx[k] = 0;
    }
  }
  return dst;
}

// A directly-constructed BoundPlan (bypasses decode/bind for the small
// transpose/reshape/pad cases — the executor only needs BoundPlan).
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
  // L: innermost unit-stride run.
  b.L = (!b.extents.empty() && b.srcStrides.back() == 1 &&
         b.dstStrides.back() == 1)
            ? b.extents.back()
            : 1;
  return b;
}

// Run executeH2D and assert byte-exact vs the reference, for a src of the
// given elementSize.
void expectH2DExact(const BoundPlan &bound) {
  int64_t srcElems = product(bound.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, bound.elementSize);
  std::vector<uint8_t> expected = referenceH2D(bound, src);
  std::vector<uint8_t> dst(expected.size(), 0xAB); // poison
  reloc::executeH2D(bound, src.data(), dst.data());
  EXPECT_EQ(dst, expected);
}

TEST(Execute, TransposeFp32) {
  // dst [32,64] <- src [64,32] transpose: axis0 (ext 32, src 1, dst 64),
  // axis1 (ext 64, src 32, dst 1). Innermost strided (L=1).
  expectH2DExact(makeBound({32, 64}, {1, 32}, {64, 1}, 4));
}

TEST(Execute, ContiguousInnerRunFp32) {
  // [8, 128] identity-ish: innermost unit stride both sides (L=128), the
  // contiguous copyRun path.
  expectH2DExact(makeBound({8, 128}, {128, 1}, {128, 1}, 4));
}

TEST(Execute, TransposeInt8AndFp16) {
  expectH2DExact(makeBound({32, 64}, {1, 32}, {64, 1}, 1));
  expectH2DExact(makeBound({32, 64}, {1, 32}, {64, 1}, 2));
}

TEST(Execute, PadFp32FillVerified) {
  // One axis extent 6, padded lo=1 hi=1 -> physical extent 8; fill = the
  // f32 bit pattern for 0.0 is 0; use a nonzero pattern to prove fill.
  PadRegion pad{0, 1, 1};
  pad.fillBits = 0xDEADBEEF;
  BoundPlan b = makeBound({6}, {1}, {1}, 4, {pad});
  expectH2DExact(b);
  // Explicitly check a pad cell and a valid cell in the produced dst.
  std::vector<uint8_t> src = iotaBytes(6, 4);
  std::vector<uint8_t> dst(8 * 4, 0);
  reloc::executeH2D(b, src.data(), dst.data());
  uint32_t leading;
  std::memcpy(&leading, &dst[0], 4); // dst[0] is leading pad
  EXPECT_EQ(leading, 0xDEADBEEFu);
  EXPECT_EQ(std::memcmp(&dst[1 * 4], &src[0], 4), 0); // valid starts at lo=1
}

TEST(Execute, ViewPublishNoCopy) {
  BoundPlan b = makeBound({8, 128}, {128, 1}, {128, 1}, 4);
  b.noCopy = true;
  int dummy = 0;
  ViewDescriptor view = reloc::executeView(b, &dummy);
  EXPECT_EQ(view.base, &dummy);
  EXPECT_EQ(view.extents, b.extents);
  EXPECT_EQ(view.strides, b.srcStrides);
  EXPECT_EQ(view.elementSize, 4u);
}

TEST(Execute, GatherChunkCoversOuterSubrange) {
  // gatherChunk over the whole outer range == executeH2D; a partial range
  // writes only its outer slice. Verify the [0,2) slice of a [4,3] plan
  // matches the corresponding rows of the full result.
  BoundPlan b = makeBound({4, 3}, {3, 1}, {3, 1}, 4);
  std::vector<uint8_t> src = iotaBytes(12, 4);
  std::vector<uint8_t> full(12 * 4, 0);
  reloc::executeH2D(b, src.data(), full.data());
  std::vector<uint8_t> partial(12 * 4, 0xAB);
  reloc::gatherChunk(b, src.data(), partial.data(), 0, 2);
  // Rows 0..1 (outer indices 0,1) written; check they match `full`.
  EXPECT_EQ(std::memcmp(partial.data(), full.data(), 2 * 3 * 4), 0);
}

TEST(Execute, ScatterChunkCoversOuterSubrange) {
  // scatterChunk over the whole outer range == executeD2H; a partial range
  // reconstructs only its outer slice. Mirror of
  // GatherChunkCoversOuterSubrange.
  BoundPlan b = makeBound({4, 3}, {3, 1}, {3, 1}, 4);
  std::vector<uint8_t> src = iotaBytes(12, 4);
  std::vector<uint8_t> dst = referenceH2D(b, src);
  std::vector<uint8_t> full(12 * 4, 0xAB);
  reloc::executeD2H(b, dst.data(), full.data());
  std::vector<uint8_t> partial(12 * 4, 0xAB);
  reloc::scatterChunk(b, dst.data(), partial.data(), 0, 2);
  // Valid outer rows 0..1 reconstructed; check they match executeD2H's output.
  EXPECT_EQ(std::memcmp(partial.data(), full.data(), 2 * 3 * 4), 0);
}

TEST(Execute, ReferencePlanN4096) {
  // End-to-end: decode the reference golden, bind, execute. This plan's
  // axis formulas (block tiling of size 64, with the n0/n1 stride tied to
  // the SAME 64 constant) only stay within the declared [N,N] src tensor
  // bounds for N/64 >= 64, i.e. N >= 4096 -- for smaller N (e.g. 256) the
  // computed src offsets run past the tensor's element count and any
  // executor (including this test's own referenceH2D oracle) reading
  // through them segfaults. N=4096 is the smallest value that is both
  // divisible by 64 (the plan's hard constraint) and in-bounds; it is in
  // fact an exact bijection onto [0, N*N). This is a property of the
  // fixed example plan, not a relaxation of the byte-exact oracle bar.
  const char *kReferenceHex =
      "52504c4e0000000001000000010000004e0200000001000000"
      "00000000000100000000000000"
      "00020000000100000000000000000100000001010000000000"
      "00000100000001000000000000"
      "00000020000000040000000300000000000000000140000000"
      "00000000050100000001400000"
      "00000000000100000001400000000000000003000000000000"
      "00000140000000000000000500"
      "00000001000000010000000000000000002000000004000000"
      "01000000000000000200000003"
      "00000004000000020000006e30030000000000000000014000"
      "00000000000005010000000140"
      "00000000000000050000000100100000000000000000000000"
      "01400000000000000005040200"
      "00006230010000000140000000000000000300000001400000"
      "00000000000000000000040500"
      "00000140000000000000000000000000014000000000000000"
      "05040200000062310100000001"
      "40000000000000000100000000000000000300000000000000"
      "00014000000000000000050200"
      "00006e31030000000000000000014000000000000000050100"
      "00000101000000000000000100"
      "00000101000000000000000000000001000000010000000000"
      "00000040000000000000000000"
      "00000400000000000001000004000000040000000100000007"
      "01000000010000000700000000"
      "010000000702000000010000000703000000";
  std::vector<uint8_t> bytes;
  for (const char *p = kReferenceHex; p[0] && p[1]; p += 2) {
    auto nib = [](char c) { return c <= '9' ? c - '0' : 10 + c - 'a'; };
    bytes.push_back(static_cast<uint8_t>(nib(p[0]) << 4 | nib(p[1])));
  }
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  ASSERT_NE(plan, nullptr);
  auto bound = reloc::bind(*plan, {{"N", 4096}});
  auto *bp = std::get_if<BoundPlan>(&bound);
  ASSERT_NE(bp, nullptr);
  expectH2DExact(*bp);
}

TEST(CopyRun, Avx2MatchesScalarAllLengths) {
  // For every length 0..300 bytes and several src/dst alignment offsets,
  // the AVX2 path must be byte-identical to the scalar path.
  std::vector<uint8_t> srcbuf(512), a(512), b(512);
  for (size_t i = 0; i < srcbuf.size(); ++i)
    srcbuf[i] = static_cast<uint8_t>(i * 7 + 3);
  for (size_t n = 0; n <= 300; ++n)
    for (size_t soff : {size_t(0), size_t(1), size_t(3), size_t(16)})
      for (size_t doff : {size_t(0), size_t(1), size_t(7), size_t(32)}) {
        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        reloc::copyRunScalar(a.data() + doff, srcbuf.data() + soff, n);
        reloc::copyRun(b.data() + doff, srcbuf.data() + soff, n);
        ASSERT_EQ(std::memcmp(a.data(), b.data(), 512), 0)
            << "n=" << n << " soff=" << soff << " doff=" << doff;
      }
}

TEST(Execute, MisalignedSourceStaysExact) {
  // Offset the source buffer by 1 byte so the contiguous inner run reads
  // from a misaligned address; result must remain byte-exact.
  BoundPlan b = makeBound({8, 128}, {128, 1}, {128, 1}, 4); // L=128
  int64_t elems = product(b.extents);
  std::vector<uint8_t> raw = iotaBytes(elems + 1, b.elementSize);
  const uint8_t *misaligned = raw.data() + 1; // 1-byte-misaligned src
  std::vector<uint8_t> expected(elems * 4), dst(elems * 4, 0xAB);
  // reference over the same misaligned source
  {
    std::vector<uint8_t> srcView(misaligned, misaligned + elems * 4);
    expected = referenceH2D(b, srcView);
  }
  reloc::executeH2D(b, misaligned, dst.data());
  EXPECT_EQ(dst, expected);
}

TEST(Execute, ThreadedMatchesSingleThread) {
  // Strategy 2 vs Strategy 3 must be bit-identical. Use a plan with a
  // sizable outer axis so partitioning is exercised.
  BoundPlan b = makeBound({64, 3}, {3, 1}, {3, 1}, 4);
  std::vector<uint8_t> src = iotaBytes(product(b.extents), 4);
  std::vector<uint8_t> single(product(b.extents) * 4, 0);
  std::vector<uint8_t> multi(product(b.extents) * 4, 0);
  reloc::executeH2D(b, src.data(), single.data());
  reloc::executeH2DThreaded(b, src.data(), multi.data(), /*threads=*/4);
  EXPECT_EQ(single, multi);
}

TEST(Execute, ThreadedTransposeAndPadExact) {
  reloc::BoundPlan t = makeBound({32, 64}, {1, 32}, {64, 1}, 4);
  std::vector<uint8_t> src = iotaBytes(product(t.extents), 4);
  std::vector<uint8_t> got(product(t.extents) * 4, 0);
  reloc::executeH2DThreaded(t, src.data(), got.data(), 3);
  EXPECT_EQ(got, referenceH2D(t, src));

  PadRegion pad{0, 1, 1};
  pad.fillBits = 0x11223344;
  reloc::BoundPlan p = makeBound({6, 4}, {4, 1}, {4, 1}, 4, {pad}); // pad outer
  std::vector<uint8_t> psrc = iotaBytes(product(p.extents), 4);
  int64_t pElems = (6 + 2) * 4;
  std::vector<uint8_t> pgot(pElems * 4, 0);
  reloc::executeH2DThreaded(p, psrc.data(), pgot.data(), 4);
  EXPECT_EQ(pgot, referenceH2D(p, psrc));

  // More threads than outer rows must be safe.
  reloc::BoundPlan tiny = makeBound({2, 5}, {5, 1}, {5, 1}, 4);
  std::vector<uint8_t> tsrc = iotaBytes(product(tiny.extents), 4);
  std::vector<uint8_t> tgot(product(tiny.extents) * 4, 0);
  reloc::executeH2DThreaded(tiny, tsrc.data(), tgot.data(), 16);
  EXPECT_EQ(tgot, referenceH2D(tiny, tsrc));
}

TEST(Execute, ThreadedOverlappingDstMatchesSingleThread) {
  // dstStrides[0]=1 < inner span (3) -> rows alias; threaded MUST fall back
  // to single-thread and stay byte-identical (and race-free).
  reloc::BoundPlan b = makeBound({8, 3}, {3, 1}, {1, 1}, 4);
  std::vector<uint8_t> src = iotaBytes(product(b.extents), 4);
  std::vector<uint8_t> single(product(b.extents) * 4, 0);
  std::vector<uint8_t> multi(product(b.extents) * 4, 0);
  reloc::executeH2D(b, src.data(), single.data());
  reloc::executeH2DThreaded(b, src.data(), multi.data(), 8);
  EXPECT_EQ(single, multi);
}

// Reference D2H: reconstruct src from a dst-layout buffer over the valid
// index space (inverse of referenceH2D on the valid region).
std::vector<uint8_t> referenceD2H(const BoundPlan &bound,
                                  const std::vector<uint8_t> &dst) {
  const uint32_t es = bound.elementSize;
  const size_t r = bound.extents.size();
  std::vector<int64_t> lo(r, 0);
  for (const PadRegion &p : bound.padRegions)
    lo[p.axis] = p.lo;
  std::vector<uint8_t> src(static_cast<size_t>(product(bound.extents)) * es);
  std::vector<int64_t> idx(r, 0);
  int64_t validElems = product(bound.extents);
  for (int64_t n = 0; n < validElems; ++n) {
    int64_t srcOff = 0, dstOff = 0;
    for (size_t k = 0; k < r; ++k) {
      srcOff += idx[k] * bound.srcStrides[k];
      dstOff += (idx[k] + lo[k]) * bound.dstStrides[k];
    }
    std::memcpy(&src[srcOff * es], &dst[dstOff * es], es);
    for (int64_t k = static_cast<int64_t>(r) - 1; k >= 0; --k) {
      if (++idx[k] < bound.extents[k])
        break;
      idx[k] = 0;
    }
  }
  return src;
}

TEST(Execute, D2HRoundTripsH2D) {
  // H2D then D2H must reconstruct the original source exactly (valid
  // region is a bijection). Cover transpose / contiguous / pad / reference.
  auto roundTrip = [](const BoundPlan &b) {
    int64_t validElems = product(b.extents);
    std::vector<uint8_t> src = iotaBytes(validElems, b.elementSize);
    std::vector<uint8_t> dst = referenceH2D(b, src); // known-good dst layout
    std::vector<uint8_t> back(validElems * b.elementSize, 0xAB);
    reloc::executeD2H(b, dst.data(), back.data());
    EXPECT_EQ(back, src);
    EXPECT_EQ(back, referenceD2H(b, dst));
  };
  roundTrip(makeBound({32, 64}, {1, 32}, {64, 1}, 4));   // transpose
  roundTrip(makeBound({8, 128}, {128, 1}, {128, 1}, 4)); // contiguous
  roundTrip(makeBound({32, 64}, {1, 32}, {64, 1}, 1));   // int8 transpose
  PadRegion pad{0, 1, 1};
  pad.fillBits = 0x55667788;
  roundTrip(makeBound({6, 4}, {4, 1}, {4, 1}, 4, {pad})); // padded (valid only)
}

TEST(Execute, D2HRank1RoundTripsH2D) {
  BoundPlan b = makeBound({256}, {1}, {1}, 4);
  int64_t validElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(validElems, b.elementSize);
  std::vector<uint8_t> dst = referenceH2D(b, src);
  std::vector<uint8_t> back(validElems * b.elementSize, 0xAB);
  reloc::executeD2H(b, dst.data(), back.data());
  EXPECT_EQ(back, src);
  EXPECT_EQ(back, referenceD2H(b, dst));
}

} // namespace
