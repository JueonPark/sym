//===- PrefoldTest.cpp - P4 pre-fold component (issue #98) ----------------===//

#include "reloc/Prefold.h"

#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"

#include "gtest/gtest.h"

#include <cstring>
#include <vector>

namespace {

using reloc::prefold::prefoldWins;

TEST(PrefoldWins, AmortizationArithmetic) {
  // pre-fold wins iff nReuse * tTransform > tPrefold + penalty (strict).
  EXPECT_TRUE(prefoldWins(2, 10.0, 10.0, 0.0));   // 20 > 10
  EXPECT_FALSE(prefoldWins(1, 10.0, 10.0, 0.0));  // 10 > 10 is false (boundary)
  EXPECT_FALSE(prefoldWins(1, 10.0, 10.0, 5.0));  // 10 > 15 is false
  EXPECT_TRUE(prefoldWins(4, 10.0, 10.0, 25.0));  // 40 > 35
  EXPECT_FALSE(prefoldWins(3, 10.0, 10.0, 25.0)); // 30 > 35 is false
}

TEST(PrefoldWins, DegenerateReuseCounts) {
  EXPECT_FALSE(prefoldWins(0, 100.0, 0.0, 0.0));
  EXPECT_FALSE(prefoldWins(-4, 100.0, 0.0, 0.0));
}

using reloc::prefold::OutputSpec;
using reloc::prefold::prefoldArtifact;
using reloc::prefold::PrefoldArtifact;

// Hand-authored plans (the CudaKernelsTest convention; never the frozen
// golden blob, issue #63).
reloc::BoundPlan transpose4() {
  reloc::BoundPlan b;
  b.extents = {4, 4};
  b.srcStrides = {1, 4};
  b.dstStrides = {4, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 4 * 4;
  b.L = 1;
  return b;
}

reloc::BoundPlan identity4() {
  reloc::BoundPlan b;
  b.extents = {4, 4};
  b.srcStrides = {4, 1};
  b.dstStrides = {4, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 4 * 4;
  b.L = 4;
  return b;
}

std::vector<float> testSrc16() {
  std::vector<float> s(16);
  for (int i = 0; i < 16; ++i)
    s[static_cast<size_t>(i)] = static_cast<float>(i * 7 - 40) * 0.9f;
  return s;
}

TEST(PrefoldArtifactTest, GatherQuantMatchesScalarKernel) {
  reloc::HostBackend backend;
  reloc::GatherPool pool(2);
  const reloc::BoundPlan b = transpose4();
  const std::vector<float> src = testSrc16();
  const std::vector<float> inv(4, 1.0f);

  std::vector<int8_t> want(16);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), want.data(), inv.data(),
                                    0, 4, reloc::quant::Variant::Scalar);

  PrefoldArtifact a =
      prefoldArtifact(b, src.data(), OutputSpec::S8GatherQuant, inv.data(),
                      backend, pool);
  ASSERT_TRUE(a.valid());
  ASSERT_EQ(a.bytes(), 16);
  EXPECT_EQ(0, std::memcmp(a.data(), want.data(), 16));
}

TEST(PrefoldArtifactTest, QuantPackMatchesScalarKernel) {
  reloc::HostBackend backend;
  reloc::GatherPool pool(2);
  const reloc::BoundPlan b = identity4();
  const std::vector<float> src = testSrc16();
  const std::vector<float> inv(4, 0.5f);

  std::vector<int8_t> want(16);
  reloc::quant::quantizePackF32S8(src.data(), want.data(), 4, 4, inv.data(),
                                  reloc::quant::Variant::Scalar);

  PrefoldArtifact a = prefoldArtifact(b, src.data(), OutputSpec::S8QuantPack,
                                      inv.data(), backend, pool);
  ASSERT_TRUE(a.valid());
  EXPECT_EQ(0, std::memcmp(a.data(), want.data(), 16));
}

TEST(PrefoldArtifactTest, PreconditionViolationsReturnInvalid) {
  reloc::HostBackend backend;
  reloc::GatherPool pool(2);
  const std::vector<float> src = testSrc16();
  const std::vector<float> inv(4, 1.0f);

  // Non-fp32 element size.
  reloc::BoundPlan e8 = transpose4();
  e8.elementSize = 8;
  EXPECT_FALSE(prefoldArtifact(e8, src.data(), OutputSpec::S8GatherQuant,
                               inv.data(), backend, pool)
                   .valid());

  // Non-packed dst.
  reloc::BoundPlan sparse = transpose4();
  sparse.dstStrides = {8, 1};
  EXPECT_FALSE(prefoldArtifact(sparse, src.data(),
                               OutputSpec::S8GatherQuant, inv.data(),
                               backend, pool)
                   .valid());

  // QuantPack demands an identity plan.
  EXPECT_FALSE(prefoldArtifact(transpose4(), src.data(),
                               OutputSpec::S8QuantPack, inv.data(), backend,
                               pool)
                   .valid());

  // Null invScales.
  EXPECT_FALSE(prefoldArtifact(transpose4(), src.data(),
                               OutputSpec::S8GatherQuant, nullptr, backend,
                               pool)
                   .valid());
}

TEST(PrefoldArtifactTest, MoveTransfersOwnership) {
  reloc::HostBackend backend;
  reloc::GatherPool pool(2);
  const reloc::BoundPlan b = identity4();
  const std::vector<float> src = testSrc16();
  const std::vector<float> inv(4, 1.0f);

  PrefoldArtifact a = prefoldArtifact(b, src.data(), OutputSpec::S8QuantPack,
                                      inv.data(), backend, pool);
  ASSERT_TRUE(a.valid());
  const void *p = a.data();

  PrefoldArtifact m(std::move(a));
  EXPECT_FALSE(a.valid());
  EXPECT_EQ(m.data(), p);

  PrefoldArtifact n;
  n = std::move(m);
  EXPECT_FALSE(m.valid());
  EXPECT_EQ(n.data(), p);
  // n's destructor frees exactly once; ASAN/valgrind-clean is the check.
}

} // namespace
