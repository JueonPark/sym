//===- PrefoldTest.cpp - P4 pre-fold component (issue #98) ----------------===//

#include "reloc/Prefold.h"

#include "gtest/gtest.h"

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

} // namespace
