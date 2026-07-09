//===- BindTest.cpp - bind() constraint + coalescing + acceptance ---------===//

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using reloc::BindError;
using reloc::BoundPlan;
using reloc::RelocationPlan;
using reloc::Strategy;

std::vector<uint8_t> fromHex(const char *hex) {
  std::vector<uint8_t> bytes;
  auto nibble = [](char c) -> uint8_t {
    if (c >= '0' && c <= '9')
      return c - '0';
    return 10 + (c - 'a');
  };
  for (const char *p = hex; p[0] && p[1]; p += 2)
    bytes.push_back(static_cast<uint8_t>(nibble(p[0]) << 4 | nibble(p[1])));
  return bytes;
}

// Copied verbatim from serialize.mlir's `plan_hex(reference):` CHECK line.
const char *kReferenceHex = "52504c4e0000000001000000010000004e0200000001000000"
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

// Copied verbatim from serialize.mlir's `plan_hex(degraded):` CHECK line.
const char *kDegradedHex = "52504c4e0000000001000000010000004e01000000010000000"
                           "0000000000000000001000000"
                           "010000000000000000002000000001000000070000000140000"
                           "000000000000000000000013f"
                           "000000000000000201400000000000000005040000000001000"
                           "0000100000000000000000020"
                           "000000010000000000000001000000010000007801000000000"
                           "0000000010000000101000000"
                           "000000000100000001010000000000000001000000000000000"
                           "1000000010000000000000000"
                           "090000000140000000000000000000000000013f00000000000"
                           "0000201400000000000000005"
                           "040000000000030020000000000000000000000000000000000"
                           "0000000000000000101000000"
                           "01000000010000000700000000";

RelocationPlan decoded(const char *hex) {
  std::vector<uint8_t> bytes = fromHex(hex);
  auto result = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<RelocationPlan>(&result);
  EXPECT_NE(plan, nullptr);
  return plan ? *plan : RelocationPlan{};
}

BoundPlan bindOk(const RelocationPlan &plan, const reloc::SymbolMap &symbols,
                 Strategy override = Strategy::Auto) {
  auto result = reloc::bind(plan, symbols, override);
  auto *bound = std::get_if<BoundPlan>(&result);
  EXPECT_NE(bound, nullptr) << (std::get_if<BindError>(&result)
                                    ? std::get_if<BindError>(&result)->message
                                    : "<none>");
  return bound ? *bound : BoundPlan{};
}

std::string bindErr(const RelocationPlan &plan,
                    const reloc::SymbolMap &symbols) {
  auto result = reloc::bind(plan, symbols, Strategy::Auto);
  auto *error = std::get_if<BindError>(&result);
  EXPECT_NE(error, nullptr) << "bind unexpectedly succeeded";
  return error ? error->message : std::string{};
}

TEST(Bind, RejectsUnboundSymbol) {
  EXPECT_NE(bindErr(decoded(kReferenceHex), {}).find("N"), std::string::npos);
}

TEST(Bind, RejectsExtraSymbol) {
  EXPECT_NE(bindErr(decoded(kReferenceHex), {{"N", 32768}, {"bogus", 1}})
                .find("bogus"),
            std::string::npos);
}

TEST(Bind, DivisibilityViolationIsHardError) {
  // N = 1000 violates divisible(N, 64) -> bind error, NOT a degraded plan.
  EXPECT_NE(bindErr(decoded(kReferenceHex), {{"N", 1000}}).find("divisib"),
            std::string::npos);
}

TEST(Bind, DegradedPadRangePassesForAlignedFormula) {
  // The alignment pad is self-consistent for every N; bind succeeds.
  BoundPlan bound = bindOk(decoded(kDegradedHex), {{"N", 1000}});
  EXPECT_FALSE(bound.noCopy);
  ASSERT_EQ(bound.padRegions.size(), 1u);
  EXPECT_EQ(bound.padRegions[0].lo, 0);
  EXPECT_EQ(bound.padRegions[0].hi, 24); // 64*ceil(1000/64)=1024; 1024-1000
}

TEST(Bind, StrategyOverrideBypassesHeuristic) {
  BoundPlan bound =
      bindOk(decoded(kDegradedHex), {{"N", 1000}}, Strategy::MultiThreadTiled);
  EXPECT_EQ(bound.strategy, Strategy::MultiThreadTiled);
}

TEST(Bind, RejectsNonByteMultipleBitwidth) {
  // A spec-legal-but-non-byte width (12) must be rejected, not silently
  // truncated to 1 byte (which would corrupt totalBytes / the heuristic).
  RelocationPlan plan = decoded(kDegradedHex);
  plan.dst.elementType.bitwidth = 12;
  EXPECT_NE(bindErr(plan, {{"N", 1000}}).find("bitwidth"), std::string::npos);
}

TEST(Bind, RejectsPadAxisOutOfRangeForDstExtents) {
  // A hand-crafted plan whose dst rank is shorter than its axis count but
  // still carries a pad_fill must be rejected (guarding the OOB), not crash.
  RelocationPlan plan = decoded(kDegradedHex);
  ASSERT_FALSE(plan.padFill.empty());
  plan.dst.extents.clear();
  EXPECT_NE(bindErr(plan, {{"N", 1000}}).find("out of range"),
            std::string::npos);
}

} // namespace
