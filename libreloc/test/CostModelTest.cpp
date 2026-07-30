//===- CostModelTest.cpp - P3b cost-model component (issue #97) -----------===//

#include "reloc/CostModel.h"

#include "gtest/gtest.h"

#include <string>
#include <variant>

namespace {

using reloc::costmodel::CostModel;

const char *kMinimal = R"(# costmodel calibration v0
# machine: testbox
pcie.h2d_gbps 13.07
cpu.t8.contiguous.quantize_pack_gbps 22.76  # trailing comment
hbm.bw_gbps 544
)";

CostModel mustParse(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<CostModel>(r)) << std::get<std::string>(r);
  return std::get<CostModel>(r);
}

std::string mustFail(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<std::string>(r));
  return std::holds_alternative<std::string>(r) ? std::get<std::string>(r)
                                                : std::string();
}

TEST(CostModelParse, MinimalFileRoundTrips) {
  CostModel m = mustParse(kMinimal);
  EXPECT_EQ(m.machine(), "testbox");
  EXPECT_TRUE(m.has("pcie.h2d_gbps"));
  EXPECT_DOUBLE_EQ(m.at("pcie.h2d_gbps"), 13.07);
  EXPECT_DOUBLE_EQ(m.at("cpu.t8.contiguous.quantize_pack_gbps"), 22.76);
  EXPECT_FALSE(m.has("absent.key"));
  EXPECT_DOUBLE_EQ(m.get("absent.key", -1.0), -1.0);
}

TEST(CostModelParse, RejectsMissingOrWrongVersionHeader) {
  EXPECT_NE(mustFail("pcie.h2d_gbps 13.07\n").find("version"),
            std::string::npos);
  EXPECT_NE(mustFail("# costmodel calibration v1\nk 1\n").find("version"),
            std::string::npos);
}

TEST(CostModelParse, RejectsMalformedLines) {
  const std::string base = "# costmodel calibration v0\n";
  // No value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps\n").find("line 2"),
            std::string::npos);
  // Non-numeric value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps fast\n").find("line 2"),
            std::string::npos);
  // Trailing junk after the value that is not a comment.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps 13.07 junk\n").find("line 2"),
            std::string::npos);
}

TEST(CostModelParse, RejectsDuplicateKeys) {
  const std::string text = "# costmodel calibration v0\n"
                           "pcie.h2d_gbps 13.07\n"
                           "pcie.h2d_gbps 26.79\n";
  EXPECT_NE(mustFail(text).find("duplicate"), std::string::npos);
}

TEST(CostModelParse, LoadReportsUnreadablePath) {
  auto r = CostModel::load("/nonexistent/path.cal");
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
}

} // namespace
