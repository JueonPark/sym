//===- DecodeTest.cpp - wire-format v0 decoder tests ----------------------===//
//
// Golden blobs are copied VERBATIM from test/dialect/reloc/serialize.mlir
// (the lit-frozen encoder output). Structural expectations follow
// docs/reloc-plan-format.md; the identity plan's expected struct is the
// format doc's annotated worked example.
//
//===----------------------------------------------------------------------===//

#include "reloc/Decode.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using reloc::DecodeError;
using reloc::decodePlan;
using reloc::ElementType;
using reloc::ElementTypeKind;
using reloc::ExprOp;
using reloc::ExprStream;
using reloc::ExprToken;
using reloc::RelocationPlan;

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

// Copied verbatim from serialize.mlir's `plan_hex(identity):` CHECK line.
const char *kIdentityHex =
    "52504c4e000000000000000001000000010000000108000000000000000000000001000000"
    "01000000000000000002200000000100000001000000010800000000000000000000000100"
    "00000100000000000000000220000000010000000000000001000000010000007801000000"
    "01080000000000000001000000010100000000000000010000000101000000000000000000"
    "000000000000000000000000000000000100000001000000010000000700000000";

RelocationPlan decodeOk(const std::vector<uint8_t> &bytes) {
  reloc::DecodeResult result = decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<RelocationPlan>(&result);
  EXPECT_NE(plan, nullptr) << "decode failed: "
                           << (std::get_if<DecodeError>(&result)
                                   ? std::get_if<DecodeError>(&result)->message
                                   : "<no error>");
  return plan ? *plan : RelocationPlan{};
}

DecodeError decodeErr(const std::vector<uint8_t> &bytes) {
  reloc::DecodeResult result = decodePlan(bytes.data(), bytes.size());
  auto *error = std::get_if<DecodeError>(&result);
  EXPECT_NE(error, nullptr) << "decode unexpectedly succeeded";
  return error ? *error : DecodeError{};
}

ExprStream constant(int64_t value) {
  return {ExprToken{ExprOp::PushConst, value}};
}
ExprStream symbol(uint32_t index) {
  return {ExprToken{ExprOp::PushSym, index}};
}
ExprStream dim(uint32_t index) { return {ExprToken{ExprOp::PushDim, index}}; }

void expectStreamEq(const ExprStream &actual, const ExprStream &expected,
                    const std::string &what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].op, expected[i].op) << what << " token " << i;
    EXPECT_EQ(actual[i].value, expected[i].value) << what << " token " << i;
  }
}

TEST(Decode, IdentityGoldenDecodesStructurally) {
  RelocationPlan plan = decodeOk(fromHex(kIdentityHex));

  EXPECT_TRUE(plan.symbols.empty());

  ASSERT_EQ(plan.src.extents.size(), 1u);
  expectStreamEq(plan.src.extents[0], constant(8), "src extent");
  EXPECT_TRUE(plan.src.strides.empty()); // canonical row-major
  expectStreamEq(plan.src.offset, constant(0), "src offset");
  EXPECT_EQ(plan.src.elementType.kind, ElementTypeKind::Integer);
  EXPECT_EQ(plan.src.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.dst.extents.size(), 1u);
  expectStreamEq(plan.dst.extents[0], constant(8), "dst extent");
  EXPECT_TRUE(plan.dst.strides.empty());

  ASSERT_EQ(plan.perm.size(), 1u);
  EXPECT_EQ(plan.perm[0], 0u);

  ASSERT_EQ(plan.axes.size(), 1u);
  EXPECT_EQ(plan.axes[0].name, "x");
  expectStreamEq(plan.axes[0].extent, constant(8), "axis extent");
  expectStreamEq(plan.axes[0].srcStride, constant(1), "axis src stride");
  expectStreamEq(plan.axes[0].dstStride, constant(1), "axis dst stride");

  EXPECT_TRUE(plan.padFill.empty());
  EXPECT_TRUE(plan.divisibility.empty());
  EXPECT_TRUE(plan.alignment.empty());
  EXPECT_TRUE(plan.contiguity.empty());
  EXPECT_FALSE(plan.noCopy);
  EXPECT_FALSE(plan.runtimePadCheck);

  ASSERT_EQ(plan.inverse.size(), 1u);
  expectStreamEq(plan.inverse[0], dim(0), "inverse result 0");
}

TEST(Decode, RejectsBadMagic) {
  std::vector<uint8_t> bytes = fromHex(kIdentityHex);
  bytes[0] = 'X';
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 0u);
  EXPECT_NE(error.message.find("magic"), std::string::npos);
}

TEST(Decode, RejectsWrongVersion) {
  std::vector<uint8_t> bytes = fromHex(kIdentityHex);
  bytes[4] = 1; // version u32 little-endian at offset 4
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 4u);
  EXPECT_NE(error.message.find("version"), std::string::npos);
}

TEST(Decode, RejectsTruncation) {
  std::vector<uint8_t> bytes = fromHex(kIdentityHex);
  bytes.resize(bytes.size() / 2);
  DecodeError error = decodeErr(bytes);
  EXPECT_NE(error.message.find("truncated"), std::string::npos);
}

TEST(Decode, RejectsTrailingBytes) {
  std::vector<uint8_t> bytes = fromHex(kIdentityHex);
  bytes.push_back(0);
  DecodeError error = decodeErr(bytes);
  EXPECT_NE(error.message.find("trailing"), std::string::npos);
}

TEST(Decode, RejectsHostileCountWithoutAllocating) {
  // Symbol-table count 0xFFFFFFFF at offset 8: must be rejected against
  // the remaining-bytes budget BEFORE any reserve happens.
  std::vector<uint8_t> bytes = fromHex(kIdentityHex);
  bytes[8] = 0xff;
  bytes[9] = 0xff;
  bytes[10] = 0xff;
  bytes[11] = 0xff;
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 8u);
  EXPECT_NE(error.message.find("count"), std::string::npos);
}

} // namespace
