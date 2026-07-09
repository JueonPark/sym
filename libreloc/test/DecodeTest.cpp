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

// Copied verbatim from serialize.mlir's `plan_hex(pad):` CHECK line.
const char *kPadHex = "52504c4e00000000000000000100000001000000011e000000000000"
                      "00000000000100000001"
                      "00000000000000000020000000010000000100000001200000000000"
                      "00000000000001000000"
                      "01000000000000000000200000000100000000000000010000000100"
                      "00007801000000011e00"
                      "00000000000001000000010100000000000000010000000101000000"
                      "00000000010000000000"
                      "00000100000001000000000000000001000000010200000000000000"
                      "00200000000000000000"
                      "00000000000000010000000000000040000000000000000000000000"
                      "00010000000100000001"
                      "0000000700000000";

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

// Reference plan (build doc §2.1): symbolic N, divisibility, contiguity.
// #reloc.plan<src = tensor<[N, N], f32, strides = [N, 1]>,
//             dst = tensor<[N floordiv 64, 64, 64, N floordiv 64], f32>,
//             perm = [1, 0, 2, 3],
//             axes = [{name = "n0", extent = N floordiv 64, src_stride = 64,
//                      dst_stride = 4096 * (N floordiv 64)},
//                     {name = "b0", extent = 64, src_stride = 64 * N,
//                      dst_stride = 64 * (N floordiv 64)},
//                     {name = "b1", extent = 64, src_stride = N,
//                      dst_stride = N floordiv 64},
//                     {name = "n1", extent = N floordiv 64, src_stride = 1,
//                      dst_stride = 1}],
//             constraints = {divisible(N, 64),
//                             contiguous = [false, false, false, true],
//                             no_copy = false},
//             inverse = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>>
TEST(Decode, ReferenceGoldenDecodesStructurally) {
  RelocationPlan plan = decodeOk(fromHex(kReferenceHex));

  ASSERT_EQ(plan.symbols.size(), 1u);
  EXPECT_EQ(plan.symbols[0], "N");

  // N floordiv 64, as a postfix stream: PUSH_SYM 0, PUSH_CONST 64, FLOORDIV.
  ExprStream nFloorDiv64 = {ExprToken{ExprOp::PushSym, 0},
                            ExprToken{ExprOp::PushConst, 64},
                            ExprToken{ExprOp::FloorDiv, 0}};

  ASSERT_EQ(plan.src.extents.size(), 2u);
  expectStreamEq(plan.src.extents[0], symbol(0), "src extent 0");
  expectStreamEq(plan.src.extents[1], symbol(0), "src extent 1");
  ASSERT_EQ(plan.src.strides.size(), 2u); // strides = [N, 1] (explicit)
  expectStreamEq(plan.src.strides[0], symbol(0), "src stride 0");
  expectStreamEq(plan.src.strides[1], constant(1), "src stride 1");
  expectStreamEq(plan.src.offset, constant(0), "src offset");
  EXPECT_EQ(plan.src.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.src.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.dst.extents.size(), 4u);
  expectStreamEq(plan.dst.extents[0], nFloorDiv64, "dst extent 0");
  expectStreamEq(plan.dst.extents[1], constant(64), "dst extent 1");
  expectStreamEq(plan.dst.extents[2], constant(64), "dst extent 2");
  expectStreamEq(plan.dst.extents[3], nFloorDiv64, "dst extent 3");
  EXPECT_TRUE(plan.dst.strides.empty()); // canonical row-major (no strides=)
  expectStreamEq(plan.dst.offset, constant(0), "dst offset");
  EXPECT_EQ(plan.dst.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.dst.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.perm.size(), 4u);
  EXPECT_EQ(plan.perm[0], 1u);
  EXPECT_EQ(plan.perm[1], 0u);
  EXPECT_EQ(plan.perm[2], 2u);
  EXPECT_EQ(plan.perm[3], 3u);

  ASSERT_EQ(plan.axes.size(), 4u);

  EXPECT_EQ(plan.axes[0].name, "n0");
  expectStreamEq(plan.axes[0].extent, nFloorDiv64, "axis n0 extent");
  expectStreamEq(plan.axes[0].srcStride, constant(64), "axis n0 src stride");
  // 4096 * (N floordiv 64): PUSH_CONST 4096, [nFloorDiv64], MUL.
  ExprStream n0Dst = {
      ExprToken{ExprOp::PushConst, 4096}, ExprToken{ExprOp::PushSym, 0},
      ExprToken{ExprOp::PushConst, 64}, ExprToken{ExprOp::FloorDiv, 0},
      ExprToken{ExprOp::Mul, 0}};
  expectStreamEq(plan.axes[0].dstStride, n0Dst, "axis n0 dst stride");

  EXPECT_EQ(plan.axes[1].name, "b0");
  expectStreamEq(plan.axes[1].extent, constant(64), "axis b0 extent");
  // 64 * N: PUSH_CONST 64, PUSH_SYM 0, MUL.
  ExprStream b0Src = {ExprToken{ExprOp::PushConst, 64},
                      ExprToken{ExprOp::PushSym, 0}, ExprToken{ExprOp::Mul, 0}};
  expectStreamEq(plan.axes[1].srcStride, b0Src, "axis b0 src stride");
  // 64 * (N floordiv 64): PUSH_CONST 64, [nFloorDiv64], MUL.
  ExprStream b0Dst = {
      ExprToken{ExprOp::PushConst, 64}, ExprToken{ExprOp::PushSym, 0},
      ExprToken{ExprOp::PushConst, 64}, ExprToken{ExprOp::FloorDiv, 0},
      ExprToken{ExprOp::Mul, 0}};
  expectStreamEq(plan.axes[1].dstStride, b0Dst, "axis b0 dst stride");

  EXPECT_EQ(plan.axes[2].name, "b1");
  expectStreamEq(plan.axes[2].extent, constant(64), "axis b1 extent");
  expectStreamEq(plan.axes[2].srcStride, symbol(0), "axis b1 src stride");
  expectStreamEq(plan.axes[2].dstStride, nFloorDiv64, "axis b1 dst stride");

  EXPECT_EQ(plan.axes[3].name, "n1");
  expectStreamEq(plan.axes[3].extent, nFloorDiv64, "axis n1 extent");
  expectStreamEq(plan.axes[3].srcStride, constant(1), "axis n1 src stride");
  expectStreamEq(plan.axes[3].dstStride, constant(1), "axis n1 dst stride");

  EXPECT_TRUE(plan.padFill.empty());

  // Mandatory named asserts (issue #42 acceptance highlights).
  ASSERT_EQ(plan.divisibility.size(), 1u);
  expectStreamEq(plan.divisibility[0].expr, symbol(0), "divisibility expr");
  EXPECT_EQ(plan.divisibility[0].divisor, 64);
  ASSERT_EQ(plan.symbols.size(), 1u);
  EXPECT_EQ(plan.symbols[0], "N");

  EXPECT_TRUE(plan.alignment.empty());

  ASSERT_EQ(plan.contiguity.size(), 4u); // [false, false, false, true]
  EXPECT_EQ(plan.contiguity[0], 0u);
  EXPECT_EQ(plan.contiguity[1], 0u);
  EXPECT_EQ(plan.contiguity[2], 0u);
  EXPECT_EQ(plan.contiguity[3], 1u);

  EXPECT_FALSE(plan.noCopy);
  EXPECT_FALSE(plan.runtimePadCheck);

  // inverse = (d0, d1, d2, d3) -> (d1, d0, d2, d3).
  ASSERT_EQ(plan.inverse.size(), 4u);
  expectStreamEq(plan.inverse[0], dim(1), "inverse result 0");
  expectStreamEq(plan.inverse[1], dim(0), "inverse result 1");
  expectStreamEq(plan.inverse[2], dim(2), "inverse result 2");
  expectStreamEq(plan.inverse[3], dim(3), "inverse result 3");
}

// Pad plan: exercises typed_value (f32 fill) and alignment.
// #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0],
//             axes = [{name = "x", extent = 30, src_stride = 1,
//                      dst_stride = 1}],
//             pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}],
//             constraints = {align(0, 64), no_copy = false},
//             inverse = affine_map<(d0) -> (d0)>>
TEST(Decode, PadGoldenDecodesStructurally) {
  RelocationPlan plan = decodeOk(fromHex(kPadHex));

  EXPECT_TRUE(plan.symbols.empty());

  ASSERT_EQ(plan.src.extents.size(), 1u);
  expectStreamEq(plan.src.extents[0], constant(30), "src extent");
  EXPECT_TRUE(plan.src.strides.empty());
  expectStreamEq(plan.src.offset, constant(0), "src offset");
  EXPECT_EQ(plan.src.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.src.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.dst.extents.size(), 1u);
  expectStreamEq(plan.dst.extents[0], constant(32), "dst extent");
  EXPECT_TRUE(plan.dst.strides.empty());
  expectStreamEq(plan.dst.offset, constant(0), "dst offset");
  EXPECT_EQ(plan.dst.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.dst.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.perm.size(), 1u);
  EXPECT_EQ(plan.perm[0], 0u);

  ASSERT_EQ(plan.axes.size(), 1u);
  EXPECT_EQ(plan.axes[0].name, "x");
  expectStreamEq(plan.axes[0].extent, constant(30), "axis extent");
  expectStreamEq(plan.axes[0].srcStride, constant(1), "axis src stride");
  expectStreamEq(plan.axes[0].dstStride, constant(1), "axis dst stride");

  ASSERT_EQ(plan.padFill.size(), 1u);
  EXPECT_EQ(plan.padFill[0].dstAxis, 0u);
  expectStreamEq(plan.padFill[0].lo, constant(0), "pad lo");
  expectStreamEq(plan.padFill[0].hi, constant(2), "pad hi");
  EXPECT_EQ(plan.padFill[0].fillType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.padFill[0].fillType.bitwidth, 32u);
  EXPECT_EQ(plan.padFill[0].fillBits, 0u); // 0.0f bit pattern is all zero.

  EXPECT_TRUE(plan.divisibility.empty());

  ASSERT_EQ(plan.alignment.size(), 1u); // align(0, 64)
  EXPECT_EQ(plan.alignment[0].axis, 0u);
  EXPECT_EQ(plan.alignment[0].bytes, 64);

  EXPECT_TRUE(plan.contiguity.empty());
  EXPECT_FALSE(plan.noCopy);
  EXPECT_FALSE(plan.runtimePadCheck);

  ASSERT_EQ(plan.inverse.size(), 1u);
  expectStreamEq(plan.inverse[0], dim(0), "inverse result 0");
}

// Degraded plan: runtime_pad_check auto-set at parse; symbols in pads.
// #reloc.plan<src = tensor<[N], f32>,
//             dst = tensor<[64 * ((N + 63) floordiv 64)], f32>, perm = [0],
//             axes = [{name = "x", extent = N, src_stride = 1,
//                      dst_stride = 1}],
//             pad_fill = [{dst_axis = 0, lo = 0,
//                          hi = (64 * ((N + 63) floordiv 64)) - N,
//                          value = 0.0 : f32}],
//             inverse = affine_map<(d0) -> (d0)>>
// The attribute never spells `runtime_pad_check = true`; the encoder sets
// it because the pad width `hi` is symbolic (only known at runtime), which
// is exactly what the mandatory named assert below pins.
TEST(Decode, DegradedGoldenDecodesStructurally) {
  RelocationPlan plan = decodeOk(fromHex(kDegradedHex));

  ASSERT_EQ(plan.symbols.size(), 1u);
  EXPECT_EQ(plan.symbols[0], "N");

  ASSERT_EQ(plan.src.extents.size(), 1u);
  expectStreamEq(plan.src.extents[0], symbol(0), "src extent");
  EXPECT_TRUE(plan.src.strides.empty());
  expectStreamEq(plan.src.offset, constant(0), "src offset");
  EXPECT_EQ(plan.src.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.src.elementType.bitwidth, 32u);

  // 64 * ((N + 63) floordiv 64), postfix: PUSH_CONST 64, PUSH_SYM 0,
  // PUSH_CONST 63, ADD, PUSH_CONST 64, FLOORDIV, MUL.
  ExprStream dstExtent = {
      ExprToken{ExprOp::PushConst, 64}, ExprToken{ExprOp::PushSym, 0},
      ExprToken{ExprOp::PushConst, 63}, ExprToken{ExprOp::Add, 0},
      ExprToken{ExprOp::PushConst, 64}, ExprToken{ExprOp::FloorDiv, 0},
      ExprToken{ExprOp::Mul, 0}};
  ASSERT_EQ(plan.dst.extents.size(), 1u);
  expectStreamEq(plan.dst.extents[0], dstExtent, "dst extent");
  EXPECT_TRUE(plan.dst.strides.empty());
  expectStreamEq(plan.dst.offset, constant(0), "dst offset");
  EXPECT_EQ(plan.dst.elementType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.dst.elementType.bitwidth, 32u);

  ASSERT_EQ(plan.perm.size(), 1u);
  EXPECT_EQ(plan.perm[0], 0u);

  ASSERT_EQ(plan.axes.size(), 1u);
  EXPECT_EQ(plan.axes[0].name, "x");
  expectStreamEq(plan.axes[0].extent, symbol(0), "axis extent");
  expectStreamEq(plan.axes[0].srcStride, constant(1), "axis src stride");
  expectStreamEq(plan.axes[0].dstStride, constant(1), "axis dst stride");

  ASSERT_EQ(plan.padFill.size(), 1u);
  EXPECT_EQ(plan.padFill[0].dstAxis, 0u);
  expectStreamEq(plan.padFill[0].lo, constant(0), "pad lo");
  // hi = (64 * ((N + 63) floordiv 64)) - N: [dstExtent], PUSH_SYM 0, SUB.
  ExprStream padHi = dstExtent;
  padHi.push_back(ExprToken{ExprOp::PushSym, 0});
  padHi.push_back(ExprToken{ExprOp::Sub, 0});
  expectStreamEq(plan.padFill[0].hi, padHi, "pad hi");
  EXPECT_EQ(plan.padFill[0].fillType.kind, ElementTypeKind::Float);
  EXPECT_EQ(plan.padFill[0].fillType.bitwidth, 32u);
  EXPECT_EQ(plan.padFill[0].fillBits, 0u); // 0.0f bit pattern is all zero.

  EXPECT_TRUE(plan.divisibility.empty());
  EXPECT_TRUE(plan.alignment.empty());
  EXPECT_TRUE(plan.contiguity.empty());
  EXPECT_FALSE(plan.noCopy);
  // Mandatory named assert (issue #42 acceptance highlights).
  EXPECT_TRUE(plan.runtimePadCheck);

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

// ---------------------------------------------------------------------------
// Negative corpus: one test per validation rule not already covered above.
//
// Offsets below are derived from the identity golden's byte layout, walked
// field-by-field per docs/reloc-plan-format.md's worked example (which
// itself annotates every field's position for the identity plan):
//   magic          @0    (4B)
//   version        @4    (4B)
//   symbol count   @8    (4B, = 0 -> no symbol entries)
//   src.rank       @12   (4B, = 1)
//   src.extent[0]: op_count @16 (4B) opcode @20 (1B) operand @21..28 (8B)
//   src.stride_count        @29 (4B, = 0)
//   src.offset:    op_count @33 (4B) opcode @37 (1B) operand @38..45 (8B)
//   src.elem_type: kind     @46 (1B) bitwidth @47..50 (4B)
//   dst.rank       @51   (4B, = 1)
//   dst.extent[0]: op_count @55 (4B) opcode @59 (1B) operand @60..67 (8B)
//   dst.stride_count        @68 (4B, = 0)
//   dst.offset:    op_count @72 (4B) opcode @76 (1B) operand @77..84 (8B)
//   dst.elem_type: kind     @85 (1B) bitwidth @86..89 (4B)
//   perm count     @90   (4B, = 1)
//   perm[0]        @94   (4B, = 0)
//   axes count     @98   (4B, = 1)
//   axis[0].name   @102  (4B len + 1B "x")
//   axis[0].extent:     op_count @107 (4B) opcode @111 (1B) operand @112..119
//   axis[0].src_stride: op_count @120 (4B) opcode @124 (1B) operand @125..132
//   axis[0].dst_stride: op_count @133 (4B) opcode @137 (1B) operand @138..145
//   pad count      @146  (4B, = 0)
//   divisibility count @150 (4B, = 0)
//   alignment count     @154 (4B, = 0)
//   contiguity count    @158 (4B, = 0)
//   no_copy        @162  (1B)
//   runtime_pad_check @163 (1B)
//   inverse.num_dims    @164 (4B, = 1)
//   inverse.num_results @168 (4B, = 1)
//   inverse.result[0]: op_count @172 (4B) opcode @176 (1B) operand @177..180
// These arithmetic derivations were cross-checked against a from-scratch
// re-implementation of the section walk over all four goldens; every
// computed offset matched the byte where the real decoder actually failed.

// Byte-patch a golden at a fixed offset (issue #42 brief's mutation
// helper).
std::vector<uint8_t> patched(const char *hex, size_t at,
                             std::initializer_list<uint8_t> bytes) {
  std::vector<uint8_t> buffer = fromHex(hex);
  size_t i = at;
  for (uint8_t b : bytes)
    buffer[i++] = b;
  return buffer;
}

// Little-endian byte appenders for the hand-built buffers below.
void putU32(std::vector<uint8_t> &buf, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void putI64(std::vector<uint8_t> &buf, int64_t v) {
  uint64_t u = static_cast<uint64_t>(v);
  for (int i = 0; i < 8; ++i)
    buf.push_back(static_cast<uint8_t>(u >> (8 * i)));
}

// magic "RPLN" + version 0 + symbol table count 0 (12 bytes: offsets 0-11).
std::vector<uint8_t> buildHeader() {
  std::vector<uint8_t> buf{'R', 'P', 'L', 'N'};
  putU32(buf, 0); // version
  putU32(buf, 0); // symbol table count
  return buf;
}

TEST(Decode, RejectsBadStrideCount) {
  // Patch src.stride_count @29 (rank=1) from 0 to 2: neither 0 (canonical)
  // nor == rank (1), so the stride_count/rank cross-check must reject it.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 29, {2, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 29u);
  EXPECT_NE(error.message.find("stride"), std::string::npos);
}

TEST(Decode, RejectsEmptyExpressionStream) {
  // Patch src.extent[0]'s op_count @16 from 1 to 0: a valid expr stream
  // must have op_count >= 1.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 16, {0, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 16u);
  EXPECT_NE(error.message.find("empty expression"), std::string::npos);
}

TEST(Decode, RejectsStackUnderflow) {
  // Patch src.offset's opcode @37 from PUSH_CONST (0x01) to ADD (0x02).
  // The expr has just started (depth 0); ADD needs depth >= 2, so the
  // underflow check must fire before the (now-garbage) operand bytes are
  // even interpreted.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 37, {0x02});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 37u);
  EXPECT_NE(error.message.find("underflow"), std::string::npos);
}

TEST(Decode, RejectsUnknownOpcode) {
  // Same byte as above (@37), but to an opcode outside 0x00-0x07.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 37, {0x99});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 37u);
  EXPECT_NE(error.message.find("unknown"), std::string::npos);
}

TEST(Decode, RejectsStackSurplus) {
  // Hand-built: header (offsets 0-11) + src rank=1 (offset 12-15) + a
  // 2-op extent stream [PUSH_CONST 1, PUSH_CONST 2] (op_count @16) that
  // leaves depth 2, not 1, on the stack.
  std::vector<uint8_t> bytes = buildHeader();
  putU32(bytes, 1); // src.rank = 1
  putU32(bytes, 2); // extent[0].op_count = 2
  bytes.push_back(0x01);
  putI64(bytes, 1); // PUSH_CONST 1
  bytes.push_back(0x01);
  putI64(bytes, 2); // PUSH_CONST 2
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 16u); // expr start = 4+4+4+4 (header + rank)
  EXPECT_NE(error.message.find("exactly one value"), std::string::npos);
}

TEST(Decode, RejectsPushSymOutOfRange) {
  // Hand-built: empty symbol table (symbolCount = 0) + a 1-op extent
  // stream [PUSH_SYM 0]; index 0 is out of range against 0 symbols.
  std::vector<uint8_t> bytes = buildHeader();
  putU32(bytes, 1); // src.rank = 1
  putU32(bytes, 1); // extent[0].op_count = 1
  bytes.push_back(0x00);
  putU32(bytes, 0); // PUSH_SYM 0
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 20u); // opcode = 16 (op_count) + 4
  EXPECT_NE(error.message.find("out of range"), std::string::npos);
}

TEST(Decode, RejectsPushDimInPlanContext) {
  // Hand-built: a 1-op extent stream [PUSH_DIM 0] in the src descriptor
  // (Plan context); PUSH_DIM is only legal inside inverse expressions.
  std::vector<uint8_t> bytes = buildHeader();
  putU32(bytes, 1); // src.rank = 1
  putU32(bytes, 1); // extent[0].op_count = 1
  bytes.push_back(0x07);
  putU32(bytes, 0); // PUSH_DIM 0
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 20u);
  EXPECT_NE(error.message.find("only allowed in inverse"), std::string::npos);
}

TEST(Decode, RejectsPushSymInInverse) {
  // Patch the identity inverse result[0]'s opcode @176 from PUSH_DIM
  // (0x07) to PUSH_SYM (0x00); PUSH_SYM is never allowed in inverse
  // expressions (v0). The 4 operand bytes that follow (originally dim
  // index 0) are reused unchanged as a would-be symbol index.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 176, {0x00});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 176u);
  EXPECT_NE(error.message.find("not allowed in inverse"), std::string::npos);
}

TEST(Decode, RejectsBadElementKind) {
  // Patch src.elem_type's kind byte @46 (12 symbol-table + 4 rank + 13
  // extent + 4 stride_count + 13 offset = 46) from 2 (integer) to 9,
  // which is not one of the four defined kinds.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 46, {9});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 46u);
  EXPECT_NE(error.message.find("element type"), std::string::npos);
}

TEST(Decode, RejectsBadBitwidthCombo) {
  // Same kind byte @46, set to 1 (bfloat) while leaving the existing
  // bitwidth bytes (32) untouched: bfloat only accepts bitwidth 16.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 46, {1});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 46u);
  EXPECT_NE(error.message.find("element type"), std::string::npos);
}

TEST(Decode, RejectsPermOutOfRange) {
  // Patch identity's single perm entry @94 from 0 to 5; axis count is 1,
  // so 5 is out of [0, axis count). The cross-check reports the perm
  // SECTION start offset (@90 = perm count), not the entry offset.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 94, {5, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 90u);
  EXPECT_NE(error.message.find("permutation"), std::string::npos);
}

TEST(Decode, RejectsInverseArityMismatch) {
  // Patch inverse.num_dims @164 from 1 to 2; axis count is 1, so num_dims
  // must equal 1.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 164, {2, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 164u);
  EXPECT_NE(error.message.find("inverse"), std::string::npos);
}

TEST(Decode, RejectsBadContiguityByte) {
  // The identity golden has contiguity count 0 (nothing to patch), so use
  // the reference golden, whose contiguity section holds 4 real flag
  // bytes. contiguity[0] is at offset 462 (derived by walking the same
  // section order as above through the reference plan's larger src/dst
  // descriptors, axes, and divisibility section; contiguity count is at
  // 458, so entry 0 follows immediately at 462). Patch it to 2.
  std::vector<uint8_t> bytes = patched(kReferenceHex, 462, {2});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 462u);
  EXPECT_NE(error.message.find("contiguity"), std::string::npos);
}

TEST(Decode, RejectsBadFlagByte) {
  // Patch identity's no_copy flag @162 (12 symtable + 39 src + 39 dst + 8
  // perm + 48 axes + 4 pad + 4 div + 4 align + 4 contiguity = 162) to 2;
  // both flag bytes must be 0 or 1.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 162, {2});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 162u);
  EXPECT_NE(error.message.find("no_copy"), std::string::npos);
}

TEST(Decode, RejectsPadFillDstAxisOutOfRange) {
  // The pad golden's pad_fill section: pad count @146 (same prefix layout
  // as identity through the axes section: 12 symtable + 39 src + 39 dst +
  // 8 perm + 48 axes = 146), so pad[0].dst_axis is @150. Patch it from 0
  // to 5; axis count is 1, so 5 is out of range.
  std::vector<uint8_t> bytes = patched(kPadHex, 150, {5, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 150u);
  EXPECT_NE(error.message.find("dst_axis"), std::string::npos);
}

TEST(Decode, RejectsAlignmentAxisOutOfRange) {
  // The pad golden's alignment section: after pad_fill count @146 comes
  // one 47-byte pad entry (4 dst_axis + 13 lo + 13 hi + 5 fill type + 8
  // fill bits), so divisibility count is @197 - 4 = 193, alignment count
  // @197, and align[0].axis @201. Patch it from 0 to 5; axis count is 1.
  std::vector<uint8_t> bytes = patched(kPadHex, 201, {5, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 201u);
  EXPECT_NE(error.message.find("alignment axis"), std::string::npos);
}

TEST(Decode, RejectsPushDimIndexOutOfRange) {
  // Patch the identity inverse result[0]'s PUSH_DIM operand (u32 at
  // @177..180, right after the opcode @176) from 0 to 5; num_dims is 1,
  // so dim index 5 is out of range. The error is reported at the opcode's
  // offset (176), matching parseExpr's opAt convention.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 177, {5, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 176u);
  EXPECT_NE(error.message.find("dim index out of range"), std::string::npos);
}

TEST(Decode, RejectsPermSizeMismatch) {
  // Distinct from the bijection check: perm COUNT != axis count. Patching
  // the identity perm count would shift every later section, so hand-build
  // a buffer with an EMPTY perm section but one axis. Layout: header
  // (0-11) + rank-0 src desc @12 (4 rank + 4 stride_count + 13 offset + 5
  // elem = 26B, ends @38) + rank-0 dst desc @38 (26B, ends @64) + perm
  // count 0 @64 + axes count 1 @68 + one axis @72 (5B name "x" + 3x13B
  // exprs = 44B, ends @116). The size cross-check fires after the axes
  // section but reports the perm SECTION offset (@64), like the bijection
  // check.
  std::vector<uint8_t> bytes = buildHeader();
  auto putDesc = [&bytes]() {
    putU32(bytes, 0); // rank 0 (no extents)
    putU32(bytes, 0); // stride_count 0
    putU32(bytes, 1); // offset expr: op_count 1
    bytes.push_back(0x01);
    putI64(bytes, 0);      // PUSH_CONST 0
    bytes.push_back(0x02); // elem kind 2 (int)
    putU32(bytes, 32);     // bitwidth 32
  };
  putDesc();        // src
  putDesc();        // dst
  putU32(bytes, 0); // perm count 0
  putU32(bytes, 1); // axes count 1
  putU32(bytes, 1); // axis name length 1
  bytes.push_back('x');
  for (int i = 0; i < 3; ++i) { // extent, src_stride, dst_stride
    putU32(bytes, 1);           // op_count 1
    bytes.push_back(0x01);
    putI64(bytes, 1); // PUSH_CONST 1
  }
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 64u);
  EXPECT_NE(error.message.find("perm size"), std::string::npos);
}

TEST(Decode, RejectsBadContiguityCount) {
  // Patch the reference golden's contiguity count @458 (derived from the
  // same section walk as RejectsBadContiguityByte: entry 0 is @462, so
  // the count u32 is @458) from 4 to 2: nonzero and != the axis count
  // (4). The count/axis-count check fires before any flag byte is read.
  std::vector<uint8_t> bytes = patched(kReferenceHex, 458, {2, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 458u);
  EXPECT_NE(error.message.find("contiguity count"), std::string::npos);
}

TEST(Decode, RejectsBadRuntimePadCheckFlag) {
  // Patch identity's runtime_pad_check flag @163 (= no_copy @162 + 1) to
  // 2; distinct byte and message from the no_copy case above.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 163, {2});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 163u);
  EXPECT_NE(error.message.find("runtime_pad_check"), std::string::npos);
}

TEST(Decode, RejectsInverseNumResultsMismatch) {
  // Patch identity's inverse num_results @168 (= num_dims @164 + 4) from
  // 1 to 2; axis count is 1. Distinct from the num_dims case above, which
  // leaves num_results untouched.
  std::vector<uint8_t> bytes = patched(kIdentityHex, 168, {2, 0, 0, 0});
  DecodeError error = decodeErr(bytes);
  EXPECT_EQ(error.offset, 168u);
  EXPECT_NE(error.message.find("num_results"), std::string::npos);
}

// Deterministic (seeded) hostile-input sweep over all four goldens: random
// truncations and bit flips/bursts. Success criterion is "no crash / no
// sanitizer report" (also exercised standalone under ASan+UBSan, see the
// commit's report); the decode result value is irrelevant.
TEST(Decode, FuzzTruncationAndBitFlipsNoCrash) {
  const char *goldens[] = {kIdentityHex, kReferenceHex, kPadHex, kDegradedHex};
  uint64_t state = 0x5DEECE66DULL; // fixed seed: reproducible corpus
  auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  };
  for (const char *hex : goldens) {
    const std::vector<uint8_t> golden = fromHex(hex);
    for (int i = 0; i < 25000; ++i) {
      std::vector<uint8_t> mutated = golden;
      switch (next() % 3) {
      case 0: // truncate
        mutated.resize(next() % (mutated.size() + 1));
        break;
      case 1: // single bit flip
        mutated[next() % mutated.size()] ^= 1u << (next() % 8);
        break;
      default: // burst: flip up to 8 bytes
        for (int b = 0; b < 8; ++b)
          mutated[next() % mutated.size()] ^= static_cast<uint8_t>(next());
      }
      (void)decodePlan(mutated.data(), mutated.size());
    }
  }
}

} // namespace
