//===- TypedDecodeTest.cpp - wire-format v1 typed decoder tests -----------===//
//
// Golden blobs are copied VERBATIM from test/dialect/reloc/typed_serialize.mlir
// (the lit-frozen typed encoder output). Structural expectations follow
// docs/reloc-plan-format.md "Wire Format v1". Negative fixtures mutate the
// pad_quantize golden at offsets derived from its section walk; every such
// test first asserts the byte it is about to change, so a layout drift is
// caught as a self-check failure rather than a misleading pass.
//
//===----------------------------------------------------------------------===//

#include "TypedGoldens.h"
#include "reloc/Decode.h"
#include "reloc/Quant.h"
#include "reloc/TypedValue.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using reloc::DecodeError;
using reloc::ElementType;
using reloc::ElementTypeKind;
using reloc::ExprOp;
using reloc::NumericPolicyKind;
using reloc::ParamKind;
using reloc::Signedness;
using reloc::TypedRelocationPlan;
using reloc::ValueTransformKind;

using typed_goldens::fromHex;

using typed_goldens::kDequantCastHex;
using typed_goldens::kPadQuantizeHex;
using typed_goldens::kQuantizeTransposeHex;

// The v0 identity golden (serialize.mlir), to prove the typed decoder never
// accepts v0.
const char *kIdentityV0Hex =
    "52504c4e000000000000000001000000010000000108000000000000000000000001000000"
    "01000000000000000002200000000100000001000000010800000000000000000000000100"
    "00000100000000000000000220000000010000000000000001000000010000007801000000"
    "01080000000000000001000000010100000000000000010000000101000000000000000000"
    "000000000000000000000000000000000100000001000000010000000700000000";

TypedRelocationPlan decodeTypedOk(const std::vector<uint8_t> &bytes) {
  auto result = reloc::decodeTypedPlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<TypedRelocationPlan>(&result);
  EXPECT_NE(plan, nullptr) << (std::get_if<DecodeError>(&result)
                                   ? std::get_if<DecodeError>(&result)->message
                                   : "<none>");
  return plan ? *plan : TypedRelocationPlan{};
}

DecodeError decodeTypedErr(const std::vector<uint8_t> &bytes) {
  auto result = reloc::decodeTypedPlan(bytes.data(), bytes.size());
  auto *error = std::get_if<DecodeError>(&result);
  EXPECT_NE(error, nullptr) << "typed decode unexpectedly succeeded";
  return error ? *error : DecodeError{};
}

bool isType(ElementType type, ElementTypeKind kind, uint32_t bitwidth) {
  return type.kind == kind && type.bitwidth == bitwidth;
}

// pad_quantize section walk (docs/reloc-plan-format.md v1): header 12,
// source desc 39 -> 51, result desc 39 -> 90, layout body: src 39 -> 129,
// dst 39 -> 168, perm 8 -> 176, axes 49 -> 225, pad_fill 47 -> 272,
// divisibility 4 -> 276, alignment 4 -> 280, contiguity 5 -> 285, flags 2
// -> 287, inverse 17 -> 304; stages: count -> 308, then transform, policy,
// input type (5) + signedness, output type (5) + signedness, rank u32 @322,
// shape expr (13) -> 339, scale param: kind @339, rank @340, type @341..345,
// count @346, value @350..357, zero_point kind @358, axis @359..366,
// has_channel @367; fills: count @368, dst_axis @372, stage @376, type
// @380..384, bits @385..392; total 393.
constexpr size_t kPadFusedBitsAt = 264;
constexpr size_t kStageCountAt = 304;
constexpr size_t kTransformAt = 308;
constexpr size_t kPolicyAt = 309;
constexpr size_t kInputSignAt = 315;
constexpr size_t kOutputSignAt = 321;
constexpr size_t kScaleKindAt = 339;
constexpr size_t kScaleCountAt = 346;
constexpr size_t kScaleBitsAt = 350;
constexpr size_t kAxisAt = 359;
constexpr size_t kHasChannelAt = 367;
constexpr size_t kFillDstAxisAt = 372;
constexpr size_t kFillStageAt = 376;
constexpr size_t kFillTypeAt = 380;
constexpr size_t kFillBitsAt = 385;

std::vector<uint8_t> padQuantize() {
  std::vector<uint8_t> bytes = fromHex(kPadQuantizeHex);
  // Self-check the walk before any mutation test relies on it.
  EXPECT_EQ(bytes.size(), 393u);
  EXPECT_EQ(bytes[kTransformAt], 1);        // quantize
  EXPECT_EQ(bytes[kPolicyAt], 2);           // symmetric_rne
  EXPECT_EQ(bytes[kInputSignAt], 0);        // f32 signless
  EXPECT_EQ(bytes[kOutputSignAt], 1);       // int8 signed
  EXPECT_EQ(bytes[kScaleKindAt], 1);        // inline
  EXPECT_EQ(bytes[kScaleCountAt], 1);       // one value
  EXPECT_EQ(bytes[kScaleBitsAt + 3], 0x3f); // 0.5f = 0x3f000000
  EXPECT_EQ(bytes[kAxisAt], 0xff);          // -1
  EXPECT_EQ(bytes[kHasChannelAt], 0);
  EXPECT_EQ(bytes[kFillStageAt], 0);
  EXPECT_EQ(bytes[kFillBitsAt + 3], 0x3f); // 1.0f = 0x3f800000
  EXPECT_EQ(bytes[kPadFusedBitsAt], 2);    // fused code 2
  return bytes;
}

void putU32(std::vector<uint8_t> &bytes, size_t at, uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes[at + i] = static_cast<uint8_t>(value >> (8 * i));
}
void putU64(std::vector<uint8_t> &bytes, size_t at, uint64_t value) {
  for (int i = 0; i < 8; ++i)
    bytes[at + i] = static_cast<uint8_t>(value >> (8 * i));
}

//===----------------------------------------------------------------------===//
// Goldens decode structurally
//===----------------------------------------------------------------------===//

TEST(TypedDecode, QuantizeTransposeGoldenDecodesStructurally) {
  TypedRelocationPlan plan = decodeTypedOk(fromHex(kQuantizeTransposeHex));
  ASSERT_EQ(plan.symbols, std::vector<std::string>{"B"});
  EXPECT_EQ(plan.layout.symbols, plan.symbols);
  ASSERT_EQ(plan.source.extents.size(), 2u);
  EXPECT_TRUE(isType(plan.source.elementType, ElementTypeKind::Float, 32));
  ASSERT_EQ(plan.result.extents.size(), 2u);
  EXPECT_TRUE(isType(plan.result.elementType, ElementTypeKind::Integer, 8));
  EXPECT_EQ(plan.layout.perm, (std::vector<uint32_t>{1, 0}));
  EXPECT_TRUE(isType(plan.layout.src.elementType, ElementTypeKind::Float, 32));
  EXPECT_TRUE(isType(plan.layout.dst.elementType, ElementTypeKind::Integer, 8));
  ASSERT_EQ(plan.stages.size(), 1u);
  const reloc::ValueStage &stage = plan.stages[0];
  EXPECT_EQ(stage.transform, ValueTransformKind::Quantize);
  EXPECT_EQ(stage.policy, NumericPolicyKind::SymmetricRne);
  EXPECT_TRUE(isType(stage.input.type, ElementTypeKind::Float, 32));
  EXPECT_EQ(stage.input.signedness, Signedness::Signless);
  EXPECT_TRUE(isType(stage.output.type, ElementTypeKind::Integer, 8));
  EXPECT_EQ(stage.output.signedness, Signedness::Signed);
  ASSERT_EQ(stage.shape.size(), 2u);
  EXPECT_EQ(stage.shape[0][0].op, ExprOp::PushSym); // B
  EXPECT_EQ(stage.shape[1][0].op, ExprOp::PushConst);
  EXPECT_EQ(stage.shape[1][0].value, 3);
  EXPECT_EQ(stage.scale.kind, ParamKind::Binding);
  EXPECT_EQ(stage.scale.bindingName, "s");
  EXPECT_EQ(stage.scale.rank, 1);
  ASSERT_EQ(stage.scale.bindingExtents.size(), 1u);
  EXPECT_EQ(stage.scale.bindingExtents[0][0].value, 3);
  EXPECT_TRUE(isType(stage.scale.elementType, ElementTypeKind::Float, 32));
  EXPECT_EQ(stage.zeroPoint.kind, ParamKind::None);
  EXPECT_EQ(stage.axis, 1);
  ASSERT_TRUE(stage.hasChannel);
  // (d0, d1)[s0] -> (d0 mod s0): PUSH_DIM 0, PUSH_SYM 0, MOD.
  ASSERT_EQ(stage.channel.size(), 3u);
  EXPECT_EQ(stage.channel[0].op, ExprOp::PushDim);
  EXPECT_EQ(stage.channel[0].value, 0);
  EXPECT_EQ(stage.channel[1].op, ExprOp::PushSym);
  EXPECT_EQ(stage.channel[1].value, 0);
  EXPECT_EQ(stage.channel[2].op, ExprOp::Mod);
  EXPECT_TRUE(plan.fills.empty());
}

TEST(TypedDecode, PadQuantizeGoldenDecodesStructurally) {
  TypedRelocationPlan plan = decodeTypedOk(padQuantize());
  EXPECT_TRUE(plan.symbols.empty());
  ASSERT_EQ(plan.layout.padFill.size(), 1u);
  EXPECT_EQ(plan.layout.padFill[0].fillBits, 2u); // fused code 2 (i8)
  EXPECT_TRUE(
      isType(plan.layout.padFill[0].fillType, ElementTypeKind::Integer, 8));
  ASSERT_EQ(plan.stages.size(), 1u);
  const reloc::ValueStage &stage = plan.stages[0];
  EXPECT_EQ(stage.axis, -1);
  EXPECT_FALSE(stage.hasChannel);
  EXPECT_EQ(stage.scale.kind, ParamKind::Inline);
  EXPECT_EQ(stage.scale.rank, 0);
  ASSERT_EQ(stage.scale.inlineBits.size(), 1u);
  EXPECT_FLOAT_EQ(reloc::typed::scaleAt(stage.scale, 0), 0.5f);
  ASSERT_EQ(plan.fills.size(), 1u);
  EXPECT_EQ(plan.fills[0].dstAxis, 0u);
  EXPECT_EQ(plan.fills[0].stage, 0u);
  EXPECT_TRUE(isType(plan.fills[0].type, ElementTypeKind::Float, 32));
  EXPECT_EQ(plan.fills[0].bits, 0x3f800000u); // 1.0f
}

TEST(TypedDecode, DequantCastGoldenDecodesStructurally) {
  TypedRelocationPlan plan = decodeTypedOk(fromHex(kDequantCastHex));
  EXPECT_TRUE(isType(plan.source.elementType, ElementTypeKind::Integer, 8));
  EXPECT_TRUE(isType(plan.result.elementType, ElementTypeKind::Float, 16));
  ASSERT_EQ(plan.stages.size(), 2u);
  const reloc::ValueStage &dequant = plan.stages[0];
  EXPECT_EQ(dequant.transform, ValueTransformKind::Dequantize);
  EXPECT_EQ(dequant.policy, NumericPolicyKind::Affine);
  EXPECT_EQ(dequant.input.signedness, Signedness::Signed);
  EXPECT_FLOAT_EQ(reloc::typed::scaleAt(dequant.scale, 0), 0.25f);
  EXPECT_EQ(dequant.zeroPoint.kind, ParamKind::Inline);
  EXPECT_TRUE(
      isType(dequant.zeroPoint.elementType, ElementTypeKind::Integer, 32));
  EXPECT_EQ(reloc::typed::zeroPointAt(dequant.zeroPoint, 0), -3);
  const reloc::ValueStage &cast = plan.stages[1];
  EXPECT_EQ(cast.transform, ValueTransformKind::Cast);
  EXPECT_EQ(cast.policy, NumericPolicyKind::IeeeRne);
  EXPECT_TRUE(isType(cast.input.type, ElementTypeKind::Float, 32));
  EXPECT_TRUE(isType(cast.output.type, ElementTypeKind::Float, 16));
  EXPECT_EQ(cast.scale.kind, ParamKind::None);
}

//===----------------------------------------------------------------------===//
// Version dispatch: v0 and v1 decoders never accept each other's blobs
//===----------------------------------------------------------------------===//

TEST(TypedDecode, LayoutDecoderRejectsV1WithUnsupportedVersion) {
  // The exact behavior of every pre-v1 runtime on a typed blob.
  std::vector<uint8_t> bytes = fromHex(kPadQuantizeHex);
  auto result = reloc::decodePlan(bytes.data(), bytes.size());
  auto *error = std::get_if<DecodeError>(&result);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->offset, 4u);
  EXPECT_NE(error->message.find("unsupported wire format version"),
            std::string::npos);
  EXPECT_EQ(reloc::peekWireVersion(bytes.data(), bytes.size()),
            std::optional<uint32_t>(1u));
}

TEST(TypedDecode, TypedDecoderRejectsV0) {
  std::vector<uint8_t> bytes = fromHex(kIdentityV0Hex);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, 4u);
  EXPECT_NE(error.message.find("unsupported wire format version"),
            std::string::npos);
  EXPECT_EQ(reloc::peekWireVersion(bytes.data(), bytes.size()),
            std::optional<uint32_t>(0u));
}

TEST(TypedDecode, PeekVersionRejectsBadHeaders) {
  const char *garbage = "not a plan";
  EXPECT_FALSE(reloc::peekWireVersion(
      reinterpret_cast<const uint8_t *>(garbage), std::strlen(garbage)));
  std::vector<uint8_t> shortHeader = fromHex("52504c4e0100");
  EXPECT_FALSE(reloc::peekWireVersion(shortHeader.data(), shortHeader.size()));
}

//===----------------------------------------------------------------------===//
// Malformed fixtures (spec "Decoder-enforced invariants")
//===----------------------------------------------------------------------===//

TEST(TypedDecode, RejectsUnknownTransform) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kTransformAt] = 5;
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kTransformAt);
  EXPECT_NE(error.message.find("unknown stage transform"), std::string::npos);
}

TEST(TypedDecode, RejectsUnknownPolicy) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kPolicyAt] = 9;
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kPolicyAt);
  EXPECT_NE(error.message.find("unknown numerical policy"), std::string::npos);
}

TEST(TypedDecode, RejectsPolicyThatDoesNotMatchTheTransform) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kPolicyAt] = 3; // affine on a quantize
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kTransformAt);
  EXPECT_NE(error.message.find("invalid quantize signature"),
            std::string::npos);
}

TEST(TypedDecode, RejectsSignlessQuantizeOutput) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kOutputSignAt] = 0;
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("invalid quantize signature"),
            std::string::npos);
}

TEST(TypedDecode, RejectsInvalidSignednessByte) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kInputSignAt] = 3;
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kInputSignAt);
  EXPECT_NE(error.message.find("signedness"), std::string::npos);
}

TEST(TypedDecode, RejectsInvalidParameterKind) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kScaleKindAt] = 7;
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kScaleKindAt);
  EXPECT_NE(error.message.find("parameter kind"), std::string::npos);
}

TEST(TypedDecode, RejectsInvalidInlineScales) {
  for (uint64_t bits : {uint64_t{0x00000000}, uint64_t{0x7fc00000},
                        uint64_t{0xbf000000}, uint64_t{0x7f800000}}) {
    std::vector<uint8_t> bytes = padQuantize();
    putU64(bytes, kScaleBitsAt, bits);
    DecodeError error = decodeTypedErr(bytes);
    EXPECT_NE(error.message.find("finite and strictly positive"),
              std::string::npos)
        << "bits " << bits;
  }
}

TEST(TypedDecode, RejectsPerChannelAxisWithRankZeroScale) {
  std::vector<uint8_t> bytes = padQuantize();
  putU64(bytes, kAxisAt, 0);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("rank-1 scale"), std::string::npos);
}

TEST(TypedDecode, RejectsChannelFlagWithoutAMap) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes[kHasChannelAt] = 1; // the map's num_dims/expr are not there
  // The decoder now reads the fills section as a channel map: the fill
  // count becomes num_dims and the fill's dst_axis (0) an empty expression
  // stream. Whichever way the bytes fall, the error must land in the stage
  // section, never in a silently accepted plan.
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_GE(error.offset, kHasChannelAt) << error.message;
  EXPECT_TRUE(error.message.find("truncated") != std::string::npos ||
              error.message.find("empty expression stream") !=
                  std::string::npos)
      << error.message;
}

TEST(TypedDecode, RejectsFillStageOutOfRange) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kFillStageAt, 2);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kFillDstAxisAt);
  EXPECT_NE(error.message.find("fill stage out of range"), std::string::npos);
}

TEST(TypedDecode, RejectsFillWithoutALayoutPad) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kFillDstAxisAt, 1);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("no layout pad"), std::string::npos);
}

TEST(TypedDecode, RejectsFillTypeNotMatchingItsEntryStage) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kFillTypeAt + 1, 16); // f32 -> f16 fill before the quantize
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("type entering its stage"), std::string::npos);
}

TEST(TypedDecode, RejectsFillThatDoesNotFoldToTheFusedFill) {
  std::vector<uint8_t> bytes = padQuantize();
  putU64(bytes, kFillBitsAt, 0x3f000000); // 0.5 at scale 0.5 -> code 1 != 2
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("fill fold mismatch"), std::string::npos);
  bytes = padQuantize();
  putU64(bytes, kPadFusedBitsAt, 3); // fused code 3 != fold(1.0) = 2
  error = decodeTypedErr(bytes);
  EXPECT_NE(error.message.find("fill fold mismatch"), std::string::npos);
}

TEST(TypedDecode, RejectsZeroStages) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kStageCountAt, 0);
  // With no stage the following bytes are read as the fills section and
  // trailing data; either way the plan is refused.
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_FALSE(error.message.empty());
}

TEST(TypedDecode, RejectsRankZeroInlineParameterWithSeveralValues) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kScaleCountAt, 2);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kScaleCountAt);
  EXPECT_NE(error.message.find("exactly one value"), std::string::npos);
}

TEST(TypedDecode, RejectsHostileStageCountWithoutAllocating) {
  std::vector<uint8_t> bytes = padQuantize();
  putU32(bytes, kStageCountAt, 0xffffffffu);
  DecodeError error = decodeTypedErr(bytes);
  EXPECT_EQ(error.offset, kStageCountAt);
  EXPECT_NE(error.message.find("count"), std::string::npos);
}

TEST(TypedDecode, RejectsTruncationAndTrailingBytes) {
  std::vector<uint8_t> bytes = padQuantize();
  bytes.resize(bytes.size() - 1);
  EXPECT_NE(decodeTypedErr(bytes).message.find("truncated"), std::string::npos);
  bytes = padQuantize();
  bytes.push_back(0);
  EXPECT_NE(decodeTypedErr(bytes).message.find("trailing"), std::string::npos);
}

TEST(TypedDecode, FuzzTruncationAndBitFlipsNoCrash) {
  const char *goldens[] = {kQuantizeTransposeHex, kPadQuantizeHex,
                           kDequantCastHex};
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
      (void)reloc::decodeTypedPlan(mutated.data(), mutated.size());
      (void)reloc::decodePlan(mutated.data(), mutated.size());
    }
  }
}

//===----------------------------------------------------------------------===//
// Reference arithmetic (reloc/TypedValue.h) matches the C1 tables
//===----------------------------------------------------------------------===//

reloc::ValueStage perTensorQuantize(float scale) {
  reloc::ValueStage stage;
  stage.transform = ValueTransformKind::Quantize;
  stage.policy = NumericPolicyKind::SymmetricRne;
  stage.input = {ElementType{ElementTypeKind::Float, 32}, Signedness::Signless};
  stage.output = {ElementType{ElementTypeKind::Integer, 8}, Signedness::Signed};
  stage.scale.kind = ParamKind::Inline;
  stage.scale.rank = 0;
  stage.scale.elementType = ElementType{ElementTypeKind::Float, 32};
  uint32_t bits;
  std::memcpy(&bits, &scale, sizeof(bits));
  stage.scale.inlineBits = {bits};
  return stage;
}

uint64_t f32Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

TEST(TypedValue, FoldFillFollowsTheC1Tables) {
  const ElementType f32{ElementTypeKind::Float, 32};
  const ElementType i8{ElementTypeKind::Integer, 8};
  reloc::ValueStage half = perTensorQuantize(0.5f);
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(1.0f), f32, half),
            std::optional<uint64_t>(2u));
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(200.0f), f32, half),
            std::optional<uint64_t>(127u));
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(-200.0f), f32, half),
            std::optional<uint64_t>(0x80u)); // -128 as the low byte
  EXPECT_EQ(reloc::typed::foldFill(0x7fc00000u, f32, half),
            std::optional<uint64_t>(0x80u)); // NaN -> -128
  reloc::ValueStage unit = perTensorQuantize(1.0f);
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(2.5f), f32, unit),
            std::optional<uint64_t>(2u)); // ties to even
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(-2.5f), f32, unit),
            std::optional<uint64_t>(0xfeu)); // -2
  // Wrong entry type: not foldable.
  EXPECT_FALSE(reloc::typed::foldFill(1, i8, half));
  // Per-channel: not foldable.
  reloc::ValueStage perChannel = half;
  perChannel.axis = 0;
  perChannel.scale.rank = 1;
  EXPECT_FALSE(reloc::typed::foldFill(f32Bits(1.0f), f32, perChannel));
  // Binding: not foldable.
  reloc::ValueStage bound = half;
  bound.scale.kind = ParamKind::Binding;
  bound.scale.bindingName = "s";
  EXPECT_FALSE(reloc::typed::foldFill(f32Bits(1.0f), f32, bound));

  // Casts.
  reloc::ValueStage narrow;
  narrow.transform = ValueTransformKind::Cast;
  narrow.policy = NumericPolicyKind::IeeeRne;
  narrow.input = {f32, Signedness::Signless};
  narrow.output = {ElementType{ElementTypeKind::Float, 16},
                   Signedness::Signless};
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(1.5f), f32, narrow),
            std::optional<uint64_t>(0x3e00u));
  EXPECT_EQ(reloc::typed::foldFill(f32Bits(65520.0f), f32, narrow),
            std::optional<uint64_t>(0x7c00u));
  reloc::ValueStage widen;
  widen.transform = ValueTransformKind::Cast;
  widen.policy = NumericPolicyKind::Exact;
  widen.input = {ElementType{ElementTypeKind::Float, 16}, Signedness::Signless};
  widen.output = {f32, Signedness::Signless};
  EXPECT_EQ(reloc::typed::foldFill(0x0001u, widen.input.type, widen),
            std::optional<uint64_t>(0x33800000u));
  EXPECT_EQ(reloc::typed::foldFill(0x7bffu, widen.input.type, widen),
            std::optional<uint64_t>(0x477fe000u));

  // Dequantize with a zero point: q = -128, zp = 127, scale 0.3.
  reloc::ValueStage dequant;
  dequant.transform = ValueTransformKind::Dequantize;
  dequant.policy = NumericPolicyKind::Affine;
  dequant.input = {i8, Signedness::Signed};
  dequant.output = {f32, Signedness::Signless};
  dequant.scale.kind = ParamKind::Inline;
  dequant.scale.elementType = f32;
  dequant.scale.inlineBits = {f32Bits(0.3f)};
  dequant.zeroPoint.kind = ParamKind::Inline;
  dequant.zeroPoint.elementType = ElementType{ElementTypeKind::Integer, 32};
  dequant.zeroPoint.inlineBits = {127u};
  EXPECT_EQ(reloc::typed::foldFill(0x80u, i8, dequant),
            std::optional<uint64_t>(0xc2990000u));
}

TEST(TypedValue, WideningIsExactOnTheWitnessBits) {
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x0001)), 0x33800000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x0400)), 0x38800000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x7bff)), 0x477fe000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x7c00)), 0x7f800000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0xfc00)), 0xff800000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x8000)), 0x80000000u);
  EXPECT_EQ(f32Bits(reloc::quant::widenF16F32(0x0200)), 0x38000000u); // 2^-15
  EXPECT_TRUE(std::isnan(reloc::quant::widenF16F32(0x7e00)));
  // Round trip through the narrowing kernel for every finite half.
  for (uint32_t h = 0; h < 0x10000u; ++h) {
    if (((h >> 10) & 0x1f) == 0x1f)
      continue; // inf / NaN
    EXPECT_EQ(reloc::quant::narrowF32F16(
                  reloc::quant::widenF16F32(static_cast<uint16_t>(h))),
              static_cast<uint16_t>(h))
        << "h=" << h;
  }
}

} // namespace
