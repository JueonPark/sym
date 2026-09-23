//===- TypedExecuteTest.cpp - scalar reference execution of typed plans
//----===//
//
// R3 (issue #147), Task 1. Expected values come from C1's published tables
// (docs/reloc-typed-semantics.md §3, §6) written out in this file, never
// from the executor under test: casts use the witness bit patterns, the
// quantize/dequantize oracle is the four-line reference formula.
//
//===----------------------------------------------------------------------===//

#include "reloc/TypedExecute.h"

#include "TypedGoldens.h"
#include "reloc/Decode.h"
#include "reloc/GatherPool.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::ElementType;
using reloc::ElementTypeKind;
using reloc::ParameterMap;
using reloc::ParameterValue;
using reloc::SymbolMap;
using reloc::TypedBoundPlan;
using reloc::typed::ExecutionError;
using reloc::typed::Program;
using typed_goldens::fromHex;

//===----------------------------------------------------------------------===//
// Independent oracle (C1 §3.3 / §3.4 written plainly).
//===----------------------------------------------------------------------===//

int8_t oracleQuantize(float x, float scale) {
  const float inv = 1.0f / scale;
  const float t = x * inv;
  float c;
  if (std::isnan(t))
    c = -128.0f;
  else
    c = std::min(std::max(t, -128.0f), 127.0f);
  return static_cast<int8_t>(std::nearbyint(c)); // default env: ties to even
}

float oracleDequantize(int8_t q, int32_t zeroPoint, float scale) {
  return static_cast<float>(static_cast<int32_t>(q) - zeroPoint) * scale;
}

uint32_t bitsOf(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
}

float floatOf(uint32_t bits) {
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}

template <typename T>
std::vector<uint8_t> bytesOf(const std::vector<T> &values) {
  std::vector<uint8_t> out(values.size() * sizeof(T));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

template <typename T>
std::vector<T> valuesOf(const std::vector<uint8_t> &bytes) {
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}

ParameterValue f32Param(std::vector<float> values, bool perChannel) {
  ParameterValue v;
  v.elementType = ElementType{ElementTypeKind::Float, 32};
  if (perChannel)
    v.extents = {static_cast<int64_t>(values.size())};
  v.bytes = bytesOf(values);
  return v;
}

ParameterValue i32Param(int32_t value) {
  ParameterValue v;
  v.elementType = ElementType{ElementTypeKind::Integer, 32};
  v.bytes = bytesOf(std::vector<int32_t>{value});
  return v;
}

TypedBoundPlan mustBind(const char *hex, const SymbolMap &symbols,
                        const ParameterMap &parameters = {}) {
  std::vector<uint8_t> bytes = fromHex(hex);
  auto decoded = reloc::decodeTypedPlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::TypedRelocationPlan>(&decoded);
  if (plan == nullptr) {
    ADD_FAILURE() << "decode failed: "
                  << std::get<reloc::DecodeError>(decoded).message;
    return {};
  }
  auto bound = reloc::bindTyped(*plan, symbols, parameters);
  auto *result = std::get_if<TypedBoundPlan>(&bound);
  if (result == nullptr) {
    ADD_FAILURE() << "bind failed: "
                  << std::get<reloc::BindError>(bound).message;
    return {};
  }
  return *result;
}

Program mustPrepare(const TypedBoundPlan &plan) {
  auto prepared = reloc::typed::prepareProgram(plan);
  auto *program = std::get_if<Program>(&prepared);
  if (program == nullptr) {
    ADD_FAILURE() << "prepare failed: "
                  << std::get<ExecutionError>(prepared).message;
    return {};
  }
  return *program;
}

std::vector<uint8_t> run(const Program &program, uint32_t from, uint32_t to,
                         const std::vector<uint8_t> &src, unsigned threads = 1,
                         reloc::GatherPool *pool = nullptr) {
  std::vector<uint8_t> dst(
      static_cast<size_t>(reloc::typed::bytesAt(program, to, true)), 0xAB);
  auto error = reloc::typed::executeHost(program, from, to, src.data(),
                                         dst.data(), pool, threads);
  EXPECT_FALSE(error.has_value()) << (error ? error->message : "");
  return dst;
}

//===----------------------------------------------------------------------===//
// Two lossy stages retain the intermediate rounding.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, QuantizeThenDequantizeRetainsTheIntermediateRounding) {
  // quant_dequant_sym: [N] f32 -> s8 (scale 0.5) -> f32 (scale 0.5).
  Program program =
      mustPrepare(mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}}));
  ASSERT_EQ(program.stages.size(), 2u);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  // Ties and limits at scale 0.5 (x * 2 lands on C1's unit-scale vectors),
  // the specials, a value that is not a multiple of the step, and signed zero.
  const std::vector<float> x = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f,
                                -64.5f, -64.0f, -63.5f, 63.0f, 63.5f, 64.0f,
                                nan,    inf,    -inf,   0.3f};
  std::vector<int8_t> expectedQ;
  std::vector<float> expectedY;
  for (float v : x) {
    const int8_t q = oracleQuantize(v, 0.5f);
    expectedQ.push_back(q);
    expectedY.push_back(oracleDequantize(q, 0, 0.5f));
  }
  EXPECT_EQ(std::vector<int8_t>(expectedQ.begin(), expectedQ.begin() + 6),
            (std::vector<int8_t>{-2, -2, 0, 0, 2, 2}));
  EXPECT_EQ(std::vector<int8_t>(expectedQ.begin() + 6, expectedQ.begin() + 12),
            (std::vector<int8_t>{-128, -128, -127, 126, 127, 127}));
  EXPECT_EQ(std::vector<int8_t>(expectedQ.begin() + 12, expectedQ.end()),
            (std::vector<int8_t>{-128, 127, -128, 1}));

  const std::vector<uint8_t> src = bytesOf(x);
  // The whole program.
  EXPECT_EQ(valuesOf<float>(run(program, 0, 2, src)), expectedY);
  // Stopping at boundary 1 exposes the s8 wire; resuming from it agrees.
  std::vector<uint8_t> wire = run(program, 0, 1, src);
  EXPECT_EQ(valuesOf<int8_t>(wire), expectedQ);
  EXPECT_EQ(valuesOf<float>(run(program, 1, 2, wire)), expectedY);
  // 0.3 -> code 1 -> 0.5: the pair is not the identity (C1 §5).
  EXPECT_EQ(expectedY.back(), 0.5f);
  EXPECT_EQ(reloc::typed::bytesAt(program, 0, true), 64);
  EXPECT_EQ(reloc::typed::bytesAt(program, 1, true), 16);
  EXPECT_EQ(reloc::typed::bytesAt(program, 2, true), 64);
}

//===----------------------------------------------------------------------===//
// Pad order: pad -> quantize versus quantize -> pad.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, PadBeforeAndAfterQuantizeCarryDifferentFills) {
  Program before = mustPrepare(mustBind(typed_goldens::kPadQuantizeHex, {}));
  Program after = mustPrepare(mustBind(typed_goldens::kQuantizePadHex, {}));
  const std::vector<float> x = {0.3f, 0.75f, -0.75f, 2.0f, -128.0f, 100.0f};
  std::vector<int8_t> body;
  for (float v : x)
    body.push_back(oracleQuantize(v, 0.5f));

  // f32 fill 1.0 before the quantize at scale 0.5 becomes code 2 ...
  std::vector<int8_t> out = valuesOf<int8_t>(run(before, 0, 1, bytesOf(x)));
  ASSERT_EQ(out.size(), 8u);
  EXPECT_EQ(out.front(), 2);
  EXPECT_EQ(out.back(), 2);
  EXPECT_EQ(std::vector<int8_t>(out.begin() + 1, out.end() - 1), body);
  // ... while an s8 fill of 1 after the quantize stays 1.
  out = valuesOf<int8_t>(run(after, 0, 1, bytesOf(x)));
  EXPECT_EQ(out.front(), 1);
  EXPECT_EQ(out.back(), 1);
  EXPECT_EQ(std::vector<int8_t>(out.begin() + 1, out.end() - 1), body);

  // Boundary bookkeeping: the f32 pad exists at boundary 0 with its f32
  // bits; the s8 pad has not entered at boundary 0.
  auto fill0 = reloc::typed::fillAt(before, 0);
  ASSERT_TRUE(std::holds_alternative<std::optional<uint64_t>>(fill0));
  EXPECT_EQ(std::get<std::optional<uint64_t>>(fill0).value(), 0x3f800000u);
  auto fill1 = reloc::typed::fillAt(before, 1);
  EXPECT_EQ(std::get<std::optional<uint64_t>>(fill1).value(), 2u);
  EXPECT_TRUE(reloc::typed::padsSettledBy(before, 0));
  EXPECT_FALSE(reloc::typed::padsSettledBy(after, 0));
  EXPECT_TRUE(reloc::typed::padsSettledBy(after, 1));
  auto missing = reloc::typed::fillAt(after, 0);
  ASSERT_TRUE(std::holds_alternative<ExecutionError>(missing));
  EXPECT_EQ(std::get<ExecutionError>(missing).code, "pad_not_entered");
  EXPECT_EQ(
      std::get<std::optional<uint64_t>>(reloc::typed::fillAt(after, 1)).value(),
      1u);
  // A cut before the pad has no padded layout to write into.
  std::vector<uint8_t> dst(64);
  auto error =
      reloc::typed::executeHost(after, 0, 0, bytesOf(x).data(), dst.data());
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "pads_not_settled");
  // Footprints: the f32 pad is counted at boundary 0 in the result layout,
  // never in the source layout.
  EXPECT_EQ(reloc::typed::bytesAt(before, 0, true), 32);
  EXPECT_EQ(reloc::typed::bytesAt(before, 0, false), 24);
  EXPECT_EQ(reloc::typed::bytesAt(before, 1, true), 8);
}

//===----------------------------------------------------------------------===//
// Casts: C1 §6 witness bits, and a pad that enters after the cast.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, NarrowingCastWitnessBitsAndPadAfterCast) {
  Program program = mustPrepare(mustBind(typed_goldens::kCastPadHex, {}));
  const std::vector<float> x = {std::ldexp(1.0f, -24),
                                std::ldexp(1.0f, -25),
                                1.5f * std::ldexp(1.0f, -25),
                                std::ldexp(1.0f, -14),
                                65504.0f,
                                65520.0f};
  std::vector<uint16_t> out =
      valuesOf<uint16_t>(run(program, 0, 1, bytesOf(x)));
  EXPECT_EQ(out, (std::vector<uint16_t>{0x3C00, 0x0001, 0x0000, 0x0001, 0x0400,
                                        0x7BFF, 0x7C00, 0x3C00}));
  const std::vector<float> more = {1.0f + std::ldexp(1.0f, -11),
                                   1.0f + 3.0f * std::ldexp(1.0f, -12),
                                   -0.0f,
                                   65519.99f,
                                   1e5f,
                                   -std::numeric_limits<float>::infinity()};
  out = valuesOf<uint16_t>(run(program, 0, 1, bytesOf(more)));
  EXPECT_EQ(out, (std::vector<uint16_t>{0x3C00, 0x3C00, 0x3C01, 0x8000, 0x7BFF,
                                        0x7C00, 0xFC00, 0x3C00}));
  const std::vector<float> nan = {
      std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0, 0};
  out = valuesOf<uint16_t>(run(program, 0, 1, bytesOf(nan)));
  EXPECT_TRUE((out[1] & 0x7C00) == 0x7C00 && (out[1] & 0x03FF) != 0)
      << "NaN-ness is promised, the payload is not";
  EXPECT_FALSE(reloc::typed::padsSettledBy(program, 0));
  EXPECT_TRUE(reloc::typed::padsSettledBy(program, 1));
}

TEST(TypedExecute, WideningCastIsExactOnTheWitnessBits) {
  Program program = mustPrepare(mustBind(typed_goldens::kCastWidenHex, {}));
  const std::vector<uint16_t> h = {0x0001, 0x0400, 0x7BFF, 0xFC00};
  std::vector<uint32_t> out =
      valuesOf<uint32_t>(run(program, 0, 1, bytesOf(h)));
  EXPECT_EQ(out, (std::vector<uint32_t>{0x33800000, 0x38800000, 0x477FE000,
                                        0xFF800000}));
  const std::vector<uint16_t> more = {0x8000, 0x7C00, 0x3C00, 0x0000};
  out = valuesOf<uint32_t>(run(program, 0, 1, bytesOf(more)));
  EXPECT_EQ(out,
            (std::vector<uint32_t>{0x80000000, 0x7F800000, 0x3F800000, 0}));
}

//===----------------------------------------------------------------------===//
// Per-channel parameters under a layout that moves the channel axis.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, PerChannelIndexFollowsTheResultCoordinates) {
  // [2, 3, 4] f32, quantize axis 1 with scales [0.5, 1, 2], transpose
  // [1, 0, 2] -> [3, 2, 4] s8. x[i] = i - 11.5 is C1's witness input.
  Program program =
      mustPrepare(mustBind(typed_goldens::kWitnessChannelTransposeHex, {}));
  ASSERT_TRUE(program.stages[0].perChannel);
  EXPECT_TRUE(program.stages[0].channelIsDim);
  EXPECT_EQ(program.stages[0].channelDim, 0u);
  std::vector<float> x(24);
  for (int i = 0; i < 24; ++i)
    x[static_cast<size_t>(i)] = static_cast<float>(i) - 11.5f;
  const float scales[3] = {0.5f, 1.0f, 2.0f};
  std::vector<int8_t> expected(24);
  for (int b = 0; b < 2; ++b)
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 4; ++k)
        expected[static_cast<size_t>(c * 8 + b * 4 + k)] = oracleQuantize(
            x[static_cast<size_t>(b * 12 + c * 4 + k)], scales[c]);
  std::vector<int8_t> out = valuesOf<int8_t>(run(program, 0, 1, bytesOf(x)));
  EXPECT_EQ(out, expected);
  // The C1 witness slice [0, :, :] of the operand, read back through the
  // transpose: result[c][0][:] for c = 0, 1, 2.
  EXPECT_EQ(std::vector<int8_t>(out.begin(), out.begin() + 4),
            (std::vector<int8_t>{-23, -21, -19, -17}));
  EXPECT_EQ(std::vector<int8_t>(out.begin() + 8, out.begin() + 12),
            (std::vector<int8_t>{-8, -6, -6, -4}));
  EXPECT_EQ(std::vector<int8_t>(out.begin() + 16, out.begin() + 20),
            (std::vector<int8_t>{-2, -1, -1, 0}));
  // Parallel execution partitions the outer rows and is bit-identical.
  reloc::GatherPool pool(3);
  EXPECT_EQ(valuesOf<int8_t>(run(program, 0, 1, bytesOf(x), 1, &pool)),
            expected);
  EXPECT_EQ(valuesOf<int8_t>(run(program, 0, 1, bytesOf(x), 4)), expected);
}

TEST(TypedExecute, EqualSizedAxesDoNotConcealAWrongChannelMapping) {
  // quantize_transpose: [B, 3] f32, quantize axis 1 with a runtime scale
  // over [3], transpose -> [3, B]. The channel map is (d0 mod B): with
  // B = 3 both axes have extent 3, so a mapping that used the row of the
  // OPERAND (b) instead of the result row (c) would still be in range.
  ParameterMap parameters;
  parameters["s"] = f32Param({1.0f, 0.5f, 0.25f}, /*perChannel=*/true);
  Program program = mustPrepare(
      mustBind(typed_goldens::kQuantizeTransposeHex, {{"B", 3}}, parameters));
  ASSERT_TRUE(program.stages[0].perChannel);
  EXPECT_FALSE(program.stages[0].channelIsDim)
      << "(d0 mod B) is not a bare dim";
  std::vector<float> x(9);
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
      x[static_cast<size_t>(b * 3 + c)] = static_cast<float>(b * 10 + c);
  std::vector<int8_t> out = valuesOf<int8_t>(run(program, 0, 1, bytesOf(x)));
  // result[c][b] = rne(x[b][c] / scale[c]).
  EXPECT_EQ(out, (std::vector<int8_t>{0, 10, 20, 2, 22, 42, 8, 48, 88}));
  std::vector<int8_t> wrong;
  for (int c = 0; c < 3; ++c)
    for (int b = 0; b < 3; ++b)
      wrong.push_back(oracleQuantize(
          x[static_cast<size_t>(b * 3 + c)],
          std::vector<float>{1.0f, 0.5f, 0.25f}[static_cast<size_t>(b)]));
  EXPECT_NE(out, wrong) << "the test input distinguishes the two mappings";
}

//===----------------------------------------------------------------------===//
// Dequantize: runtime scale and zero point, and a following lossy cast.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, RuntimeScaleAndZeroPointDequantizeBits) {
  ParameterMap parameters;
  parameters["s"] = f32Param({0.3f}, /*perChannel=*/false);
  parameters["zp"] = i32Param(0);
  Program program = mustPrepare(
      mustBind(typed_goldens::kDequantBindingHex, {{"N", 5}}, parameters));
  const std::vector<int8_t> q = {-128, -1, 0, 1, 127};
  std::vector<uint32_t> out =
      valuesOf<uint32_t>(run(program, 0, 1, bytesOf(q)));
  EXPECT_EQ(out, (std::vector<uint32_t>{0xC219999A, 0xBE99999A, 0x00000000,
                                        0x3E99999A, 0x42186667}));
  parameters["zp"] = i32Param(127);
  program = mustPrepare(
      mustBind(typed_goldens::kDequantBindingHex, {{"N", 5}}, parameters));
  out = valuesOf<uint32_t>(run(program, 0, 1, bytesOf(q)));
  EXPECT_EQ(out[0], 0xC2990000u); // (-255) * 0.3 = -76.5 exactly
  for (size_t i = 0; i < q.size(); ++i)
    EXPECT_EQ(out[i], bitsOf(oracleDequantize(q[i], 127, 0.3f)));
}

TEST(TypedExecute, DequantizeThenNarrowingCastKeepsBothRoundings) {
  // dequant_cast: [4] s8 -> f32 (scale 0.25, zero point -3) -> f16.
  Program program = mustPrepare(mustBind(typed_goldens::kDequantCastHex, {}));
  const std::vector<int8_t> q = {-128, 0, 5, 127};
  // (q + 3) * 0.25 is exact in f32 and representable in f16, so the second
  // rounding is the identity and the expected bits are hand-computable.
  std::vector<uint32_t> wire =
      valuesOf<uint32_t>(run(program, 0, 1, bytesOf(q)));
  EXPECT_EQ(wire, (std::vector<uint32_t>{bitsOf(-31.25f), bitsOf(0.75f),
                                         bitsOf(2.0f), bitsOf(32.5f)}));
  std::vector<uint16_t> out =
      valuesOf<uint16_t>(run(program, 0, 2, bytesOf(q)));
  EXPECT_EQ(out, (std::vector<uint16_t>{0xCFD0, 0x3A00, 0x4000, 0x5010}));
  EXPECT_EQ(valuesOf<uint16_t>(run(program, 1, 2, bytesOf(wire))), out);
}

//===----------------------------------------------------------------------===//
// Rejections happen before or instead of writing a result.
//===----------------------------------------------------------------------===//

TEST(TypedExecute, UnsupportedStagesAreRejectedAtPreparation) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 4}});
  TypedBoundPlan wrongPolicy = plan;
  wrongPolicy.stages[0].policy = reloc::NumericPolicyKind::Affine;
  auto prepared = reloc::typed::prepareProgram(wrongPolicy);
  ASSERT_TRUE(std::holds_alternative<ExecutionError>(prepared));
  EXPECT_EQ(std::get<ExecutionError>(prepared).code, "unsupported_stage");

  TypedBoundPlan wrongType = plan;
  wrongType.stages[1].output.type = ElementType{ElementTypeKind::Float, 64};
  prepared = reloc::typed::prepareProgram(wrongType);
  ASSERT_TRUE(std::holds_alternative<ExecutionError>(prepared));
  EXPECT_EQ(std::get<ExecutionError>(prepared).code, "unsupported_stage");

  TypedBoundPlan badScale = plan;
  badScale.stages[0].scale.bytes = bytesOf(std::vector<float>{0.0f});
  prepared = reloc::typed::prepareProgram(badScale);
  ASSERT_TRUE(std::holds_alternative<ExecutionError>(prepared));
  EXPECT_EQ(std::get<ExecutionError>(prepared).code, "invalid_parameter");
}

TEST(TypedExecute, ChannelSelectionOutsideTheParameterFails) {
  Program program =
      mustPrepare(mustBind(typed_goldens::kWitnessChannelTransposeHex, {}));
  // Force the slow path with a constant channel beyond the three scales.
  program.stages[0].channelIsDim = false;
  program.plan.stages[0].channel = {
      reloc::ExprToken{reloc::ExprOp::PushConst, 7}};
  std::vector<float> x(24, 1.0f);
  std::vector<uint8_t> dst(24, 0);
  auto error =
      reloc::typed::executeHost(program, 0, 1, bytesOf(x).data(), dst.data());
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "channel_out_of_range");
  auto bad =
      reloc::typed::executeHost(program, 2, 1, bytesOf(x).data(), dst.data());
  ASSERT_TRUE(bad.has_value());
  EXPECT_EQ(bad->code, "invalid_boundary");
}

TEST(TypedExecute, ApplyStageIsTheReferenceArithmetic) {
  Program program =
      mustPrepare(mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 2}}));
  const reloc::typed::StageArithmetic &quantize = program.stages[0];
  EXPECT_EQ(quantize.invScale, (std::vector<float>{2.0f}));
  EXPECT_EQ(reloc::typed::applyStage(quantize, bitsOf(0.75f), 0), 2u);
  EXPECT_EQ(reloc::typed::applyStage(quantize, bitsOf(-0.75f), 0), 0xFEu);
  const reloc::typed::StageArithmetic &dequantize = program.stages[1];
  EXPECT_EQ(reloc::typed::applyStage(dequantize, 0xFE, 0), bitsOf(-1.0f));
  EXPECT_EQ(reloc::typed::typeAt(program, 1).bitwidth, 8u);
  EXPECT_EQ(reloc::typed::widthAt(program, 2), 4u);
}

} // namespace
