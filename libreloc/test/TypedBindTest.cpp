//===- TypedBindTest.cpp - bindTyped guards and footprints (C3) -----------===//
//
// Typed binding over the lit-frozen v1 goldens (TypedGoldens.h): symbol
// and parameter guards, owned parameter snapshots, distinct checked byte
// footprints, re-validation on rebinding, and the refusal of typed layouts
// by the layout-only transfer path. Nothing here executes a kernel.
//
//===----------------------------------------------------------------------===//

#include "TypedGoldens.h"
#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/Transfer.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::BindError;
using reloc::ElementType;
using reloc::ElementTypeKind;
using reloc::ParameterMap;
using reloc::ParameterValue;
using reloc::TypedBoundPlan;
using reloc::TypedRelocationPlan;
using typed_goldens::fromHex;

TypedRelocationPlan decoded(const char *hex) {
  std::vector<uint8_t> bytes = fromHex(hex);
  auto result = reloc::decodeTypedPlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<TypedRelocationPlan>(&result);
  EXPECT_NE(plan, nullptr)
      << (std::get_if<reloc::DecodeError>(&result)
              ? std::get_if<reloc::DecodeError>(&result)->message
              : "<none>");
  return plan ? *plan : TypedRelocationPlan{};
}

TypedBoundPlan bindOk(const TypedRelocationPlan &plan,
                      const reloc::SymbolMap &symbols,
                      const ParameterMap &parameters = {}) {
  auto result = reloc::bindTyped(plan, symbols, parameters);
  auto *bound = std::get_if<TypedBoundPlan>(&result);
  EXPECT_NE(bound, nullptr) << (std::get_if<BindError>(&result)
                                    ? std::get_if<BindError>(&result)->message
                                    : "<none>");
  return bound ? *bound : TypedBoundPlan{};
}

std::string bindErr(const TypedRelocationPlan &plan,
                    const reloc::SymbolMap &symbols,
                    const ParameterMap &parameters = {}) {
  auto result = reloc::bindTyped(plan, symbols, parameters);
  auto *error = std::get_if<BindError>(&result);
  EXPECT_NE(error, nullptr) << "typed bind unexpectedly succeeded";
  return error ? error->message : std::string{};
}

ParameterValue f32Values(std::vector<float> values, bool rankOne = true) {
  ParameterValue value;
  value.elementType = ElementType{ElementTypeKind::Float, 32};
  if (rankOne)
    value.extents = {static_cast<int64_t>(values.size())};
  value.bytes.resize(values.size() * 4);
  std::memcpy(value.bytes.data(), values.data(), value.bytes.size());
  return value;
}

ParameterValue i32Scalar(int32_t value) {
  ParameterValue out;
  out.elementType = ElementType{ElementTypeKind::Integer, 32};
  out.bytes.resize(4);
  std::memcpy(out.bytes.data(), &value, 4);
  return out;
}

bool isType(ElementType type, ElementTypeKind kind, uint32_t bitwidth) {
  return type.kind == kind && type.bitwidth == bitwidth;
}

//===----------------------------------------------------------------------===//
// Footprints (issue #143 Task 2's declared-path regression)
//===----------------------------------------------------------------------===//

TEST(TypedBind, SourceWireAndDestinationBytesAreDistinct) {
  // N = 192, f32 -> s8 -> f32: source 768, the s8 boundary 192, destination
  // 768; the two inline scales (4 bytes each) are recorded apart.
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantDequantSymHex);
  TypedBoundPlan bound = bindOk(plan, {{"N", 192}});
  EXPECT_EQ(bound.sourceBytes, 768);
  EXPECT_EQ(bound.destinationBytes, 768);
  EXPECT_EQ(bound.parameterBytes, 8);
  ASSERT_EQ(bound.cuts.size(), 3u);
  EXPECT_EQ(bound.cuts[0].bytes, 768);
  EXPECT_TRUE(isType(bound.cuts[0].elementType, ElementTypeKind::Float, 32));
  EXPECT_EQ(bound.cuts[1].bytes, 192);
  EXPECT_EQ(bound.cuts[1].elements, 192);
  EXPECT_TRUE(isType(bound.cuts[1].elementType, ElementTypeKind::Integer, 8));
  EXPECT_EQ(bound.cuts[2].bytes, 768);
  EXPECT_EQ(bound.layout.totalBytes, bound.destinationBytes);
  EXPECT_EQ(bound.layout.elementSize, 4u); // the RESULT width, never the wire's
  EXPECT_TRUE(bound.layout.typed);
  ASSERT_EQ(bound.stages.size(), 2u);
  EXPECT_TRUE(bound.stages[0].scale.present);
  EXPECT_EQ(bound.stages[0].scale.length, 1);
  EXPECT_TRUE(bound.stages[0].scale.bindingName.empty()); // inline constant
  ASSERT_EQ(bound.requirements.size(), 1u);
  EXPECT_EQ(bound.requirements[0], "typed_execution_dispatch");
}

TEST(TypedBind, PaddedFootprintCountsPadsFromTheirEntryStage) {
  // pad(f32) then quantize: the pad enters at stage 0, so the tensor
  // entering the quantize already has 8 elements (32 bytes as f32), the
  // result has 8 bytes as s8; the caller's SOURCE stays 6 * 4 = 24 bytes.
  TypedRelocationPlan plan = decoded(typed_goldens::kPadQuantizeHex);
  TypedBoundPlan bound = bindOk(plan, {});
  EXPECT_EQ(bound.sourceBytes, 24);
  ASSERT_EQ(bound.cuts.size(), 2u);
  EXPECT_EQ(bound.cuts[0].elements, 8);
  EXPECT_EQ(bound.cuts[0].bytes, 32);
  EXPECT_EQ(bound.cuts[1].bytes, 8);
  EXPECT_EQ(bound.destinationBytes, 8);
  EXPECT_EQ(bound.layout.totalBytes, 8);
}

TEST(TypedBind, ZeroNegativeAndOverflowingExtentsFailBeforeAnything) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantDequantSymHex);
  EXPECT_NE(bindErr(plan, {{"N", 0}}).find("extent"), std::string::npos);
  EXPECT_NE(bindErr(plan, {{"N", -4}}).find("extent"), std::string::npos);
  // 2^61 elements * 4 bytes overflows int64 in the byte multiplication.
  std::string error = bindErr(plan, {{"N", int64_t{1} << 61}});
  EXPECT_NE(error.find("overflow"), std::string::npos) << error;
}

TEST(TypedBind, SymbolMismatchesAreHardErrors) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantDequantSymHex);
  EXPECT_NE(bindErr(plan, {}).find("unbound symbol"), std::string::npos);
  EXPECT_NE(bindErr(plan, {{"N", 4}, {"M", 1}}).find("unknown symbol"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

TEST(TypedBind, RuntimeParameterIsSnapshottedAndValidated) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantizeTransposeHex);
  ParameterMap parameters{{"s", f32Values({0.5f, 0.25f, 0.125f})}};
  TypedBoundPlan bound = bindOk(plan, {{"B", 2}}, parameters);
  ASSERT_EQ(bound.stages.size(), 1u);
  const reloc::BoundParameter &scale = bound.stages[0].scale;
  EXPECT_TRUE(scale.present);
  EXPECT_EQ(scale.bindingName, "s");
  EXPECT_EQ(scale.length, 3);
  EXPECT_EQ(scale.bytes.size(), 12u);
  EXPECT_EQ(bound.parameterBytes, 12);
  // Owned snapshot: mutating the caller's copy afterwards changes nothing.
  parameters["s"].bytes[0] = 0xff;
  float first;
  std::memcpy(&first, scale.bytes.data(), 4);
  EXPECT_EQ(first, 0.5f);
  EXPECT_FALSE(bound.stages[0].zeroPoint.present);
  EXPECT_TRUE(bound.stages[0].hasChannel);
  EXPECT_EQ(bound.stages[0].axis, 1);
  EXPECT_EQ(bound.stages[0].shape, (std::vector<int64_t>{2, 3}));
}

TEST(TypedBind, MissingExtraAndMismatchedParametersFail) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantizeTransposeHex);
  const reloc::SymbolMap symbols{{"B", 2}};
  EXPECT_NE(bindErr(plan, symbols, {}).find("unbound parameter: s"),
            std::string::npos);
  ParameterMap extra{{"s", f32Values({0.5f, 0.25f, 0.125f})},
                     {"t", f32Values({1.0f})}};
  EXPECT_NE(bindErr(plan, symbols, extra).find("unknown parameter"),
            std::string::npos);
  ParameterValue wrongType = f32Values({0.5f, 0.25f, 0.125f});
  wrongType.elementType = ElementType{ElementTypeKind::Float, 16};
  EXPECT_NE(bindErr(plan, symbols, {{"s", wrongType}}).find("element type"),
            std::string::npos);
  ParameterValue wrongRank = f32Values({0.5f}, /*rankOne=*/false);
  EXPECT_NE(bindErr(plan, symbols, {{"s", wrongRank}}).find("rank"),
            std::string::npos);
  EXPECT_NE(
      bindErr(plan, symbols, {{"s", f32Values({0.5f, 0.25f, 0.125f, 1.f})}})
          .find("has length 4 but the channel extent is 3"),
      std::string::npos);
  ParameterValue shortBytes = f32Values({0.5f, 0.25f, 0.125f});
  shortBytes.bytes.pop_back();
  EXPECT_NE(bindErr(plan, symbols, {{"s", shortBytes}}).find("bytes"),
            std::string::npos);
  EXPECT_NE(bindErr(plan, symbols, {{"s", f32Values({0.5f, 0.0f, 0.125f})}})
                .find("finite and strictly positive"),
            std::string::npos);
  EXPECT_NE(bindErr(plan, symbols, {{"s", f32Values({0.5f, -0.25f, 0.125f})}})
                .find("finite and strictly positive"),
            std::string::npos);
  EXPECT_NE(
      bindErr(plan, symbols, {{"s", f32Values({0.5f, std::nanf(""), 0.125f})}})
          .find("finite and strictly positive"),
      std::string::npos);
  EXPECT_NE(bindErr(plan, symbols, {{"s", f32Values({0.5f, INFINITY, 0.125f})}})
                .find("finite and strictly positive"),
            std::string::npos);
}

TEST(TypedBind, RebindingASymbolicChannelRevalidatesTheLength) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantizeChannelSymHex);
  ParameterMap two{{"s", f32Values({0.5f, 0.25f})}};
  TypedBoundPlan bound = bindOk(plan, {{"B", 2}}, two);
  EXPECT_EQ(bound.stages[0].scale.length, 2);
  EXPECT_EQ(bound.sourceBytes, 2 * 3 * 4);
  EXPECT_EQ(bound.destinationBytes, 2 * 3 * 1);
  // The same parameter with B = 3 is the wrong length: the previous bind's
  // validation carries nothing over.
  std::string error = bindErr(plan, {{"B", 3}}, two);
  EXPECT_NE(error.find("has length 2 but the channel extent is 3"),
            std::string::npos)
      << error;
  bindOk(plan, {{"B", 3}}, {{"s", f32Values({0.5f, 0.25f, 0.125f})}});
}

TEST(TypedBind, RuntimeZeroPointIsRangeChecked) {
  TypedRelocationPlan plan = decoded(typed_goldens::kDequantBindingHex);
  ParameterMap ok{{"s", f32Values({0.3f}, /*rankOne=*/false)},
                  {"zp", i32Scalar(-3)}};
  TypedBoundPlan bound = bindOk(plan, {{"N", 5}}, ok);
  EXPECT_TRUE(bound.stages[0].zeroPoint.present);
  EXPECT_EQ(bound.stages[0].zeroPoint.bindingName, "zp");
  EXPECT_EQ(bound.parameterBytes, 8);
  EXPECT_EQ(bound.sourceBytes, 5);
  EXPECT_EQ(bound.destinationBytes, 20);
  ParameterMap high = ok;
  high["zp"] = i32Scalar(200);
  EXPECT_NE(bindErr(plan, {{"N", 5}}, high).find("[-128, 127]"),
            std::string::npos);
  ParameterMap low = ok;
  low["zp"] = i32Scalar(-129);
  EXPECT_NE(bindErr(plan, {{"N", 5}}, low).find("[-128, 127]"),
            std::string::npos);
  ParameterMap missingZp{{"s", f32Values({0.3f}, /*rankOne=*/false)}};
  EXPECT_NE(bindErr(plan, {{"N", 5}}, missingZp).find("unbound parameter: zp"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Legacy layout-only paths refuse a typed layout
//===----------------------------------------------------------------------===//

TEST(TypedBind, LayoutOnlyTransferPathRefusesTheTypedLayout) {
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantDequantSymHex);
  TypedBoundPlan bound = bindOk(plan, {{"N", 8}});
  ASSERT_TRUE(bound.layout.typed);
  std::vector<float> src(8), dst(8);
  reloc::BufferView source;
  source.base = reinterpret_cast<uintptr_t>(src.data());
  source.capacityBytes = src.size() * 4;
  source.extents = {8};
  source.strides = {1};
  source.elementSize = 4;
  reloc::BufferView destination = source;
  destination.base = reinterpret_cast<uintptr_t>(dst.data());
  auto validated =
      reloc::validateTransfer(bound.layout, source, destination,
                              reloc::TransferDirection::HostToDevice);
  auto *error = std::get_if<reloc::TransferError>(&validated);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->code, "typed_unsupported");
  auto sourceOnly = reloc::validateTransferSource(
      bound.layout, source, reloc::TransferDirection::HostToDevice);
  auto *sourceError = std::get_if<reloc::TransferError>(&sourceOnly);
  ASSERT_NE(sourceError, nullptr);
  EXPECT_EQ(sourceError->code, "typed_unsupported");
}

TEST(TypedBind, LayoutOnlyBindOfTheSamePlanIsNotTyped) {
  // The v0 binder over the typed plan's layout would happily produce a
  // BoundPlan; bindTyped is what marks it. Consumers must never route the
  // plain v0 result of a typed layout to an executor either, which is why
  // typed plans are only reachable through decodeTypedPlan/bindTyped.
  TypedRelocationPlan plan = decoded(typed_goldens::kQuantDequantSymHex);
  auto layoutOnly = reloc::bind(plan.layout, {{"N", 8}});
  ASSERT_TRUE(std::holds_alternative<reloc::BoundPlan>(layoutOnly));
  EXPECT_FALSE(std::get<reloc::BoundPlan>(layoutOnly).typed);
  EXPECT_TRUE(bindOk(plan, {{"N", 8}}).layout.typed);
}

} // namespace
