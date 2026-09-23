//===- DispatchTest.cpp - plan-driven typed dispatch (R3) -----------------===//
//
// Capability is pure and lists only implemented rows; policy selects among
// them (forced CPU baseline, auto with and without a calibration, explicit
// row); execution through HostBackend is byte-identical to the scalar
// reference and reports what it moved; failures propagate without a second
// path. The CUDA section runs every eligible row on a real device against
// the same reference and is skipped without one.
//
//===----------------------------------------------------------------------===//

#include "reloc/Dispatch.h"

#include "TypedGoldens.h"
#include "reloc/Decode.h"
#include "reloc/HostBackend.h"
#include "gtest/gtest.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::BufferView;
using reloc::MemoryKind;
using reloc::TransferDirection;
using reloc::TransferError;
using reloc::TransferOptions;
using reloc::TypedBoundPlan;
using reloc::dispatch::Capability;
using reloc::dispatch::DispatchRequest;
using reloc::dispatch::Options;
using reloc::dispatch::Policy;
using reloc::typed::Program;
using typed_goldens::fromHex;

const char *kSynth = R"(# costmodel calibration v0
pcie.h2d_gbps 10
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.contiguous.convert_f32_f16_gbps 20
cpu.t8.contiguous.quantize_pack_gbps 20
cpu.t8.contiguous.pack_s8_s4_gbps 20
cpu.t8.blocked.gather_f32_gbps 5
cpu.t8.blocked.convert_f32_f16_gbps 20
cpu.t8.blocked.gather_quantize_gbps 4
cpu.t8.blocked.pack_s8_s4_gbps 20
hbm.bw_gbps 100
hbm.m.contiguous 1
hbm.m.blocked 2
overhead.a_ms 0.5
overhead.b_ms 0.1
)";

template <typename T>
std::vector<uint8_t> bytesOf(const std::vector<T> &values) {
  std::vector<uint8_t> out(values.size() * sizeof(T));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

reloc::ParameterValue f32Param(std::vector<float> values, bool perChannel) {
  reloc::ParameterValue v;
  v.elementType = reloc::ElementType{reloc::ElementTypeKind::Float, 32};
  if (perChannel)
    v.extents = {static_cast<int64_t>(values.size())};
  v.bytes = bytesOf(values);
  return v;
}

reloc::ParameterValue i32Param(int32_t value) {
  reloc::ParameterValue v;
  v.elementType = reloc::ElementType{reloc::ElementTypeKind::Integer, 32};
  v.bytes = bytesOf(std::vector<int32_t>{value});
  return v;
}

TypedBoundPlan mustBind(const char *hex, const reloc::SymbolMap &symbols,
                        const reloc::ParameterMap &parameters = {}) {
  std::vector<uint8_t> bytes = fromHex(hex);
  auto decoded = reloc::decodeTypedPlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::TypedRelocationPlan>(&decoded);
  if (plan == nullptr) {
    ADD_FAILURE() << std::get<reloc::DecodeError>(decoded).message;
    return {};
  }
  auto bound = reloc::bindTyped(*plan, symbols, parameters);
  auto *result = std::get_if<TypedBoundPlan>(&bound);
  if (result == nullptr) {
    ADD_FAILURE() << std::get<reloc::BindError>(bound).message;
    return {};
  }
  return *result;
}

Program mustPrepare(const TypedBoundPlan &plan) {
  auto prepared = reloc::typed::prepareProgram(plan);
  auto *program = std::get_if<Program>(&prepared);
  if (program == nullptr) {
    ADD_FAILURE() << std::get<reloc::typed::ExecutionError>(prepared).message;
    return {};
  }
  return *program;
}

BufferView view(const void *base, size_t bytes, std::vector<int64_t> extents,
                uint32_t width, MemoryKind kind = MemoryKind::Host,
                int device = -1) {
  BufferView v;
  v.base = reinterpret_cast<uintptr_t>(base);
  v.capacityBytes = bytes;
  v.offsetBytes = 0;
  v.extents = extents;
  v.strides.assign(extents.size(), 1);
  for (size_t k = extents.size(); k-- > 1;)
    v.strides[k - 1] = v.strides[k] * extents[k];
  v.elementSize = width;
  v.kind = kind;
  v.device = device;
  return v;
}

std::vector<uint8_t> reference(const Program &program,
                               const std::vector<uint8_t> &src) {
  std::vector<uint8_t> dst(static_cast<size_t>(program.plan.destinationBytes),
                           0);
  auto error = reloc::typed::executeHost(
      program, 0, static_cast<uint32_t>(program.stages.size()), src.data(),
      dst.data());
  EXPECT_FALSE(error.has_value()) << (error ? error->message : "");
  return dst;
}

std::set<std::string> labels(const Capability &capability) {
  std::set<std::string> out;
  for (const auto &row : capability.eligible)
    out.insert(row.label());
  return out;
}

std::string exclusion(const Capability &capability, const std::string &id) {
  for (const auto &row : capability.excluded)
    if (row.id == id)
      return row.reason;
  return "<eligible>";
}

std::vector<float> randomFloats(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-70.0f, 70.0f);
  std::vector<float> v(n);
  for (float &x : v)
    x = dist(rng);
  if (n >= 4) {
    v[0] = std::numeric_limits<float>::quiet_NaN();
    v[1] = std::numeric_limits<float>::infinity();
    v[2] = -std::numeric_limits<float>::infinity();
    v[3] = -0.0f;
  }
  return v;
}

DispatchRequest mustPrepareDispatch(const TypedBoundPlan &plan,
                                    const BufferView &source,
                                    const BufferView &destination,
                                    TransferDirection direction,
                                    const Options &options) {
  auto prepared = reloc::dispatch::prepareDispatch(plan, source, destination,
                                                   direction, options);
  auto *request = std::get_if<DispatchRequest>(&prepared);
  if (request == nullptr) {
    ADD_FAILURE() << std::get<TransferError>(prepared).code << ": "
                  << std::get<TransferError>(prepared).message;
    return {};
  }
  return std::move(*request);
}

//===----------------------------------------------------------------------===//
// Capability.
//===----------------------------------------------------------------------===//

TEST(Dispatch, CapabilityListsOnlyImplementedRows) {
  Program qd =
      mustPrepare(mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}}));
  // Without a CUDA device only the CPU reference exists.
  Capability host = reloc::dispatch::queryCapability(
      qd, TransferDirection::HostToDevice, /*cuda=*/false);
  EXPECT_EQ(labels(host), (std::set<std::string>{"cpu_reference"}));
  EXPECT_EQ(exclusion(host, "cuda_relocate_f32"), "no_cuda_device");
  EXPECT_EQ(exclusion(host, "cpu_stages_cuda_stages@1"), "no_cuda_device");
  // f32 -> s8 -> f32, per tensor: every element-wise kernel exists, the
  // f32 source can be relocated on the device, no dequantize leads.
  Capability cuda = reloc::dispatch::queryCapability(
      qd, TransferDirection::HostToDevice, /*cuda=*/true);
  EXPECT_EQ(
      labels(cuda),
      (std::set<std::string>{"cpu_reference", "cpu_stages_cuda_stages@0",
                             "cpu_stages_cuda_stages@1", "cuda_relocate_f32"}));
  EXPECT_EQ(exclusion(cuda, "cuda_dequant_relocate"),
            "stage 0 is not a dequantize");
  for (const auto &row : cuda.eligible) {
    if (row.label() == "cpu_stages_cuda_stages@1")
      EXPECT_EQ(row.wireBytes, 16); // the s8 intermediate
    if (row.label() == "cpu_reference")
      EXPECT_EQ(row.wireBytes, 64);
    if (row.id == "cuda_relocate_f32")
      EXPECT_EQ(row.method, "B");
  }
  Capability d2h = reloc::dispatch::queryCapability(
      qd, TransferDirection::DeviceToHost, /*cuda=*/true);
  EXPECT_EQ(labels(d2h),
            (std::set<std::string>{"cpu_reference", "cuda_stages_then_cpu@1",
                                   "cuda_stages_then_cpu@2"}));

  // A narrowing cast has no CUDA kernel: only the reference.
  Program dc = mustPrepare(mustBind(typed_goldens::kDequantCastHex, {}));
  Capability cast = reloc::dispatch::queryCapability(
      dc, TransferDirection::HostToDevice, true);
  EXPECT_EQ(labels(cast), (std::set<std::string>{"cpu_reference"}));
  EXPECT_EQ(exclusion(cast, "cpu_stages_cuda_stages@1"),
            "stage 1: no_cuda_kernel:cast_f32_f16");
  EXPECT_EQ(exclusion(cast, "cuda_dequant_relocate"), "rank below two");

  // Pads: no cut before the pad enters, no fused relocate kernel.
  Program cp = mustPrepare(mustBind(typed_goldens::kCastPadHex, {}));
  Capability pads = reloc::dispatch::queryCapability(
      cp, TransferDirection::HostToDevice, true);
  EXPECT_EQ(labels(pads), (std::set<std::string>{"cpu_reference"}));
  EXPECT_EQ(exclusion(pads, "cpu_stages_cuda_stages@0"), "pads_not_settled");
  EXPECT_EQ(exclusion(pads, "cuda_relocate_f32"), "layout has pads");

  // Per-channel quantize whose channel is result axis 0 after a transpose:
  // element-wise on the result layout is fine, on the source layout it is
  // not (the channel is source axis 1).
  Program wt =
      mustPrepare(mustBind(typed_goldens::kWitnessChannelTransposeHex, {}));
  Capability witness = reloc::dispatch::queryCapability(
      wt, TransferDirection::HostToDevice, true);
  EXPECT_EQ(labels(witness),
            (std::set<std::string>{"cpu_reference", "cpu_stages_cuda_stages@0",
                                   "cuda_relocate_f32"}));
  Capability witnessD2H = reloc::dispatch::queryCapability(
      wt, TransferDirection::DeviceToHost, true);
  EXPECT_EQ(labels(witnessD2H), (std::set<std::string>{"cpu_reference"}));
  EXPECT_EQ(exclusion(witnessD2H, "cuda_stages_then_cpu@1"),
            "stage 0: channel_not_source_outer_axis");

  // A nonzero zero point has no CUDA dequantize; zero point 0 does.
  reloc::ParameterMap params;
  params["s"] = f32Param({0.3f}, false);
  params["zp"] = i32Param(5);
  Program zp = mustPrepare(
      mustBind(typed_goldens::kDequantBindingHex, {{"N", 8}}, params));
  Capability nonzero = reloc::dispatch::queryCapability(
      zp, TransferDirection::HostToDevice, true);
  EXPECT_EQ(labels(nonzero), (std::set<std::string>{"cpu_reference"}));
  EXPECT_EQ(exclusion(nonzero, "cpu_stages_cuda_stages@0"),
            "stage 0: no_cuda_kernel:nonzero_zero_point");
  params["zp"] = i32Param(0);
  Program zero = mustPrepare(
      mustBind(typed_goldens::kDequantBindingHex, {{"N", 8}}, params));
  Capability affine = reloc::dispatch::queryCapability(
      zero, TransferDirection::HostToDevice, true);
  EXPECT_EQ(labels(affine), (std::set<std::string>{
                                "cpu_reference", "cpu_stages_cuda_stages@0"}));
  EXPECT_EQ(exclusion(affine, "cuda_dequant_relocate"), "rank below two");
  EXPECT_EQ(exclusion(affine, "cuda_relocate_f32"), "source is not f32");
}

//===----------------------------------------------------------------------===//
// Host execution: the forced baseline through the backend interface.
//===----------------------------------------------------------------------===//

TEST(Dispatch, HostToHostReferenceMatchesExecuteHostAndReports) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}});
  Program program = mustPrepare(plan);
  std::vector<uint8_t> src = bytesOf(randomFloats(16, 7));
  std::vector<uint8_t> dst(64, 0xCD);
  Options options;
  options.policy = Policy::OriginalCpu;
  DispatchRequest request =
      mustPrepareDispatch(plan, view(src.data(), src.size(), {16}, 4),
                          view(dst.data(), dst.size(), {16}, 4),
                          TransferDirection::HostToDevice, options);
  EXPECT_EQ(request.report.implementation, "cpu_reference");
  EXPECT_EQ(request.report.policy, "original_cpu");
  EXPECT_EQ(request.report.placementReason, "forced");
  EXPECT_EQ(request.report.method, "A");
  EXPECT_EQ(request.report.sourceBytes, 64);
  EXPECT_EQ(request.report.wireBytes, 64);
  EXPECT_EQ(request.report.destinationBytes, 64);
  EXPECT_EQ(request.report.parameterBytes, 8);
  EXPECT_EQ(request.report.wireBoundary, 2u);
  EXPECT_FALSE(request.report.executed);
  reloc::HostBackend backend(2);
  auto error =
      reloc::dispatch::executeDispatch(request, backend, TransferOptions{});
  ASSERT_FALSE(error.has_value()) << error->message;
  EXPECT_EQ(dst, reference(program, src));
  EXPECT_TRUE(request.report.executed);
  EXPECT_EQ(request.report.payloadBytesTransferred, 64);
  EXPECT_TRUE(request.consumed);
  error = reloc::dispatch::executeDispatch(request, backend, TransferOptions{});
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "already_executed");
}

TEST(Dispatch, DeviceToHostReferenceStagesThenExecutes) {
  reloc::ParameterMap params;
  params["s"] = f32Param({1.0f, 0.5f, 0.25f}, true);
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantizeTransposeHex, {{"B", 5}}, params);
  Program program = mustPrepare(plan);
  std::vector<uint8_t> src = bytesOf(randomFloats(15, 11));
  std::vector<uint8_t> dst(15, 0xCD);
  Options options;
  options.policy = Policy::OriginalCpu;
  DispatchRequest request =
      mustPrepareDispatch(plan, view(src.data(), src.size(), {5, 3}, 4),
                          view(dst.data(), dst.size(), {3, 5}, 1),
                          TransferDirection::DeviceToHost, options);
  EXPECT_EQ(request.report.wireBytes, 60); // the dense source crosses
  EXPECT_EQ(request.report.wireBoundary, 0u);
  EXPECT_EQ(request.report.method, "B");
  reloc::HostBackend backend(1);
  TransferOptions transfer;
  transfer.gatherThreads = 3;
  auto error = reloc::dispatch::executeDispatch(request, backend, transfer);
  ASSERT_FALSE(error.has_value()) << error->message;
  EXPECT_EQ(dst, reference(program, src));
  EXPECT_EQ(request.report.payloadBytesTransferred, 60);
}

//===----------------------------------------------------------------------===//
// Policy.
//===----------------------------------------------------------------------===//

TEST(Dispatch, AutoFallsBackToTheReferenceWithARecordedReason) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}});
  std::vector<uint8_t> src(64), dst(64);
  BufferView s = view(src.data(), 64, {16}, 4),
             d = view(dst.data(), 64, {16}, 4);
  Options options;
  options.policy = Policy::Auto;
  DispatchRequest request =
      mustPrepareDispatch(plan, s, d, TransferDirection::HostToDevice, options);
  EXPECT_EQ(request.report.implementation, "cpu_reference");
  EXPECT_EQ(request.report.policy, "auto");
  EXPECT_EQ(request.report.placementReason, "only_qualified_path");
  options.cuda = true; // alternatives exist, but nothing prices them
  request =
      mustPrepareDispatch(plan, s, d, TransferDirection::HostToDevice, options);
  EXPECT_EQ(request.report.implementation, "cpu_reference");
  EXPECT_EQ(request.report.placementReason, "no_calibration");
}

TEST(Dispatch, AutoTranslatesAdviceOnlyToEligibleRows) {
  auto parsed = reloc::costmodel::CostModel::parse(kSynth);
  ASSERT_TRUE(std::holds_alternative<reloc::costmodel::CostModel>(parsed));
  const auto &model = std::get<reloc::costmodel::CostModel>(parsed);
  Options options;
  options.policy = Policy::Auto;
  options.cuda = true;
  options.model = &model;
  // Small tensor: the synthetic calibration prefers B (transfer, then the
  // device works) -> the f32 source is relocated and quantized on the GPU.
  {
    const int64_t n = 1 << 10;
    TypedBoundPlan plan =
        mustBind(typed_goldens::kQuantDequantSymHex, {{"N", n}});
    std::vector<uint8_t> src(n * 4), dst(n * 4);
    DispatchRequest request =
        mustPrepareDispatch(plan, view(src.data(), src.size(), {n}, 4),
                            view(dst.data(), dst.size(), {n}, 4),
                            TransferDirection::HostToDevice, options);
    EXPECT_EQ(request.report.implementation, "cuda_relocate_f32");
    EXPECT_EQ(request.report.placementReason, "cost_model_prefers_b");
    EXPECT_EQ(request.report.wireBytes, n * 4);
    EXPECT_EQ(request.report.method, "B");
  }
  // Large tensor: A wins -> the cheapest CPU-first cut, the s8 wire.
  {
    const int64_t n = 1 << 22;
    TypedBoundPlan plan =
        mustBind(typed_goldens::kQuantDequantSymHex, {{"N", n}});
    // Views only need to be declared; nothing is touched at preparation.
    std::vector<uint8_t> src(16), dst(16);
    BufferView s = view(src.data(), n * 4, {n}, 4);
    BufferView d = view(dst.data(), n * 4, {n}, 4);
    DispatchRequest request = mustPrepareDispatch(
        plan, s, d, TransferDirection::HostToDevice, options);
    EXPECT_EQ(request.report.implementation, "cpu_stages_cuda_stages@1");
    EXPECT_EQ(request.report.placementReason, "cost_model_prefers_a");
    EXPECT_EQ(request.report.wireBytes, n);
    EXPECT_EQ(request.report.method, "A");
  }
  // Advice for a class with no eligible row falls back and says so; a
  // class the model prices is never invented for a row that does not exist.
  {
    reloc::ParameterMap params;
    params["s"] = f32Param({0.3f}, false);
    params["zp"] = i32Param(0);
    const int64_t n = 1 << 10;
    TypedBoundPlan plan =
        mustBind(typed_goldens::kDequantBindingHex, {{"N", n}}, params);
    Program program = mustPrepare(plan);
    std::vector<uint8_t> src(n), dst(n * 4);
    DispatchRequest request =
        mustPrepareDispatch(plan, view(src.data(), src.size(), {n}, 1),
                            view(dst.data(), dst.size(), {n}, 4),
                            TransferDirection::HostToDevice, options);
    Capability capability = reloc::dispatch::queryCapability(
        program, TransferDirection::HostToDevice, true);
    EXPECT_EQ(
        labels(capability),
        (std::set<std::string>{"cpu_reference", "cpu_stages_cuda_stages@0"}));
    auto decision = reloc::costmodel::decide(
        model, reloc::costmodel::classify(program.plan.layout), n, 1.0, 8);
    ASSERT_TRUE(decision.has_value());
    if (decision->method == reloc::costmodel::MethodDecision::Method::B) {
      EXPECT_EQ(request.report.placementReason, "advice_unavailable_path");
    } else {
      EXPECT_EQ(request.report.placementReason, "cost_model_prefers_a");
    }
    EXPECT_EQ(request.report.implementation, "cpu_stages_cuda_stages@0");
    EXPECT_TRUE(labels(capability).count(request.report.implementation));
  }
  // An explicit row must be eligible.
  {
    TypedBoundPlan plan = mustBind(typed_goldens::kDequantCastHex, {});
    std::vector<uint8_t> src(4), dst(8);
    Options explicitRow = options;
    explicitRow.implementation = "cpu_stages_cuda_stages@1";
    auto prepared = reloc::dispatch::prepareDispatch(
        plan, view(src.data(), 4, {4}, 1), view(dst.data(), 8, {4}, 2),
        TransferDirection::HostToDevice, explicitRow);
    ASSERT_TRUE(std::holds_alternative<TransferError>(prepared));
    EXPECT_EQ(std::get<TransferError>(prepared).code,
              "implementation_unavailable");
    explicitRow.implementation = "cpu_reference";
    DispatchRequest request = mustPrepareDispatch(
        plan, view(src.data(), 4, {4}, 1), view(dst.data(), 8, {4}, 2),
        TransferDirection::HostToDevice, explicitRow);
    EXPECT_EQ(request.report.policy, "explicit");
  }
}

//===----------------------------------------------------------------------===//
// Rejections and failures.
//===----------------------------------------------------------------------===//

TEST(Dispatch, ViewsAndProgramsAreCheckedBeforeSelection) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}});
  std::vector<uint8_t> src(64), dst(64);
  Options options;
  auto code = [&](const BufferView &s, const BufferView &d) {
    auto prepared = reloc::dispatch::prepareDispatch(
        plan, s, d, TransferDirection::HostToDevice, options);
    return std::holds_alternative<TransferError>(prepared)
               ? std::get<TransferError>(prepared).code
               : std::string("ok");
  };
  EXPECT_EQ(code(view(src.data(), 64, {16}, 4), view(dst.data(), 64, {16}, 4)),
            "ok");
  // Source width is the program's source type, not the destination's.
  EXPECT_EQ(code(view(src.data(), 64, {32}, 2), view(dst.data(), 64, {16}, 4)),
            "plan_mismatch");
  // Destination must span exactly the padded result.
  EXPECT_EQ(code(view(src.data(), 64, {16}, 4), view(dst.data(), 64, {8}, 4)),
            "plan_mismatch");
  EXPECT_EQ(code(view(src.data(), 63, {16}, 4), view(dst.data(), 64, {16}, 4)),
            "insufficient_capacity");
  BufferView strided = view(src.data(), 64, {8}, 4);
  strided.strides = {2};
  EXPECT_EQ(code(strided, view(dst.data(), 64, {16}, 4)), "unsupported_layout");
  BufferView cudaSource = view(src.data(), 64, {16}, 4, MemoryKind::Cuda, 0);
  EXPECT_EQ(code(cudaSource, view(dst.data(), 64, {16}, 4)),
            "direction_mismatch");
  // An unsupported stage is refused before any view is examined.
  TypedBoundPlan bad = plan;
  bad.stages[0].policy = reloc::NumericPolicyKind::Affine;
  auto prepared = reloc::dispatch::prepareDispatch(
      bad, view(src.data(), 64, {16}, 4), view(dst.data(), 64, {16}, 4),
      TransferDirection::HostToDevice, options);
  ASSERT_TRUE(std::holds_alternative<TransferError>(prepared));
  EXPECT_EQ(std::get<TransferError>(prepared).code, "unsupported_stage");
  options.policy = Policy::OriginalCpu;
  prepared = reloc::dispatch::prepareDispatch(
      bad, view(src.data(), 64, {16}, 4), view(dst.data(), 64, {16}, 4),
      TransferDirection::HostToDevice, options);
  ASSERT_TRUE(std::holds_alternative<TransferError>(prepared));
  EXPECT_EQ(std::get<TransferError>(prepared).code, "unsupported_stage")
      << "an unavailable baseline is never reported as forced";
}

class FailingStagingBackend : public reloc::HostBackend {
public:
  FailingStagingBackend() : HostBackend(1) {}
  void *allocStaging(size_t) override { return nullptr; }
};

TEST(Dispatch, BackendFailuresPropagateWithoutASecondPath) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}});
  std::vector<uint8_t> src = bytesOf(randomFloats(16, 3));
  std::vector<uint8_t> dst(64, 0xCD);
  Options options;
  options.policy = Policy::OriginalCpu;
  FailingStagingBackend backend;
  for (TransferDirection direction :
       {TransferDirection::HostToDevice, TransferDirection::DeviceToHost}) {
    DispatchRequest request = mustPrepareDispatch(
        plan, view(src.data(), src.size(), {16}, 4),
        view(dst.data(), dst.size(), {16}, 4), direction, options);
    auto error =
        reloc::dispatch::executeDispatch(request, backend, TransferOptions{});
    ASSERT_TRUE(error.has_value());
    EXPECT_EQ(error->code, "backend_failure");
    EXPECT_TRUE(request.consumed);
    EXPECT_FALSE(request.report.executed);
    EXPECT_EQ(std::count(dst.begin(), dst.end(), 0xCD),
              static_cast<long>(dst.size()));
  }
  // A CUDA-only row cannot run on a host backend.
  options.policy = Policy::Auto;
  options.cuda = true;
  options.implementation = "cpu_stages_cuda_stages@1";
  DispatchRequest request =
      mustPrepareDispatch(plan, view(src.data(), src.size(), {16}, 4),
                          view(dst.data(), dst.size(), {16}, 4),
                          TransferDirection::HostToDevice, options);
  reloc::HostBackend host(1);
  auto error =
      reloc::dispatch::executeDispatch(request, host, TransferOptions{});
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "backend_mismatch");
}

//===----------------------------------------------------------------------===//
// Prefold capability for T4, and the pure selection entry point.
//===----------------------------------------------------------------------===//

TEST(Dispatch, PrefoldSpecMatchesOnlyTheS8QuantizeVariants) {
  // C1's channel witness transposed to [3, 2, 4]: the channel is the
  // coalesced outer axis (a bare result dimension), the destination is
  // packed, source strides differ -> the fused gather variant with
  // inv = fl32(1/scale) formed from the declared scales.
  TypedBoundPlan witness =
      mustBind(typed_goldens::kWitnessChannelTransposeHex, {});
  auto spec = reloc::dispatch::prefoldSpecFor(witness);
  ASSERT_TRUE(std::holds_alternative<reloc::dispatch::PrefoldSpec>(spec))
      << std::get<TransferError>(spec).message;
  EXPECT_EQ(std::get<reloc::dispatch::PrefoldSpec>(spec).spec,
            reloc::prefold::OutputSpec::S8GatherQuant);
  EXPECT_EQ(std::get<reloc::dispatch::PrefoldSpec>(spec).invScales,
            (std::vector<float>{2.0f, 1.0f, 0.5f}));
  // Per-channel quantize on axis 0 of an identity [B, 3] layout: the binder
  // coalesces the identity to ONE axis, so no distinct channel axis is left
  // for the pack variant (the T4 handoff fact: s8_quant_pack needs a
  // channel-preserving plan).
  reloc::ParameterMap params;
  params["s"] = f32Param({0.5f, 0.25f, 0.125f, 1.0f}, true);
  TypedBoundPlan channel =
      mustBind(typed_goldens::kQuantizeChannelSymHex, {{"B", 4}}, params);
  spec = reloc::dispatch::prefoldSpecFor(channel);
  ASSERT_TRUE(std::holds_alternative<TransferError>(spec));
  EXPECT_EQ(std::get<TransferError>(spec).code, "prefold_unavailable");
  EXPECT_NE(std::get<TransferError>(spec).message.find("rank >= 2"),
            std::string::npos);
  auto reason = [](const TypedBoundPlan &p) {
    auto s = reloc::dispatch::prefoldSpecFor(p);
    return std::holds_alternative<TransferError>(s)
               ? std::get<TransferError>(s).code + ": " +
                     std::get<TransferError>(s).message
               : std::string("<available>");
  };
  // Two stages, pads: no prefolder implements them.
  EXPECT_NE(reason(mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 8}}))
                .find("exactly one"),
            std::string::npos);
  EXPECT_NE(reason(mustBind(typed_goldens::kPadQuantizeHex, {}))
                .find("prefold_unavailable: layout has pads"),
            std::string::npos);
  // A channel map that is not a bare result dimension ((d0 mod B) here,
  // equal to d0 in value) is conservatively unavailable: the prefolder
  // indexes channels by the outer axis and nothing proves the map is it.
  params.clear();
  params["s"] = f32Param({0.5f, 0.25f, 0.125f}, true);
  EXPECT_NE(
      reason(mustBind(typed_goldens::kQuantizeTransposeHex, {{"B", 8}}, params))
          .find("not the coalesced outer axis"),
      std::string::npos);
}

TEST(Dispatch, SelectionWithoutBuffersMatchesPreparation) {
  TypedBoundPlan plan =
      mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 16}});
  Options options;
  options.policy = Policy::OriginalCpu;
  auto selected = reloc::dispatch::selectImplementation(
      plan, TransferDirection::HostToDevice, options);
  ASSERT_TRUE(std::holds_alternative<reloc::dispatch::Selection>(selected));
  EXPECT_EQ(std::get<reloc::dispatch::Selection>(selected).row.label(),
            "cpu_reference");
  EXPECT_EQ(std::get<reloc::dispatch::Selection>(selected).policy,
            "original_cpu");
  options.policy = Policy::Auto;
  options.cuda = true;
  options.implementation = "cuda_dequant_relocate";
  selected = reloc::dispatch::selectImplementation(
      plan, TransferDirection::HostToDevice, options);
  ASSERT_TRUE(std::holds_alternative<TransferError>(selected));
  EXPECT_EQ(std::get<TransferError>(selected).code,
            "implementation_unavailable");
}

//===----------------------------------------------------------------------===//
// CUDA rows on a real device (local only; skipped without a GPU).
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

bool haveGpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct DeviceBuffer {
  void *p = nullptr;
  size_t bytes = 0;
  explicit DeviceBuffer(size_t n) : bytes(n) { cudaMalloc(&p, n); }
  ~DeviceBuffer() { cudaFree(p); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  std::vector<uint8_t> download() const {
    std::vector<uint8_t> h(bytes);
    EXPECT_EQ(cudaSuccess,
              cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    return h;
  }
  void upload(const std::vector<uint8_t> &h) const {
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice));
  }
  void fill(uint8_t value) const { cudaMemset(p, value, bytes); }
};

struct Case {
  const char *name;
  TypedBoundPlan plan;
  std::vector<uint8_t> src;
  std::vector<int64_t> sourceExtents;
  std::vector<int64_t> resultExtents;
};

std::vector<Case> cudaCases() {
  std::vector<Case> cases;
  {
    TypedBoundPlan plan =
        mustBind(typed_goldens::kQuantDequantSymHex, {{"N", 1000}});
    cases.push_back({"quant_dequant_sym",
                     plan,
                     bytesOf(randomFloats(1000, 21)),
                     {1000},
                     {1000}});
  }
  {
    TypedBoundPlan plan =
        mustBind(typed_goldens::kWitnessChannelTransposeHex, {});
    std::vector<float> x(24);
    for (int i = 0; i < 24; ++i)
      x[static_cast<size_t>(i)] = static_cast<float>(i) - 11.5f;
    cases.push_back(
        {"witness_channel_transpose", plan, bytesOf(x), {2, 3, 4}, {3, 2, 4}});
  }
  {
    reloc::ParameterMap params;
    params["s"] = f32Param({0.3f}, false);
    params["zp"] = i32Param(0);
    TypedBoundPlan plan =
        mustBind(typed_goldens::kDequantBindingHex, {{"N", 777}}, params);
    std::vector<int8_t> q(777);
    for (size_t i = 0; i < q.size(); ++i)
      q[i] = static_cast<int8_t>(static_cast<int>(i * 37) % 256 - 128);
    cases.push_back({"dequant_binding", plan, bytesOf(q), {777}, {777}});
  }
  {
    reloc::ParameterMap params;
    params["s"] = f32Param({1.0f, 0.5f, 0.25f}, true);
    TypedBoundPlan plan =
        mustBind(typed_goldens::kQuantizeTransposeHex, {{"B", 64}}, params);
    cases.push_back({"quantize_transpose",
                     plan,
                     bytesOf(randomFloats(192, 5)),
                     {64, 3},
                     {3, 64}});
  }
  return cases;
}

TEST(CudaDispatch, EveryEligibleHostToDeviceRowMatchesTheReference) {
  if (!haveGpu())
    GTEST_SKIP() << "no CUDA device";
  reloc::CudaBackend backend(2);
  ASSERT_FALSE(backend.failed()) << backend.error();
  int rows = 0;
  for (const Case &c : cudaCases()) {
    Program program = mustPrepare(c.plan);
    std::vector<uint8_t> expected = reference(program, c.src);
    Capability capability = reloc::dispatch::queryCapability(
        program, TransferDirection::HostToDevice, true);
    for (const auto &row : capability.eligible) {
      DeviceBuffer dst(expected.size());
      dst.fill(0xCD);
      Options options;
      options.cuda = true;
      options.implementation = row.label();
      DispatchRequest request = mustPrepareDispatch(
          c.plan,
          view(c.src.data(), c.src.size(), c.sourceExtents,
               reloc::typed::widthAt(program, 0)),
          view(dst.p, dst.bytes, c.resultExtents,
               reloc::typed::widthAt(
                   program, static_cast<uint32_t>(program.stages.size())),
               MemoryKind::Cuda, backend.device()),
          TransferDirection::HostToDevice, options);
      auto error =
          reloc::dispatch::executeDispatch(request, backend, TransferOptions{});
      ASSERT_FALSE(error.has_value())
          << c.name << " " << row.label() << ": " << error->message;
      EXPECT_EQ(dst.download(), expected) << c.name << " " << row.label();
      EXPECT_TRUE(request.report.executed);
      EXPECT_EQ(request.report.implementation, row.label());
      EXPECT_GE(request.report.payloadBytesTransferred, row.wireBytes)
          << row.label();
      EXPECT_EQ(request.report.wireBytes, row.wireBytes);
      ++rows;
    }
  }
  EXPECT_GE(rows, 10);
}

TEST(CudaDispatch, EveryEligibleDeviceToHostRowMatchesTheReference) {
  if (!haveGpu())
    GTEST_SKIP() << "no CUDA device";
  reloc::CudaBackend backend(2);
  ASSERT_FALSE(backend.failed()) << backend.error();
  int rows = 0;
  for (const Case &c : cudaCases()) {
    Program program = mustPrepare(c.plan);
    std::vector<uint8_t> expected = reference(program, c.src);
    DeviceBuffer src(c.src.size());
    src.upload(c.src);
    Capability capability = reloc::dispatch::queryCapability(
        program, TransferDirection::DeviceToHost, true);
    for (const auto &row : capability.eligible) {
      std::vector<uint8_t> dst(expected.size(), 0xCD);
      Options options;
      options.cuda = true;
      options.implementation = row.label();
      DispatchRequest request = mustPrepareDispatch(
          c.plan,
          view(src.p, src.bytes, c.sourceExtents,
               reloc::typed::widthAt(program, 0), MemoryKind::Cuda,
               backend.device()),
          view(dst.data(), dst.size(), c.resultExtents,
               reloc::typed::widthAt(
                   program, static_cast<uint32_t>(program.stages.size()))),
          TransferDirection::DeviceToHost, options);
      auto error =
          reloc::dispatch::executeDispatch(request, backend, TransferOptions{});
      ASSERT_FALSE(error.has_value())
          << c.name << " " << row.label() << ": " << error->message;
      EXPECT_EQ(dst, expected) << c.name << " " << row.label();
      EXPECT_GE(request.report.payloadBytesTransferred, row.wireBytes);
      ++rows;
    }
  }
  EXPECT_GE(rows, 6);
}

#endif // RELOC_ENABLE_CUDA

} // namespace
