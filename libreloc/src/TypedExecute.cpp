//===- TypedExecute.cpp - scalar reference execution of typed plans -------===//

#include "reloc/TypedExecute.h"

#include "reloc/GatherPool.h"
#include "reloc/Quant.h"
#include "reloc/TypedValue.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>

namespace reloc {
namespace typed {
namespace {

ExecutionError fail(const char *code, std::string message) {
  return ExecutionError{code, std::move(message)};
}

bool isF32(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 32;
}
bool isF16(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 16;
}
bool isS8(ElementType t) {
  return t.kind == ElementTypeKind::Integer && t.bitwidth == 8;
}

float f32FromBits(uint64_t bits) {
  uint32_t b = static_cast<uint32_t>(bits);
  float x;
  std::memcpy(&x, &b, sizeof(x));
  return x;
}

uint64_t bitsFromF32(float x) {
  uint32_t b;
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

/// Read `count` little-endian scalars of `param` as host values.
bool readScales(const BoundParameter &param, std::vector<float> &out,
                std::string &why) {
  if (!param.present) {
    why = "scale is absent";
    return false;
  }
  if (!isF32(param.elementType)) {
    why = "scale is not f32";
    return false;
  }
  if (param.bytes.size() != static_cast<size_t>(param.length) * 4) {
    why = "scale byte size disagrees with its length";
    return false;
  }
  out.resize(static_cast<size_t>(param.length));
  for (int64_t i = 0; i < param.length; ++i) {
    uint32_t b;
    std::memcpy(&b, param.bytes.data() + i * 4, 4);
    out[static_cast<size_t>(i)] = f32FromBits(b);
    if (!std::isfinite(out[static_cast<size_t>(i)]) ||
        out[static_cast<size_t>(i)] <= 0.0f) {
      why = "scale is not finite and strictly positive";
      return false;
    }
  }
  return true;
}

bool readZeroPoints(const BoundParameter &param, std::vector<int32_t> &out,
                    std::string &why) {
  if (!param.present) {
    out.assign(1, 0);
    return true;
  }
  if (param.elementType.kind != ElementTypeKind::Integer ||
      param.elementType.bitwidth == 0 || param.elementType.bitwidth > 64 ||
      param.elementType.bitwidth % 8 != 0) {
    why = "zero point is not a byte-multiple integer type";
    return false;
  }
  const size_t width = param.elementType.bitwidth / 8;
  if (param.bytes.size() != static_cast<size_t>(param.length) * width) {
    why = "zero point byte size disagrees with its length";
    return false;
  }
  out.resize(static_cast<size_t>(param.length));
  for (int64_t i = 0; i < param.length; ++i) {
    uint64_t raw = 0;
    std::memcpy(&raw, param.bytes.data() + i * width, width);
    // Sign-extend from the declared width.
    const unsigned shift = 64 - param.elementType.bitwidth;
    int64_t value =
        static_cast<int64_t>(raw << shift) >> static_cast<int64_t>(shift);
    if (value < -128 || value > 127) {
      why = "zero point outside [-128, 127]";
      return false;
    }
    out[static_cast<size_t>(i)] = static_cast<int32_t>(value);
  }
  return true;
}

/// Write `count` copies of the low `width` bytes of `bits`.
void fillPattern(uint8_t *dst, uint64_t bits, uint32_t width, int64_t count) {
  uint8_t pattern[8];
  for (uint32_t b = 0; b < width; ++b)
    pattern[b] = static_cast<uint8_t>(bits >> (8 * b));
  for (int64_t i = 0; i < count; ++i)
    std::memcpy(dst + i * width, pattern, width);
}

uint64_t loadBits(const uint8_t *p, uint32_t width) {
  uint64_t bits = 0;
  std::memcpy(&bits, p, width);
  return bits;
}

void storeBits(uint8_t *p, uint64_t bits, uint32_t width) {
  std::memcpy(p, &bits, width);
}

struct StageRange {
  uint32_t from;
  uint32_t to;
  bool needsCoordinates;
};

/// Per-walk shared state: the first error wins, workers stop.
struct WalkState {
  const Program &program;
  StageRange range;
  const uint8_t *src;
  uint8_t *dst;
  uint32_t widthFrom;
  uint32_t widthTo;
  std::vector<int64_t> lo; // pad lo per coalesced axis
  std::atomic<bool> failed{false};
  std::mutex mutex;
  ExecutionError error;

  void setError(ExecutionError e) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!failed.load()) {
      error = std::move(e);
      failed.store(true);
    }
  }
};

/// One element: load, run the stage range, store. Returns false on error.
bool element(WalkState &state, int64_t srcOff, int64_t dstOff,
             std::vector<int64_t> &coords) {
  const Program &program = state.program;
  uint64_t bits =
      loadBits(state.src + srcOff * state.widthFrom, state.widthFrom);
  if (state.range.needsCoordinates) {
    int64_t rest = dstOff;
    for (size_t j = 0; j < coords.size(); ++j) {
      coords[j] = rest / program.resultStrides[j];
      rest -= coords[j] * program.resultStrides[j];
    }
  }
  for (uint32_t k = state.range.from; k < state.range.to; ++k) {
    const StageArithmetic &stage = program.stages[k];
    int64_t channel = 0;
    if (stage.perChannel) {
      if (stage.channelIsDim) {
        channel = coords[stage.channelDim];
      } else {
        std::string why;
        if (!evalChannel(program.plan.stages[k].channel, program.plan.symbols,
                         coords, channel, why)) {
          state.setError(fail("channel_evaluation_failed",
                              "stage " + std::to_string(k) + ": " + why));
          return false;
        }
      }
      if (channel < 0 || channel >= stage.channelLength) {
        state.setError(fail("channel_out_of_range",
                            "stage " + std::to_string(k) +
                                " selected channel " + std::to_string(channel) +
                                " of " + std::to_string(stage.channelLength)));
        return false;
      }
    }
    bits = applyStage(stage, bits, channel);
  }
  storeBits(state.dst + dstOff * state.widthTo, bits, state.widthTo);
  return true;
}

void walk(WalkState &state, size_t depth, int64_t iBegin, int64_t iEnd,
          int64_t srcOff, int64_t dstOff, std::vector<int64_t> &coords) {
  const BoundPlan &b = state.program.plan.layout;
  const size_t r = b.extents.size();
  if (depth == r - 1) {
    for (int64_t i = iBegin; i < iEnd; ++i) {
      if (state.failed.load(std::memory_order_relaxed))
        return;
      if (!element(state, srcOff + i * b.srcStrides[depth],
                   dstOff + (i + state.lo[depth]) * b.dstStrides[depth],
                   coords))
        return;
    }
    return;
  }
  for (int64_t i = iBegin; i < iEnd; ++i) {
    if (state.failed.load(std::memory_order_relaxed))
      return;
    walk(state, depth + 1, 0, b.extents[depth + 1],
         srcOff + i * b.srcStrides[depth],
         dstOff + (i + state.lo[depth]) * b.dstStrides[depth], coords);
  }
}

void walkRows(WalkState &state, int64_t begin, int64_t end) {
  std::vector<int64_t> coords(state.program.plan.resultExtents.size(), 0);
  walk(state, 0, begin, end, 0, 0, coords);
}

} // namespace

std::variant<Program, ExecutionError>
prepareProgram(const TypedBoundPlan &plan) {
  Program program;
  program.plan = plan;
  const size_t rank = plan.resultExtents.size();
  program.resultStrides.assign(rank, 1);
  for (size_t j = rank; j-- > 1;)
    program.resultStrides[j - 1] =
        program.resultStrides[j] * plan.resultExtents[j];
  program.resultElements = 1;
  for (int64_t e : plan.resultExtents)
    program.resultElements *= e;
  program.sourceElements = 1;
  for (int64_t e : plan.sourceExtents)
    program.sourceElements *= e;

  for (size_t k = 0; k < plan.stages.size(); ++k) {
    const BoundStage &bound = plan.stages[k];
    StageArithmetic stage;
    stage.transform = bound.transform;
    stage.policy = bound.policy;
    stage.input = bound.input.type;
    stage.output = bound.output.type;
    const std::string where = "stage " + std::to_string(k) + ": ";
    std::string why;
    switch (bound.transform) {
    case ValueTransformKind::Cast: {
      const bool narrow = isF32(stage.input) && isF16(stage.output) &&
                          bound.policy == NumericPolicyKind::IeeeRne;
      const bool widen = isF16(stage.input) && isF32(stage.output) &&
                         bound.policy == NumericPolicyKind::Exact;
      if (!narrow && !widen)
        return fail("unsupported_stage",
                    where + "only cast f32->f16 ieee_rne and f16->f32 exact "
                            "have a reference implementation");
      break;
    }
    case ValueTransformKind::Quantize: {
      if (!isF32(stage.input) || !isS8(stage.output) ||
          bound.policy != NumericPolicyKind::SymmetricRne)
        return fail("unsupported_stage",
                    where + "only quantize f32->s8 symmetric_rne has a "
                            "reference implementation");
      if (!readScales(bound.scale, stage.scale, why))
        return fail("invalid_parameter", where + why);
      std::vector<int32_t> zeroPoints;
      if (!readZeroPoints(bound.zeroPoint, zeroPoints, why))
        return fail("invalid_parameter", where + why);
      for (int32_t zp : zeroPoints)
        if (zp != 0)
          return fail("invalid_parameter",
                      where + "symmetric_rne admits only zero point 0");
      stage.invScale.resize(stage.scale.size());
      for (size_t i = 0; i < stage.scale.size(); ++i)
        stage.invScale[i] = 1.0f / stage.scale[i]; // fl32(1/scale), once
      break;
    }
    case ValueTransformKind::Dequantize: {
      if (!isS8(stage.input) || !isF32(stage.output) ||
          bound.policy != NumericPolicyKind::Affine)
        return fail("unsupported_stage",
                    where + "only dequantize s8->f32 affine has a reference "
                            "implementation");
      if (!readScales(bound.scale, stage.scale, why))
        return fail("invalid_parameter", where + why);
      if (!readZeroPoints(bound.zeroPoint, stage.zeroPoint, why))
        return fail("invalid_parameter", where + why);
      break;
    }
    }
    stage.perChannel = bound.hasChannel;
    if (bound.hasChannel) {
      const size_t entries = stage.scale.size();
      if (entries == 0)
        return fail("invalid_parameter",
                    where + "per-channel stage without scales");
      stage.channelLength = static_cast<int64_t>(entries);
      if (stage.zeroPoint.size() > 1 &&
          stage.zeroPoint.size() != stage.scale.size())
        return fail("invalid_parameter",
                    where + "zero point length disagrees with the scale");
      const ExprStream &channel = bound.channel;
      if (channel.size() == 1 && channel.front().op == ExprOp::PushDim &&
          channel.front().value >= 0 &&
          static_cast<size_t>(channel.front().value) < rank) {
        stage.channelIsDim = true;
        stage.channelDim = static_cast<uint32_t>(channel.front().value);
      }
      program.needsCoordinates = true;
    } else {
      if (stage.scale.size() > 1 || stage.zeroPoint.size() > 1)
        return fail("invalid_parameter",
                    where + "per-tensor stage with more than one parameter");
    }
    program.stages.push_back(std::move(stage));
  }
  return program;
}

ElementType typeAt(const Program &program, uint32_t boundary) {
  if (boundary == 0)
    return program.plan.sourceType;
  return program.stages[boundary - 1].output;
}

uint32_t widthAt(const Program &program, uint32_t boundary) {
  return byteWidth(typeAt(program, boundary));
}

bool padsSettledBy(const Program &program, uint32_t boundary) {
  for (const TypedFill &fill : program.plan.fills)
    if (fill.stage > boundary)
      return false;
  return true;
}

std::variant<std::optional<uint64_t>, ExecutionError>
fillAt(const Program &program, uint32_t boundary) {
  if (boundary > program.stages.size())
    return fail("invalid_boundary", "boundary beyond the last stage");
  if (program.plan.fills.empty())
    return std::optional<uint64_t>();
  std::optional<uint64_t> agreed;
  for (const TypedFill &fill : program.plan.fills) {
    if (fill.stage > boundary)
      return fail("pad_not_entered",
                  "a pad enters at stage " + std::to_string(fill.stage) +
                      ", after boundary " + std::to_string(boundary));
    uint64_t bits = fill.bits;
    ElementType type = fill.type;
    for (uint32_t k = fill.stage; k < boundary; ++k) {
      const StageArithmetic &stage = program.stages[k];
      if (stage.perChannel)
        return fail("fill_crosses_channel_stage",
                    "a pad fill would cross per-channel stage " +
                        std::to_string(k));
      if (!sameType(type, stage.input))
        return fail("fill_type_mismatch",
                    "pad fill type disagrees with stage " + std::to_string(k));
      bits = applyStage(stage, bits, 0);
      type = stage.output;
    }
    if (!sameType(type, typeAt(program, boundary)))
      return fail("fill_type_mismatch",
                  "pad fill type disagrees with boundary " +
                      std::to_string(boundary));
    if (agreed && *agreed != bits)
      return fail("mixed_fills_unsupported",
                  "pads carry different values at boundary " +
                      std::to_string(boundary));
    agreed = bits;
  }
  if (boundary == program.stages.size() &&
      !program.plan.layout.padRegions.empty() &&
      program.plan.layout.padRegions.front().fillBits != *agreed)
    return fail("fill_mismatch",
                "folded fill disagrees with the layout's fused fill");
  return agreed;
}

int64_t bytesAt(const Program &program, uint32_t boundary, bool resultLayout) {
  const int64_t elements =
      resultLayout ? program.resultElements : program.sourceElements;
  return elements * static_cast<int64_t>(widthAt(program, boundary));
}

uint64_t applyStage(const StageArithmetic &stage, uint64_t bits,
                    int64_t channel) {
  assert(channel >= 0 && (!stage.perChannel || channel < stage.channelLength));
  const size_t c = stage.perChannel ? static_cast<size_t>(channel) : 0;
  switch (stage.transform) {
  case ValueTransformKind::Cast:
    if (isF32(stage.input))
      return quant::narrowF32F16(f32FromBits(bits));
    return bitsFromF32(quant::widenF16F32(static_cast<uint16_t>(bits)));
  case ValueTransformKind::Quantize: {
    const int8_t q =
        quant::quantizeOneF32S8(f32FromBits(bits), stage.invScale[c]);
    return static_cast<uint8_t>(q);
  }
  case ValueTransformKind::Dequantize: {
    const int8_t q = static_cast<int8_t>(static_cast<uint8_t>(bits & 0xffu));
    const int32_t zp =
        stage.zeroPoint.size() == 1 ? stage.zeroPoint[0] : stage.zeroPoint[c];
    // d is an exact small integer; one binary32 multiply follows.
    const float d = static_cast<float>(static_cast<int32_t>(q) - zp);
    return bitsFromF32(d * stage.scale[c]);
  }
  }
  return bits;
}

std::optional<ExecutionError> executeHost(const Program &program, uint32_t from,
                                          uint32_t to, const void *src,
                                          void *dst, GatherPool *pool,
                                          unsigned threads) {
  const uint32_t stageCount = static_cast<uint32_t>(program.stages.size());
  if (from > to || to > stageCount)
    return fail("invalid_boundary", "stage range [" + std::to_string(from) +
                                        ", " + std::to_string(to) +
                                        ") is not inside the program");
  if (!padsSettledBy(program, to))
    return fail("pads_not_settled",
                "a pad enters after boundary " + std::to_string(to) +
                    "; the padded result layout does not exist there");
  auto fill = fillAt(program, to);
  if (auto *error = std::get_if<ExecutionError>(&fill))
    return *error;
  const uint32_t widthTo = widthAt(program, to);
  if (const auto &bits = std::get<std::optional<uint64_t>>(fill))
    fillPattern(static_cast<uint8_t *>(dst), *bits, widthTo,
                program.resultElements);

  const BoundPlan &layout = program.plan.layout;
  WalkState state{program,
                  StageRange{from, to, false},
                  static_cast<const uint8_t *>(src),
                  static_cast<uint8_t *>(dst),
                  widthAt(program, from),
                  widthTo,
                  std::vector<int64_t>(layout.extents.size(), 0)};
  for (uint32_t k = from; k < to; ++k)
    if (program.stages[k].perChannel)
      state.range.needsCoordinates = true;
  for (const PadRegion &p : layout.padRegions)
    state.lo[p.axis] = p.lo;

  const int64_t outer = layout.extents.front();
  int64_t innerSpan = 0;
  for (size_t k = 1; k < layout.dstStrides.size(); ++k)
    innerSpan += (layout.extents[k] - 1) * layout.dstStrides[k];
  const bool rowsDisjoint = layout.dstStrides.front() >= innerSpan + 1;

  std::unique_ptr<GatherPool> owned;
  if (pool == nullptr && threads != 1) {
    owned = std::make_unique<GatherPool>(threads);
    pool = owned.get();
  }
  if (pool != nullptr && rowsDisjoint && pool->threadCount() > 1 && outer > 1) {
    pool->parallelFor(
        0, outer, /*minPerWorker=*/1,
        [&](int64_t begin, int64_t end) { walkRows(state, begin, end); });
  } else {
    walkRows(state, 0, outer);
  }
  if (state.failed.load())
    return state.error;
  return std::nullopt;
}

} // namespace typed
} // namespace reloc
