//===- TypedExecute.cpp - CPU reference execution of typed plans -------===//

#include "reloc/TypedExecute.h"
#include "Trace.h"
#include "TypedKernels.h"

#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"
#include "reloc/TypedValue.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RELOC_POOLED_AVX2 1
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
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
bool isBF16(ElementType t) {
  return t.kind == ElementTypeKind::BFloat && t.bitwidth == 16;
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
  bool narrowRun = false;
  bool bufferedRun = false;
  int64_t destinationOrigin = 0; // global element offset of this local window
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
  storeBits(state.dst + (dstOff - state.destinationOrigin) * state.widthTo,
            bits, state.widthTo);
  return true;
}

void walk(WalkState &state, size_t depth, int64_t iBegin, int64_t iEnd,
          int64_t srcOff, int64_t dstOff, std::vector<int64_t> &coords) {
  const BoundPlan &b = state.program.plan.layout;
  const size_t r = b.extents.size();
  if (depth == r - 1) {
    if (state.narrowRun) {
      const auto *src = state.src + (srcOff + iBegin) * sizeof(float);
      auto *dst = state.dst + (dstOff + iBegin + state.lo[depth] -
                               state.destinationOrigin) *
                                  sizeof(uint16_t);
      // Unaligned external views use the buffered memcpy loads below.
      if (reinterpret_cast<uintptr_t>(src) % alignof(float) == 0 &&
          reinterpret_cast<uintptr_t>(dst) % alignof(uint16_t) == 0) {
        quant::convertF32F16(reinterpret_cast<const float *>(src),
                             reinterpret_cast<uint16_t *>(dst), iEnd - iBegin);
        return;
      }
    }
    if (state.bufferedRun) {
      const int64_t offset = dstOff + iBegin + state.lo[depth];
      detail::stages(state.program, state.range.from, state.range.to,
                     state.src + (srcOff + iBegin) * state.widthFrom,
                     state.dst +
                         (offset - state.destinationOrigin) * state.widthTo,
                     iEnd - iBegin, offset);
      return;
    }
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
      const bool narrow = isF32(stage.input) &&
                          (isF16(stage.output) || isBF16(stage.output)) &&
                          bound.policy == NumericPolicyKind::IeeeRne;
      const bool widen = (isF16(stage.input) || isBF16(stage.input)) &&
                         isF32(stage.output) &&
                         bound.policy == NumericPolicyKind::Exact;
      if (!narrow && !widen)
        return fail("unsupported_stage",
                    where + "only cast f32->f16/bf16 ieee_rne and "
                            "f16/bf16->f32 exact have a reference "
                            "implementation");
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
      if (!channel.empty() && channel.front().op == ExprOp::PushDim &&
          channel.front().value >= 0 &&
          static_cast<size_t>(channel.front().value) < rank) {
        stage.channelDim = static_cast<uint32_t>(channel.front().value);
        stage.channelIsDim = channel.size() == 1;
        // Exported reshapes can leave (dN mod extent). Prove identity over
        // the full logical result, including padding; never infer an axis
        // from equal parameter lengths or simplify a wrapping expression.
        if (channel.size() == 3 && channel[2].op == ExprOp::Mod &&
            (channel[1].op == ExprOp::PushConst ||
             channel[1].op == ExprOp::PushSym)) {
          int64_t modulus = 0;
          if (evalChannel({channel[1]}, plan.symbols, {}, modulus, why) &&
              modulus > 0 && plan.resultExtents[stage.channelDim] <= modulus)
            stage.channelIsDim = true;
        }
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
      return isBF16(stage.output) ? quant::narrowF32BF16(f32FromBits(bits))
                                  : quant::narrowF32F16(f32FromBits(bits));
    if (isBF16(stage.input))
      return bitsFromF32(quant::widenBF16F32(static_cast<uint16_t>(bits)));
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

HostKernel hostKernel(const Program &p, uint32_t from, uint32_t to) {
  if (from > to || to > p.stages.size())
    return HostKernel::Generic;
  if (p.indexedSourceRows)
    return HostKernel::IndexedRows;
  if (!p.pooledTables.empty())
    return HostKernel::PooledBags;
  const auto &l = p.plan.layout;
  if (!detail::simpleChannels(p, from, to, l.extents.back()))
    return HostKernel::Generic;
  if (l.extents.size() == 2 && l.srcStrides[0] == 1 &&
      l.srcStrides[1] == l.extents[0] && l.dstStrides[1] == 1 &&
      l.dstStrides[0] >= l.extents[1])
    return HostKernel::TiledTranspose;
  if (l.extents.size() == 3 && l.padRegions.empty() && l.srcStrides[2] == 1 &&
      l.dstStrides[2] == 1 && l.extents[2] >= 8)
    return HostKernel::StridedRuns;
  if (l.srcStrides.back() == 1 && l.dstStrides.back() == 1) {
    if (to == from + 1 &&
        p.stages[from].transform == ValueTransformKind::Cast &&
        isF32(p.stages[from].input) && isF16(p.stages[from].output))
      return HostKernel::ContiguousCast;
    return HostKernel::ContiguousStages;
  }
  return HostKernel::Generic;
}

const char *hostKernelName(HostKernel kernel) {
  switch (kernel) {
  case HostKernel::Generic:
    return "generic";
  case HostKernel::ContiguousCast:
    return "contiguous_cast";
  case HostKernel::ContiguousStages:
    return "contiguous_stages";
  case HostKernel::TiledTranspose:
    return "tiled_transpose";
  case HostKernel::IndexedRows:
    return "indexed_rows";
  case HostKernel::StridedRuns:
    return "strided_runs";
  case HostKernel::PooledBags:
    return "pooled_bags";
  }
  return "generic";
}

namespace {
// A chunk gets its own worker decision. Do not wake every retained worker for
// a tiny window; also avoid constructing an ephemeral pool for inline work.
int64_t workerFloor(int64_t bytesPerUnit,
                    int64_t minBytes = kMinGatherBytesPerWorker) {
  return std::max<int64_t>(1, (minBytes + bytesPerUnit - 1) / bytesPerUnit);
}

// Typed windows are at most a few MiB (the chunked H2D ring targets 1 MiB
// slots), so the layout pipeline's 1 MiB per-worker floor would hand a chunk
// to two or three workers and leave the rest of a retained pool idle. 64 KiB
// keeps tiny windows inline while a default chunk still feeds eight workers.
constexpr int64_t kMinTypedBytesPerWorker = 64 * 1024;

// Rows of an indexed gather are independent random reads; issuing the next
// rows' cache lines early keeps several misses in flight per worker.
// RELOC_INDEXED_PREFETCH overrides the distance (0 disables) for experiments.
int64_t indexedPrefetchRows() {
  static const int64_t value = [] {
    if (const char *env = std::getenv("RELOC_INDEXED_PREFETCH"))
      return std::max<int64_t>(0, std::atoll(env));
    return int64_t(32);
  }();
  return value;
}

// Sum one bag of `dim` floats per table into `acc`, in index order (matching
// an unweighted EmbeddingBag sum). Returns false on an index out of range.
#ifdef RELOC_POOLED_AVX2
__attribute__((target("avx2,f16c"))) bool
pooledBagAvx2(const Program::PooledTable &table, int64_t begin, int64_t end,
              int64_t dim, float *acc) {
  for (int64_t d = 0; d + 8 <= dim; d += 8)
    _mm256_storeu_ps(acc + d, _mm256_setzero_ps());
  for (int64_t d = dim & ~int64_t(7); d < dim; ++d)
    acc[d] = 0.f;
  for (int64_t k = begin; k < end; ++k) {
    const int64_t row = table.indices[k];
    if (row < 0 || row >= table.rowCount)
      return false;
    const float *src =
        reinterpret_cast<const float *>(table.rows + row * dim * sizeof(float));
    int64_t d = 0;
    for (; d + 8 <= dim; d += 8)
      _mm256_storeu_ps(acc + d, _mm256_add_ps(_mm256_loadu_ps(acc + d),
                                              _mm256_loadu_ps(src + d)));
    for (; d < dim; ++d)
      acc[d] += src[d];
  }
  return true;
}
#endif

bool pooledBagScalar(const Program::PooledTable &table, int64_t begin,
                     int64_t end, int64_t dim, float *acc) {
  for (int64_t d = 0; d < dim; ++d)
    acc[d] = 0.f;
  for (int64_t k = begin; k < end; ++k) {
    const int64_t row = table.indices[k];
    if (row < 0 || row >= table.rowCount)
      return false;
    const float *src =
        reinterpret_cast<const float *>(table.rows + row * dim * sizeof(float));
    for (int64_t d = 0; d < dim; ++d)
      acc[d] += src[d];
  }
  return true;
}

/// Result rows [first, last) of a pooled program into `dst` (the window's
/// own storage): dst[(b - first), t, :] = stages(sum of bag (t, b)).
std::optional<ExecutionError>
executePooledBags(const Program &program, void *dst, GatherPool *pool,
                  unsigned threads, const Chunk *chunk, uint64_t chunkIndex) {
  const uint32_t stageCount = static_cast<uint32_t>(program.stages.size());
  const int64_t bags = program.pooledBags, dim = program.pooledDim;
  const int64_t tables = static_cast<int64_t>(program.pooledTables.size());
  const uint32_t outputWidth = widthAt(program, stageCount);
  const bool narrow = stageCount == 1 && isF32(program.stages[0].input) &&
                      isF16(program.stages[0].output);
  if (stageCount > 1 || (stageCount == 1 && !narrow))
    return fail("unsupported_stage",
                "pooled programs support no stage or one f32 -> f16 cast");
  const int64_t rowBytes = tables * dim * outputWidth;
  int64_t first = 0, last = bags;
  if (chunk) {
    if (chunk->validBegin < 0 || chunk->validEnd <= chunk->validBegin ||
        chunk->validEnd > bags || chunk->paddedBegin != chunk->validBegin ||
        chunk->paddedEnd != chunk->validEnd ||
        chunk->byteOffset != chunk->validBegin * rowBytes ||
        chunk->bytes !=
            size_t((chunk->validEnd - chunk->validBegin) * rowBytes))
      return fail("invalid_chunk", "pooled chunk must contain complete bags");
    first = chunk->validBegin;
    last = chunk->validEnd;
  }
  std::atomic<bool> outOfRange{false};
#ifdef RELOC_POOLED_AVX2
  const bool avx2 = quant::cpuSupports(quant::Variant::AVX2);
#else
  const bool avx2 = false;
#endif
  auto rows = [&](int64_t begin, int64_t end) {
    reloc::detail::TraceRange trace("reloc.typed.transform.work", chunkIndex);
    alignas(64) float acc[1024];
    std::vector<float> spill;
    float *sum = acc;
    if (dim > 1024) {
      spill.resize(static_cast<size_t>(dim));
      sum = spill.data();
    }
    // Table-major: one table's bags [begin, end) read a contiguous run of
    // its index stream, so the prefetch window can run ahead across bag
    // boundaries (bags are short) and keep several row misses in flight.
    constexpr int64_t kAhead = 12;
    const int64_t rowStride = dim * int64_t(sizeof(float));
    for (int64_t t = 0; t < tables; ++t) {
      const auto &table = program.pooledTables[static_cast<size_t>(t)];
      const int64_t streamEnd =
          end < bags ? table.offsets[end] : table.indexCount;
      auto touch = [&](int64_t k) {
        if (k < streamEnd) {
          const int64_t row = table.indices[k];
          if (row >= 0 && row < table.rowCount) {
            const uint8_t *p = table.rows + row * rowStride;
            for (int64_t byte = 0; byte < rowStride; byte += 64)
              __builtin_prefetch(p + byte, 0, 1);
          }
        }
      };
      for (int64_t k = table.offsets[begin]; k < table.offsets[begin] + kAhead;
           ++k)
        touch(k);
      for (int64_t b = begin; b < end; ++b) {
        const int64_t lo = table.offsets[b];
        const int64_t hi =
            b + 1 < bags ? table.offsets[b + 1] : table.indexCount;
        for (int64_t k = lo; k < hi; ++k)
          touch(k + kAhead);
        bool ok;
#ifdef RELOC_POOLED_AVX2
        if (avx2)
          ok = pooledBagAvx2(table, lo, hi, dim, sum);
        else
#endif
          ok = pooledBagScalar(table, lo, hi, dim, sum);
        if (!ok) {
          outOfRange.store(true);
          return;
        }
        uint8_t *cell = static_cast<uint8_t *>(dst) + (b - first) * rowBytes +
                        t * dim * outputWidth;
        if (narrow)
          quant::convertF32F16(sum, reinterpret_cast<uint16_t *>(cell), dim);
        else
          std::memcpy(cell, sum, static_cast<size_t>(dim) * sizeof(float));
      }
    }
  };
  std::unique_ptr<GatherPool> owned;
  const int64_t grain = workerFloor(rowBytes * 3, kMinTypedBytesPerWorker);
  if (!pool && threads != 1 && last - first >= 2 * grain) {
    owned = std::make_unique<GatherPool>(threads);
    pool = owned.get();
  }
  if (pool)
    pool->parallelFor(first, last, grain, rows);
  else
    rows(first, last);
  if (outOfRange.load())
    return fail("index_out_of_range", "pooled bag index is outside its table");
  return std::nullopt;
}

std::optional<ExecutionError>
executeHostWindow(const Program &program, uint32_t from, uint32_t to,
                  const void *src, void *dst, GatherPool *pool,
                  unsigned threads, const Chunk *chunk, uint64_t chunkIndex) {
  const uint32_t stageCount = static_cast<uint32_t>(program.stages.size());
  if (from > to || to > stageCount)
    return fail("invalid_boundary", "stage range [" + std::to_string(from) +
                                        ", " + std::to_string(to) +
                                        ") is not inside the program");
  if (!program.pooledTables.empty()) {
    if (from != 0 || to != stageCount)
      return fail("invalid_boundary",
                  "pooled programs execute all host stages together");
    return executePooledBags(program, dst, pool, threads, chunk, chunkIndex);
  }
  if (program.indexedSourceRows) {
    if (from != 0 || to != stageCount)
      return fail("invalid_boundary",
                  "indexed programs execute all host stages together");
    const uint32_t inputWidth = widthAt(program, 0),
                   outputWidth = widthAt(program, to);
    const int64_t columns = program.indexedRowElements;
    const int64_t rowCount = static_cast<int64_t>(program.rowIndices.size());
    const int64_t rowBytes = columns * outputWidth;
    int64_t first = 0, last = rowCount;
    if (chunk) {
      if (chunk->validBegin < 0 || chunk->validEnd <= chunk->validBegin ||
          chunk->validEnd > rowCount ||
          chunk->paddedBegin != chunk->validBegin ||
          chunk->paddedEnd != chunk->validEnd ||
          chunk->byteOffset != chunk->validBegin * rowBytes ||
          chunk->bytes !=
              size_t((chunk->validEnd - chunk->validBegin) * rowBytes))
        return fail("invalid_chunk",
                    "indexed chunk must contain complete selected rows");
      first = chunk->validBegin;
      last = chunk->validEnd;
    }
    auto rows = [&](int64_t begin, int64_t end) {
      reloc::detail::TraceRange trace("reloc.typed.transform.work", chunkIndex);
      const int64_t rowInputBytes = columns * inputWidth;
      const int64_t ahead = indexedPrefetchRows();
      for (int64_t row = begin; row < end; ++row) {
        if (ahead > 0 && row + ahead < end) {
          const auto *next = static_cast<const uint8_t *>(src) +
                             program.rowIndices[row + ahead] * rowInputBytes;
          for (int64_t b = 0; b < rowInputBytes; b += 64)
            __builtin_prefetch(next + b, 0, 1);
        }
        const auto *input = static_cast<const uint8_t *>(src) +
                            program.rowIndices[row] * columns * inputWidth;
        auto *output = static_cast<uint8_t *>(dst) + (row - first) * rowBytes;
        if (stageCount == 0) {
          std::memcpy(output, input, columns * inputWidth);
        } else if (isF32(program.plan.sourceType) &&
                   reinterpret_cast<uintptr_t>(input) % alignof(float) == 0 &&
                   reinterpret_cast<uintptr_t>(output) % alignof(uint16_t) ==
                       0) {
          quant::convertF32F16(reinterpret_cast<const float *>(input),
                               reinterpret_cast<uint16_t *>(output), columns);
        } else {
          for (int64_t column = 0; column < columns; ++column)
            storeBits(output + column * outputWidth,
                      applyStage(
                          program.stages[0],
                          loadBits(input + column * inputWidth, inputWidth), 0),
                      outputWidth);
        }
      }
    };
    std::unique_ptr<GatherPool> owned;
    const int64_t grain = workerFloor(columns * (inputWidth + outputWidth),
                                      kMinTypedBytesPerWorker);
    if (!pool && threads != 1 && last - first >= 2 * grain) {
      owned = std::make_unique<GatherPool>(threads);
      pool = owned.get();
    }
    if (pool)
      pool->parallelFor(first, last, grain, rows);
    else
      rows(first, last);
    return std::nullopt;
  }
  if (!padsSettledBy(program, to))
    return fail("pads_not_settled",
                "a pad enters after boundary " + std::to_string(to) +
                    "; the padded result layout does not exist there");
  auto fill = fillAt(program, to);
  if (auto *error = std::get_if<ExecutionError>(&fill))
    return *error;
  const uint32_t widthTo = widthAt(program, to);
  const BoundPlan &layout = program.plan.layout;
  int64_t begin = 0, end = layout.extents.front(), origin = 0;
  int64_t elements = program.resultElements;
  if (chunk) {
    auto wire = layout;
    wire.elementSize = widthTo;
    wire.totalBytes = bytesAt(program, to, true);
    // A maximal target produces one window and checks row disjointness.
    const auto whole =
        planChunks(wire, 1, static_cast<size_t>(wire.totalBytes));
    const auto &full = whole.chunks.front();
    if (whole.serialized || full.bytes != size_t(wire.totalBytes) ||
        chunk->paddedBegin < 0 || chunk->paddedEnd <= chunk->paddedBegin ||
        chunk->paddedEnd > full.paddedEnd ||
        chunk->byteOffset != chunk->paddedBegin * whole.rowBytes ||
        chunk->bytes !=
            size_t((chunk->paddedEnd - chunk->paddedBegin) * whole.rowBytes))
      return fail("invalid_chunk",
                  "chunk is not a disjoint physical row window");
    begin = std::clamp(chunk->paddedBegin - whole.outerLo, int64_t(0), end);
    end = std::clamp(chunk->paddedEnd - whole.outerLo, begin, end);
    if (chunk->validBegin != begin || chunk->validEnd != end)
      return fail("invalid_chunk",
                  "chunk valid rows differ from its physical window");
    origin = chunk->byteOffset / widthTo;
    elements = static_cast<int64_t>(chunk->bytes / widthTo);
  }
  if (const auto &bits = std::get<std::optional<uint64_t>>(fill)) {
    reloc::detail::TraceRange trace("reloc.typed.fill", chunkIndex);
    fillPattern(static_cast<uint8_t *>(dst), *bits, widthTo, elements);
  }

  WalkState state{program,
                  StageRange{from, to, false},
                  static_cast<const uint8_t *>(src),
                  static_cast<uint8_t *>(dst),
                  widthAt(program, from),
                  widthTo,
                  std::vector<int64_t>(layout.extents.size(), 0)};
  state.destinationOrigin = origin;
  for (uint32_t k = from; k < to; ++k)
    if (program.stages[k].perChannel)
      state.range.needsCoordinates = true;
  for (const PadRegion &p : layout.padRegions)
    state.lo[p.axis] = p.lo;
  const HostKernel kernel = hostKernel(program, from, to);
  state.narrowRun = kernel == HostKernel::ContiguousCast;
  state.bufferedRun = state.narrowRun || kernel == HostKernel::ContiguousStages;

  int64_t innerSpan = 0;
  for (size_t k = 1; k < layout.dstStrides.size(); ++k)
    innerSpan += (layout.extents[k] - 1) * layout.dstStrides[k];
  const bool rowsDisjoint = layout.dstStrides.front() >= innerSpan + 1;

  const bool tiled = kernel == HostKernel::TiledTranspose;
  const bool runs = kernel == HostKernel::StridedRuns;
  const bool narrowCast =
      to == from + 1 &&
      program.stages[from].transform == ValueTransformKind::Cast &&
      isF32(program.stages[from].input) && isF16(program.stages[from].output);
  const int64_t units =
      tiled ? (end - begin + detail::kTypedTile - 1) / detail::kTypedTile
            : end - begin;
  int64_t rowElements = 1;
  for (size_t k = 1; k < layout.extents.size(); ++k)
    rowElements *= layout.extents[k];
  const int64_t unitBytes =
      std::min<int64_t>(rowElements, kMinGatherBytesPerWorker) *
      (state.widthFrom + state.widthTo) * (tiled ? detail::kTypedTile : 1);
  // Scalar coordinate evaluation is compute-bound even for small byte
  // windows. Apply the bandwidth-oriented floor only to qualified kernels.
  // Tiling and multiple conversions do more work per byte than a direct
  // cast/copy. 256 KiB lets a default 1 MiB wire chunk use the worker budget
  // while keeping tiny windows inline. Direct contiguous casts use the
  // existing gather/copy floor of 1 MiB.
  const int64_t grain = kernel == HostKernel::Generic
                            ? 1
                            : workerFloor(unitBytes, kMinTypedBytesPerWorker);
  std::unique_ptr<GatherPool> owned;
  if (pool == nullptr && threads != 1 && rowsDisjoint && units >= 2 * grain) {
    owned = std::make_unique<GatherPool>(threads);
    pool = owned.get();
  }
  auto work = [&](int64_t first, int64_t last) {
    reloc::detail::TraceRange trace("reloc.typed.transform.work", chunkIndex);
    if (runs)
      detail::stridedRuns(program, from, to, state.src, state.dst,
                          begin + first, begin + last, origin, narrowCast);
    else if (tiled)
      detail::transpose(program, from, to, state.src, state.dst,
                        begin + first * detail::kTypedTile,
                        std::min(end, begin + last * detail::kTypedTile),
                        origin, state.lo[0], state.lo[1]);
    else
      walkRows(state, begin + first, begin + last);
  };
  if (pool != nullptr && rowsDisjoint && pool->threadCount() > 1 && units > 1)
    pool->parallelFor(0, units, grain, work);
  else if (end > begin)
    work(0, units);
  if (state.failed.load())
    return state.error;
  return std::nullopt;
}
} // namespace

std::optional<ExecutionError> executeHost(const Program &program, uint32_t from,
                                          uint32_t to, const void *src,
                                          void *dst, GatherPool *pool,
                                          unsigned threads) {
  return executeHostWindow(program, from, to, src, dst, pool, threads, nullptr,
                           0);
}

std::optional<ExecutionError>
executeHostChunk(const Program &program, uint32_t from, uint32_t to,
                 const void *src, void *dst, const Chunk &chunk,
                 GatherPool *pool, unsigned threads, uint64_t chunkIndex) {
  return executeHostWindow(program, from, to, src, dst, pool, threads, &chunk,
                           chunkIndex);
}

} // namespace typed
} // namespace reloc
