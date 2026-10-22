//===- TypedExecute.h - CPU reference execution of typed plans -*- C++
//-*-===//
//
// R3 (issue #147), Task 1: the standalone CPU reference for C3's typed bound
// plans. A Program is the checked, immutable form of a TypedBoundPlan: every
// stage's C1 arithmetic with its parameters as host scalars (the reciprocal
// of a quantize scale formed once, in binary32, from the declared scale),
// plus the coordinate bookkeeping that per-channel stages need. Qualified
// contiguous runs and dense matrix transposes use buffered/tiled SIMD stages;
// the checked scalar walker remains the fallback for other layouts and
// channel maps. Every intermediate rounding is preserved. executeHost
// runs the layout and any contiguous range of stages on dense host buffers;
// the same function is the forced `original_cpu` baseline and the CPU half
// of every partitioned path in Dispatch.h. MLIR/Torch/GPU-free; no cost
// model is consulted here.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_TYPEDEXECUTE_H
#define RELOC_TYPEDEXECUTE_H

#include "reloc/Bind.h"
#include "reloc/ChunkSchedule.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reloc {

class GatherPool;

namespace typed {

/// Stable reason code plus a human-readable detail. Codes:
///   unsupported_stage, invalid_parameter, pads_not_settled,
///   pad_not_entered, fill_crosses_channel_stage, fill_type_mismatch,
///   mixed_fills_unsupported, fill_mismatch, invalid_boundary,
///   channel_out_of_range, channel_evaluation_failed.
struct ExecutionError {
  std::string code;
  std::string message;
};

/// One stage's checked arithmetic (docs/reloc-typed-semantics.md §3):
///   cast f32 -> f16 ieee_rne, cast f16 -> f32 exact,
///   quantize f32 -> s8 symmetric_rne (zero point 0),
///   dequantize s8 -> f32 affine (any zero point in range).
/// Parameters are host scalars indexed by channel (one entry per tensor).
struct StageArithmetic {
  ValueTransformKind transform = ValueTransformKind::Cast;
  NumericPolicyKind policy = NumericPolicyKind::IeeeRne;
  ElementType input{};
  ElementType output{};
  std::vector<float> scale;       // dequantize: the declared scale
  std::vector<float> invScale;    // quantize: fl32(1 / scale), formed once
  std::vector<int32_t> zeroPoint; // dequantize: broadcast when size() == 1
  bool perChannel = false;
  bool channelIsDim = false; // channel map is exactly one result coordinate
  uint32_t channelDim = 0;
  int64_t channelLength = 1; // parameter entries (1 per tensor)
};

/// The checked program. Owns its plan; immutable after prepareProgram.
struct Program {
  TypedBoundPlan plan;
  std::vector<StageArithmetic> stages;
  std::vector<int64_t> resultStrides; // dense row-major over resultExtents
  int64_t sourceElements = 0;
  int64_t resultElements = 0; // padded logical result
  bool needsCoordinates = false;
  // Runtime row indirection over a dense host source. The portable plan still
  // describes the selected logical tensor; indices are owned by this request.
  int64_t indexedSourceRows = 0;
  int64_t indexedRowElements = 0;
  std::vector<int64_t> rowIndices;
  // Fused embedding-bag producer: result row b holds, per table t, the
  // float32 sum of that table's rows in bag b (indices[offsets[b] ..
  // offsets[b+1])), followed by the program's stages (an f16 cast or none).
  // Tables, indices and offsets are caller-owned for the request's lifetime;
  // the kernel range-checks every index. Empty unless the program is pooled.
  struct PooledTable {
    const uint8_t *rows = nullptr; // dense float32 [rowCount, pooledDim]
    int64_t rowCount = 0;
    const int64_t *indices = nullptr;
    int64_t indexCount = 0;
    const int64_t *offsets = nullptr; // pooledBags entries, nondecreasing
  };
  std::vector<PooledTable> pooledTables;
  int64_t pooledBags = 0;
  int64_t pooledDim = 0;
};

/// Check every stage against the implemented C1 tables and materialize its
/// parameters. Fails (never partially) on an unsupported transform/policy/
/// type pair or an inconsistent parameter. Consults nothing but the plan.
std::variant<Program, ExecutionError>
prepareProgram(const TypedBoundPlan &plan);

/// Boundary helpers. Boundary 0 is the source side, stages.size() the result.
ElementType typeAt(const Program &program, uint32_t boundary);
uint32_t widthAt(const Program &program, uint32_t boundary);

/// True when every pad has entered by `boundary`, i.e. the dense padded
/// result layout exists at that boundary (a legal transfer cut).
bool padsSettledBy(const Program &program, uint32_t boundary);

/// The bit pattern every pad carries at `boundary`: the original fill folded
/// through the stages it crossed. nullopt when the plan has no pads. Errors:
/// a pad has not entered yet (pad_not_entered), the fold would cross a
/// per-channel stage (fill_crosses_channel_stage), pads disagree
/// (mixed_fills_unsupported), or the final fold contradicts the layout's
/// fused fill (fill_mismatch; the decoder proves this never happens).
std::variant<std::optional<uint64_t>, ExecutionError>
fillAt(const Program &program, uint32_t boundary);

/// Bytes of the tensor at `boundary`: the padded result in result layout,
/// the pad-free source elements in source layout.
int64_t bytesAt(const Program &program, uint32_t boundary, bool resultLayout);

/// Apply one stage to one value: `bits` is the input bit pattern (zero
/// extended), the result the output bit pattern. `channel` selects the
/// per-channel parameter entry (0 per tensor). Pure; asserts the range.
uint64_t applyStage(const StageArithmetic &stage, uint64_t bits,
                    int64_t channel);

enum class HostKernel {
  Generic,
  ContiguousCast,
  ContiguousStages,
  TiledTranspose,
  IndexedRows,
  PooledBags,
  /// Rank-3 pad-free layouts whose innermost axis is contiguous on both
  /// sides: every (outer, middle) pair is one SIMD run, iterated so that the
  /// source side streams sequentially (the strided side is the write side).
  StridedRuns
};
/// Pure plan/range qualification. Complex channel expressions retain the
/// generic checked walker. No code generation or shape cache is involved.
HostKernel hostKernel(const Program &, uint32_t from, uint32_t to);
const char *hostKernelName(HostKernel);

/// Execute the layout and stages [from, to) on the host. `src` is the dense
/// SOURCE-layout buffer of boundary-`from` values (no pads), `dst` the dense
/// RESULT-layout buffer of boundary-`to` values; every pad is written with
/// fillAt(to). Requires from <= to <= stages.size() and padsSettledBy(to).
/// Outer rows are partitioned across `pool` (or `threads` workers when no
/// pool is given; 1 = inline) when they are provably disjoint in `dst`.
/// Reports channel_out_of_range / channel_evaluation_failed with `dst`
/// unspecified. Never throws.
std::optional<ExecutionError> executeHost(const Program &program, uint32_t from,
                                          uint32_t to, const void *src,
                                          void *dst, GatherPool *pool = nullptr,
                                          unsigned threads = 1);

/// Execute one physical outer-row window from planChunks over a layout whose
/// elementSize/totalBytes describe boundary `to`. `dst` points to the chunk's
/// own storage, not an artificially rebased pointer. Channel maps still use
/// global result coordinates. Rejects non-partitionable layouts or invalid
/// windows before writing; use executeHost for the whole-buffer fallback.
std::optional<ExecutionError>
executeHostChunk(const Program &, uint32_t from, uint32_t to, const void *src,
                 void *dst, const Chunk &, GatherPool *pool = nullptr,
                 unsigned threads = 1, uint64_t chunkIndex = 0);

} // namespace typed
} // namespace reloc

#endif // RELOC_TYPEDEXECUTE_H
