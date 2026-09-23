//===- Dispatch.h - plan-driven typed transform dispatch --------*- C++ -*-===//
//
// R3 (issue #147): execute C3's typed bound plans through qualified paths.
// Capability is a pure question answered from the checked program alone
// (which implementations are semantically equivalent to the reference and
// actually exist for this direction and device); policy chooses among the
// eligible ones (`original_cpu` forces the CPU reference pipeline, `auto`
// consults the central cost model and translates its advice to an eligible
// row or falls back with a recorded reason); execution runs exactly the
// selected row through R2's CopyBackend and reports what it moved. Cost
// estimates never grant capability, and no path may change the requested
// precision or skip a stage. Torch/MLIR-free; CUDA rows exist only under
// RELOC_ENABLE_CUDA and need a CudaBackend.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_DISPATCH_H
#define RELOC_DISPATCH_H

#include "reloc/Backend.h"
#include "reloc/CostModel.h"
#include "reloc/Prefold.h"
#include "reloc/Transfer.h"
#include "reloc/TypedExecute.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reloc {
namespace dispatch {

/// `OriginalCpu`: the declared layout and every stage run on the CPU, then
/// the necessary cross-device copy (transform then H2D, or D2H of the
/// logical source then the forward CPU transform). `Auto`: enumerate the
/// eligible equivalent rows, consult the cost model when one is given, and
/// otherwise take the CPU baseline.
enum class Policy { OriginalCpu, Auto };

/// Row identifiers (docs/runtime-dispatch.md capability table).
inline constexpr const char *kCpuReference = "cpu_reference";
inline constexpr const char *kCpuStagesCudaStages = "cpu_stages_cuda_stages";
inline constexpr const char *kCudaStagesThenCpu = "cuda_stages_then_cpu";
inline constexpr const char *kCudaDequantRelocate = "cuda_dequant_relocate";
inline constexpr const char *kCudaRelocateF32 = "cuda_relocate_f32";

/// One executable path with the stage partition it uses. The wire tensor is
/// the tensor at `wireBoundary`, in the result layout (padded) or the
/// source layout (pad-free); `wireBytes` is what crosses the link.
struct Implementation {
  std::string id;
  uint32_t wireBoundary = 0;
  bool wireResultLayout = true;
  int64_t wireBytes = 0;
  int64_t deviceTempBytes = 0; // device scratch the row allocates itself
  std::string method;          // cost-model class: "A" or "B"
  /// "<id>" for whole-program rows, "<id>@<boundary>" for partitions.
  std::string label() const;
};

struct Exclusion {
  std::string id; // label of the row that is not available
  std::string reason;
};

struct Capability {
  std::vector<Implementation> eligible;
  std::vector<Exclusion> excluded;
};

/// Pure capability query: which rows can run `program` in `direction` when
/// the device end is (`cuda`) a CUDA backend. Never consults a cost model.
Capability queryCapability(const typed::Program &program,
                           TransferDirection direction, bool cuda);

/// The scalar-only execution report (no live references, no pointers).
struct Report {
  std::string implementation; // Implementation::label()
  std::string policy;         // "original_cpu", "auto" or "explicit"
  std::string placementReason;
  std::string method; // "A", "B" or ""
  uint32_t wireBoundary = 0;
  int64_t sourceBytes = 0;
  int64_t wireBytes = 0;
  int64_t destinationBytes = 0;
  int64_t parameterBytes = 0;
  /// Observed link traffic of the executed row: the wire tensor plus every
  /// device parameter upload the row needed. Set by executeDispatch.
  int64_t payloadBytesTransferred = 0;
  int64_t deviceTempBytes = 0;
  uint32_t artifactVersion = 1; // the typed wire format version
  bool executed = false;
};

struct Options {
  Policy policy = Policy::Auto;
  /// Central cost model (optional). Advice only: it ranks eligible rows.
  const costmodel::CostModel *model = nullptr;
  int threads = 8; // the cost model's CPU thread key
  /// The device end will be a CudaBackend (enables the CUDA rows).
  bool cuda = false;
  /// Explicit row selection by label (tests, conformance evidence); must be
  /// eligible or preparation fails with implementation_unavailable.
  std::string implementation;
};

/// A prepared, single-use request: the checked program, both views, the
/// selected row and the report skeleton.
struct DispatchRequest {
  typed::Program program;
  BufferView source;
  BufferView destination;
  TransferDirection direction = TransferDirection::HostToDevice;
  Implementation selected;
  Report report;
  bool consumed = false;
};

/// The outcome of applying a policy to the eligible rows.
struct Selection {
  Implementation row;
  std::string policy; // "original_cpu", "auto" or "explicit"
  std::string reason; // stable placement reason
};

/// Pure selection without views: capability for `direction`/`options.cuda`,
/// then the policy. Lets a bridge fix the row at preparation time and force
/// exactly that row when the destination exists (Options::implementation).
/// Fails with the program's own code or implementation_unavailable.
std::variant<Selection, TransferError>
selectImplementation(const TypedBoundPlan &plan, TransferDirection direction,
                     const Options &options);

/// Validate the typed program and both views, enumerate capability, apply
/// the policy. Fails (TransferError) with: the program's own code
/// (unsupported_stage, invalid_parameter), invalid_view / unsupported_layout
/// / insufficient_capacity / integer_overflow / plan_mismatch /
/// direction_mismatch for the views, or implementation_unavailable when the
/// forced baseline or an explicit row is not eligible. Allocates and
/// launches nothing.
std::variant<DispatchRequest, TransferError>
prepareDispatch(const TypedBoundPlan &plan, const BufferView &source,
                const BufferView &destination, TransferDirection direction,
                const Options &options);

/// Run the selected row through `backend` and block until this request's
/// work completed. Marks the request consumed before launching; a consumed
/// request fails with already_executed. CUDA rows need a CudaBackend
/// (backend_mismatch otherwise). Backend and kernel failures surface as
/// backend_failure; reference-path failures keep their own code. No second
/// path is tried after data dispatch. Never throws.
std::optional<TransferError> executeDispatch(DispatchRequest &request,
                                             CopyBackend &backend,
                                             const TransferOptions &options);

//===----------------------------------------------------------------------===//
// Prefold capability (Task 4, shared with T4)
//===----------------------------------------------------------------------===//

/// The existing S8 prefolder variant a typed program maps onto, with the
/// reciprocal scales it needs, formed once from the DECLARED scales (a
/// caller never supplies reciprocals on the typed path).
struct PrefoldSpec {
  prefold::OutputSpec spec = prefold::OutputSpec::S8GatherQuant;
  std::vector<float> invScales; // one per coalesced outer channel
};

/// Pure: succeeds only when the program is exactly one quantize
/// (symmetric_rne, f32 -> s8) whose channel the prefolder indexes (per
/// tensor, or the coalesced outer axis that is logical result axis 0), over
/// a pad-free, packed, rank >= 2 layout. Everything else is
/// TransferError{"prefold_unavailable", reason}; T4 then keeps the normal
/// layout preparation and never quantizes an f32 weight to use the
/// prefolder.
std::variant<PrefoldSpec, TransferError>
prefoldSpecFor(const TypedBoundPlan &plan);

} // namespace dispatch
} // namespace reloc

#endif // RELOC_DISPATCH_H
