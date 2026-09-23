//===- Transfer.h - validated forward transfer requests ---------*- C++ -*-===//
//
// R2 (issue #146): a Torch-free description of the buffers a framework hands
// the runtime, a checked validator that proves every plan access fits the
// declared allocations before anything is allocated or launched, and a
// blocking forward executor over the CopyBackend interface. "Forward" means
// the plan's own src->dst relocation in both directions: H2D reuses the
// pinned/stream pipeline, D2H copies the dense device source into owned
// pinned staging and applies the same host gather into the destination.
// The existing inverse-scatter D2H APIs (executeD2H*) are a separate contract.
// No exceptions cross this interface; failures are reported by value.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_TRANSFER_H
#define RELOC_TRANSFER_H

#include "reloc/Backend.h"
#include "reloc/Bind.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reloc {

class GatherPool;

/// Where a buffer lives. A Host view may serve as either end of a transfer so
/// the whole forward path is exercisable through HostBackend in CI.
enum class MemoryKind : uint8_t { Host, Cuda };

/// Direction of the plan's forward relocation.
enum class TransferDirection : uint8_t { HostToDevice, DeviceToHost };

/// A framework buffer as the caller declares it: the allocation the logical
/// tensor lives in, plus the logical view over it. Trust boundary: the
/// validator proves that the *declared* view and plan fit the *declared*
/// allocation; it cannot verify raw addresses, capacities or device ordinals
/// against the allocator. Callers are responsible for truthful declarations
/// (the Torch adapter derives them from the tensor's storage and proves the
/// ordinal with cudaPointerGetAttributes) and for keeping owners alive
/// through the blocking call.
struct BufferView {
  uintptr_t base = 0;           // allocation base address
  size_t capacityBytes = 0;     // bytes owned from `base`
  size_t offsetBytes = 0;       // byte offset of logical element 0 from `base`
  std::vector<int64_t> extents; // logical extents (all >= 1)
  std::vector<int64_t> strides; // element strides per extent
  uint32_t elementSize = 0;
  MemoryKind kind = MemoryKind::Host;
  int device = -1; // CUDA ordinal for Cuda views, -1 for Host
};

/// A stable reason code plus a human-readable detail. Codes:
///   invalid_view, unsupported_layout, insufficient_capacity,
///   integer_overflow, plan_mismatch, direction_mismatch, already_executed,
///   device_mismatch, backend_failure.
struct TransferError {
  std::string code;
  std::string message;
};

/// A validated request. Owns copies of the bound plan and both views for the
/// whole operation; single-use.
struct TransferRequest {
  BoundPlan bound;
  BufferView source;
  BufferView destination;
  TransferDirection direction = TransferDirection::HostToDevice;
  size_t sourceSpanBytes = 0;  // bytes the plan may read from src offset
  size_t destinationBytes = 0; // == bound.totalBytes
  bool consumed = false;
};

/// Byte span a view addresses: elementSize * (1 + sum((extent-1)*stride)),
/// after rejecting empty, negative-stride, broadcast and overlapping views.
/// Every arithmetic step is overflow-checked.
std::variant<size_t, TransferError> viewSpanBytes(const BufferView &view,
                                                  const char *role);

/// Preflight half: prove the bound plan's valid source accesses fit the
/// declared source view and that the view is admissible for `direction`.
/// Returns the source span in bytes.
std::variant<size_t, TransferError>
validateTransferSource(const BoundPlan &bound, const BufferView &source,
                       TransferDirection direction);

/// Full validation with the destination: dense destination view of exactly
/// bound.totalBytes, plan writes inside it, capacities sufficient.
std::variant<TransferRequest, TransferError>
validateTransfer(const BoundPlan &bound, const BufferView &source,
                 const BufferView &destination, TransferDirection direction);

struct TransferOptions {
  int nBuffers = 4;             // pinned staging ring size (H2D)
  size_t chunkSizeOverride = 0; // 0 = heuristic
  unsigned gatherThreads = 1;   // per-call gather parallelism when no pool
  GatherPool *gather = nullptr; // caller-owned pool (wins over threads)
  bool hasCallerStream = false; // order after callerStream (may be 0 ==
                                // the legacy default stream)
  const void *callerStream = nullptr;
};

/// Execute a validated request through `backend` and block until this
/// request's work has completed (never a device-wide synchronization).
/// Marks the request consumed before launching; a consumed request fails
/// with already_executed before any work. A CUDA view whose ordinal differs
/// from backend.device() fails with device_mismatch before any work (host
/// backends, device() < 0, accept every ordinal). Staging allocation and
/// backend failures surface as backend_failure with the backend's
/// diagnostic. Never throws.
std::optional<TransferError> executeTransfer(TransferRequest &request,
                                             CopyBackend &backend,
                                             const TransferOptions &options);

} // namespace reloc

#endif // RELOC_TRANSFER_H
