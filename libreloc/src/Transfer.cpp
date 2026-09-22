//===- Transfer.cpp - validated forward transfer requests -----------------===//

#include "reloc/Transfer.h"

#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"

#include <algorithm>
#include <utility>

namespace reloc {
namespace {

TransferError fail(const char *code, std::string message) {
  return TransferError{code, std::move(message)};
}

bool mulOk(size_t a, size_t b, size_t &out) {
  return !__builtin_mul_overflow(a, b, &out);
}

bool addOk(size_t a, size_t b, size_t &out) {
  return !__builtin_add_overflow(a, b, &out);
}

bool mulOk(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_mul_overflow(a, b, &out);
}

bool addOk(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_add_overflow(a, b, &out);
}

/// Product of extents, overflow-checked.
bool elementCount(const std::vector<int64_t> &extents, int64_t &out) {
  out = 1;
  for (int64_t e : extents)
    if (e < 1 || !mulOk(out, e, out))
      return false;
  return true;
}

bool isDense(const BufferView &view) {
  int64_t expected = 1;
  for (size_t k = view.extents.size(); k-- > 0;) {
    if (view.extents[k] != 1 && view.strides[k] != expected)
      return false;
    if (!mulOk(expected, view.extents[k], expected))
      return false;
  }
  return true;
}

std::string role(const char *what) { return std::string(what); }

} // namespace

std::variant<size_t, TransferError> viewSpanBytes(const BufferView &view,
                                                  const char *what) {
  if (view.base == 0)
    return fail("invalid_view", role(what) + " base pointer is null");
  if (view.elementSize == 0)
    return fail("invalid_view", role(what) + " element size is zero");
  if (view.extents.empty())
    return fail("invalid_view", role(what) + " has rank zero");
  if (view.extents.size() != view.strides.size())
    return fail("invalid_view", role(what) + " extents/strides rank mismatch");
  if (view.kind == MemoryKind::Cuda && view.device < 0)
    return fail("invalid_view",
                role(what) + " CUDA view lacks a device ordinal");
  if (view.kind == MemoryKind::Host && view.device >= 0)
    return fail("invalid_view",
                role(what) + " host view declares a CUDA device");

  // Admitted subset: nonempty, non-negative strides, injective addressing.
  std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent), extent>1
  for (size_t k = 0; k < view.extents.size(); ++k) {
    if (view.extents[k] < 1)
      return fail("invalid_view", role(what) + " has an extent below one");
    if (view.strides[k] < 0)
      return fail("unsupported_layout", role(what) + " has a negative stride");
    if (view.extents[k] == 1)
      continue;
    if (view.strides[k] == 0)
      return fail("unsupported_layout",
                  role(what) + " broadcasts a stride of zero");
    axes.emplace_back(view.strides[k], view.extents[k]);
  }
  std::sort(axes.begin(), axes.end());
  int64_t span = 0; // largest element offset reachable so far
  for (const auto &[stride, extent] : axes) {
    if (stride <= span)
      return fail("unsupported_layout",
                  role(what) + " has overlapping strides");
    int64_t reach = 0;
    if (!mulOk(extent - 1, stride, reach) || !addOk(span, reach, span))
      return fail("integer_overflow", role(what) + " span overflows");
  }
  size_t elements = static_cast<size_t>(span) + 1;
  size_t bytes = 0, last = 0;
  if (!mulOk(elements, static_cast<size_t>(view.elementSize), bytes) ||
      !addOk(view.offsetBytes, bytes, last))
    return fail("integer_overflow", role(what) + " byte span overflows");
  if (last > view.capacityBytes)
    return fail("insufficient_capacity",
                role(what) + " needs " + std::to_string(last) +
                    " bytes from its allocation base but only " +
                    std::to_string(view.capacityBytes) + " are declared");
  return bytes;
}

namespace {

std::optional<TransferError> checkKinds(const BufferView &source,
                                        const BufferView &destination,
                                        TransferDirection direction) {
  // Host views are admitted on either end so HostBackend can exercise the
  // forward path; a CUDA view must sit on the device end of its direction.
  if (direction == TransferDirection::HostToDevice &&
      source.kind == MemoryKind::Cuda)
    return fail("direction_mismatch",
                "host-to-device source must be host memory");
  if (direction == TransferDirection::DeviceToHost &&
      destination.kind == MemoryKind::Cuda)
    return fail("direction_mismatch",
                "device-to-host destination must be host memory");
  return std::nullopt;
}

std::optional<TransferError> checkSourceKind(const BufferView &source,
                                             TransferDirection direction) {
  if (direction == TransferDirection::HostToDevice &&
      source.kind == MemoryKind::Cuda)
    return fail("direction_mismatch",
                "host-to-device source must be host memory");
  return std::nullopt;
}

/// Plan-side proofs shared by both validators: element size, element count,
/// and the plan's maximal source read inside the source span.
std::optional<TransferError> checkPlanAgainstSource(const BoundPlan &bound,
                                                    const BufferView &source,
                                                    size_t sourceSpanBytes) {
  if (bound.extents.empty() ||
      bound.extents.size() != bound.srcStrides.size() ||
      bound.extents.size() != bound.dstStrides.size())
    return fail("plan_mismatch", "bound plan has inconsistent axes");
  if (bound.elementSize == 0 || bound.elementSize != source.elementSize)
    return fail("plan_mismatch",
                "plan element size differs from the source view");
  int64_t planElements = 0, viewElements = 0;
  if (!elementCount(bound.extents, planElements))
    return fail("integer_overflow", "plan element count overflows");
  if (!elementCount(source.extents, viewElements))
    return fail("integer_overflow", "source element count overflows");
  if (planElements != viewElements)
    return fail("plan_mismatch", "plan relocates " +
                                     std::to_string(planElements) +
                                     " elements but the source view holds " +
                                     std::to_string(viewElements));
  int64_t maxRead = 0;
  for (size_t k = 0; k < bound.extents.size(); ++k) {
    if (bound.srcStrides[k] < 0)
      return fail("plan_mismatch", "plan has a negative source stride");
    int64_t reach = 0;
    if (!mulOk(bound.extents[k] - 1, bound.srcStrides[k], reach) ||
        !addOk(maxRead, reach, maxRead))
      return fail("integer_overflow", "plan source reach overflows");
  }
  size_t needed = 0;
  if (!mulOk(static_cast<size_t>(maxRead) + 1,
             static_cast<size_t>(bound.elementSize), needed))
    return fail("integer_overflow", "plan source footprint overflows");
  if (needed > sourceSpanBytes)
    return fail("plan_mismatch", "plan reads " + std::to_string(needed) +
                                     " bytes beyond a source view spanning " +
                                     std::to_string(sourceSpanBytes));
  return std::nullopt;
}

std::optional<TransferError>
checkPlanAgainstDestination(const BoundPlan &bound,
                            const BufferView &destination, size_t spanBytes) {
  if (bound.elementSize != destination.elementSize)
    return fail("plan_mismatch",
                "plan element size differs from the destination view");
  if (!isDense(destination))
    return fail("unsupported_layout",
                "destination view must be dense row-major");
  if (bound.totalBytes <= 0)
    return fail("plan_mismatch", "bound plan has no destination footprint");
  if (spanBytes != static_cast<size_t>(bound.totalBytes))
    return fail("plan_mismatch", "destination view spans " +
                                     std::to_string(spanBytes) +
                                     " bytes but the plan writes " +
                                     std::to_string(bound.totalBytes));
  // Plan writes: max padded offset across coalesced axes stays inside.
  std::vector<int64_t> padded = bound.extents;
  for (const PadRegion &p : bound.padRegions) {
    if (p.axis >= padded.size() || p.lo < 0 || p.hi < 0)
      return fail("plan_mismatch", "bound plan has an invalid pad region");
    if (!addOk(padded[p.axis], p.lo, padded[p.axis]) ||
        !addOk(padded[p.axis], p.hi, padded[p.axis]))
      return fail("integer_overflow", "padded extent overflows");
  }
  int64_t maxWrite = 0;
  for (size_t k = 0; k < padded.size(); ++k) {
    if (bound.dstStrides[k] < 0)
      return fail("plan_mismatch", "plan has a negative destination stride");
    int64_t reach = 0;
    if (!mulOk(padded[k] - 1, bound.dstStrides[k], reach) ||
        !addOk(maxWrite, reach, maxWrite))
      return fail("integer_overflow", "plan destination reach overflows");
  }
  size_t needed = 0;
  if (!mulOk(static_cast<size_t>(maxWrite) + 1,
             static_cast<size_t>(bound.elementSize), needed))
    return fail("integer_overflow", "plan destination footprint overflows");
  if (needed > static_cast<size_t>(bound.totalBytes))
    return fail("plan_mismatch",
                "plan writes beyond its declared destination footprint");
  return std::nullopt;
}

} // namespace

std::variant<size_t, TransferError>
validateTransferSource(const BoundPlan &bound, const BufferView &source,
                       TransferDirection direction) {
  if (auto error = checkSourceKind(source, direction))
    return *error;
  auto span = viewSpanBytes(source, "source");
  if (auto *error = std::get_if<TransferError>(&span))
    return *error;
  size_t bytes = std::get<size_t>(span);
  if (auto error = checkPlanAgainstSource(bound, source, bytes))
    return *error;
  return bytes;
}

std::variant<TransferRequest, TransferError>
validateTransfer(const BoundPlan &bound, const BufferView &source,
                 const BufferView &destination, TransferDirection direction) {
  if (auto error = checkKinds(source, destination, direction))
    return *error;
  auto sourceSpan = validateTransferSource(bound, source, direction);
  if (auto *error = std::get_if<TransferError>(&sourceSpan))
    return *error;
  auto destinationSpan = viewSpanBytes(destination, "destination");
  if (auto *error = std::get_if<TransferError>(&destinationSpan))
    return *error;
  if (auto error = checkPlanAgainstDestination(
          bound, destination, std::get<size_t>(destinationSpan)))
    return *error;
  TransferRequest request;
  request.bound = bound;
  request.source = source;
  request.destination = destination;
  request.direction = direction;
  request.sourceSpanBytes = std::get<size_t>(sourceSpan);
  request.destinationBytes = static_cast<size_t>(bound.totalBytes);
  return request;
}

namespace {

// Forward host gather over the whole plan: pads first, then valid cells,
// partitioned across the caller's pool when outer rows are provably disjoint
// in dst (same conservative test as executeH2DThreaded).
void forwardHostGather(const BoundPlan &bound, const void *src, void *dst,
                       const TransferOptions &options) {
  if (options.gather == nullptr) {
    if (options.gatherThreads == 1)
      executeH2D(bound, src, dst);
    else
      executeH2DThreaded(bound, src, dst, options.gatherThreads);
    return;
  }
  fillDst(bound, dst);
  int64_t innerSpan = 0;
  for (size_t k = 1; k < bound.dstStrides.size(); ++k)
    innerSpan += (bound.extents[k] - 1) * bound.dstStrides[k];
  const bool rowsDisjoint = bound.dstStrides[0] >= innerSpan + 1;
  if (!rowsDisjoint || options.gather->threadCount() <= 1) {
    gatherChunk(bound, src, dst, 0, bound.extents[0]);
    return;
  }
  const int64_t rowBytes = std::max<int64_t>(
      1, bound.dstStrides[0] * static_cast<int64_t>(bound.elementSize));
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(kMinGatherBytesPerWorker) / rowBytes);
  options.gather->parallelFor(0, bound.extents[0], minRows,
                              [&](int64_t begin, int64_t end) {
                                gatherChunk(bound, src, dst, begin, end);
                              });
}

struct StagingGuard {
  CopyBackend &backend;
  void *buffer = nullptr;
  ~StagingGuard() {
    if (buffer)
      backend.freeStaging(buffer);
  }
};

std::optional<TransferError> backendFailure(const CopyBackend &backend,
                                            const char *phase) {
  return fail("backend_failure", std::string(phase) + ": " + backend.error());
}

} // namespace

std::optional<TransferError> executeTransfer(TransferRequest &request,
                                             CopyBackend &backend,
                                             const TransferOptions &options) {
  if (request.consumed)
    return fail("already_executed", "transfer request was already executed");
  if (backend.failed())
    return backendFailure(backend, "backend unusable before launch");
  request.consumed = true;

  const auto *src = reinterpret_cast<const uint8_t *>(request.source.base) +
                    request.source.offsetBytes;
  auto *dst = reinterpret_cast<uint8_t *>(request.destination.base) +
              request.destination.offsetBytes;

  // Every private queue orders after the caller's producer stream before
  // touching source or destination storage.
  if (options.hasCallerStream && !backend.waitStream(options.callerStream))
    return backendFailure(backend, "ordering after the caller stream failed");

  if (request.direction == TransferDirection::HostToDevice) {
    if (options.gather != nullptr)
      executeH2DPipelined(request.bound, src, dst, backend, options.nBuffers,
                          options.chunkSizeOverride, *options.gather);
    else
      executeH2DPipelined(request.bound, src, dst, backend, options.nBuffers,
                          options.chunkSizeOverride, options.gatherThreads);
    if (backend.failed())
      return backendFailure(backend, "host-to-device pipeline failed");
    return std::nullopt;
  }

  // Forward D2H: land the dense logical source in owned pinned staging, wait
  // for exactly that copy, then run the same forward gather on the host.
  StagingGuard staging{backend, backend.allocStaging(request.sourceSpanBytes)};
  if (staging.buffer == nullptr)
    return fail("backend_failure", "pinned staging allocation failed for " +
                                       std::to_string(request.sourceSpanBytes) +
                                       " bytes");
  backend.copyAsync(0, staging.buffer, src, request.sourceSpanBytes,
                    CopyDir::DeviceToHost);
  EventHandle ev = backend.recordEvent(0);
  backend.waitEvent(ev);
  if (backend.failed())
    return backendFailure(backend, "device-to-host staging copy failed");
  forwardHostGather(request.bound, staging.buffer, dst, options);
  return std::nullopt;
}

} // namespace reloc
