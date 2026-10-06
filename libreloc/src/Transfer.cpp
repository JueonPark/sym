//===- Transfer.cpp - validated forward transfer requests -----------------===//

#include "reloc/Transfer.h"
#include "TransferInternal.h"

#include "reloc/ChunkSchedule.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/PinnedBufferPool.h"
#include "reloc/Pipeline.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
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
  // C3: the layout part of a typed plan changes the element width along
  // the way; this layout-only path would copy source-width bytes into a
  // destination-width allocation. Typed execution is R3's dispatch.
  if (bound.typed)
    return fail("typed_unsupported",
                "bound plan is the layout of a typed plan; the layout-only "
                "transfer path cannot execute it");
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

std::variant<size_t, TransferError>
validateStackedSources(const BoundPlan &bound,
                       const std::vector<BufferView> &sources,
                       TransferDirection direction) {
  if (direction != TransferDirection::HostToDevice)
    return fail("direction_mismatch",
                "stacked sources support host-to-device transfers only");
  if (sources.empty())
    return fail("plan_mismatch", "a stacked request needs at least one source");
  int64_t segment = 0;
  size_t total = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    const BufferView &view = sources[i];
    const std::string what = "stacked source " + std::to_string(i);
    if (auto error = checkSourceKind(view, direction))
      return *error;
    auto span = viewSpanBytes(view, what.c_str());
    if (auto *error = std::get_if<TransferError>(&span))
      return *error;
    if (!isDense(view))
      return fail("unsupported_layout", what + " must be dense row-major");
    int64_t elements = 0;
    if (!elementCount(view.extents, elements))
      return fail("integer_overflow", what + " element count overflows");
    if (i == 0)
      segment = elements;
    else if (elements != segment)
      return fail("plan_mismatch", what + " holds " + std::to_string(elements) +
                                       " elements but source 0 holds " +
                                       std::to_string(segment));
    if (view.elementSize != sources.front().elementSize)
      return fail("plan_mismatch",
                  what + " element size differs from source 0");
    if (!addOk(total, std::get<size_t>(span), total))
      return fail("integer_overflow", "stacked source span overflows");
  }
  // The plan reads the logical source: a dense row-major view of N * Z
  // elements. Prove reads against it exactly as for an ordinary source.
  int64_t logicalElements = 0;
  size_t logicalBytes = 0;
  if (!mulOk(segment, static_cast<int64_t>(sources.size()), logicalElements) ||
      !mulOk(static_cast<size_t>(logicalElements),
             static_cast<size_t>(sources.front().elementSize), logicalBytes))
    return fail("integer_overflow", "stacked logical source overflows");
  BufferView logical;
  logical.base = sources.front().base; // never dereferenced
  logical.capacityBytes = logicalBytes;
  logical.extents = {logicalElements};
  logical.strides = {1};
  logical.elementSize = sources.front().elementSize;
  if (auto error = checkPlanAgainstSource(bound, logical, logicalBytes))
    return *error;
  return total;
}

std::variant<TransferRequest, TransferError> validateStackedTransfer(
    const BoundPlan &bound, const std::vector<BufferView> &sources,
    const BufferView &destination, TransferDirection direction) {
  auto sourceSpan = validateStackedSources(bound, sources, direction);
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
  request.destination = destination;
  request.direction = direction;
  request.sourceSpanBytes = std::get<size_t>(sourceSpan);
  request.destinationBytes = static_cast<size_t>(bound.totalBytes);
  request.stackSources = sources;
  elementCount(sources.front().extents, request.stackSegmentElements);
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

std::optional<TransferError> backendFailure(const CopyBackend &backend,
                                            const char *phase) {
  return fail("backend_failure", std::string(phase) + ": " + backend.error());
}

// A CUDA view must belong to the device the backend's queues run on; the
// validator cannot know the backend, so this is the executor's check.
std::optional<TransferError> checkDevice(const BufferView &view,
                                         const CopyBackend &backend,
                                         const char *what) {
  if (view.kind != MemoryKind::Cuda || backend.device() < 0 ||
      view.device == backend.device())
    return std::nullopt;
  return fail("device_mismatch", std::string(what) + " view is on device " +
                                     std::to_string(view.device) +
                                     " but the backend runs on device " +
                                     std::to_string(backend.device()));
}

} // namespace

namespace detail {

void quarantine(std::unique_ptr<QuarantineNode> resources) noexcept {
  static std::atomic<QuarantineNode *> head{nullptr};
  auto *node = resources.release();
  node->next = head.load(std::memory_order_relaxed);
  while (!head.compare_exchange_weak(
      node->next, node, std::memory_order_release, std::memory_order_relaxed)) {
  }
}

CompletionGuard::~CompletionGuard() {
  if (completion_ != TransferCompletion::Unknown)
    return;
  try {
    if (backend_.quiesce() == QueueCompletion::Complete)
      completion_ = TransferCompletion::Complete;
  } catch (...) {
    // A throwing backend violates CopyBackend's contract, but still cannot
    // authorize releasing possibly in-flight storage during stack unwinding.
  }
}

std::optional<TransferError>
checkTransferBackend(const TransferRequest &request,
                     const CopyBackend &backend) {
  if (request.consumed)
    return fail("already_executed", "transfer request was already executed");
  if (backend.failed())
    return backendFailure(backend, "backend unusable before launch");
  if (auto error = checkDevice(request.source, backend, "source"))
    return error;
  return checkDevice(request.destination, backend, "destination");
}

std::variant<TransferRequirements, TransferError>
describeTransfer(const TransferRequest &request,
                 const TransferOptions &options) {
  if (request.consumed)
    return fail("already_executed", "transfer request was already executed");
  // Public request fields can change after validation. Never trust the cached
  // span or an earlier capacity proof when allocating/reusing staging.
  auto checked =
      request.stackSources.empty()
          ? validateTransfer(request.bound, request.source, request.destination,
                             request.direction)
          : validateStackedTransfer(request.bound, request.stackSources,
                                    request.destination, request.direction);
  if (auto *error = std::get_if<TransferError>(&checked))
    return *error;
  TransferRequirements r;
  r.sourceBytes = std::get<TransferRequest>(checked).sourceSpanBytes;
  r.stackSegmentElements =
      std::get<TransferRequest>(checked).stackSegmentElements;
  r.gatherThreads = options.gather ? 1 : options.gatherThreads;
  if (r.gatherThreads == 0)
    r.gatherThreads = std::max(1u, std::thread::hardware_concurrency());
  if (r.gatherThreads > unsigned(std::numeric_limits<int>::max()))
    return fail("invalid_options", "gather thread count exceeds native range");

  const auto &b = request.bound;
  int64_t rowBytes = 0;
  if (!mulOk(b.dstStrides[0], int64_t(b.elementSize), rowBytes))
    return fail("integer_overflow", "destination row size overflows");
  std::vector<int64_t> padded = b.extents;
  std::vector<bool> seen(padded.size(), false);
  for (const auto &p : b.padRegions) {
    if (seen[p.axis] || b.elementSize > sizeof(uint64_t) ||
        p.fillBits != b.padRegions.front().fillBits)
      return fail("unsupported_layout", "unsupported padding representation");
    seen[p.axis] = true;
    // validateTransfer already checked each padded extent for overflow.
    padded[p.axis] += p.lo + p.hi;
  }
  int64_t physicalElements = 0, physicalBytes = 0;
  if (!elementCount(padded, physicalElements) ||
      !mulOk(physicalElements, int64_t(b.elementSize), physicalBytes))
    return fail("integer_overflow", "physical destination size overflows");
  if (physicalBytes != b.totalBytes)
    return fail("plan_mismatch", "physical destination size differs from plan");
  BufferView physical = request.destination;
  physical.extents = padded;
  physical.strides = b.dstStrides;
  auto physicalSpan = viewSpanBytes(physical, "plan destination");
  if (auto *error = std::get_if<TransferError>(&physicalSpan))
    return *error;
  if (std::get<size_t>(physicalSpan) != size_t(b.totalBytes))
    return fail("plan_mismatch",
                "plan does not cover its physical destination");

  if (request.direction == TransferDirection::HostToDevice) {
    // Bound destination reach was checked above, including padded inner axes.
    // Prove row products before the planner evaluates signed byte arithmetic.
    int64_t innerSpan = 0;
    for (size_t k = 1; k < padded.size(); ++k)
      innerSpan += (padded[k] - 1) * b.dstStrides[k];
    if (b.dstStrides[0] >= innerSpan + 1) {
      int64_t covered = 0;
      if (!mulOk(padded[0], rowBytes, covered))
        return fail("integer_overflow", "chunk coverage overflows");
      if (covered != b.totalBytes)
        return fail("plan_mismatch",
                    "outer-row chunks do not cover destination");
    }
    const int configured = std::max(1, options.nBuffers);
    r.schedule = planChunks(b, configured, options.chunkSizeOverride);
    r.activeSlots = static_cast<int>(
        std::min<size_t>(configured, r.schedule.chunks.size()));
    r.slotBytes = r.schedule.maxChunkBytes;
  } else {
    r.slotBytes = r.sourceBytes;
  }
  if (r.activeSlots < 1 || r.slotBytes == 0 ||
      !mulOk(r.slotBytes, size_t(r.activeSlots), r.stagingBytes))
    return fail("integer_overflow", "staging capacity overflows");
  return r;
}

std::optional<TransferError>
executePreparedTransfer(const TransferRequest &request,
                        const TransferRequirements &r,
                        const TransferOptions &options, CopyBackend &backend,
                        PinnedBufferPool &pool, GatherPool *gather,
                        TransferCompletion &completion) {
  CompletionGuard guard(backend, completion);
  if (!pool.usesBackend(backend) || !pool.valid() ||
      pool.nBuffers() < r.activeSlots || pool.bufferBytes() < r.slotBytes)
    return fail("insufficient_capacity", "staging does not cover this request");
  const auto *src = reinterpret_cast<const uint8_t *>(request.source.base) +
                    request.source.offsetBytes;
  auto *dst = reinterpret_cast<uint8_t *>(request.destination.base) +
              request.destination.offsetBytes;
  if (options.hasCallerStream) {
    completion = TransferCompletion::Unknown;
    if (!backend.waitStream(options.callerStream) || backend.failed())
      return backendFailure(backend, "ordering after the caller stream failed");
  }
  if (request.direction == TransferDirection::HostToDevice) {
    if (request.stackSources.empty())
      return executeH2DPrepared(request.bound, src, dst, backend, pool,
                                r.schedule, r.activeSlots, gather, completion);
    // Stacked: one base per input, resolved through the input pointer table.
    // Z comes from describeTransfer's revalidation, never the public field.
    std::vector<const uint8_t *> bases;
    bases.reserve(request.stackSources.size());
    for (const BufferView &view : request.stackSources)
      bases.push_back(reinterpret_cast<const uint8_t *>(view.base) +
                      view.offsetBytes);
    const StackedSource stacked{bases.data(),
                                static_cast<int64_t>(bases.size()),
                                r.stackSegmentElements};
    return executeH2DPrepared(request.bound, nullptr, dst, backend, pool,
                              r.schedule, r.activeSlots, gather, completion,
                              &stacked);
  }

  // Forward D2H uses the validated source span, not destination size or an
  // inverse-scatter schedule. Only gather after the download proves complete.
  completion = TransferCompletion::Unknown;
  pool.markPending();
  backend.copyAsync(0, pool.buffer(0), src, r.sourceBytes,
                    CopyDir::DeviceToHost);
  if (backend.failed())
    return backendFailure(backend, "device-to-host copy submission failed");
  EventHandle event = backend.recordEvent(0);
  if (!event || backend.failed())
    return backendFailure(backend, "device-to-host event recording failed");
  pool.setEvent(0, event);
  pool.drain();
  if (backend.failed())
    return backendFailure(backend, "device-to-host copy completion failed");
  TransferOptions gathering;
  gathering.gather = gather;
  forwardHostGather(request.bound, pool.buffer(0), dst, gathering);
  completion = TransferCompletion::Complete;
  return std::nullopt;
}

} // namespace detail

std::optional<TransferError> executeTransfer(TransferRequest &request,
                                             CopyBackend &backend,
                                             const TransferOptions &options) {
  struct Ephemeral : detail::QuarantineNode {
    std::unique_ptr<PinnedBufferPool> pool;
    std::unique_ptr<GatherPool> workers;
  };
  std::unique_ptr<Ephemeral> resources;
  TransferCompletion completion = TransferCompletion::NotLaunched;
  struct Retire {
    std::unique_ptr<Ephemeral> &resources;
    TransferCompletion &completion;
    ~Retire() {
      // Even diagnostic allocation can throw while reporting an earlier
      // failure. Persistent ownership must not depend on constructing it.
      if (resources && completion == TransferCompletion::Unknown)
        detail::quarantine(std::move(resources));
    }
  } retire{resources, completion};
  std::optional<TransferError> error;
  try {
    if (auto error = detail::checkTransferBackend(request, backend))
      return error;
    auto description = detail::describeTransfer(request, options);
    if (auto *error = std::get_if<TransferError>(&description))
      return *error;
    const auto &r = std::get<detail::TransferRequirements>(description);
    request.consumed = true;
    resources = std::make_unique<Ephemeral>();
    resources->pool =
        std::make_unique<PinnedBufferPool>(backend, r.activeSlots, r.slotBytes);
    if (!resources->pool->valid() || backend.failed())
      return fail("backend_failure",
                  "pinned staging allocation failed: " + backend.error());
    if (!options.gather && r.gatherThreads > 1)
      resources->workers = std::make_unique<GatherPool>(r.gatherThreads);
    error = detail::executePreparedTransfer(
        request, r, options, backend, *resources->pool,
        options.gather ? options.gather : resources->workers.get(), completion);
  } catch (const std::exception &e) {
    error = fail("backend_failure", e.what());
  } catch (...) {
    error = fail("backend_failure", "unexpected transfer execution failure");
  }
  if (completion == TransferCompletion::Unknown) {
    detail::quarantine(std::move(resources));
    return fail("completion_unknown", error ? error->message : backend.error());
  }
  if (resources && resources->pool)
    resources->pool->markComplete();
  return error;
}

} // namespace reloc
