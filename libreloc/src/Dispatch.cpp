//===- Dispatch.cpp - plan-driven typed transform dispatch ----------------===//

#include "reloc/Dispatch.h"
#include "DispatchGroupInternal.h"
#include "Trace.h"
#include "TransferInternal.h"

#include "reloc/ChunkSchedule.h"
#include "reloc/GatherPool.h"
#include "reloc/PinnedBufferPool.h"
#include "reloc/Pipeline.h"
#include "reloc/TypedValue.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#include "reloc/CudaKernels.h"
#endif

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

namespace reloc {
namespace dispatch {

std::string Implementation::label() const {
  if (id == kCpuStagesCudaStages || id == kCudaStagesThenCpu)
    return id + "@" + std::to_string(wireBoundary);
  return id;
}

namespace {

using typed::Program;

TransferError fail(const char *code, std::string message) {
  return TransferError{code, std::move(message)};
}

TransferError fromExecution(const typed::ExecutionError &error) {
  return TransferError{error.code, error.message};
}

bool isF32(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 32;
}
bool isF16(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 16;
}

std::string label(const char *id, uint32_t boundary) {
  return std::string(id) + "@" + std::to_string(boundary);
}

int64_t product(const std::vector<int64_t> &v, size_t from = 0) {
  int64_t out = 1;
  for (size_t i = from; i < v.size(); ++i)
    out *= v[i];
  return out;
}

//===----------------------------------------------------------------------===//
// Qualification of the existing CUDA kernels against one stage.
//===----------------------------------------------------------------------===//

/// The layout exposes source axis 0 as one coalesced axis that is exactly
/// logical result axis `dim`: the source-layout flat index then yields the
/// same channel as the result coordinate `dim` (D2H device stages).
bool sourceOuterIsResultDim(const Program &p, uint32_t dim) {
  const BoundPlan &l = p.plan.layout;
  if (!l.padRegions.empty() || dim >= p.plan.resultExtents.size())
    return false;
  const int64_t outerExtent = p.plan.sourceExtents.front();
  const int64_t outerSrcStride = product(p.plan.sourceExtents, 1);
  if (p.plan.resultExtents[dim] != outerExtent)
    return false;
  for (size_t a = 0; a < l.extents.size(); ++a)
    if (l.extents[a] == outerExtent && l.srcStrides[a] == outerSrcStride &&
        l.dstStrides[a] == p.resultStrides[dim])
      return true;
  return false;
}

/// Empty when the element-wise CUDA kernel for stage `k` implements exactly
/// the stage's C1 semantics over a dense buffer in the result (H2D) or the
/// source (D2H) layout; otherwise the stable exclusion reason.
std::string deviceElementwiseReason(const Program &p, uint32_t k,
                                    bool resultLayout) {
  const typed::StageArithmetic &stage = p.stages[k];
  switch (stage.transform) {
  case ValueTransformKind::Cast:
    if (isF16(stage.input) && isF32(stage.output))
      return {}; // convertF16F32, exact
    return "no_cuda_kernel:cast_f32_f16";
  case ValueTransformKind::Quantize:
    break; // quantizeF32S8, bit-identical to the CPU scalar contract
  case ValueTransformKind::Dequantize:
    for (int32_t zp : stage.zeroPoint)
      if (zp != 0)
        return "no_cuda_kernel:nonzero_zero_point";
    break; // dequantS8F32: q * scale, exact
  }
  if (!stage.perChannel)
    return {};
  // The kernels index the channel by flat / channelSize: the outermost
  // axis of the buffer they run over.
  if (!stage.channelIsDim)
    return resultLayout ? "channel_not_outer_result_axis"
                        : "channel_not_source_outer_axis";
  if (resultLayout)
    return stage.channelDim == 0 ? std::string()
                                 : "channel_not_outer_result_axis";
  return sourceOuterIsResultDim(p, stage.channelDim)
             ? std::string()
             : "channel_not_source_outer_axis";
}

/// Reason the whole device stage range [from, to) is not element-wise
/// qualified, or empty.
std::string deviceRangeReason(const Program &p, uint32_t from, uint32_t to,
                              bool resultLayout) {
  for (uint32_t j = from; j < to; ++j)
    if (std::string why = deviceElementwiseReason(p, j, resultLayout);
        !why.empty())
      return "stage " + std::to_string(j) + ": " + why;
  return {};
}

int64_t tempBytes(const Program &p, uint32_t from, uint32_t to,
                  bool resultLayout) {
  int64_t bytes = 0;
  for (uint32_t j = from; j < to; ++j)
    bytes += typed::bytesAt(p, j, resultLayout);
  return bytes;
}

//===----------------------------------------------------------------------===//
// View validation for typed programs: dense source of the program's own
// descriptor, dense destination of exactly the padded result.
//===----------------------------------------------------------------------===//

bool isDense(const BufferView &view) {
  int64_t expected = 1;
  for (size_t k = view.extents.size(); k-- > 0;) {
    if (view.extents[k] != 1 && view.strides[k] != expected)
      return false;
    if (__builtin_mul_overflow(expected, view.extents[k], &expected))
      return false;
  }
  return true;
}

std::optional<TransferError> checkView(const Program &program,
                                       const BufferView &view, bool isSource,
                                       TransferDirection direction) {
  const char *what = isSource ? "source" : "destination";
  if (direction == TransferDirection::HostToDevice && isSource &&
      view.kind == MemoryKind::Cuda)
    return fail("direction_mismatch",
                "host-to-device source must be host memory");
  if (direction == TransferDirection::DeviceToHost && !isSource &&
      view.kind == MemoryKind::Cuda)
    return fail("direction_mismatch",
                "device-to-host destination must be host memory");
  auto span = viewSpanBytes(view, what);
  if (auto *error = std::get_if<TransferError>(&span))
    return *error;
  const uint32_t boundary =
      isSource ? 0 : static_cast<uint32_t>(program.stages.size());
  if (view.elementSize != typed::widthAt(program, boundary))
    return fail("plan_mismatch",
                std::string(what) +
                    " element size differs from the typed "
                    "program's " +
                    std::string(isSource ? "source" : "result") + " type");
  if (!isDense(view))
    return fail("unsupported_layout",
                std::string(what) + " view must be dense row-major");
  if (isSource && program.indexedSourceRows) {
    auto extents = program.plan.sourceExtents;
    extents.front() = program.indexedSourceRows;
    if (direction != TransferDirection::HostToDevice || view.extents != extents)
      return fail("plan_mismatch",
                  "indexed source requires the prepared host row shape");
    return std::nullopt; // viewSpanBytes already proved the full allocation.
  }
  const int64_t expectedBytes = typed::bytesAt(program, boundary, !isSource);
  if (std::get<size_t>(span) != static_cast<size_t>(expectedBytes))
    return fail("plan_mismatch", std::string(what) + " view spans " +
                                     std::to_string(std::get<size_t>(span)) +
                                     " bytes but the typed program needs " +
                                     std::to_string(expectedBytes));
  if (isSource) {
    int64_t elements = 1;
    for (int64_t e : view.extents)
      elements *= e;
    if (elements != program.sourceElements)
      return fail("plan_mismatch",
                  "source view element count differs from the program's "
                  "source descriptor");
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Execution helpers.
//===----------------------------------------------------------------------===//

/// Identity layout over a dense buffer, for the pinned/stream pipeline.
BoundPlan densePlan(int64_t elements, uint32_t width) {
  BoundPlan b;
  b.extents = {elements};
  b.srcStrides = {1};
  b.dstStrides = {1};
  b.perm = {0};
  b.elementSize = width;
  b.totalBytes = elements * static_cast<int64_t>(width);
  b.L = elements;
  return b;
}

std::optional<TransferError> backendFailure(const CopyBackend &backend,
                                            const char *phase) {
  return fail("backend_failure", std::string(phase) + ": " + backend.error());
}

/// Host-to-device copy of a dense host buffer through the staging pipeline
/// (byte-identical to executeH2D; the same machinery R2 uses).
std::optional<TransferError>
pipelineToDevice(const void *hostSrc, void *deviceDst, int64_t elements,
                 uint32_t width, CopyBackend &backend,
                 const TransferOptions &options, GroupReport *group = nullptr) {
  if (options.directDenseUpload) {
    backend.copyAsync(0, deviceDst, hostSrc,
                      static_cast<size_t>(elements) * width,
                      CopyDir::HostToDevice);
    if (group)
      ++group->payloadCopyCalls;
    else
      backend.waitEvent(backend.recordEvent(0));
    if (backend.failed()) {
      backend.quiesce();
      return backendFailure(backend, "dense host-to-device copy failed");
    }
    return std::nullopt;
  }
  BoundPlan dense = densePlan(elements, width);
  const int nBuffers = std::max(1, options.nBuffers);
  ChunkSchedule sched = planChunks(dense, nBuffers, options.chunkSizeOverride);
  PinnedBufferPool pool(backend, std::min<int>(nBuffers, sched.chunks.size()),
                        sched.maxChunkBytes);
  if (!pool.valid()) {
    std::string detail = "pinned staging allocation failed for " +
                         std::to_string(nBuffers) + " x " +
                         std::to_string(sched.maxChunkBytes) + " bytes";
    if (!backend.error().empty())
      detail += ": " + backend.error();
    return fail("backend_failure", detail);
  }
  if (options.gather != nullptr) {
    executeH2DPipelined(dense, hostSrc, deviceDst, backend, pool,
                        options.chunkSizeOverride, options.gather);
  } else if (options.gatherThreads == 1) {
    executeH2DPipelined(dense, hostSrc, deviceDst, backend, pool,
                        options.chunkSizeOverride, nullptr);
  } else {
    GatherPool gather(options.gatherThreads);
    executeH2DPipelined(dense, hostSrc, deviceDst, backend, pool,
                        options.chunkSizeOverride, &gather);
  }
  if (backend.failed())
    return backendFailure(backend, "host-to-device pipeline failed");
  return std::nullopt;
}

struct StagingGuard {
  CopyBackend &backend;
  void *buffer = nullptr;
  ~StagingGuard() {
    if (buffer)
      backend.freeStaging(buffer);
  }
};

/// Device-to-host copy into owned pinned staging, waited for exactly.
std::optional<TransferError>
stageFromDevice(const void *deviceSrc, int64_t bytes, CopyBackend &backend,
                StagingGuard &staging, GroupReport *group = nullptr) {
  staging.buffer = backend.allocStaging(static_cast<size_t>(bytes));
  if (staging.buffer == nullptr)
    return fail("backend_failure", "pinned staging allocation failed for " +
                                       std::to_string(bytes) + " bytes");
  backend.copyAsync(0, staging.buffer, deviceSrc, static_cast<size_t>(bytes),
                    CopyDir::DeviceToHost);
  if (group)
    ++group->payloadCopyCalls;
  else {
    EventHandle ev = backend.recordEvent(0);
    backend.waitEvent(ev);
  }
  if (backend.failed())
    return backendFailure(backend, "device-to-host staging copy failed");
  return std::nullopt;
}

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

std::optional<TransferError> hostProgram(const Program &program, uint32_t from,
                                         uint32_t to, const void *src,
                                         void *dst,
                                         const TransferOptions &options) {
  auto error = typed::executeHost(program, from, to, src, dst, options.gather,
                                  options.gather ? 1 : options.gatherThreads);
  if (error)
    return fromExecution(*error);
  return std::nullopt;
}

// Transform directly into ring slots. Completion of a slot's previous copy
// gates its reuse; the worker barrier completes before that slot is submitted.
// Pool cleanup independently drains on failure, while Resources retains or
// quarantines the arena and all tensor owners when completion is unknown.
std::optional<TransferError>
transformToDevice(const Program &program, uint32_t boundary, const void *src,
                  void *dst, CopyBackend &backend,
                  const TransferOptions &options, Report &report) {
  const auto bytes =
      static_cast<size_t>(typed::bytesAt(program, boundary, true));
  auto wire = program.plan.layout;
  wire.elementSize = typed::widthAt(program, boundary);
  wire.totalBytes = static_cast<int64_t>(bytes);
  report.hostChunks = 1;
  report.hostChunkBytes = bytes;
  report.hostBuffers = 1;
  auto whole = [&]() -> std::optional<TransferError> {
    StagingGuard result{backend, backend.allocStaging(bytes)};
    if (!result.buffer)
      return backendFailure(backend, "host scratch allocation failed");
    if (auto error =
            hostProgram(program, 0, boundary, src, result.buffer, options))
      return error;
    return pipelineToDevice(result.buffer, dst, program.resultElements,
                            wire.elementSize, backend, options);
  };
  if (!options.pipelineTypedH2D) {
    report.hostPipeline = "whole_disabled";
    return whole();
  }
  // Keep small transfers whole. Larger typed transfers target 32 chunks with
  // a 1 MiB floor: measured conversion/DMA schedules favor smaller slots than
  // the layout gather default (see docs/typed-h2d-pipeline.md). Physical rows
  // remain indivisible, and callers can override this hardware-sensitive knob.
  // The schedule is independent of ring size, including the one-slot control.
  size_t target = options.chunkSizeOverride;
  if (!target)
    target = bytes <= kMinChunkBytes
                 ? bytes
                 : std::clamp<size_t>(bytes / 32, 1ull << 20, kMaxChunkBytes);
  const auto schedule = planChunks(wire, 2, target);
  const bool partitionable =
      !schedule.serialized && size_t(schedule.chunks.back().byteOffset) +
                                      schedule.chunks.back().bytes ==
                                  bytes;
  report.hostPipeline = !partitionable ? "whole_non_partitionable"
                        : schedule.chunks.size() == 1 ? "whole_single_chunk"
                                                      : "chunked";
  if (report.hostPipeline != "chunked")
    return whole();
  const int slots = static_cast<int>(
      std::min<size_t>(std::max(1, options.nBuffers), schedule.chunks.size()));
  if (backend.numQueues() < 1)
    return backendFailure(backend, "backend has no copy queue");
  report.hostChunks = schedule.chunks.size();
  report.hostChunkBytes = schedule.maxChunkBytes;
  report.hostBuffers = slots;
  PinnedBufferPool ring(backend, slots, schedule.maxChunkBytes);
  if (!ring.valid() || backend.failed())
    return backendFailure(backend, "typed staging ring allocation failed");
  // Raw callers without an owned/borrowed pool still create workers once per
  // transfer, never once per chunk. Resources normally supplies its pool.
  std::unique_ptr<GatherPool> owned;
  GatherPool *workers = options.gather;
  if (!workers && options.gatherThreads != 1) {
    owned = std::make_unique<GatherPool>(options.gatherThreads);
    workers = owned.get();
  }
  for (size_t k = 0; k < schedule.chunks.size(); ++k) {
    const auto &chunk = schedule.chunks[k];
    const int slot = ring.acquire();
    if (slot < 0)
      return backendFailure(backend, "typed staging reuse failed");
    if (auto error = typed::executeHostChunk(
            program, 0, boundary, src, ring.buffer(slot), chunk, workers, 1, k))
      return fromExecution(*error);
    const int queue = static_cast<int>(k % size_t(backend.numQueues()));
    ring.markPending();
    {
      detail::TraceRange submit("reloc.typed.h2d.submit", k);
      backend.copyAsync(queue, static_cast<uint8_t *>(dst) + chunk.byteOffset,
                        ring.buffer(slot), chunk.bytes, CopyDir::HostToDevice);
    }
    if (backend.failed())
      return backendFailure(backend, "typed chunk copy failed");
    const auto event = backend.recordEvent(queue);
    ring.setEvent(slot, event);
    if (!event || backend.failed())
      return backendFailure(backend, "typed chunk event failed");
  }
  ring.drain();
  if (backend.failed())
    return backendFailure(backend, "typed chunk completion failed");
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CUDA rows.
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

/// Device scratch and pinned parameter staging, freed together.
struct DeviceScratch {
  CudaBackend &backend;
  std::vector<void *> device;
  std::vector<void *> staging;
  GroupReport *group = nullptr;
  std::unordered_map<std::string, const float *> parameters;
  ~DeviceScratch() {
    for (void *p : staging)
      backend.freeStaging(p);
    for (void *p : device)
      backend.freeDevice(p);
  }
  void *allocDevice(int64_t bytes) {
    void *p =
        backend.allocDevice(static_cast<size_t>(std::max<int64_t>(bytes, 1)));
    if (p)
      device.push_back(p);
    return p;
  }
  /// Upload `values` to device memory through pinned staging on queue 0
  /// (ordered before every later launch on stream 0). Adds the bytes to
  /// `payload`; nullptr on failure.
  const float *uploadFloats(const std::vector<float> &values,
                            int64_t &payload) {
    const size_t bytes = values.size() * sizeof(float);
    std::string key;
    if (group) {
      key.assign(reinterpret_cast<const char *>(values.data()), bytes);
      auto found = parameters.find(key);
      if (found != parameters.end()) {
        ++group->parameterReuses;
        return found->second;
      }
    }
    void *host = backend.allocStaging(bytes);
    if (host == nullptr)
      return nullptr;
    staging.push_back(host);
    std::memcpy(host, values.data(), bytes);
    void *dev = allocDevice(static_cast<int64_t>(bytes));
    if (dev == nullptr)
      return nullptr;
    backend.copyAsync(0, dev, host, bytes, CopyDir::HostToDevice);
    payload += static_cast<int64_t>(bytes);
    if (group) {
      ++group->parameterUploads;
      group->parameterUploadBytes += bytes;
      parameters.emplace(std::move(key), static_cast<const float *>(dev));
    }
    return static_cast<const float *>(dev);
  }
};

/// Element-wise device stages [from, to) over a dense buffer in the result
/// (H2D) or source (D2H) layout: `dIn` holds boundary-`from` values, `dOut`
/// receives boundary-`to` values; intermediates are scratch. Launches on
/// stream 0 after the uploads enqueued there.
std::optional<TransferError> runDeviceStages(const Program &program,
                                             uint32_t from, uint32_t to,
                                             bool resultLayout, const void *dIn,
                                             void *dOut, CudaBackend &backend,
                                             DeviceScratch &scratch,
                                             int64_t &payload) {
  const int64_t elements =
      resultLayout ? program.resultElements : program.sourceElements;
  const int64_t outer = resultLayout ? program.plan.resultExtents.front()
                                     : program.plan.sourceExtents.front();
  void *stream = backend.stream(0);
  const void *input = dIn;
  for (uint32_t j = from; j < to; ++j) {
    const typed::StageArithmetic &stage = program.stages[j];
    void *output = dOut;
    if (j + 1 < to) {
      output =
          scratch.allocDevice(typed::bytesAt(program, j + 1, resultLayout));
      if (output == nullptr)
        return backendFailure(backend, "device scratch allocation failed");
    }
    const int64_t channels = stage.perChannel ? outer : 1;
    const int64_t channelSize = elements / channels;
    switch (stage.transform) {
    case ValueTransformKind::Cast:
      cuda::convertF16F32(static_cast<const uint16_t *>(input),
                          static_cast<float *>(output), elements, stream);
      break;
    case ValueTransformKind::Quantize: {
      std::vector<float> inv = stage.perChannel
                                   ? stage.invScale
                                   : std::vector<float>{stage.invScale[0]};
      const float *dInv = scratch.uploadFloats(inv, payload);
      if (dInv == nullptr)
        return backendFailure(backend, "parameter upload failed");
      cuda::quantizeF32S8(static_cast<const float *>(input),
                          static_cast<int8_t *>(output), channels, channelSize,
                          dInv, stream);
      break;
    }
    case ValueTransformKind::Dequantize: {
      std::vector<float> scale =
          stage.perChannel ? stage.scale : std::vector<float>{stage.scale[0]};
      const float *dScale = scratch.uploadFloats(scale, payload);
      if (dScale == nullptr)
        return backendFailure(backend, "parameter upload failed");
      cuda::dequantS8F32(static_cast<const int8_t *>(input),
                         static_cast<float *>(output), channels, channelSize,
                         dScale, stream);
      break;
    }
    }
    if (scratch.group)
      ++scratch.group->kernelLaunches;
    if (!backend.recordLaunchStatus("typed stage kernel launch"))
      return backendFailure(backend, "kernel launch failed");
    input = output;
  }
  return std::nullopt;
}

std::optional<TransferError> finishQueue(CudaBackend &backend) {
  EventHandle ev = backend.recordEvent(0);
  backend.waitEvent(ev);
  if (backend.failed())
    return backendFailure(backend, "device work failed");
  return std::nullopt;
}

/// A layout copy the f32 relocate kernels accept: the typed layout keeps
/// the RESULT width as elementSize, the kernels move f32 elements.
BoundPlan asF32Layout(const Program &program) {
  BoundPlan l = program.plan.layout;
  l.elementSize = 4;
  l.totalBytes = program.resultElements * 4;
  l.typed = false;
  return l;
}

struct GroupExecution {
  DeviceScratch scratch;
  std::vector<std::function<std::optional<TransferError>()>> host;
};

std::optional<TransferError> executeCuda(DispatchRequest &request,
                                         CudaBackend &backend,
                                         const TransferOptions &options,
                                         GroupExecution *grouped = nullptr) {
  const Program &program = request.program;
  const uint32_t stageCount = static_cast<uint32_t>(program.stages.size());
  const Implementation &row = request.selected;
  const auto *src = reinterpret_cast<const uint8_t *>(request.source.base) +
                    request.source.offsetBytes;
  auto *dst = reinterpret_cast<uint8_t *>(request.destination.base) +
              request.destination.offsetBytes;
  int64_t payload = 0;
  DeviceScratch local{backend};
  DeviceScratch &scratch = grouped ? grouped->scratch : local;

  if (row.id == kCpuStagesCudaStages) {
    const uint32_t k = row.wireBoundary;
    const size_t wireBytes =
        static_cast<size_t>(typed::bytesAt(program, k, true));
    void *dWire = scratch.allocDevice(static_cast<int64_t>(wireBytes));
    if (dWire == nullptr)
      return backendFailure(backend, "device scratch allocation failed");
    const bool directWire =
        k == 0 && options.directDenseUpload &&
        program.plan.layout.padRegions.empty() &&
        program.plan.layout.srcStrides == program.plan.layout.dstStrides;
    if (directWire) {
      // An owned prepacked buffer is already in the wire layout. The same
      // identity proof applies to ordinary dense typed inputs; their existing
      // source-owner contract keeps bytes alive until completion.
      request.report.hostPipeline = "direct_dense";
      request.report.hostChunks = 1;
      request.report.hostBuffers = 0;
      request.report.hostChunkBytes = wireBytes;
      if (auto error = pipelineToDevice(src, dWire, program.resultElements,
                                        typed::widthAt(program, k), backend,
                                        options, scratch.group))
        return error;
    } else if (grouped) {
      // Groups retain their one completion barrier and whole-member staging.
      StagingGuard wire{backend, backend.allocStaging(wireBytes)};
      if (!wire.buffer)
        return backendFailure(backend, "host scratch allocation failed");
      if (auto error = hostProgram(program, 0, k, src, wire.buffer, options))
        return error;
      scratch.group->hostTransformBytes += wireBytes;
      request.report.hostPipeline = "whole_group";
      request.report.hostChunks = request.report.hostBuffers = 1;
      request.report.hostChunkBytes = wireBytes;
      if (auto error = pipelineToDevice(
              wire.buffer, dWire, program.resultElements,
              typed::widthAt(program, k), backend, options, scratch.group))
        return error;
    } else if (auto error = transformToDevice(program, k, src, dWire, backend,
                                              options, request.report))
      return error;
    payload += static_cast<int64_t>(wireBytes);
    if (auto error = runDeviceStages(program, k, stageCount, true, dWire, dst,
                                     backend, scratch, payload))
      return error;
  } else if (row.id == kCudaDequantRelocate || row.id == kCudaRelocateF32) {
    void *dSrc = scratch.allocDevice(program.plan.sourceBytes);
    if (dSrc == nullptr)
      return backendFailure(backend, "device scratch allocation failed");
    if (auto error = pipelineToDevice(src, dSrc, program.sourceElements,
                                      typed::widthAt(program, 0), backend,
                                      options, scratch.group))
      return error;
    payload += program.plan.sourceBytes;
    BoundPlan layout = asF32Layout(program);
    if (row.id == kCudaDequantRelocate) {
      const typed::StageArithmetic &stage = program.stages[0];
      std::vector<float> scales(static_cast<size_t>(layout.extents.front()),
                                stage.scale[0]);
      if (stage.perChannel)
        scales = stage.scale;
      const float *dScales = scratch.uploadFloats(scales, payload);
      if (dScales == nullptr)
        return backendFailure(backend, "parameter upload failed");
      void *out = dst;
      if (stageCount > 1) {
        out = scratch.allocDevice(typed::bytesAt(program, 1, true));
        if (out == nullptr)
          return backendFailure(backend, "device scratch allocation failed");
      }
      cuda::dequantRelocateS8F32(layout, static_cast<const int8_t *>(dSrc),
                                 static_cast<float *>(out), dScales,
                                 backend.stream(0));
      if (scratch.group)
        ++scratch.group->kernelLaunches;
      if (!backend.recordLaunchStatus("dequantRelocateS8F32 launch"))
        return backendFailure(backend, "kernel launch failed");
      if (stageCount > 1)
        if (auto error = runDeviceStages(program, 1, stageCount, true, out, dst,
                                         backend, scratch, payload))
          return error;
    } else {
      void *relocated = scratch.allocDevice(program.resultElements * 4);
      if (relocated == nullptr)
        return backendFailure(backend, "device scratch allocation failed");
      cuda::relocateF32(layout, static_cast<const float *>(dSrc),
                        static_cast<float *>(relocated), backend.stream(0));
      if (scratch.group)
        ++scratch.group->kernelLaunches;
      if (!backend.recordLaunchStatus("relocateF32 launch"))
        return backendFailure(backend, "kernel launch failed");
      if (auto error = runDeviceStages(program, 0, stageCount, true, relocated,
                                       dst, backend, scratch, payload))
        return error;
    }
  } else if (row.id == kCudaStagesThenCpu) {
    const uint32_t k = row.wireBoundary;
    void *dWire = scratch.allocDevice(typed::bytesAt(program, k, false));
    if (dWire == nullptr)
      return backendFailure(backend, "device scratch allocation failed");
    if (auto error = runDeviceStages(program, 0, k, false, src, dWire, backend,
                                     scratch, payload))
      return error;
    if (!grouped)
      if (auto error = finishQueue(backend))
        return error;
    StagingGuard staging{backend};
    const int64_t wireBytes = typed::bytesAt(program, k, false);
    if (auto error =
            stageFromDevice(dWire, wireBytes, backend, staging, scratch.group))
      return error;
    payload += wireBytes;
    request.report.payloadBytesTransferred = payload;
    if (grouped) {
      void *buffer = staging.buffer;
      grouped->host.emplace_back(
          [&program, k, stageCount, buffer, dst, options] {
            return hostProgram(program, k, stageCount, buffer, dst, options);
          });
      scratch.group->hostTransformBytes += program.plan.destinationBytes;
      return std::nullopt;
    }
    return hostProgram(program, k, stageCount, staging.buffer, dst, options);
  } else {
    return fail("implementation_unavailable",
                "unknown CUDA row " + row.label());
  }
  if (!grouped)
    if (auto error = finishQueue(backend))
      return error;
  request.report.payloadBytesTransferred = payload;
  return std::nullopt;
}

#endif // RELOC_ENABLE_CUDA

//===----------------------------------------------------------------------===//
// Selection.
//===----------------------------------------------------------------------===//

const Implementation *findRow(const Capability &capability,
                              const std::string &label) {
  for (const Implementation &row : capability.eligible)
    if (row.label() == label)
      return &row;
  return nullptr;
}

/// The cheapest-wire row of a cost-model class, preferring whole-program
/// rows on ties (fewer device stages, no scratch).
const Implementation *bestOfClass(const Capability &capability,
                                  const std::string &method) {
  const Implementation *best = nullptr;
  for (const Implementation &row : capability.eligible) {
    if (row.method != method)
      continue;
    if (best == nullptr || row.wireBytes < best->wireBytes ||
        (row.wireBytes == best->wireBytes &&
         row.deviceTempBytes < best->deviceTempBytes))
      best = &row;
  }
  return best;
}

struct Choice {
  const Implementation *row = nullptr;
  std::string policy;
  std::string reason;
};

std::variant<Choice, TransferError> select(const Program &program,
                                           const Capability &capability,
                                           const Options &options) {
  Choice out;
  if (!options.implementation.empty()) {
    out.row = findRow(capability, options.implementation);
    if (out.row == nullptr)
      return fail("implementation_unavailable",
                  "row '" + options.implementation +
                      "' is not eligible for this program");
    out.policy = "explicit";
    out.reason = "explicit_selection";
    return out;
  }
  const Implementation *reference = findRow(capability, kCpuReference);
  if (reference == nullptr)
    return fail("implementation_unavailable",
                "the CPU reference pipeline cannot run this program");
  if (options.policy == Policy::OriginalCpu) {
    out.row = reference;
    out.policy = "original_cpu";
    out.reason = "forced";
    return out;
  }
  out.policy = "auto";
  if (capability.eligible.size() == 1) {
    out.row = reference;
    out.reason = "only_qualified_path";
    return out;
  }
  if (options.model == nullptr) {
    out.row = reference;
    out.reason = "no_calibration";
    return out;
  }
  const Implementation *bestA = bestOfClass(capability, "A");
  const Implementation *bestB = bestOfClass(capability, "B");
  const Implementation *transformFirst = bestA ? bestA : reference;
  const double r =
      static_cast<double>(transformFirst->wireBytes) /
      static_cast<double>(std::max<int64_t>(1, program.plan.sourceBytes));
  auto decision = costmodel::decide(
      *options.model, costmodel::classify(program.plan.layout),
      program.plan.sourceBytes, r, options.threads);
  if (!decision) {
    out.row = reference;
    out.reason = "model_missing_keys";
    return out;
  }
  switch (decision->method) {
  case costmodel::MethodDecision::Method::A:
    out.row = transformFirst;
    out.reason = "cost_model_prefers_a";
    return out;
  case costmodel::MethodDecision::Method::APrefold:
    out.row = transformFirst;
    out.reason = "prefold_requires_prepared_artifact";
    return out;
  case costmodel::MethodDecision::Method::B:
    if (bestB != nullptr) {
      out.row = bestB;
      out.reason = "cost_model_prefers_b";
    } else {
      out.row = transformFirst;
      out.reason = "advice_unavailable_path";
    }
    return out;
  }
  out.row = reference;
  out.reason = "only_qualified_path";
  return out;
}

// Populate only after both current views and the immutable selection are
// qualified. Never reuse a request's mutable report/consumed state.
DispatchRequest makeRequest(const Program &program, const BufferView &source,
                            const BufferView &destination,
                            TransferDirection direction,
                            const Selection &choice) {
  DispatchRequest request;
  request.program = program;
  request.source = source;
  request.destination = destination;
  request.direction = direction;
  request.selected = choice.row;
  Report &report = request.report;
  report.implementation = choice.row.label();
  report.policy = choice.policy;
  report.placementReason = choice.reason;
  report.method = choice.row.method;
  report.wireBoundary = choice.row.wireBoundary;
  report.sourceBytes = program.indexedSourceRows
                           ? program.indexedSourceRows *
                                 program.indexedRowElements *
                                 typed::widthAt(program, 0)
                           : program.plan.sourceBytes;
  report.wireBytes = choice.row.wireBytes;
  report.destinationBytes = program.plan.destinationBytes;
  report.parameterBytes = program.plan.parameterBytes;
  report.deviceTempBytes = choice.row.deviceTempBytes;
  report.artifactVersion = program.indexedSourceRows ? 2 : 1;
  return request;
}

} // namespace

//===----------------------------------------------------------------------===//
// Capability.
//===----------------------------------------------------------------------===//

std::variant<Program, TransferError>
prepareIndexSelect(const IndexedBoundPlan &bound, const BufferView &source) {
  auto prepared = typed::prepareProgram(bound.selected);
  if (auto *error = std::get_if<typed::ExecutionError>(&prepared))
    return fromExecution(*error);
  auto program = std::get<Program>(std::move(prepared));
  if (bound.sourceExtents.empty() || bound.indices.empty())
    return fail("plan_mismatch",
                "indexed binding requires nonempty source and indices");
  program.indexedSourceRows = bound.sourceExtents.front();
  program.indexedRowElements = bound.rowElements;
  program.rowIndices = bound.indices;
  if (auto error =
          checkView(program, source, true, TransferDirection::HostToDevice))
    return *error;
  return program;
}

Capability queryCapability(const Program &program, TransferDirection direction,
                           bool cuda) {
  Capability out;
  const uint32_t S = static_cast<uint32_t>(program.stages.size());
  const bool h2d = direction == TransferDirection::HostToDevice;

  Implementation reference;
  reference.id = kCpuReference;
  reference.wireBoundary = h2d ? S : 0;
  reference.wireResultLayout = h2d;
  reference.wireBytes =
      h2d ? program.plan.destinationBytes : program.plan.sourceBytes;
  reference.method = h2d ? "A" : "B";
  out.eligible.push_back(reference);

  // Indirection must run on the host before upload. Existing GPU rows assume
  // that the logical source is one dense allocation, so none qualify here.
  if (program.indexedSourceRows) {
    if (!h2d)
      out.eligible.clear();
    out.excluded.push_back({kCudaRelocateF32, "host_index_select"});
    return out;
  }

  auto exclude = [&](std::string id, std::string reason) {
    out.excluded.push_back(Exclusion{std::move(id), std::move(reason)});
  };

  if (!cuda) {
    if (h2d) {
      for (uint32_t k = 0; k < S; ++k)
        exclude(label(kCpuStagesCudaStages, k), "no_cuda_device");
      exclude(kCudaDequantRelocate, "no_cuda_device");
      exclude(kCudaRelocateF32, "no_cuda_device");
    } else {
      for (uint32_t k = 1; k <= S; ++k)
        exclude(label(kCudaStagesThenCpu, k), "no_cuda_device");
    }
    return out;
  }

  const BoundPlan &layout = program.plan.layout;
  if (h2d) {
    for (uint32_t k = 0; k < S; ++k) {
      if (!typed::padsSettledBy(program, k)) {
        exclude(label(kCpuStagesCudaStages, k), "pads_not_settled");
        continue;
      }
      if (std::string why = deviceRangeReason(program, k, S, true);
          !why.empty()) {
        exclude(label(kCpuStagesCudaStages, k), why);
        continue;
      }
      Implementation row;
      row.id = kCpuStagesCudaStages;
      row.wireBoundary = k;
      row.wireResultLayout = true;
      row.wireBytes = typed::bytesAt(program, k, true);
      row.deviceTempBytes = tempBytes(program, k, S, true);
      row.method = "A";
      out.eligible.push_back(row);
    }
    // cuda_dequant_relocate: stage 0 dequantizes with zero point 0 and a
    // channel the fused kernel can index (the coalesced outer axis, which
    // must be logical result axis 0), the layout has no pads, rank >= 2 and
    // a unit inner destination stride; later stages run element-wise.
    {
      std::string why;
      if (S == 0 ||
          program.stages[0].transform != ValueTransformKind::Dequantize)
        why = "stage 0 is not a dequantize";
      else if (!layout.padRegions.empty())
        why = "layout has pads";
      else if (layout.extents.size() < 2)
        why = "rank below two";
      else if (layout.dstStrides.back() != 1)
        why = "inner destination stride is not one";
      else if (std::string s = deviceElementwiseReason(program, 0, true);
               !s.empty())
        why = "stage 0: " + s;
      else if (program.stages[0].perChannel &&
               (layout.extents.front() != program.plan.resultExtents.front() ||
                layout.dstStrides.front() != program.resultStrides.front()))
        why = "channel axis is not the coalesced outer axis";
      else if (std::string s = deviceRangeReason(program, 1, S, true);
               !s.empty())
        why = s;
      if (why.empty()) {
        Implementation row;
        row.id = kCudaDequantRelocate;
        row.wireBoundary = 0;
        row.wireResultLayout = false;
        row.wireBytes = program.plan.sourceBytes;
        row.deviceTempBytes =
            program.plan.sourceBytes + tempBytes(program, 1, S, true);
        row.method = "B";
        out.eligible.push_back(row);
      } else {
        exclude(kCudaDequantRelocate, why);
      }
    }
    // cuda_relocate_f32: an f32 source relocated on the device, then every
    // stage element-wise there.
    {
      std::string why;
      if (!isF32(program.plan.sourceType))
        why = "source is not f32";
      else if (!layout.padRegions.empty())
        why = "layout has pads";
      else if (layout.extents.size() > 8)
        why = "rank above eight";
      else if (std::string s = deviceRangeReason(program, 0, S, true);
               !s.empty())
        why = s;
      if (why.empty()) {
        Implementation row;
        row.id = kCudaRelocateF32;
        row.wireBoundary = 0;
        row.wireResultLayout = false;
        row.wireBytes = program.plan.sourceBytes;
        row.deviceTempBytes =
            program.plan.sourceBytes + tempBytes(program, 0, S, true);
        row.method = "B";
        out.eligible.push_back(row);
      } else {
        exclude(kCudaRelocateF32, why);
      }
    }
    return out;
  }

  // Device to host: the device runs stages [0, k) element-wise in the
  // source layout, the wire is the boundary-k source tensor, the host runs
  // the layout and the rest.
  for (uint32_t k = 1; k <= S; ++k) {
    if (std::string why = deviceRangeReason(program, 0, k, false);
        !why.empty()) {
      exclude(label(kCudaStagesThenCpu, k), why);
      continue;
    }
    Implementation row;
    row.id = kCudaStagesThenCpu;
    row.wireBoundary = k;
    row.wireResultLayout = false;
    row.wireBytes = typed::bytesAt(program, k, false);
    row.deviceTempBytes = tempBytes(program, 1, k + 1, false);
    row.method = "A";
    out.eligible.push_back(row);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Prepare / execute.
//===----------------------------------------------------------------------===//

std::variant<Selection, TransferError>
selectImplementation(const TypedBoundPlan &plan, TransferDirection direction,
                     const Options &options) {
  auto prepared = typed::prepareProgram(plan);
  if (auto *error = std::get_if<typed::ExecutionError>(&prepared))
    return fromExecution(*error);
  return selectImplementation(std::get<Program>(prepared), direction, options);
}

std::variant<Selection, TransferError>
selectImplementation(const Program &program, TransferDirection direction,
                     const Options &options) {
  Capability capability = queryCapability(program, direction, options.cuda);
  auto selected = select(program, capability, options);
  if (auto *error = std::get_if<TransferError>(&selected))
    return *error;
  const Choice &choice = std::get<Choice>(selected);
  return Selection{*choice.row, choice.policy, choice.reason};
}

std::variant<DispatchRequest, TransferError>
prepareDispatch(const TypedBoundPlan &plan, const BufferView &source,
                const BufferView &destination, TransferDirection direction,
                const Options &options) {
  auto prepared = typed::prepareProgram(plan);
  if (auto *error = std::get_if<typed::ExecutionError>(&prepared))
    return fromExecution(*error);
  return prepareDispatch(std::get<Program>(prepared), source, destination,
                         direction, options);
}

std::variant<DispatchRequest, TransferError>
prepareDispatch(const Program &program, const BufferView &source,
                const BufferView &destination, TransferDirection direction,
                const Options &options) {
  if (auto error = checkView(program, source, true, direction))
    return *error;
  if (auto error = checkView(program, destination, false, direction))
    return *error;
  auto selected = selectImplementation(program, direction, options);
  if (auto *error = std::get_if<TransferError>(&selected))
    return *error;
  return makeRequest(program, source, destination, direction,
                     std::get<Selection>(selected));
}

std::variant<DispatchTemplate, TransferError>
prepareDispatchTemplate(const TypedBoundPlan &bound,
                        TransferDirection direction, const Options &options) {
  auto program = typed::prepareProgram(bound);
  if (auto *error = std::get_if<typed::ExecutionError>(&program))
    return fromExecution(*error);
  DispatchTemplate result;
  result.program = std::get<Program>(std::move(program));
  result.capability = queryCapability(result.program, direction, options.cuda);
  auto selected = select(result.program, result.capability, options);
  if (auto *error = std::get_if<TransferError>(&selected))
    return *error;
  const auto &choice = std::get<Choice>(selected);
  result.selection = Selection{*choice.row, choice.policy, choice.reason};
  result.direction = direction;
  result.cuda = options.cuda;
  return result;
}

std::variant<DispatchRequest, TransferError>
prepareDispatch(const DispatchTemplate &prepared, const BufferView &source,
                const BufferView &destination) {
  if (auto error =
          checkView(prepared.program, source, true, prepared.direction))
    return *error;
  if (auto error =
          checkView(prepared.program, destination, false, prepared.direction))
    return *error;
  const bool cuda =
      source.kind == MemoryKind::Cuda || destination.kind == MemoryKind::Cuda;
  if (cuda != prepared.cuda)
    return fail("backend_mismatch",
                "views do not match the prepared device end");
  return makeRequest(prepared.program, source, destination, prepared.direction,
                     prepared.selection);
}

std::variant<GroupRequest, TransferError>
prepareDispatchGroup(const std::vector<GroupEntry> &entries) {
  if (entries.empty() || entries.size() > 256)
    return fail("invalid_group",
                "a group must contain between 1 and 256 items");
  GroupRequest group;
  group.items.reserve(entries.size());
  group.reports.reserve(entries.size());
  struct Interval {
    uintptr_t begin, end;
    MemoryKind kind;
    int device;
  };
  std::vector<Interval> inputs, outputs;
  auto interval = [](const BufferView &view, size_t bytes,
                     Interval &out) -> bool {
    out.kind = view.kind;
    out.device = view.device;
    return !__builtin_add_overflow(view.base, view.offsetBytes, &out.begin) &&
           !__builtin_add_overflow(out.begin, bytes, &out.end);
  };
  auto overlaps = [](const Interval &a, const Interval &b) {
    return a.kind == b.kind && a.device == b.device && a.begin < b.end &&
           b.begin < a.end;
  };
  for (const auto &entry : entries) {
    Report report;
    size_t inputBytes = 0, outputBytes = 0;
    if (const auto *plan = std::get_if<DispatchTemplate>(&entry.plan)) {
      if (plan->direction != entry.direction)
        return fail("direction_mismatch",
                    "group direction differs from template");
      auto prepared = prepareDispatch(*plan, entry.source, entry.destination);
      if (const auto *error = std::get_if<TransferError>(&prepared))
        return *error;
      auto request = std::get<DispatchRequest>(std::move(prepared));
      inputBytes = request.program.plan.sourceBytes;
      outputBytes = request.program.plan.destinationBytes;
      report = request.report;
      group.items.emplace_back(std::move(request));
    } else {
      auto prepared =
          validateTransfer(std::get<BoundPlan>(entry.plan), entry.source,
                           entry.destination, entry.direction);
      if (const auto *error = std::get_if<TransferError>(&prepared))
        return *error;
      auto request = std::get<TransferRequest>(std::move(prepared));
      inputBytes = request.sourceSpanBytes;
      outputBytes = request.destinationBytes;
      report.implementation = "layout_cpu";
      report.policy = "layout";
      report.placementReason = "group_forward_layout";
      report.method = "A";
      report.sourceBytes = inputBytes;
      report.destinationBytes = outputBytes;
      report.wireBytes = entry.direction == TransferDirection::HostToDevice
                             ? outputBytes
                             : inputBytes;
      report.artifactVersion = 0;
      group.items.emplace_back(std::move(request));
    }
    for (const auto *view : {&entry.source, &entry.destination}) {
      if (view->kind == MemoryKind::Cuda) {
        if (group.device >= 0 && group.device != view->device)
          return fail("device_mismatch", "a group must use one CUDA device");
        group.device = view->device;
      }
    }
    Interval input{}, output{};
    if (!interval(entry.source, inputBytes, input) ||
        !interval(entry.destination, outputBytes, output) ||
        __builtin_add_overflow(group.report.outputBytes, uint64_t(outputBytes),
                               &group.report.outputBytes))
      return fail("integer_overflow",
                  "group view interval or output size overflow");
    inputs.push_back(input);
    outputs.push_back(output);
    group.reports.push_back(std::move(report));
  }
  if (group.device >= 0)
    for (const auto &entry : entries)
      if (entry.source.kind != MemoryKind::Cuda &&
          entry.destination.kind != MemoryKind::Cuda)
        return fail("device_mismatch",
                    "every member must use the group's CUDA device");
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto &input : inputs)
      if (overlaps(outputs[i], input))
        return fail("group_alias", "a group output aliases an input");
    for (size_t j = 0; j < i; ++j)
      if (overlaps(outputs[i], outputs[j]))
        return fail("group_alias", "group outputs must not overlap");
  }
  return group;
}

// Resources' arena keeps even logically freed scratch busy until the group
// finishes. On any failure Resources independently drains or quarantines the
// entire arena and all borrowed buffer owners. Never run this with a backend
// that releases or recycles freeStaging/freeDevice immediately.
std::optional<TransferError>
group_detail::executeGroup(GroupRequest &group, CopyBackend &base,
                           const TransferOptions &options,
                           std::vector<HostCompletion> *deferred) {
#ifdef RELOC_ENABLE_CUDA
  auto *backend = dynamic_cast<CudaBackend *>(&base);
  if (!backend)
    return fail("backend_mismatch", "groups require owned CUDA resources");
  if (group.consumed)
    return fail("already_executed", "group was already executed");
  if (backend->failed())
    return backendFailure(*backend, "group backend unusable");
  for (const auto &item : group.items) {
    auto error = std::visit(
        [&](const auto &request) -> std::optional<TransferError> {
          if (request.consumed)
            return fail("already_executed", "group item was already executed");
          if (auto error = checkDevice(request.source, *backend, "source"))
            return error;
          return checkDevice(request.destination, *backend, "destination");
        },
        item);
    if (error)
      return error;
  }
  GroupExecution execution{DeviceScratch{*backend}, {}};
  execution.scratch.group = &group.report;
  execution.host.reserve(group.items.size());
  group.consumed = true;
  for (auto &item : group.items)
    std::visit([](auto &request) { request.consumed = true; }, item);
  if (options.hasCallerStream && !backend->waitStream(options.callerStream))
    return backendFailure(*backend, "group producer ordering failed");
  CudaBackend::LaunchScope scope(*backend);
  if (backend->failed())
    return backendFailure(*backend, "selecting group device failed");
  auto direct = options;
  direct.directDenseUpload = true;
  for (size_t i = 0; i < group.items.size(); ++i) {
    auto &item = group.items[i];
    if (auto *typed = std::get_if<DispatchRequest>(&item);
        typed && typed->selected.id != kCpuReference) {
      if (auto error = executeCuda(*typed, *backend, direct, &execution))
        return error;
      group.reports[i] = typed->report;
      continue;
    }
    // CPU transforms (typed reference or layout-only): H2D can submit each
    // completed CPU result while the next transform runs. D2H submits every
    // input copy first, then applies the forward transforms after one barrier.
    const auto source =
        std::visit([](const auto &r) { return r.source; }, item);
    const auto destination =
        std::visit([](const auto &r) { return r.destination; }, item);
    const auto direction =
        std::visit([](const auto &r) { return r.direction; }, item);
    const auto *src =
        reinterpret_cast<const uint8_t *>(source.base) + source.offsetBytes;
    auto *dst =
        reinterpret_cast<uint8_t *>(destination.base) + destination.offsetBytes;
    const auto &report = group.reports[i];
    auto transform = [&item,
                      direct](const void *input,
                              void *output) -> std::optional<TransferError> {
      if (auto *typed = std::get_if<DispatchRequest>(&item))
        return hostProgram(typed->program, 0, typed->program.stages.size(),
                           input, output, direct);
      detail::forwardHostGather(std::get<TransferRequest>(item).bound, input,
                                output, direct);
      return std::nullopt;
    };
    StagingGuard staging{*backend};
    if (direction == TransferDirection::HostToDevice) {
      if (std::holds_alternative<DispatchRequest>(item)) {
        group.reports[i].hostPipeline = "whole_group";
        group.reports[i].hostChunks = group.reports[i].hostBuffers = 1;
        group.reports[i].hostChunkBytes = report.destinationBytes;
      }
      staging.buffer = backend->allocStaging(report.destinationBytes);
      if (!staging.buffer)
        return backendFailure(*backend, "group host allocation failed");
      if (auto error = transform(src, staging.buffer))
        return error;
      backend->copyAsync(0, dst, staging.buffer, report.destinationBytes,
                         CopyDir::HostToDevice);
      ++group.report.payloadCopyCalls;
    } else {
      if (auto error = stageFromDevice(src, report.sourceBytes, *backend,
                                       staging, &group.report))
        return error;
      void *input = staging.buffer;
      execution.host.emplace_back(
          [transform, input, dst] { return transform(input, dst); });
    }
    group.report.hostTransformBytes += report.destinationBytes;
    group.reports[i].payloadBytesTransferred = report.wireBytes;
    if (backend->failed())
      return backendFailure(*backend, "group payload copy failed");
  }
  if (deferred) {
    *deferred = std::move(execution.host);
    return std::nullopt;
  }
  if (auto error = finishQueue(*backend))
    return error;
  for (auto &finish : execution.host)
    if (auto error = finish())
      return error;
  for (auto &report : group.reports)
    report.executed = true;
  return std::nullopt;
#else
  (void)group;
  (void)base;
  (void)options;
  (void)deferred;
  return fail("backend_mismatch", "group execution requires a CUDA build");
#endif
}

std::optional<TransferError> executeDispatch(DispatchRequest &request,
                                             CopyBackend &backend,
                                             const TransferOptions &options) {
  if (request.consumed)
    return fail("already_executed", "dispatch request was already executed");
  if (backend.failed())
    return backendFailure(backend, "backend unusable before launch");
  if (auto error = checkDevice(request.source, backend, "source"))
    return error;
  if (auto error = checkDevice(request.destination, backend, "destination"))
    return error;
  request.consumed = true;
  if (options.hasCallerStream && !backend.waitStream(options.callerStream))
    return backendFailure(backend, "ordering after the caller stream failed");

  const Program &program = request.program;
  const uint32_t stageCount = static_cast<uint32_t>(program.stages.size());
  const auto *src = reinterpret_cast<const uint8_t *>(request.source.base) +
                    request.source.offsetBytes;
  auto *dst = reinterpret_cast<uint8_t *>(request.destination.base) +
              request.destination.offsetBytes;

  if (request.selected.id == kCpuReference) {
    if (request.direction == TransferDirection::HostToDevice) {
      if (auto error = transformToDevice(program, stageCount, src, dst, backend,
                                         options, request.report))
        return error;
      request.report.payloadBytesTransferred = program.plan.destinationBytes;
    } else {
      StagingGuard staging{backend};
      if (auto error =
              stageFromDevice(src, program.plan.sourceBytes, backend, staging))
        return error;
      request.report.payloadBytesTransferred = program.plan.sourceBytes;
      if (auto error =
              hostProgram(program, 0, stageCount, staging.buffer, dst, options))
        return error;
    }
    request.report.executed = true;
    return std::nullopt;
  }

#ifdef RELOC_ENABLE_CUDA
  auto *cudaBackend = dynamic_cast<CudaBackend *>(&backend);
  if (cudaBackend == nullptr)
    return fail("backend_mismatch",
                "row " + request.selected.label() + " needs a CudaBackend");
  {
    CudaBackend::LaunchScope scope(*cudaBackend);
    if (cudaBackend->failed())
      return backendFailure(backend, "selecting kernel device failed");
    if (auto error = executeCuda(request, *cudaBackend, options))
      return error;
  }
  if (cudaBackend->failed())
    return backendFailure(backend, "restoring kernel device failed");
  request.report.executed = true;
  return std::nullopt;
#else
  return fail("backend_mismatch",
              "row " + request.selected.label() +
                  " needs a CUDA build (RELOC_ENABLE_CUDA)");
#endif
}

//===----------------------------------------------------------------------===//
// Prefold capability.
//===----------------------------------------------------------------------===//

std::variant<PrefoldSpec, TransferError>
prefoldSpecFor(const TypedBoundPlan &plan) {
  auto prepared = typed::prepareProgram(plan);
  if (auto *error = std::get_if<typed::ExecutionError>(&prepared))
    return fail("prefold_unavailable", error->message);
  const Program &program = std::get<Program>(prepared);
  const BoundPlan &layout = program.plan.layout;
  if (program.stages.size() != 1 ||
      program.stages[0].transform != ValueTransformKind::Quantize)
    return fail("prefold_unavailable",
                "the prefolder implements exactly one symmetric_rne quantize");
  if (!layout.padRegions.empty())
    return fail("prefold_unavailable", "layout has pads");
  if (layout.extents.size() < 2)
    return fail(
        "prefold_unavailable",
        "the prefolder needs a distinct outer channel axis (rank >= 2)");
  int64_t packed = 1;
  for (size_t k = layout.extents.size(); k-- > 0;) {
    if (layout.dstStrides[k] != packed)
      return fail("prefold_unavailable", "destination is not packed row-major");
    packed *= layout.extents[k];
  }
  const typed::StageArithmetic &stage = program.stages[0];
  PrefoldSpec spec;
  if (stage.perChannel) {
    if (!stage.channelIsDim || stage.channelDim != 0 ||
        layout.extents.front() != program.plan.resultExtents.front() ||
        layout.dstStrides.front() != program.resultStrides.front())
      return fail("prefold_unavailable",
                  "channel axis is not the coalesced outer axis");
    spec.invScales = stage.invScale;
  } else {
    spec.invScales.assign(static_cast<size_t>(layout.extents.front()),
                          stage.invScale[0]);
  }
  spec.spec = layout.srcStrides == layout.dstStrides
                  ? prefold::OutputSpec::S8QuantPack
                  : prefold::OutputSpec::S8GatherQuant;
  return spec;
}

} // namespace dispatch
} // namespace reloc
