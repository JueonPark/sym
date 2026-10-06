//===- PyTransfer.cpp - validated forward transfer bindings ---------------===//

#include "PyTransfer.h"
#include "PyPinning.h"

#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/Transfer.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#endif

#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {

/// Raised as pyreloc.TransferError; the message is "<code>: <detail>" so
/// Python can recover the stable reason code.
struct TransferException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void raise(const reloc::TransferError &error) {
  throw TransferException(error.code + ": " + error.message);
}

reloc::MemoryKind parseKind(const std::string &kind) {
  if (kind == "host")
    return reloc::MemoryKind::Host;
  if (kind == "cuda")
    return reloc::MemoryKind::Cuda;
  throw py::value_error("kind must be 'host' or 'cuda', got '" + kind + "'");
}

const char *kindName(reloc::MemoryKind kind) {
  return kind == reloc::MemoryKind::Host ? "host" : "cuda";
}

reloc::TransferDirection parseDirection(const std::string &direction) {
  if (direction == "h2d")
    return reloc::TransferDirection::HostToDevice;
  if (direction == "d2h")
    return reloc::TransferDirection::DeviceToHost;
  throw py::value_error("direction must be 'h2d' or 'd2h', got '" + direction +
                        "'");
}

const char *directionName(reloc::TransferDirection direction) {
  return direction == reloc::TransferDirection::HostToDevice ? "h2d" : "d2h";
}

reloc::BufferView makeView(uintptr_t base, size_t capacityBytes,
                           size_t offsetBytes, std::vector<int64_t> extents,
                           std::vector<int64_t> strides, uint32_t elementSize,
                           const std::string &kind, int device) {
  reloc::BufferView view;
  view.base = base;
  view.capacityBytes = capacityBytes;
  view.offsetBytes = offsetBytes;
  view.extents = std::move(extents);
  view.strides = std::move(strides);
  view.elementSize = elementSize;
  view.kind = parseKind(kind);
  view.device = device;
  return view;
}

size_t spanBytes(const reloc::BufferView &view) {
  auto result = reloc::viewSpanBytes(view, "view");
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return std::get<size_t>(result);
}

size_t validateSource(const reloc::BoundPlan &bound,
                      const reloc::BufferView &source,
                      const std::string &direction) {
  auto result =
      reloc::validateTransferSource(bound, source, parseDirection(direction));
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return std::get<size_t>(result);
}

struct PythonTransferRequest {
  reloc::TransferRequest native;
  // Accessed only under the GIL. Avoid reading native.consumed while native
  // execution can write it, and reject concurrent use of the same request.
  bool executing = false;
  std::vector<reloc::StagingDecision> staging;
};

PythonTransferRequest makeTransfer(const reloc::BoundPlan &bound,
                                   const reloc::BufferView &source,
                                   const reloc::BufferView &destination,
                                   const std::string &direction) {
  auto result = reloc::validateTransfer(bound, source, destination,
                                        parseDirection(direction));
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return {std::get<reloc::TransferRequest>(std::move(result))};
}

size_t validateStackedSourcesPy(const reloc::BoundPlan &bound,
                                const std::vector<reloc::BufferView> &sources,
                                const std::string &direction) {
  auto result =
      reloc::validateStackedSources(bound, sources, parseDirection(direction));
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return std::get<size_t>(result);
}

PythonTransferRequest
makeStackedTransfer(const reloc::BoundPlan &bound,
                    const std::vector<reloc::BufferView> &sources,
                    const reloc::BufferView &destination,
                    const std::string &direction) {
  auto result = reloc::validateStackedTransfer(bound, sources, destination,
                                               parseDirection(direction));
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return {std::get<reloc::TransferRequest>(std::move(result))};
}

void checkProcess(const reloc::TransferResourceCache &cache) {
  if (!cache.stats().processValid)
    raise({"process_mismatch", "resource cache belongs to another process"});
}

void executeTransferPy(PythonTransferRequest &wrapped,
                       const py::object &callerStream, int nBuffers,
                       int nStreams, int gatherThreads,
                       std::shared_ptr<reloc::GatherPool> pool,
                       std::shared_ptr<reloc::TransferResourceCache> resources,
                       const py::object &owners, const std::string &pinning,
                       std::optional<size_t> minPinnedBytes) {
  if (resources)
    checkProcess(*resources); // before any inherited pool locks
  if (wrapped.executing)
    raise({"already_executed", "transfer request is already executing"});
  // Claim the Python entry before argument conversion can invoke Python (for
  // example a tuple subclass). Native preflight still controls consumed.
  wrapped.executing = true;
  struct ResetEntry {
    bool &executing;
    ~ResetEntry() { executing = false; } // after reacquiring the GIL
  } reset{wrapped.executing};
  if (nBuffers < 1)
    throw py::value_error("n_buffers must be >= 1");
  if (nStreams < 1)
    throw py::value_error("n_streams must be >= 1");
  if (gatherThreads < 0)
    throw py::value_error("gather_threads must be >= 0 (0 = all cores)");
  if (pool && pool->closed())
    throw py::value_error("gather_pool is closed");
  std::shared_ptr<py::object> token;
  if (resources || !owners.is_none()) {
    if (!py::isinstance<py::tuple>(owners) || py::len(owners) != 2 ||
        owners[py::int_(0)].is_none() || owners[py::int_(1)].is_none())
      throw py::value_error(
          "owners must be a (source, destination) tuple of strong owners");
    token = std::make_shared<py::object>(owners);
  }
  auto &request = wrapped.native;
  reloc::TransferOptions options;
  options.nBuffers = nBuffers;
  options.pinning = parsePinning(pinning);
  options.minPinnedBytes = minPinnedBytes;
  wrapped.staging.clear();
  options.staging = &wrapped.staging;
  options.gatherThreads = static_cast<unsigned>(gatherThreads);
  options.gather = pool.get();
  if (!callerStream.is_none()) {
    options.hasCallerStream = true;
    options.callerStream =
        reinterpret_cast<const void *>(callerStream.cast<uintptr_t>());
  }
  const bool cuda = request.source.kind == reloc::MemoryKind::Cuda ||
                    request.destination.kind == reloc::MemoryKind::Cuda;
  const int device = !cuda ? -1
                     : request.source.kind == reloc::MemoryKind::Cuda
                         ? request.source.device
                         : request.destination.device;
  std::optional<reloc::TransferError> error;
  {
    // Keep our own token reference across the released-GIL scope. Every normal
    // last release therefore occurs with the GIL held, even on an exception.
    // Unknown completion retains its copy in native process-lifetime
    // quarantine; no Python decref or GIL acquisition is attempted at
    // interpreter shutdown.
    py::gil_scoped_release release;
    if (resources) {
      reloc::CachedTransferOptions cached;
      cached.transfer = options;
      cached.backend = {cuda ? reloc::MemoryKind::Cuda
                             : reloc::MemoryKind::Host,
                        device, nStreams};
      cached.gather = pool;
      error = reloc::executeTransferCached(request, *resources, cached, token)
                  .error;
    } else {
      auto decision = reloc::selectStaging(
          options,
          request.direction == reloc::TransferDirection::HostToDevice
              ? request.destinationBytes
              : request.sourceSpanBytes,
          false);
      if (!cuda) {
        decision.pinned = false;
        decision.reason = "host_backend";
      }
      wrapped.staging.push_back(decision);
      std::unique_ptr<reloc::CopyBackend> backend;
      if (cuda) {
#ifdef RELOC_ENABLE_CUDA
        backend = std::make_unique<reloc::CudaBackend>(nStreams, device,
                                                       decision.pinned);
#else
        raise(
            {"backend_failure", "pyreloc was built without RELOC_ENABLE_CUDA"});
#endif
      } else {
        backend = std::make_unique<reloc::HostBackend>(nStreams);
      }
      if (token) {
        reloc::TransferContext context(std::move(backend));
        error = reloc::executeTransfer(request, context, options, token).error;
      } else {
        // Legacy raw-pointer callers still own exceptional buffer lifetimes.
        error = reloc::executeTransfer(request, *backend, options);
      }
    }
  }
  if (error)
    raise(*error);
}

py::dict usageDict(const reloc::TransferResourceUsage &s) {
  py::dict d;
#define STAT(name, field) d[name] = s.field
  STAT("contexts", contexts);
  STAT("building", building);
  STAT("leased", leased);
  STAT("idle", idle);
  STAT("retiring", retiring);
  STAT("quarantined", quarantined);
  STAT("allocated_staging_bytes", allocatedStagingBytes);
  STAT("reserved_staging_bytes", reservedStagingBytes);
  STAT("retained_allocated_bytes", retainedAllocatedBytes);
  STAT("retained_reserved_bytes", retainedReservedBytes);
  STAT("background_workers", backgroundWorkers);
  STAT("reserved_workers", reservedWorkers);
  STAT("streams", streams);
  STAT("quarantine_bytes", quarantineBytes);
  STAT("outstanding_events", outstandingEvents);
#undef STAT
  return d;
}

py::dict resourceStats(const reloc::TransferResourceCache &cache) {
  const auto s = cache.stats();
  if (!s.processValid)
    raise({"process_mismatch", "resource cache belongs to another process"});
  auto d = usageDict(s);
#define STAT(name, field) d[name] = s.field
  STAT("requests", requests);
  STAT("hits", hits);
  STAT("growths", growths);
  STAT("misses", misses);
  STAT("evictions", evictions);
  STAT("ephemeral", ephemeral);
  STAT("waits", waits);
  STAT("timeouts", timeouts);
  STAT("failures", failures);
  STAT("quarantines", quarantines);
  STAT("waiters", waiters);
  STAT("staging_allocations", stagingAllocations);
  STAT("staging_frees", stagingFrees);
  STAT("stream_creations", streamCreations);
  STAT("stream_destructions", streamDestructions);
  STAT("worker_creations", workerCreations);
  STAT("worker_joins", workerJoins);
  STAT("event_creations", eventCreations);
  STAT("event_retirements", eventRetirements);
  STAT("peak_live_staging_bytes", peakLiveStagingBytes);
  STAT("peak_allocated_staging_bytes", peakAllocatedStagingBytes);
  STAT("generation", generation);
  STAT("closed", closed);
  STAT("process_valid", processValid);
#undef STAT
  py::list devices;
  for (const auto &device : s.devices) {
    auto entry = usageDict(device);
    entry["kind"] = kindName(device.kind);
    entry["device"] = device.device;
    entry["disabled"] = device.disabled;
    devices.append(entry);
  }
  d["devices"] = devices;
  return d;
}

int cudaPointerDevicePy(uintptr_t pointer) {
#ifdef RELOC_ENABLE_CUDA
  int device = -1;
  std::string error;
  if (!reloc::cudaPointerDevice(reinterpret_cast<const void *>(pointer), device,
                                error))
    throw TransferException("invalid_view: " + error);
  return device;
#else
  (void)pointer;
  throw TransferException(
      "backend_failure: pyreloc was built without RELOC_ENABLE_CUDA");
#endif
}

} // namespace

void registerTransferBindings(py::module_ &m) {
  py::register_exception<TransferException>(m, "TransferError");

  py::class_<reloc::BufferView>(
      m, "BufferView",
      "A framework buffer: the allocation it lives in (base, capacity) plus "
      "the logical view (offset, extents, element strides). kind is 'host' "
      "or 'cuda'; device is the CUDA ordinal (-1 for host).")
      .def(py::init(&makeView), py::arg("base"), py::arg("capacity_bytes"),
           py::arg("offset_bytes"), py::arg("extents"), py::arg("strides"),
           py::arg("element_size"), py::arg("kind"), py::arg("device") = -1)
      .def_readonly("base", &reloc::BufferView::base)
      .def_readonly("capacity_bytes", &reloc::BufferView::capacityBytes)
      .def_readonly("offset_bytes", &reloc::BufferView::offsetBytes)
      .def_readonly("extents", &reloc::BufferView::extents)
      .def_readonly("strides", &reloc::BufferView::strides)
      .def_readonly("element_size", &reloc::BufferView::elementSize)
      .def_property_readonly(
          "kind", [](const reloc::BufferView &v) { return kindName(v.kind); })
      .def_readonly("device", &reloc::BufferView::device)
      .def_property_readonly("span_bytes", &spanBytes,
                             "Checked byte span; raises TransferError for "
                             "empty, overlapping or overflowing views.");

  py::class_<PythonTransferRequest>(
      m, "TransferRequest",
      "A validated, single-use forward transfer owning copies of the bound "
      "plan, the destination view and either the source view or every "
      "stacked source view; source is unused for stacked requests.")
      .def_property_readonly("direction",
                             [](const PythonTransferRequest &r) {
                               return directionName(r.native.direction);
                             })
      .def_property_readonly("source_span_bytes",
                             [](const PythonTransferRequest &r) {
                               return r.native.sourceSpanBytes;
                             })
      .def_property_readonly("destination_bytes",
                             [](const PythonTransferRequest &r) {
                               return r.native.destinationBytes;
                             })
      .def_property_readonly("consumed",
                             [](const PythonTransferRequest &r) {
                               return r.executing || r.native.consumed;
                             })
      .def_property_readonly(
          "staging",
          [](const PythonTransferRequest &r) {
            if (r.executing)
              throw py::value_error(
                  "staging report is unavailable during execution");
            return stagingReport(r.staging);
          })
      .def_property_readonly(
          "source",
          [](const PythonTransferRequest &r) { return r.native.source; })
      .def_property_readonly(
          "stack_sources",
          [](const PythonTransferRequest &r) { return r.native.stackSources; })
      .def_property_readonly("stack_segment_elements",
                             [](const PythonTransferRequest &r) {
                               return r.native.stackSegmentElements;
                             })
      .def_property_readonly("destination", [](const PythonTransferRequest &r) {
        return r.native.destination;
      });

  py::class_<reloc::TransferResourceCache,
             std::shared_ptr<reloc::TransferResourceCache>>(
      m, "TransferResourceCache",
      "Explicit lazy transfer resource cache; no CUDA initialization until "
      "execution.")
      .def(py::init([](size_t retained, size_t contexts, size_t perDevice,
                       size_t workers, std::optional<size_t> live,
                       std::optional<int64_t> timeout) {
             reloc::TransferResourceLimits limits;
             limits.maxRetainedBytes = retained;
             limits.maxContexts = contexts;
             limits.maxContextsPerDevice = perDevice;
             limits.maxBackgroundWorkers = workers;
             limits.maxLiveStagingBytes = live;
             if (timeout)
               limits.acquireTimeout = std::chrono::milliseconds(*timeout);
             return reloc_python::makeResourceCache(limits);
           }),
           py::kw_only(), py::arg("max_retained_bytes") = size_t(256) << 20,
           py::arg("max_contexts") = 4, py::arg("max_contexts_per_device") = 2,
           py::arg("max_background_workers") = 64,
           py::arg("max_live_staging_bytes") = py::none(),
           py::arg("acquire_timeout_ms") = py::none())
      .def("stats", &resourceStats)
      .def(
          "close",
          [](reloc::TransferResourceCache &c) {
            if (auto e = c.close())
              raise(*e);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "clear",
          [](reloc::TransferResourceCache &c) {
            if (auto e = c.clear())
              raise(*e);
          },
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("closed",
                             [](const reloc::TransferResourceCache &c) {
                               checkProcess(c);
                               return c.stats().closed;
                             })
      .def("__enter__",
           [](std::shared_ptr<reloc::TransferResourceCache> c) {
             checkProcess(*c);
             if (c->stats().closed)
               raise({"resources_closed", "resource cache is closed"});
             return c;
           })
      .def("__exit__",
           [](reloc::TransferResourceCache &c, py::object, py::object,
              py::object) {
             std::optional<reloc::TransferError> error;
             {
               py::gil_scoped_release release;
               error = c.close();
             }
             if (error)
               raise(*error);
             return false;
           })
      .def("__reduce_ex__", [](const reloc::TransferResourceCache &, int) {
        throw py::type_error("transfer resources cannot be serialized");
      });

  m.def("validate_transfer_source", &validateSource, py::arg("bound"),
        py::arg("source"), py::arg("direction"),
        "Preflight: prove the bound plan's source accesses fit `source` and "
        "that the view is admissible for `direction`. Returns the source span "
        "in bytes; raises TransferError('<code>: <detail>') otherwise. "
        "Allocates and launches nothing.");
  m.def("make_transfer", &makeTransfer, py::arg("bound"), py::arg("source"),
        py::arg("destination"), py::arg("direction"),
        "Full validation with the dense destination view; returns the "
        "single-use TransferRequest.");
  m.def("validate_stacked_sources", &validateStackedSourcesPy, py::arg("bound"),
        py::arg("sources"), py::arg("direction"),
        "Preflight for a stacked request (torch.stack): every source is a "
        "dense host view with the same element count and size, and the plan's "
        "logical source is their concatenation in order. Returns the summed "
        "source span in bytes; raises TransferError('<code>: <detail>'). "
        "Allocates and launches nothing.");
  m.def("make_stacked_transfer", &makeStackedTransfer, py::arg("bound"),
        py::arg("sources"), py::arg("destination"), py::arg("direction"),
        "Full validation of a stacked request with its dense destination; "
        "returns the single-use TransferRequest. Execute it with "
        "execute_transfer(..., owners=(sources_owner, destination_owner)).");
  m.def(
      "execute_transfer", &executeTransferPy, py::arg("request"), py::kw_only(),
      py::arg("caller_stream") = py::none(), py::arg("n_buffers") = 4,
      py::arg("n_streams") = 2, py::arg("gather_threads") = 1,
      py::arg("gather_pool") = nullptr, py::arg("resources") = nullptr,
      py::arg("owners") = py::none(), py::arg("pinning") = "auto",
      py::arg("min_pinned_bytes") = py::none(),
      "Run the forward transfer and block until this request's work has "
      "completed. caller_stream (a cudaStream_t handle; 0 is the legacy "
      "default stream, None means no producer to order after) is recorded "
      "before any private stream touches shared storage. Raises "
      "TransferError on a consumed request or a backend failure. Cached calls "
      "require owners=(source_owner, destination_owner); successful calls "
      "release them and unknown completion quarantines them. Supplying owners "
      "without resources uses the same ownership protection with ephemeral "
      "resources. Raw callers omitting owners must retain buffers themselves "
      "on completion_unknown.");
  m.def("cuda_pointer_device", &cudaPointerDevicePy, py::arg("pointer"),
        "CUDA device ordinal owning `pointer` (cudaPointerGetAttributes); "
        "raises TransferError for host/unknown memory or CUDA-less builds.");
}
