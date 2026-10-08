//===- PyDispatch.cpp - typed dispatch bindings (R3, issue #147) ----------===//

#include "PyDispatch.h"
#include "PyPinning.h"

#include "reloc/Dispatch.h"
#include "reloc/DispatchResources.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
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

/// pyreloc.TransferError is registered by PyTransfer.cpp; raising the same
/// Python type from here keeps one exception for every request-level
/// failure ("<code>: <detail>").
[[noreturn]] void raise(const reloc::TransferError &error) {
  py::object module = py::module_::import("pyreloc._pyreloc");
  py::object type = module.attr("TransferError");
  PyErr_SetString(reinterpret_cast<PyObject *>(type.ptr()),
                  (error.code + ": " + error.message).c_str());
  throw py::error_already_set();
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

reloc::dispatch::Policy parsePolicy(const std::string &policy) {
  if (policy == "auto")
    return reloc::dispatch::Policy::Auto;
  if (policy == "original_cpu")
    return reloc::dispatch::Policy::OriginalCpu;
  throw py::value_error("policy must be 'auto' or 'original_cpu', got '" +
                        policy + "'");
}

py::dict rowDict(const reloc::dispatch::Implementation &row) {
  py::dict out;
  out["implementation"] = row.label();
  out["wire_boundary"] = row.wireBoundary;
  out["wire_result_layout"] = row.wireResultLayout;
  out["wire_bytes"] = row.wireBytes;
  out["device_temp_bytes"] = row.deviceTempBytes;
  out["method"] = row.method;
  return out;
}

py::dict reportDict(const reloc::dispatch::Report &r) {
  py::dict out;
  out["implementation"] = r.implementation;
  out["policy"] = r.policy;
  out["placement_reason"] = r.placementReason;
  out["method"] = r.method;
  out["wire_boundary"] = r.wireBoundary;
  out["source_bytes"] = r.sourceBytes;
  out["wire_bytes"] = r.wireBytes;
  out["destination_bytes"] = r.destinationBytes;
  out["parameter_bytes"] = r.parameterBytes;
  out["payload_bytes_transferred"] = r.payloadBytesTransferred;
  out["device_temp_bytes"] = r.deviceTempBytes;
  out["artifact_version"] = r.artifactVersion;
  out["executed"] = r.executed;
  out["host_pipeline"] = r.hostPipeline;
  out["host_chunks"] = r.hostChunks;
  out["host_chunk_bytes"] = r.hostChunkBytes;
  out["host_buffers"] = r.hostBuffers;
  out["staging"] = stagingReport(r.staging);
  return out;
}

reloc::typed::Program checkedProgram(const reloc::TypedBoundPlan &bound) {
  auto prepared = reloc::typed::prepareProgram(bound);
  if (auto *error = std::get_if<reloc::typed::ExecutionError>(&prepared))
    raise({error->code, error->message});
  return std::get<reloc::typed::Program>(std::move(prepared));
}
const reloc::typed::Program &
checkedProgram(const reloc::typed::Program &program) {
  return program;
}

py::dict capabilityDict(const reloc::dispatch::Capability &capability) {
  py::list eligible, excluded;
  for (const auto &row : capability.eligible)
    eligible.append(rowDict(row));
  for (const auto &row : capability.excluded) {
    py::dict entry;
    entry["implementation"] = row.id;
    entry["reason"] = row.reason;
    excluded.append(entry);
  }
  py::dict out;
  out["eligible"] = eligible;
  out["excluded"] = excluded;
  return out;
}

template <typename Plan>
py::dict queryCapability(const Plan &bound, const std::string &direction,
                         const std::string &device) {
  if (device != "host" && device != "cuda")
    throw py::value_error("device must be 'host' or 'cuda', got '" + device +
                          "'");
  const auto &program = checkedProgram(bound);
  auto capability = reloc::dispatch::queryCapability(
      program, parseDirection(direction), device == "cuda");
  return capabilityDict(capability);
}

template <typename Plan>
reloc::dispatch::DispatchRequest
prepareDispatch(const Plan &bound, const reloc::BufferView &source,
                const reloc::BufferView &destination,
                const std::string &direction, const std::string &policy,
                const reloc::costmodel::CostModel *calibration, int threads,
                const std::string &implementation) {
  if (threads < 1)
    throw py::value_error("threads must be >= 1");
  reloc::dispatch::Options options;
  options.policy = parsePolicy(policy);
  options.model = calibration;
  options.threads = threads;
  options.cuda = source.kind == reloc::MemoryKind::Cuda ||
                 destination.kind == reloc::MemoryKind::Cuda;
  options.implementation = implementation;
  auto prepared = reloc::dispatch::prepareDispatch(
      bound, source, destination, parseDirection(direction), options);
  if (auto *error = std::get_if<reloc::TransferError>(&prepared))
    raise(*error);
  return std::get<reloc::dispatch::DispatchRequest>(std::move(prepared));
}

py::dict executeDispatchPy(
    reloc::dispatch::DispatchRequest &request, const py::object &callerStream,
    int nBuffers, int nStreams, int gatherThreads,
    std::shared_ptr<reloc::GatherPool> pool, const std::string &pinning,
    std::optional<size_t> minPinnedBytes, bool directDenseUpload, bool pipeline,
    size_t chunkSize, std::shared_ptr<reloc::dispatch::Resources> resources,
    const py::object &owners) {
  if (request.executing)
    raise({"already_executed", "typed request is executing"});
  request.executing = true;
  struct Reset {
    bool &value;
    ~Reset() { value = false; }
  } reset{request.executing};
  std::shared_ptr<py::object> token;
  if (resources || !owners.is_none()) {
    if (!py::isinstance<py::tuple>(owners) || py::len(owners) != 2 ||
        owners[py::int_(0)].is_none() || owners[py::int_(1)].is_none())
      throw py::value_error("owners must strongly own source and destination");
    token = std::make_shared<py::object>(owners);
  }
  if (nBuffers < 1)
    throw py::value_error("n_buffers must be >= 1");
  if (nStreams < 1)
    throw py::value_error("n_streams must be >= 1");
  if (gatherThreads < 0)
    throw py::value_error("gather_threads must be >= 0 (0 = all cores)");
  if (pool && pool->closed())
    throw py::value_error("gather_pool is closed");
  reloc::TransferOptions options;
  options.nBuffers = nBuffers;
  options.pinning = parsePinning(pinning);
  options.minPinnedBytes = minPinnedBytes;
  request.report.staging.clear();
  options.staging = &request.report.staging;
  options.directDenseUpload = directDenseUpload;
  options.pipelineTypedH2D = pipeline;
  options.chunkSizeOverride = chunkSize;
  options.gatherThreads = static_cast<unsigned>(gatherThreads);
  options.gather = pool.get();
  if (!callerStream.is_none()) {
    options.hasCallerStream = true;
    options.callerStream =
        reinterpret_cast<const void *>(callerStream.cast<uintptr_t>());
  }
  const bool cuda = request.source.kind == reloc::MemoryKind::Cuda ||
                    request.destination.kind == reloc::MemoryKind::Cuda;
  const int device = request.source.kind == reloc::MemoryKind::Cuda
                         ? request.source.device
                         : request.destination.device;
  std::optional<reloc::TransferError> error;
  {
    // The Python caller retains every owner (tensors, parameters, request)
    // for the duration of this blocking call; only the GIL is released.
    py::gil_scoped_release release;
    if (cuda) {
#ifdef RELOC_ENABLE_CUDA
      if (resources) {
        error =
            resources->execute(request, device, nStreams, options, token).error;
      } else if (token) {
        reloc::dispatch::Resources ephemeral(0);
        error =
            ephemeral.execute(request, device, nStreams, options, token).error;
      } else {
        // Raw callers retain their buffers and backend lifetimes themselves.
        reloc::CudaBackend backend(
            nStreams, device,
            reloc::usePinnedStaging(options, request.selected.wireBytes,
                                    false));
        error = reloc::dispatch::executeDispatch(request, backend, options);
      }
#else
      (void)device;
      error = reloc::TransferError{
          "backend_failure", "pyreloc was built without RELOC_ENABLE_CUDA"};
#endif
    } else {
      reloc::HostBackend backend(nStreams);
      error = reloc::dispatch::executeDispatch(request, backend, options);
    }
  }
  if (error)
    raise(*error);
  return reportDict(request.report);
}

py::dict groupReport(const reloc::dispatch::GroupRequest &group) {
  const auto &r = group.report;
  py::dict out;
  out["logical_transfers"] = group.items.size();
  out["request_count"] = 1;
  out["completion_granularity"] = "group";
  out["payload_copy_calls"] = r.payloadCopyCalls;
  out["parameter_uploads"] = r.parameterUploads;
  out["parameter_reuses"] = r.parameterReuses;
  out["parameter_upload_bytes"] = r.parameterUploadBytes;
  out["packing_bytes"] = r.packingBytes;
  out["host_transform_bytes"] = r.hostTransformBytes;
  out["kernel_launches"] = r.kernelLaunches;
  out["output_bytes"] = r.outputBytes;
  out["copy_calls"] = r.copyCalls;
  out["event_records"] = r.eventRecords;
  out["event_waits"] = r.eventWaits;
  out["caller_waits"] = r.callerWaits;
  out["scratch_peak_bytes"] = r.scratchPeakBytes;
  out["staging"] = stagingReport(r.staging);
  py::list items;
  for (const auto &item : group.reports)
    items.append(reportDict(item));
  out["items"] = items;
  return out;
}
using GroupArguments = std::tuple<
    std::variant<reloc::dispatch::DispatchTemplate, reloc::BoundPlan>,
    reloc::BufferView, reloc::BufferView, std::string>;
reloc::dispatch::GroupRequest
prepareGroup(const std::vector<GroupArguments> &args) {
  std::vector<reloc::dispatch::GroupEntry> entries;
  entries.reserve(args.size());
  for (const auto &[plan, source, destination, direction] : args)
    entries.push_back({plan, source, destination, parseDirection(direction)});
  auto group = reloc::dispatch::prepareDispatchGroup(entries);
  if (auto *error = std::get_if<reloc::TransferError>(&group))
    raise(*error);
  return std::get<reloc::dispatch::GroupRequest>(std::move(group));
}
py::dict executeGroupPy(reloc::dispatch::GroupRequest &group,
                        const py::object &callerStream, int gatherThreads,
                        std::shared_ptr<reloc::GatherPool> pool,
                        const std::string &pinning,
                        std::optional<size_t> minPinnedBytes,
                        size_t maxScratchBytes,
                        std::shared_ptr<reloc::dispatch::Resources> resources,
                        const py::object &owners) {
  if (group.executing || group.consumed)
    raise({"already_executed", "group is executing or already executed"});
  if (!py::isinstance<py::tuple>(owners) || py::len(owners) != 2)
    throw py::value_error("owners must be (source owners, destination owners)");
  for (const auto &part : owners.cast<py::tuple>()) {
    if (!py::isinstance<py::tuple>(part) || py::len(part) != group.items.size())
      throw py::value_error("each group buffer needs a strong owner");
    for (const auto &owner : part.cast<py::tuple>())
      if (owner.is_none())
        throw py::value_error("group owner cannot be None");
  }
  if (gatherThreads < 1 || !maxScratchBytes)
    throw py::value_error(
        "gather_threads and max_scratch_bytes must be positive");
  if (pool && pool->closed())
    throw py::value_error("gather_pool is closed");
  reloc::TransferOptions options;
  options.nBuffers = 1;
  options.gatherThreads = gatherThreads;
  options.gather = pool.get();
  options.pinning = parsePinning(pinning);
  options.minPinnedBytes = minPinnedBytes;
  options.maxScratchBytes = maxScratchBytes;
  options.staging = &group.report.staging;
  if (!callerStream.is_none()) {
    options.hasCallerStream = true;
    options.callerStream =
        reinterpret_cast<const void *>(callerStream.cast<uintptr_t>());
  }
  auto token = std::make_shared<py::object>(owners);
  group.executing = true;
  struct Reset {
    bool &value;
    ~Reset() { value = false; }
  } reset{group.executing};
  std::optional<reloc::TransferError> error;
  {
    py::gil_scoped_release release;
    if (resources)
      error = resources->execute(group, options, token).error;
    else {
      reloc::dispatch::Resources ephemeral(0);
      error = ephemeral.execute(group, options, token).error;
    }
  }
  if (error)
    raise(*error);
  return groupReport(group);
}

template <typename Plan>
py::dict selectDispatch(const Plan &bound, const std::string &direction,
                        const std::string &device, const std::string &policy,
                        const reloc::costmodel::CostModel *calibration,
                        int threads, const std::string &implementation) {
  if (device != "host" && device != "cuda")
    throw py::value_error("device must be 'host' or 'cuda', got '" + device +
                          "'");
  if (threads < 1)
    throw py::value_error("threads must be >= 1");
  reloc::dispatch::Options options;
  options.policy = parsePolicy(policy);
  options.model = calibration;
  options.threads = threads;
  options.cuda = device == "cuda";
  options.implementation = implementation;
  auto selected = reloc::dispatch::selectImplementation(
      bound, parseDirection(direction), options);
  if (auto *error = std::get_if<reloc::TransferError>(&selected))
    raise(*error);
  const auto &choice = std::get<reloc::dispatch::Selection>(selected);
  py::dict out = rowDict(choice.row);
  out["policy"] = choice.policy;
  out["placement_reason"] = choice.reason;
  return out;
}

reloc::dispatch::DispatchTemplate
prepareTemplate(const reloc::TypedBoundPlan &bound,
                const std::string &direction, const std::string &device,
                const std::string &policy,
                const reloc::costmodel::CostModel *calibration, int threads,
                const std::string &implementation) {
  if (device != "host" && device != "cuda")
    throw py::value_error("device must be 'host' or 'cuda'");
  if (threads < 1)
    throw py::value_error("threads must be >= 1");
  reloc::dispatch::Options options;
  options.policy = parsePolicy(policy);
  options.model = calibration;
  options.threads = threads;
  options.cuda = device == "cuda";
  options.implementation = implementation;
  auto prepared = reloc::dispatch::prepareDispatchTemplate(
      bound, parseDirection(direction), options);
  if (auto *error = std::get_if<reloc::TransferError>(&prepared))
    raise(*error);
  return std::get<reloc::dispatch::DispatchTemplate>(std::move(prepared));
}

reloc::dispatch::DispatchRequest
fromTemplate(const reloc::dispatch::DispatchTemplate &prepared,
             const reloc::BufferView &source,
             const reloc::BufferView &destination) {
  auto request =
      reloc::dispatch::prepareDispatch(prepared, source, destination);
  if (auto *error = std::get_if<reloc::TransferError>(&request))
    raise(*error);
  return std::get<reloc::dispatch::DispatchRequest>(std::move(request));
}

py::dict prefoldSpec(const reloc::TypedBoundPlan &bound) {
  auto spec = reloc::dispatch::prefoldSpecFor(bound);
  if (auto *error = std::get_if<reloc::TransferError>(&spec))
    raise(*error);
  const auto &value = std::get<reloc::dispatch::PrefoldSpec>(spec);
  py::dict out;
  out["output_spec"] = value.spec == reloc::prefold::OutputSpec::S8GatherQuant
                           ? "s8_gather_quant"
                           : "s8_quant_pack";
  out["channels"] = static_cast<int64_t>(value.invScales.size());
  out["inv_scales"] =
      py::bytes(reinterpret_cast<const char *>(value.invScales.data()),
                value.invScales.size() * sizeof(float));
  return out;
}

} // namespace

void registerDispatchBindings(py::module_ &m) {
  py::class_<reloc::typed::Program>(m, "TypedProgram");
  m.def(
      "prepare_index_select_program",
      [](const reloc::IndexedBoundPlan &bound,
         const reloc::BufferView &source) {
        auto prepared = reloc::dispatch::prepareIndexSelect(bound, source);
        if (auto *error = std::get_if<reloc::TransferError>(&prepared))
          raise(*error);
        return std::get<reloc::typed::Program>(std::move(prepared));
      },
      py::arg("bound"), py::arg("source"));
  m.def(
      "prepare_typed_program",
      [](const reloc::TypedBoundPlan &p) { return checkedProgram(p); },
      py::arg("bound"));
  using Template = reloc::dispatch::DispatchTemplate;
  py::class_<Template>(m, "DispatchTemplate")
      .def_property_readonly(
          "program",
          [](const Template &t) -> const reloc::typed::Program & {
            return t.program;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "capability",
          [](const Template &t) { return capabilityDict(t.capability); })
      .def_property_readonly("selection", [](const Template &t) {
        py::dict result = rowDict(t.selection.row);
        result["policy"] = t.selection.policy;
        result["placement_reason"] = t.selection.reason;
        return result;
      });
  m.def("prepare_dispatch_template", &prepareTemplate, py::arg("bound"),
        py::arg("direction"), py::arg("device") = "host", py::kw_only(),
        py::arg("policy") = "auto",
        py::arg("calibration") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("threads") = 8, py::arg("implementation") = "",
        "Prepare immutable checked arithmetic, capability and selection, "
        "without views or execution state.");
  m.def("prepare_dispatch_from_template", &fromTemplate, py::arg("prepared"),
        py::arg("source"), py::arg("destination"),
        "Validate fresh views and create an independent single-use request; "
        "never reselect a row.");
  using Group = reloc::dispatch::GroupRequest;
  py::class_<Group>(m, "DispatchGroup")
      .def_property_readonly(
          "consumed", [](const Group &g) { return g.executing || g.consumed; })
      .def_property_readonly("report", [](const Group &g) {
        if (g.executing)
          throw py::value_error("report is unavailable during execution");
        return groupReport(g);
      });
  m.def("prepare_dispatch_group", &prepareGroup, py::arg("entries"),
        "Validate fresh (template or layout bound, source, destination, "
        "direction) entries together.");
  m.def("execute_dispatch_group", &executeGroupPy, py::arg("group"),
        py::kw_only(), py::arg("caller_stream") = py::none(),
        py::arg("gather_threads") = 8, py::arg("gather_pool") = nullptr,
        py::arg("pinning") = "auto", py::arg("min_pinned_bytes") = py::none(),
        py::arg("max_scratch_bytes") = size_t(64) << 20,
        py::arg("resources") = nullptr, py::arg("owners") = py::none(),
        "Submit a consumer-sized group on one owned queue and complete once; "
        "never reuse requests.");
  using Resources = reloc::dispatch::Resources;
  py::class_<Resources, std::shared_ptr<Resources>>(m, "DispatchResources")
      .def(py::init<size_t, size_t, unsigned, unsigned>(), py::kw_only(),
           py::arg("max_retained_bytes") = size_t(64) << 20,
           py::arg("max_live_bytes") = 0,
           py::arg("max_background_workers") = 64, py::arg("max_streams") = 8)
      .def("stats",
           [](Resources &r) {
             reloc::dispatch::ResourceStats s;
             {
               py::gil_scoped_release release;
               s = r.stats();
             }
             py::dict out;
             out["requests"] = s.requests;
             out["hits"] = s.hits;
             out["context_creations"] = s.contexts;
             out["device_allocations"] = s.deviceAllocations;
             out["host_allocations"] = s.hostAllocations;
             out["frees"] = s.frees;
             out["copy_calls"] = s.copyCalls;
             out["event_records"] = s.eventRecords;
             out["event_waits"] = s.eventWaits;
             out["caller_waits"] = s.callerWaits;
             out["peak_live_bytes"] = s.peakLiveBytes;
             out["retained_bytes"] = s.retainedBytes;
             out["device_bytes"] = s.deviceBytes;
             out["host_bytes"] = s.hostBytes;
             out["background_workers"] = s.backgroundWorkers;
             out["streams"] = s.streams;
             out["closed"] = s.closed;
             out["quarantined"] = s.quarantined;
             out["process_valid"] = s.processValid;
             return out;
           })
      .def("clear",
           [](Resources &r) {
             std::optional<reloc::TransferError> error;
             {
               py::gil_scoped_release release;
               error = r.clear();
             }
             if (error)
               raise(*error);
           })
      .def("close",
           [](Resources &r) {
             std::optional<reloc::TransferError> error;
             {
               py::gil_scoped_release release;
               error = r.close();
             }
             if (error)
               raise(*error);
           })
      .def("__reduce_ex__", [](const Resources &, int) {
        throw py::type_error("dispatch resources cannot be serialized");
      });
  py::class_<reloc::dispatch::DispatchRequest>(
      m, "DispatchRequest",
      "A prepared, single-use typed dispatch: the checked program, both "
      "views, the selected implementation and the scalar report.")
      .def_property_readonly("report",
                             [](const reloc::dispatch::DispatchRequest &r) {
                               if (r.executing)
                                 throw py::value_error(
                                     "report is unavailable during execution");
                               return reportDict(r.report);
                             })
      .def_property_readonly("implementation",
                             [](const reloc::dispatch::DispatchRequest &r) {
                               return r.selected.label();
                             })
      .def_property_readonly("direction",
                             [](const reloc::dispatch::DispatchRequest &r) {
                               return directionName(r.direction);
                             })
      .def_property_readonly("consumed",
                             [](const reloc::dispatch::DispatchRequest &r) {
                               return r.executing || r.consumed;
                             })
      .def_readonly("source", &reloc::dispatch::DispatchRequest::source)
      .def_readonly("destination",
                    &reloc::dispatch::DispatchRequest::destination);

  m.def("query_capability", &queryCapability<reloc::TypedBoundPlan>,
        py::arg("bound"), py::arg("direction"), py::arg("device") = "host",
        "Pure capability query for a typed bound plan: {'eligible': [rows "
        "with implementation/wire_boundary/wire_bytes/method], 'excluded': "
        "[{implementation, reason}]}. `device` is the device end ('host' or "
        "'cuda'). Consults no cost model; raises TransferError when no "
        "reference path exists (unsupported_stage).");
  m.def("query_capability", &queryCapability<reloc::typed::Program>,
        py::arg("bound"), py::arg("direction"), py::arg("device") = "host",
        "Pure capability query for a typed bound plan: {'eligible': [rows "
        "with implementation/wire_boundary/wire_bytes/method], 'excluded': "
        "[{implementation, reason}]}. `device` is the device end ('host' or "
        "'cuda'). Consults no cost model; raises TransferError when no "
        "reference path exists (unsupported_stage).");
  m.def("select_dispatch", &selectDispatch<reloc::TypedBoundPlan>,
        py::arg("bound"), py::arg("direction"), py::arg("device") = "host",
        py::kw_only(), py::arg("policy") = "auto",
        py::arg("calibration") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("threads") = 8, py::arg("implementation") = "",
        "Pure selection without buffers: the row the policy picks for this "
        "program, direction and device end, with 'policy' and "
        "'placement_reason'. A bridge fixes the row here and forces exactly "
        "it in prepare_dispatch once the destination exists.");
  m.def("select_dispatch", &selectDispatch<reloc::typed::Program>,
        py::arg("bound"), py::arg("direction"), py::arg("device") = "host",
        py::kw_only(), py::arg("policy") = "auto",
        py::arg("calibration") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("threads") = 8, py::arg("implementation") = "",
        "Pure selection without buffers: the row the policy picks for this "
        "program, direction and device end, with 'policy' and "
        "'placement_reason'. A bridge fixes the row here and forces exactly "
        "it in prepare_dispatch once the destination exists.");
  m.def("prepare_dispatch", &prepareDispatch<reloc::TypedBoundPlan>,
        py::arg("bound"), py::arg("source"), py::arg("destination"),
        py::arg("direction"), py::kw_only(), py::arg("policy") = "auto",
        py::arg("calibration") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("threads") = 8, py::arg("implementation") = "",
        "Validate the typed program and both dense views, enumerate the "
        "eligible rows and select one: policy 'original_cpu' forces the CPU "
        "reference pipeline, 'auto' consults `calibration` when given and "
        "otherwise takes the reference with a recorded reason; "
        "`implementation` forces an eligible row by label. Raises "
        "TransferError('<code>: <detail>'); allocates and launches nothing.");
  m.def("prepare_dispatch", &prepareDispatch<reloc::typed::Program>,
        py::arg("bound"), py::arg("source"), py::arg("destination"),
        py::arg("direction"), py::kw_only(), py::arg("policy") = "auto",
        py::arg("calibration") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("threads") = 8, py::arg("implementation") = "",
        "Validate the typed program and both dense views, enumerate the "
        "eligible rows and select one: policy 'original_cpu' forces the CPU "
        "reference pipeline, 'auto' consults `calibration` when given and "
        "otherwise takes the reference with a recorded reason; "
        "`implementation` forces an eligible row by label. Raises "
        "TransferError('<code>: <detail>'); allocates and launches nothing.");
  m.def("execute_dispatch", &executeDispatchPy, py::arg("request"),
        py::kw_only(), py::arg("caller_stream") = py::none(),
        py::arg("n_buffers") = 2, py::arg("n_streams") = 2,
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        py::arg("pinning") = "auto", py::arg("min_pinned_bytes") = py::none(),
        py::arg("direct_dense_upload") = true, py::arg("pipeline") = true,
        py::arg("chunk_size") = 0, py::arg("resources") = nullptr,
        py::arg("owners") = py::none(),
        "Run the selected row and block until this request's work completed; "
        "returns the report with payload_bytes_transferred filled in. "
        "caller_stream orders every private stream after the caller's "
        "producer stream. Raises TransferError; never tries another path.");
  m.def("typed_prefold_spec", &prefoldSpec, py::arg("bound"),
        "R3's prefold capability for T4: when the typed program is exactly "
        "one symmetric_rne quantize the existing S8 prefolder implements, "
        "returns {'output_spec', 'channels', 'inv_scales' (f32 bytes formed "
        "from the declared scales)}; raises TransferError("
        "'prefold_unavailable: <reason>') otherwise.");
}
