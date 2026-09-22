//===- PyTransfer.cpp - validated forward transfer bindings ---------------===//

#include "PyTransfer.h"

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

reloc::TransferRequest makeTransfer(const reloc::BoundPlan &bound,
                                    const reloc::BufferView &source,
                                    const reloc::BufferView &destination,
                                    const std::string &direction) {
  auto result = reloc::validateTransfer(bound, source, destination,
                                        parseDirection(direction));
  if (auto *error = std::get_if<reloc::TransferError>(&result))
    raise(*error);
  return std::get<reloc::TransferRequest>(std::move(result));
}

void executeTransferPy(reloc::TransferRequest &request,
                       const py::object &callerStream, int nBuffers,
                       int nStreams, int gatherThreads,
                       std::shared_ptr<reloc::GatherPool> pool) {
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
    // The Python caller retains every owner (tensors, request) for the
    // duration of this blocking call; only the GIL is released.
    py::gil_scoped_release release;
    if (cuda) {
#ifdef RELOC_ENABLE_CUDA
      reloc::CudaBackend backend(nStreams, device);
      error = reloc::executeTransfer(request, backend, options);
#else
      (void)device;
      error = reloc::TransferError{
          "backend_failure", "pyreloc was built without RELOC_ENABLE_CUDA"};
#endif
    } else {
      reloc::HostBackend backend(nStreams);
      error = reloc::executeTransfer(request, backend, options);
    }
  }
  if (error)
    raise(*error);
}

int cudaPointerDevicePy(uintptr_t pointer) {
#ifdef RELOC_ENABLE_CUDA
  int device = -1;
  std::string error;
  if (!reloc::cudaPointerDevice(reinterpret_cast<const void *>(pointer),
                                device, error))
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

  py::class_<reloc::TransferRequest>(
      m, "TransferRequest",
      "A validated, single-use forward transfer owning copies of the bound "
      "plan and both views.")
      .def_property_readonly("direction",
                             [](const reloc::TransferRequest &r) {
                               return directionName(r.direction);
                             })
      .def_readonly("source_span_bytes", &reloc::TransferRequest::sourceSpanBytes)
      .def_readonly("destination_bytes", &reloc::TransferRequest::destinationBytes)
      .def_readonly("consumed", &reloc::TransferRequest::consumed)
      .def_readonly("source", &reloc::TransferRequest::source)
      .def_readonly("destination", &reloc::TransferRequest::destination);

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
  m.def("execute_transfer", &executeTransferPy, py::arg("request"),
        py::kw_only(), py::arg("caller_stream") = py::none(),
        py::arg("n_buffers") = 4, py::arg("n_streams") = 2,
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        "Run the forward transfer and block until this request's work has "
        "completed. caller_stream (a cudaStream_t handle; 0 is the legacy "
        "default stream, None means no producer to order after) is recorded "
        "before any private stream touches shared storage. Raises "
        "TransferError on a consumed request or a backend failure.");
  m.def("cuda_pointer_device", &cudaPointerDevicePy, py::arg("pointer"),
        "CUDA device ordinal owning `pointer` (cudaPointerGetAttributes); "
        "raises TransferError for host/unknown memory or CUDA-less builds.");
}
