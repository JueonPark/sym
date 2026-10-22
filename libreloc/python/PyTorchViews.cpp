//===- PyTorchViews.cpp - libtorch-backed request builders ----------------===//
//
// Optional (RELOC_ENABLE_TORCH_EXT): reads at::Tensor metadata directly so a
// prepared transfer can be turned into a native request with one call
// instead of a dozen Python attribute reads per tensor. Types it returns
// (BufferView, DispatchRequest, DispatchGroup) are _pyreloc's; import
// pyreloc first. The validation is the same native checkView path the
// Python builders use; this file only replaces the descriptor plumbing.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>

#include "reloc/Bind.h"
#include "reloc/Dispatch.h"
#include "reloc/Transfer.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {

[[noreturn]] void raise(const reloc::TransferError &error) {
  py::object module = py::module_::import("pyreloc._pyreloc");
  py::object type = module.attr("TransferError");
  PyErr_SetString(reinterpret_cast<PyObject *>(type.ptr()),
                  (error.code + ": " + error.message).c_str());
  throw py::error_already_set();
}

const at::Tensor &unpack(const py::handle &object, const char *what) {
  if (!THPVariable_Check(object.ptr()))
    throw py::type_error(std::string(what) + " must be a torch.Tensor");
  return THPVariable_Unpack(object.ptr());
}

/// The same declaration reloc_torch.transport._storage_view derives in Python:
/// the allocation (storage base, nbytes) and the logical view over it.
reloc::BufferView viewOf(const at::Tensor &tensor, const char *what) {
  if (tensor.requires_grad())
    throw py::value_error(std::string(what) + " must not require grad");
  if (tensor.layout() != at::kStrided)
    throw py::value_error(std::string(what) + " must be a strided tensor");
  const at::Storage &storage = tensor.storage();
  reloc::BufferView view;
  view.base = reinterpret_cast<uintptr_t>(storage.data());
  view.capacityBytes = storage.nbytes();
  view.elementSize = static_cast<uint32_t>(tensor.element_size());
  view.offsetBytes =
      static_cast<size_t>(tensor.storage_offset()) * view.elementSize;
  view.extents.assign(tensor.sizes().begin(), tensor.sizes().end());
  view.strides.assign(tensor.strides().begin(), tensor.strides().end());
  if (tensor.is_cuda()) {
    view.kind = reloc::MemoryKind::Cuda;
    view.device = tensor.get_device();
  } else if (tensor.device().is_cpu()) {
    view.kind = reloc::MemoryKind::Host;
    view.device = -1;
  } else {
    throw py::value_error(std::string(what) +
                          " must live on the CPU or a CUDA device");
  }
  return view;
}

void checkDtype(const at::Tensor &tensor, const std::string &dtype,
                const char *what) {
  if (dtype.empty())
    return;
  // Python's compat.dtype_name strips the "torch." prefix.
  const std::string actual = std::string(c10::toString(tensor.scalar_type()));
  static const std::pair<const char *, const char *> names[] = {
      {"Float", "float32"}, {"Half", "float16"}, {"Char", "int8"},
      {"Int", "int32"},     {"Long", "int64"},   {"BFloat16", "bfloat16"}};
  std::string canonical = actual;
  for (const auto &[aten, torch] : names)
    if (actual == aten)
      canonical = torch;
  if (canonical != dtype)
    throw py::value_error(std::string(what) + " dtype " + canonical +
                          " does not match the prepared program (" + dtype +
                          ")");
}

reloc::TransferDirection parseDirection(const std::string &direction) {
  if (direction == "h2d")
    return reloc::TransferDirection::HostToDevice;
  if (direction == "d2h")
    return reloc::TransferDirection::DeviceToHost;
  throw py::value_error("direction must be 'h2d' or 'd2h'");
}

using Plan = std::variant<reloc::dispatch::DispatchTemplate, reloc::BoundPlan>;

/// entries: (plan, source tensor, destination tensor, direction, source dtype
/// name or "") -> DispatchGroup, validated by prepareDispatchGroup.
reloc::dispatch::GroupRequest groupFromTensors(
    const std::vector<std::tuple<Plan, py::object, py::object, std::string,
                                 std::string>> &entries) {
  std::vector<reloc::dispatch::GroupEntry> native;
  native.reserve(entries.size());
  for (const auto &[plan, source, destination, direction, dtype] : entries) {
    const at::Tensor &src = unpack(source, "source");
    const at::Tensor &dst = unpack(destination, "destination");
    checkDtype(src, dtype, "source");
    native.push_back({plan, viewOf(src, "source"), viewOf(dst, "destination"),
                      parseDirection(direction)});
  }
  auto group = reloc::dispatch::prepareDispatchGroup(native);
  if (auto *error = std::get_if<reloc::TransferError>(&group))
    raise(*error);
  return std::get<reloc::dispatch::GroupRequest>(std::move(group));
}

reloc::dispatch::DispatchRequest
requestFromTensors(const reloc::dispatch::DispatchTemplate &prepared,
                   const py::object &source, const py::object &destination,
                   const std::string &dtype) {
  const at::Tensor &src = unpack(source, "source");
  const at::Tensor &dst = unpack(destination, "destination");
  checkDtype(src, dtype, "source");
  auto request = reloc::dispatch::prepareDispatch(
      prepared, viewOf(src, "source"), viewOf(dst, "destination"));
  if (auto *error = std::get_if<reloc::TransferError>(&request))
    raise(*error);
  return std::get<reloc::dispatch::DispatchRequest>(std::move(request));
}

reloc::BufferView viewFromTensor(const py::object &tensor) {
  return viewOf(unpack(tensor, "tensor"), "tensor");
}

/// One call for the indexed gather: bind the live index tensor, prepare the
/// program over the source and select the CUDA row for the destination.
reloc::dispatch::DispatchRequest
indexSelectRequest(const reloc::IndexedRelocationPlan &plan,
                   const reloc::SymbolMap &symbols, const py::object &source,
                   const py::object &index, const py::object &destination,
                   int threads) {
  if (threads < 1)
    throw py::value_error("threads must be >= 1");
  const at::Tensor &src = unpack(source, "source");
  const at::Tensor &idx = unpack(index, "index");
  const at::Tensor &dst = unpack(destination, "destination");
  if (!idx.is_contiguous() || idx.dim() != 1 || !idx.device().is_cpu() ||
      (idx.scalar_type() != at::kLong && idx.scalar_type() != at::kInt))
    throw py::value_error("index must be a contiguous CPU int32/int64 vector");
  reloc::ElementType type{reloc::ElementTypeKind::Integer,
                          idx.scalar_type() == at::kLong ? 64u : 32u};
  std::variant<reloc::dispatch::DispatchRequest, reloc::TransferError> request;
  {
    py::gil_scoped_release release;
    auto bound = reloc::bindIndexed(plan, symbols, type, {idx.size(0)},
                                    idx.data_ptr(), idx.nbytes());
    if (auto *error = std::get_if<reloc::BindError>(&bound)) {
      py::gil_scoped_acquire gil;
      py::object module = py::module_::import("pyreloc._pyreloc");
      PyErr_SetString(
          reinterpret_cast<PyObject *>(module.attr("BindError").ptr()),
          error->message.c_str());
      throw py::error_already_set();
    }
    const reloc::BufferView view = viewOf(src, "source");
    auto program = reloc::dispatch::prepareIndexSelect(
        std::get<reloc::IndexedBoundPlan>(bound), view);
    if (auto *error = std::get_if<reloc::TransferError>(&program))
      request = *error;
    else {
      reloc::dispatch::Options options;
      options.policy = reloc::dispatch::Policy::Auto;
      options.threads = threads;
      options.cuda = true;
      request = reloc::dispatch::prepareDispatch(
          std::get<reloc::typed::Program>(std::move(program)), view,
          viewOf(dst, "destination"), reloc::TransferDirection::HostToDevice,
          options);
    }
  }
  if (auto *error = std::get_if<reloc::TransferError>(&request))
    raise(*error);
  return std::get<reloc::dispatch::DispatchRequest>(std::move(request));
}

} // namespace

PYBIND11_MODULE(_pyreloc_torch, m) {
  py::module_::import("pyreloc._pyreloc"); // registers the returned types
  m.doc() = "libtorch-backed request builders for reloc_torch.fast";
  m.def("group_from_tensors", &groupFromTensors, py::arg("entries"),
        "Validate (plan, source, destination, direction, source_dtype) entries "
        "straight from tensors; same checks as prepare_dispatch_group.");
  m.def("request_from_tensors", &requestFromTensors, py::arg("template"),
        py::arg("source"), py::arg("destination"), py::arg("source_dtype") = "",
        "prepare_dispatch_from_template straight from tensors.");
  m.def("view_from_tensor", &viewFromTensor, py::arg("tensor"),
        "The BufferView of a tensor (storage base/capacity, logical view).");
  m.def(
      "index_select_group",
      [](const reloc::IndexedRelocationPlan &plan,
         const reloc::SymbolMap &symbols, const py::object &source,
         const py::object &index, const py::object &destination, int threads) {
        auto request = indexSelectRequest(plan, symbols, source, index,
                                          destination, threads);
        // A one-item group over the prepared program, for asynchronous
        // submission: the template carries the already selected row.
        reloc::dispatch::DispatchTemplate prepared;
        prepared.program = std::move(request.program);
        prepared.capability = reloc::dispatch::queryCapability(
            prepared.program, reloc::TransferDirection::HostToDevice, true);
        prepared.selection = {request.selected, request.report.policy,
                              request.report.placementReason};
        prepared.direction = reloc::TransferDirection::HostToDevice;
        prepared.cuda = true;
        std::vector<reloc::dispatch::GroupEntry> entries;
        entries.push_back({std::move(prepared), request.source,
                           request.destination,
                           reloc::TransferDirection::HostToDevice});
        auto group = reloc::dispatch::prepareDispatchGroup(entries);
        if (auto *error = std::get_if<reloc::TransferError>(&group))
          raise(*error);
        return std::get<reloc::dispatch::GroupRequest>(std::move(group));
      },
      py::arg("plan"), py::arg("symbols"), py::arg("source"), py::arg("index"),
      py::arg("destination"), py::arg("threads") = 8,
      "index_select_request wrapped as a one-item DispatchGroup.");
  m.def("index_select_request", &indexSelectRequest, py::arg("plan"),
        py::arg("symbols"), py::arg("source"), py::arg("index"),
        py::arg("destination"), py::arg("threads") = 8,
        "bind_indexed + prepare_index_select_program + prepare_dispatch in "
        "one call straight from tensors (CUDA destination).");
}
