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
#include "reloc/GatherPool.h"
#include "reloc/Transfer.h"
#include "reloc/TypedExecute.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaKernels.h"
#include <cuda_runtime.h>
#endif

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

//===----------------------------------------------------------------------===//
// Kernels without the runtime (bench/beat_torch ablation, Reviewer 1 #4):
// the same host producers and device kernels the fast lane runs, callable
// on plain tensors so that eager Torch can move the bytes itself. No plan
// selection, no chunk pipeline, no group: what remains when only the kernel
// engineering is kept.
//===----------------------------------------------------------------------===//

reloc::GatherPool &sharedPool(int threads) {
  if (threads < 1)
    throw py::value_error("threads must be >= 1");
  // The first call fixes the worker count for the process.
  static reloc::GatherPool pool(static_cast<unsigned>(threads));
  return pool;
}

[[noreturn]] void raiseExecution(const reloc::typed::ExecutionError &error) {
  raise(reloc::TransferError{error.code, error.message});
}

const at::Tensor &hostOperand(const py::object &object, const char *what,
                              at::ScalarType type, int64_t dim) {
  const at::Tensor &tensor = unpack(object, what);
  if (!tensor.device().is_cpu() || tensor.scalar_type() != type ||
      tensor.dim() != dim || !tensor.is_contiguous())
    throw py::value_error(std::string(what) +
                          " must be a contiguous CPU tensor of the expected "
                          "dtype and rank");
  return tensor;
}

/// out[b, t, :] = sum of tables[t]'s bag b (EmbeddingBag mode="sum"), cast
/// to out's dtype (float16 ieee_rne or float32); runs the fused AVX2
/// producer on the shared pool. Returns out.
py::object pooledBags(const std::vector<py::object> &tables,
                      const std::vector<py::object> &indices,
                      const std::vector<py::object> &offsets,
                      const py::object &out, int threads) {
  const at::Tensor &dst = unpack(out, "out");
  if (!dst.device().is_cpu() || !dst.is_contiguous() || dst.dim() != 3 ||
      (dst.scalar_type() != at::kHalf && dst.scalar_type() != at::kFloat))
    throw py::value_error(
        "out must be a contiguous CPU float16/float32 [bags, tables, dim]");
  if (tables.empty() || indices.size() != tables.size() ||
      offsets.size() != tables.size())
    throw py::value_error("one index and one offset tensor per table");
  const int64_t bags = dst.size(0), dim = dst.size(2);
  if (dst.size(1) != static_cast<int64_t>(tables.size()))
    throw py::value_error("out's second extent must be the table count");
  std::vector<reloc::BufferView> views;
  std::vector<reloc::dispatch::PooledOperand> idx, off;
  for (size_t t = 0; t < tables.size(); ++t) {
    const at::Tensor &table = hostOperand(tables[t], "table", at::kFloat, 2);
    if (table.size(1) != dim)
      throw py::value_error("every table must have out's embedding dim");
    views.push_back(viewOf(table, "table"));
    const at::Tensor &index = hostOperand(indices[t], "indices", at::kLong, 1);
    const at::Tensor &offset = hostOperand(offsets[t], "offsets", at::kLong, 1);
    if (offset.size(0) != bags)
      throw py::value_error("offsets must have one entry per bag");
    idx.push_back({index.data_ptr<int64_t>(), index.size(0)});
    off.push_back({offset.data_ptr<int64_t>(), offset.size(0)});
  }
  const bool narrow = dst.scalar_type() == at::kHalf;
  void *output = dst.data_ptr();
  std::variant<reloc::typed::Program, reloc::TransferError> prepared;
  std::optional<reloc::typed::ExecutionError> failure;
  {
    py::gil_scoped_release release;
    prepared =
        reloc::dispatch::preparePooledBags(views, idx, off, bags, dim, narrow);
    if (auto *program = std::get_if<reloc::typed::Program>(&prepared)) {
      reloc::GatherPool &pool = sharedPool(threads);
      failure = reloc::typed::executeHost(
          *program, 0, static_cast<uint32_t>(program->stages.size()), nullptr,
          output, &pool, static_cast<unsigned>(threads));
    }
  }
  if (auto *error = std::get_if<reloc::TransferError>(&prepared))
    raise(*error);
  if (failure)
    raiseExecution(*failure);
  return out;
}

/// out[i, :] = cast(source[index[i], :]) through the indexed program of
/// `plan` (an index_select recipe with its cast), on the shared pool.
py::object gatherRows(const reloc::IndexedRelocationPlan &plan,
                      const reloc::SymbolMap &symbols, const py::object &source,
                      const py::object &index, const py::object &out,
                      int threads) {
  const at::Tensor &src = unpack(source, "source");
  const at::Tensor &idx = unpack(index, "index");
  const at::Tensor &dst = unpack(out, "out");
  if (!idx.is_contiguous() || idx.dim() != 1 || !idx.device().is_cpu() ||
      (idx.scalar_type() != at::kLong && idx.scalar_type() != at::kInt))
    throw py::value_error("index must be a contiguous CPU int32/int64 vector");
  if (!src.device().is_cpu() || !src.is_contiguous() ||
      !dst.device().is_cpu() || !dst.is_contiguous() || dst.dim() != 2 ||
      dst.size(0) != idx.size(0))
    throw py::value_error(
        "source and out must be contiguous CPU tensors; out has one row per "
        "index");
  reloc::ElementType type{reloc::ElementTypeKind::Integer,
                          idx.scalar_type() == at::kLong ? 64u : 32u};
  const reloc::BufferView view = viewOf(src, "source");
  const auto *input =
      reinterpret_cast<const uint8_t *>(view.base) + view.offsetBytes;
  void *output = dst.data_ptr();
  const size_t outBytes = dst.nbytes();
  std::string bindFailure;
  std::variant<reloc::typed::Program, reloc::TransferError> prepared;
  std::optional<reloc::typed::ExecutionError> failure;
  {
    py::gil_scoped_release release;
    auto bound = reloc::bindIndexed(plan, symbols, type, {idx.size(0)},
                                    idx.data_ptr(), idx.nbytes());
    if (auto *error = std::get_if<reloc::BindError>(&bound))
      bindFailure = error->message;
    else {
      prepared = reloc::dispatch::prepareIndexSelect(
          std::get<reloc::IndexedBoundPlan>(bound), view);
      if (auto *program = std::get_if<reloc::typed::Program>(&prepared)) {
        const uint32_t stages = static_cast<uint32_t>(program->stages.size());
        const size_t resultBytes =
            program->rowIndices.size() *
            static_cast<size_t>(program->indexedRowElements) *
            reloc::typed::widthAt(*program, stages);
        if (resultBytes != outBytes)
          failure = reloc::typed::ExecutionError{
              "destination_mismatch",
              "out does not have the program's result bytes"};
        else {
          reloc::GatherPool &pool = sharedPool(threads);
          failure =
              reloc::typed::executeHost(*program, 0, stages, input, output,
                                        &pool, static_cast<unsigned>(threads));
        }
      }
    }
  }
  if (!bindFailure.empty()) {
    py::object module = py::module_::import("pyreloc._pyreloc");
    PyErr_SetString(
        reinterpret_cast<PyObject *>(module.attr("BindError").ptr()),
        bindFailure.c_str());
    throw py::error_already_set();
  }
  if (auto *error = std::get_if<reloc::TransferError>(&prepared))
    raise(*error);
  if (failure)
    raiseExecution(*failure);
  return out;
}

#ifdef RELOC_ENABLE_CUDA
/// out = float(q) * scale[channel] on the device (the per-channel
/// dequantize kernel the fast lane launches), on `stream` (a cudaStream_t
/// handle, e.g. torch.cuda.current_stream().cuda_stream). Returns out.
py::object dequantS8F32(const py::object &q, const py::object &scale,
                        const py::object &out, uintptr_t stream) {
  const at::Tensor &src = unpack(q, "q");
  const at::Tensor &s = unpack(scale, "scale");
  const at::Tensor &dst = unpack(out, "out");
  if (!src.is_cuda() || src.scalar_type() != at::kChar || src.dim() != 2 ||
      !src.is_contiguous())
    throw py::value_error("q must be a contiguous CUDA int8 [channels, size]");
  if (!s.is_cuda() || s.scalar_type() != at::kFloat || s.dim() != 1 ||
      !s.is_contiguous() || s.size(0) != src.size(0))
    throw py::value_error("scale must be a contiguous CUDA float32 [channels]");
  if (!dst.is_cuda() || dst.scalar_type() != at::kFloat ||
      !dst.is_contiguous() || dst.sizes() != src.sizes() ||
      dst.get_device() != src.get_device() ||
      s.get_device() != src.get_device())
    throw py::value_error("out must be a contiguous CUDA float32 of q's shape "
                          "on q's device");
  {
    py::gil_scoped_release release;
    cudaSetDevice(src.get_device());
    reloc::cuda::dequantS8F32(static_cast<const int8_t *>(src.data_ptr()),
                              static_cast<float *>(dst.data_ptr()), src.size(0),
                              src.size(1),
                              static_cast<const float *>(s.data_ptr()),
                              reinterpret_cast<void *>(stream));
    if (cudaError_t status = cudaGetLastError(); status != cudaSuccess)
      throw std::runtime_error(std::string("dequantS8F32 launch failed: ") +
                               cudaGetErrorString(status));
  }
  return out;
}
#endif

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
  m.def("pooled_bags", &pooledBags, py::arg("tables"), py::arg("indices"),
        py::arg("offsets"), py::arg("out"), py::arg("threads") = 8,
        "Fused EmbeddingBag-sum producer into a host tensor (no transfer): "
        "the fast lane's CPU kernel alone.");
  m.def("gather_rows", &gatherRows, py::arg("plan"), py::arg("symbols"),
        py::arg("source"), py::arg("index"), py::arg("out"),
        py::arg("threads") = 8,
        "Fused row gather + cast of an index_select recipe into a host "
        "tensor (no transfer): the fast lane's CPU kernel alone.");
#ifdef RELOC_ENABLE_CUDA
  m.def("dequant_s8_f32", &dequantS8F32, py::arg("q"), py::arg("scale"),
        py::arg("out"), py::arg("stream") = 0,
        "Per-channel int8 -> float32 dequantize on the device (the fast "
        "lane's kernel alone), on the given cudaStream_t handle.");
#endif
}
