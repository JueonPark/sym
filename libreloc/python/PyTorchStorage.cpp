// Packed output ownership: consumers, rather than transfer completions, own
// views.
#include <pybind11/pybind11.h>

#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>
#ifdef RELOC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <optional>
#include <stdexcept>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace py = pybind11;

namespace {
int processId() {
#ifdef _WIN32
  return _getpid();
#else
  return getpid();
#endif
}

template <typename T>
bool soleOwner(const T *value, size_t references) {
  // Torch preserves a canonical Python wrapper while C++ owners exist. That
  // wrapper contributes one intrusive reference even when no caller owns it.
  // These methods run with the GIL; discount it only when its sole Python
  // reference is Torch's preservation reference, never a caller's handle.
  return references == 1 ||
         (references == 2 && value->pyobj_slot()->has_unique_reference());
}

class PackedStorage {
  at::Tensor backing_;
  std::optional<c10::weak_intrusive_ptr<c10::StorageImpl>> exported_;
  int pid_ = processId();
  bool recycle_;
  bool used_ = false;
  bool closed_ = false, poisoned_ = false;
  size_t barriers_ = 0;

  void checkProcess() const {
    if (processId() != pid_)
      throw std::runtime_error("packed storage belongs to another process");
  }

  void check() const {
    checkProcess();
    if (poisoned_)
      throw std::runtime_error("packed storage completion is uncertain");
  }

  void synchronize() {
    if (!backing_.is_cuda() || !used_)
      return;
#ifdef RELOC_ENABLE_CUDA
    int previous = 0;
    auto status = cudaGetDevice(&previous);
    if (status == cudaSuccess)
      status = cudaSetDevice(backing_.get_device());
    if (status == cudaSuccess)
      status = cudaDeviceSynchronize();
    const auto restored = cudaSetDevice(previous);
    if (status == cudaSuccess)
      status = restored;
    if (status != cudaSuccess) {
      poisoned_ = true;
      throw std::runtime_error(
          std::string("packed storage completion failed: ") +
          cudaGetErrorString(status));
    }
    ++barriers_;
#else
    throw std::runtime_error("packed CUDA storage requires a CUDA build");
#endif
  }

  void quarantine() noexcept {
    // Completion cannot be proved. Keep the allocation until process exit,
    // including a child process where CUDA must not be touched at all.
    if (backing_.defined())
      (void)new at::Tensor(std::move(backing_));
  }

public:
  explicit PackedStorage(const py::object &value, bool recycle)
      : recycle_(recycle) {
    if (!THPVariable_Check(value.ptr()))
      throw py::type_error("backing must be a tensor");
    backing_ = THPVariable_Unpack(value.ptr());
    if (backing_.scalar_type() != at::kByte || backing_.dim() != 1 ||
        !backing_.is_contiguous() || backing_.storage_offset() != 0 ||
        backing_.numel() <= 0 ||
        size_t(backing_.numel()) != backing_.storage().nbytes() ||
        (!backing_.device().is_cpu() && !backing_.is_cuda()))
      throw py::value_error(
          "backing must be a whole contiguous CPU/CUDA byte allocation");
    // Establish immutable capacity before publishing views. Keep the original
    // allocator DataPtr: ordinary Tensor operations and record_stream may read
    // it concurrently, outside this owner's lock and Python's GIL.
    backing_.storage().unsafeGetStorageImpl()->set_resizable(false);
  }

  ~PackedStorage() {
    try {
      close();
    } catch (...) {
      quarantine();
    }
  }

  bool expired() const {
    checkProcess();
    if (!used_)
      return true;
    if (backing_.defined())
      // Exported tensors have separate TensorImpls sharing this storage.
      // Also count a caller retaining the original backing TensorImpl.
      return soleOwner(backing_.unsafeGetTensorImpl(), backing_.use_count()) &&
             soleOwner(backing_.storage().unsafeGetStorageImpl(),
                       backing_.storage().use_count());
    return !exported_ || exported_->expired();
  }

  py::object view() {
    check();
    if (closed_ || !backing_.defined())
      throw std::runtime_error("packed storage is closed");
    if (!expired())
      throw std::runtime_error("packed storage still has live consumers");
    if (!recycle_) {
      // Packing alone uses the original allocator storage from the start.
      // Track it weakly for admission; no extra completion fence or retention.
      used_ = true;
      exported_ = backing_.storage().getWeakStorageImpl();
      auto result = std::move(backing_);
      return py::reinterpret_steal<py::object>(THPVariable_Wrap(result));
    }
    synchronize();
    const auto &storage = backing_.storage();
    exported_ = storage.getWeakStorageImpl();
    auto result = at::empty({0}, backing_.options());
    result.set_(storage, 0, {backing_.numel()}, {1});
    used_ = true;
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(result));
  }

  void close() {
    check();
    if (closed_)
      return;
    if (!backing_.defined()) {
      closed_ = true;
      return;
    }
    synchronize();
    // Retire only this owner's reference. Published storage is never mutated;
    // concurrent and future consumers retain ordinary allocator semantics.
    backing_ = at::Tensor();
    closed_ = true;
  }

  size_t barriers() const { return barriers_; }
  bool poisoned() const { return poisoned_; }
};
} // namespace

void bindPackedStorage(py::module_ &module) {
  py::class_<PackedStorage>(module, "PackedStorage")
      .def(py::init<const py::object &, bool>(), py::arg("backing"),
           py::arg("recycle") = true)
      .def("view", &PackedStorage::view)
      .def("expired", &PackedStorage::expired)
      .def("close", &PackedStorage::close)
      .def_property_readonly("poisoned", &PackedStorage::poisoned)
      .def_property_readonly("barriers", &PackedStorage::barriers);
}
