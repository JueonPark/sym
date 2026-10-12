// Build-only event timing of preallocated kernels. Device tensors and stream
// are caller-owned; the Python qualification checks metadata and output bytes.
#include "reloc/CudaKernels.h"
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct Event {
  cudaEvent_t event = nullptr;
  Event() { check(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
};
} // namespace

PYBIND11_MODULE(_cuda_kernel_test, m) {
  m.def("measure", [](int64_t rows, int64_t cols, uintptr_t source,
                      uintptr_t destination, uintptr_t scales, uintptr_t stream,
                      bool generic, unsigned samples) {
    if (rows <= 0 || cols <= 0 || rows > (1 << 20) || cols > (1 << 20) ||
        !source || !destination || !scales || !samples || samples > 10000)
      throw py::value_error("invalid kernel measurement arguments");
    int device;
    check(cudaGetDevice(&device));
    for (auto pointer : {source, destination, scales}) {
      cudaPointerAttributes attributes{};
      check(cudaPointerGetAttributes(&attributes,
                                     reinterpret_cast<void *>(pointer)));
      if (attributes.type != cudaMemoryTypeDevice ||
          attributes.device != device)
        throw py::value_error("buffers must be on the current CUDA device");
    }
    reloc::BoundPlan bound;
    bound.elementSize = 4;
    bound.extents = {cols, rows};
    bound.srcStrides = {1, cols};
    bound.dstStrides = {rows, 1};
    bound.totalBytes = rows * cols * 4;
    const auto queue = reinterpret_cast<cudaStream_t>(stream);
    auto launch = generic ? reloc::cuda::dequantRelocateNaiveS8F32
                          : reloc::cuda::dequantRelocateS8F32;
    Event begin, end;
    std::vector<float> values;
    values.reserve(samples);
    py::gil_scoped_release release;
    for (unsigned i = 0; i < samples; ++i) {
      check(cudaEventRecord(begin.event, queue));
      launch(bound, reinterpret_cast<const int8_t *>(source),
             reinterpret_cast<float *>(destination),
             reinterpret_cast<const float *>(scales), queue);
      check(cudaGetLastError());
      check(cudaEventRecord(end.event, queue));
      check(cudaEventSynchronize(end.event));
      float milliseconds;
      check(cudaEventElapsedTime(&milliseconds, begin.event, end.event));
      values.push_back(milliseconds);
    }
    return values;
  });
}
