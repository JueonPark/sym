// Test-only LD_PRELOAD shim: fail after a typed upload is enqueued, without
// corrupting the GPU context. Never linked into or installed with the runtime.
#include <atomic>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

namespace {
std::atomic<int> mode{0};
std::atomic<int> copies{0};
} // namespace
extern "C" void sym_dispatch_fault_mode(int value) {
  copies.store(0);
  mode.store(value);
}
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                                 cudaStream_t stream) {
  if (mode.load() == 1 || mode.load() == 2)
    return cudaErrorUnknown;
  static auto real = reinterpret_cast<decltype(&cudaEventRecord)>(
      dlsym(RTLD_NEXT, "cudaEventRecord"));
  return real(event, stream);
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  if (mode.load() == 2 || mode.load() == 4 || mode.load() == 6)
    return cudaErrorUnknown;
  static auto real = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
      dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
  return real(stream);
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (mode.load() == 5 || mode.load() == 6)
    return cudaErrorUnknown;
  static auto real = reinterpret_cast<decltype(&cudaEventSynchronize)>(
      dlsym(RTLD_NEXT, "cudaEventSynchronize"));
  return real(event);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void *destination,
                                                 const void *source,
                                                 size_t bytes,
                                                 cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  // Group qualification: fail the third copy after a prior member's payload,
  // scale upload and kernel were submitted. Modes 3/4 have known/unknown drain.
  if ((mode.load() == 3 || mode.load() == 4) && ++copies == 3)
    return cudaErrorUnknown;
  static auto real = reinterpret_cast<decltype(&cudaMemcpyAsync)>(
      dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
  return real(destination, source, bytes, kind, stream);
}
