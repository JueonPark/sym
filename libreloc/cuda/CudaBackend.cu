//===- CudaBackend.cu - CopyBackend over the CUDA Runtime API -------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaBackend.h"

#include <cuda_runtime.h>

namespace reloc {
namespace {

cudaStream_t asStream(void *p) { return static_cast<cudaStream_t>(p); }
cudaEvent_t asEvent(void *p) { return static_cast<cudaEvent_t>(p); }

// Select the backend's device for the enclosing scope and restore the
// caller's device afterwards. `status` keeps the first failing selection
// call so the constructor can record an invalid ordinal instead of silently
// creating resources on the caller's device. Destructors must not fail
// loudly, so the restore status is deliberately ignored.
struct DeviceScope {
  int previous = -1;
  bool switched = false;
  cudaError_t status = cudaSuccess;
  explicit DeviceScope(int device) {
    if (device < 0)
      return;
    status = cudaGetDevice(&previous);
    if (status != cudaSuccess || previous == device)
      return;
    status = cudaSetDevice(device);
    if (status == cudaSuccess)
      switched = true;
  }
  ~DeviceScope() {
    if (switched)
      (void)cudaSetDevice(previous);
  }
};

} // namespace

bool CudaBackend::check(int status, const char *what) {
  if (status == static_cast<int>(cudaSuccess))
    return true;
  if (error_.empty())
    error_ = std::string(what) + ": " +
             cudaGetErrorString(static_cast<cudaError_t>(status));
  return false;
}

CudaBackend::CudaBackend(int numStreams, int device) {
  if (numStreams < 1)
    numStreams = 1;
  if (device < 0) {
    int current = 0;
    if (check(cudaGetDevice(&current), "cudaGetDevice"))
      device = current;
  }
  device_ = device;
  DeviceScope scope(device_);
  if (!check(static_cast<int>(scope.status), "cudaSetDevice")) {
    streams_.push_back(nullptr); // keep numQueues() >= 1; failed() is set
    return;
  }
  streams_.reserve(numStreams);
  for (int i = 0; i < numStreams; ++i) {
    cudaStream_t s = nullptr;
    if (!check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags"))
      break;
    streams_.push_back(s);
  }
  if (streams_.empty())
    streams_.push_back(nullptr); // keep numQueues() >= 1; failed() is set
}

CudaBackend::~CudaBackend() {
  DeviceScope scope(device_);
  for (auto &kv : events_)
    (void)cudaEventDestroy(asEvent(kv.second));
  for (void *s : streams_)
    if (s != nullptr)
      (void)cudaStreamDestroy(asStream(s));
}

void *CudaBackend::allocStaging(size_t bytes) {
  DeviceScope scope(device_);
  void *p = nullptr;
  if (!check(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "cudaHostAlloc"))
    return nullptr;
  return p;
}

void CudaBackend::freeStaging(void *p) {
  if (p == nullptr)
    return;
  DeviceScope scope(device_);
  check(cudaFreeHost(p), "cudaFreeHost");
}

void CudaBackend::copyAsync(int queue, void *dst, const void *src, size_t bytes,
                            CopyDir dir) {
  if (failed())
    return; // never enqueue after a failure; the caller sees error()
  DeviceScope scope(device_);
  check(cudaMemcpyAsync(dst, src, bytes,
                        dir == CopyDir::HostToDevice ? cudaMemcpyHostToDevice
                                                     : cudaMemcpyDeviceToHost,
                        asStream(streams_[queue])),
        "cudaMemcpyAsync");
}

EventHandle CudaBackend::recordEvent(int queue) {
  if (failed())
    return 0;
  DeviceScope scope(device_);
  cudaEvent_t e = nullptr;
  if (!check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming),
             "cudaEventCreateWithFlags"))
    return 0;
  if (!check(cudaEventRecord(e, asStream(streams_[queue])),
             "cudaEventRecord")) {
    (void)cudaEventDestroy(e);
    return 0;
  }
  EventHandle h = nextEvent_++;
  events_[h] = e;
  return h;
}

void CudaBackend::waitEvent(EventHandle ev) {
  if (ev == 0)
    return;
  auto it = events_.find(ev);
  if (it == events_.end())
    return;
  DeviceScope scope(device_);
  check(cudaEventSynchronize(asEvent(it->second)), "cudaEventSynchronize");
  check(cudaEventDestroy(asEvent(it->second)), "cudaEventDestroy");
  events_.erase(it);
}

bool CudaBackend::queryEvent(EventHandle ev) {
  if (ev == 0)
    return true;
  auto it = events_.find(ev);
  if (it == events_.end())
    return true;
  DeviceScope scope(device_);
  cudaError_t status = cudaEventQuery(asEvent(it->second));
  if (status == cudaSuccess)
    return true;
  if (status != cudaErrorNotReady)
    check(status, "cudaEventQuery"); // a real failure, distinct from not-ready
  return false;
}

bool CudaBackend::waitStream(const void *externalStream) {
  if (failed())
    return false;
  DeviceScope scope(device_);
  // Record the producer's progress on the caller's stream, then make every
  // private queue wait for it before any copy that touches shared storage.
  cudaEvent_t e = nullptr;
  if (!check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming),
             "cudaEventCreateWithFlags(producer)"))
    return false;
  auto producer = static_cast<cudaStream_t>(const_cast<void *>(externalStream));
  bool ok = check(cudaEventRecord(e, producer), "cudaEventRecord(producer)");
  for (void *s : streams_)
    if (ok)
      ok = check(cudaStreamWaitEvent(asStream(s), e, 0), "cudaStreamWaitEvent");
  // The wait is enqueued; the event object may be destroyed immediately.
  (void)cudaEventDestroy(e);
  return ok;
}

void *CudaBackend::allocDevice(size_t bytes) {
  DeviceScope scope(device_);
  void *p = nullptr;
  if (!check(cudaMalloc(&p, bytes), "cudaMalloc"))
    return nullptr;
  return p;
}

void CudaBackend::freeDevice(void *p) {
  if (p == nullptr)
    return;
  DeviceScope scope(device_);
  check(cudaFree(p), "cudaFree");
}

bool CudaBackend::recordLaunchStatus(const char *what) {
  DeviceScope scope(device_);
  return check(cudaGetLastError(), what);
}

bool cudaPointerDevice(const void *pointer, int &device, std::string &error) {
  cudaPointerAttributes attributes{};
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError(); // clear the sticky error for the caller
    error =
        std::string("cudaPointerGetAttributes: ") + cudaGetErrorString(status);
    return false;
  }
  if (attributes.type != cudaMemoryTypeDevice &&
      attributes.type != cudaMemoryTypeManaged) {
    error = "pointer is not CUDA device memory";
    return false;
  }
  device = attributes.device;
  return true;
}

} // namespace reloc

#endif // RELOC_ENABLE_CUDA
