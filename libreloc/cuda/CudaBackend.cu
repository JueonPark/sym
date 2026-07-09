//===- CudaBackend.cu - CopyBackend over the CUDA Runtime API -------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaBackend.h"

#include <cuda_runtime.h>

namespace reloc {
namespace {
cudaStream_t asStream(void *p) { return static_cast<cudaStream_t>(p); }
cudaEvent_t asEvent(void *p) { return static_cast<cudaEvent_t>(p); }
} // namespace

CudaBackend::CudaBackend(int numStreams) {
  if (numStreams < 1)
    numStreams = 1;
  streams_.resize(numStreams);
  for (int i = 0; i < numStreams; ++i) {
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    streams_[i] = s;
  }
}

CudaBackend::~CudaBackend() {
  for (auto &kv : events_)
    cudaEventDestroy(asEvent(kv.second));
  for (void *s : streams_)
    cudaStreamDestroy(asStream(s));
}

void *CudaBackend::allocStaging(size_t bytes) {
  void *p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess)
    return nullptr;
  return p;
}

void CudaBackend::freeStaging(void *p) {
  if (p)
    cudaFreeHost(p);
}

void CudaBackend::copyAsync(int queue, void *dst, const void *src, size_t bytes,
                            CopyDir dir) {
  cudaMemcpyAsync(dst, src, bytes,
                  dir == CopyDir::HostToDevice ? cudaMemcpyHostToDevice
                                               : cudaMemcpyDeviceToHost,
                  asStream(streams_[queue]));
}

EventHandle CudaBackend::recordEvent(int queue) {
  cudaEvent_t e = nullptr;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  cudaEventRecord(e, asStream(streams_[queue]));
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
  cudaEventSynchronize(asEvent(it->second));
  cudaEventDestroy(asEvent(it->second));
  events_.erase(it);
}

bool CudaBackend::queryEvent(EventHandle ev) {
  if (ev == 0)
    return true;
  auto it = events_.find(ev);
  if (it == events_.end())
    return true;
  return cudaEventQuery(asEvent(it->second)) == cudaSuccess;
}

} // namespace reloc

#endif // RELOC_ENABLE_CUDA
