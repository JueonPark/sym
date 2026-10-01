//===- Trace.h - optional native pipeline profiling -----------------------===//
#ifndef RELOC_TRACE_H
#define RELOC_TRACE_H

#include <cstdint>

#ifdef RELOC_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace reloc::detail {
// Stack ranges stay on the executing thread. In particular gather.work is
// emitted inside each worker's gatherChunk call, not around the driver barrier.
// Disabled builds have no timestamps, environment checks or profiler calls.
class TraceRange {
public:
  TraceRange(const char *name, uint64_t chunk) {
#ifdef RELOC_ENABLE_NVTX
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    event.payload.ullValue = chunk;
    nvtxRangePushEx(&event);
#else
    (void)name;
    (void)chunk;
#endif
  }
  ~TraceRange() {
#ifdef RELOC_ENABLE_NVTX
    nvtxRangePop();
#endif
  }
  TraceRange(const TraceRange &) = delete;
  TraceRange &operator=(const TraceRange &) = delete;
};
} // namespace reloc::detail

#endif
