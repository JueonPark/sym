//===- PyTransfer.h - validated forward transfer bindings -------*- C++ -*-===//
//
// R2 (issue #146): expose reloc::BufferView / TransferRequest validation and
// the blocking forward executor to Python without any Torch dependency.
// Tensor storage is described from Python as integers (allocation base,
// capacity, offset) plus logical extents/strides -- design decision 2 applied
// to whole allocations instead of bare (ptr, nbytes) pairs.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PYTHON_PYTRANSFER_H
#define RELOC_PYTHON_PYTRANSFER_H

#include "reloc/TransferResources.h"
#include <pybind11/pybind11.h>

namespace reloc_python {
// Shared by the binding and the build-only failure-injection module. Last
// Python reference destruction must not hold the GIL while draining resources.
inline std::shared_ptr<reloc::TransferResourceCache>
makeResourceCache(reloc::TransferResourceLimits limits = {},
                  reloc::TransferBackendFactory factory = {}) {
  return {new reloc::TransferResourceCache(limits, std::move(factory)),
          [](reloc::TransferResourceCache *cache) {
            if (PyGILState_Check()) {
              pybind11::gil_scoped_release release;
              delete cache;
            } else {
              delete cache;
            }
          }};
}
} // namespace reloc_python

/// Register BufferView, TransferRequest, TransferResourceCache, TransferError
/// and the validate_transfer_source / make_transfer / execute_transfer /
/// cuda_pointer_device functions on `m`. Must run after BoundPlan is bound.
void registerTransferBindings(pybind11::module_ &m);

#endif // RELOC_PYTHON_PYTRANSFER_H
