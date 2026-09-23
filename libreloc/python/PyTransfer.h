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

#include <pybind11/pybind11.h>

/// Register BufferView, TransferRequest, TransferError and the
/// validate_transfer_source / make_transfer / execute_transfer /
/// cuda_pointer_device functions on `m`. Must run after BoundPlan is bound.
void registerTransferBindings(pybind11::module_ &m);

#endif // RELOC_PYTHON_PYTRANSFER_H
