//===- PyPrefold.h - owned validated prefold bindings -----------*- C++ -*-===//
//
// T4 Task 3 (#137): a Torch-free Python bridge to the existing load-time
// prefolder (reloc/Prefold.h). The owner keeps the allocating backend, the
// gather pool and the folded artifact alive together and validates every
// precondition before a kernel with asserted contracts runs. It defines no
// new value semantics; the typed recipe/parameter contract that decides when
// a Torch weight may take this path belongs to C3/C4/R3.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PYTHON_PYPREFOLD_H
#define RELOC_PYTHON_PYPREFOLD_H

#include <pybind11/pybind11.h>

/// Register PrefoldHandle, PrefoldError and prefold_s8 on `m`. Must run after
/// BoundPlan is bound.
void registerPrefoldBindings(pybind11::module_ &m);

#endif // RELOC_PYTHON_PYPREFOLD_H
