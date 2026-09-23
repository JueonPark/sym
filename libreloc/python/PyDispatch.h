//===- PyDispatch.h - typed dispatch bindings (R3) --------------*- C++ -*-===//

#ifndef RELOC_PYDISPATCH_H
#define RELOC_PYDISPATCH_H

#include <pybind11/pybind11.h>

/// Register query_capability / prepare_dispatch / execute_dispatch /
/// DispatchRequest / typed_prefold_spec on the pyreloc module. Must run after
/// registerTransferBindings (reuses BufferView and TransferError).
void registerDispatchBindings(pybind11::module_ &m);

#endif // RELOC_PYDISPATCH_H
