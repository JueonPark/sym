#ifndef RELOC_PYPINNING_H
#define RELOC_PYPINNING_H
#include "reloc/Transfer.h"
#include <pybind11/pybind11.h>
inline reloc::PinningPolicy parsePinning(const std::string &value) {
  if (value == "auto")
    return reloc::PinningPolicy::Auto;
  if (value == "pinned")
    return reloc::PinningPolicy::Pinned;
  if (value == "pageable")
    return reloc::PinningPolicy::Pageable;
  throw pybind11::value_error("pinning must be auto, pinned or pageable");
}
#endif
