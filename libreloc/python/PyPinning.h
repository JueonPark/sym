#ifndef RELOC_PYPINNING_H
#define RELOC_PYPINNING_H
#include "reloc/Transfer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
inline reloc::PinningPolicy parsePinning(const std::string &value) {
  if (value == "auto")
    return reloc::PinningPolicy::Auto;
  if (value == "pinned")
    return reloc::PinningPolicy::Pinned;
  if (value == "pageable")
    return reloc::PinningPolicy::Pageable;
  throw pybind11::value_error("pinning must be auto, pinned or pageable");
}
inline pybind11::list
stagingReport(const std::vector<reloc::StagingDecision> &items) {
  pybind11::list result;
  for (const auto &d : items) {
    pybind11::dict row;
    row["policy"] = d.policy == reloc::PinningPolicy::Auto     ? "auto"
                    : d.policy == reloc::PinningPolicy::Pinned ? "pinned"
                                                               : "pageable";
    row["memory_kind"] = d.pinned ? "pinned" : "pageable";
    row["wire_bytes"] = d.wireBytes;
    row["staging_capacity_bytes"] = d.capacityBytes;
    row["buffer_count"] = d.buffers;
    row["min_pinned_bytes"] = pybind11::cast(d.threshold);
    row["reason"] = d.reason;
    row["retention_eligible"] = d.retentionEligible;
    row["reused"] = d.reused;
    result.append(std::move(row));
  }
  return result;
}
#endif
