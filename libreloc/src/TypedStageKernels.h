//===- TypedStageKernels.h - bounded ordered stage specialization -*- C++
//-*-===//
#ifndef RELOC_SRC_TYPEDSTAGEKERNELS_H
#define RELOC_SRC_TYPEDSTAGEKERNELS_H

#include "reloc/TypedExecute.h"

namespace reloc::typed::detail {

// Prepared once per execution range, never per element/tile. This descriptor
// owns no executable memory, shape cache, parameters or intermediate arrays.
struct StageRun {
  enum class Kind { Buffered, CastChain, DequantCastChain };
  uint32_t from = 0;
  uint32_t to = 0;
  Kind kind = Kind::Buffered;
  uint32_t roundTrips = 0;
  bool inputHalf = false;
  bool outputHalf = false;
  int64_t channelStride = 0;
  int64_t channelExtent = 0;
};

StageRun prepareStageRun(const Program &, uint32_t from, uint32_t to);
const char *stageRunName(const StageRun &);

// False means unsupported, before any write. Eligible runs retain every
// ordered binary32 multiply and half conversion, with scalar tails/fallback.
// The offset is in the global logical destination even for chunk-local dst.
bool runStageRun(const Program &, const StageRun &, const uint8_t *src,
                 uint8_t *dst, int64_t count, int64_t destinationOffset);

} // namespace reloc::typed::detail
#endif
