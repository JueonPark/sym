//===- TypedKernels.h - bounded CPU layout/stage kernels --------*- C++ -*-===//
#ifndef RELOC_SRC_TYPEDKERNELS_H
#define RELOC_SRC_TYPEDKERNELS_H

#include "reloc/TypedExecute.h"

namespace reloc::typed::detail {

constexpr int64_t kTypedTile = 32;
constexpr int64_t kTypedRun = 256;

// True only when each channel map is a checked logical coordinate and its
// parameter stays constant over a contiguous inner span. Other expressions
// keep the generic evaluator, including its error reporting.
bool simpleChannels(const Program &, uint32_t from, uint32_t to,
                    int64_t innerSpan);

// Contiguous input/output, potentially unaligned. Bounded stack buffers keep
// every intermediate rounding without a full tensor or shape-specific code.
void stages(const Program &, uint32_t from, uint32_t to, const uint8_t *src,
            uint8_t *dst, int64_t count, int64_t destinationOffset);

// Dense matrix transpose, with optional output padding and chunk-local dst.
void transpose(const Program &, uint32_t from, uint32_t to, const uint8_t *src,
               uint8_t *dst, int64_t first, int64_t last, int64_t origin,
               int64_t outerLo, int64_t innerLo);

} // namespace reloc::typed::detail
#endif
