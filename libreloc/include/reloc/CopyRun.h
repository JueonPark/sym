//===- CopyRun.h - contiguous byte-run copy (scalar + AVX2) -----*- C++ -*-===//

#ifndef RELOC_COPYRUN_H
#define RELOC_COPYRUN_H

#include <cstddef>

namespace reloc {

/// Copy `n` contiguous bytes. Runtime-dispatched: AVX2 (scalar prologue to
/// a 32-byte dst boundary + 32-byte body + scalar tail) when available,
/// else scalar. Byte-identical to copyRunScalar on every input.
void copyRun(void *dst, const void *src, size_t n);

/// The always-correct scalar path (std::memcpy), exposed for the
/// byte-identity test.
void copyRunScalar(void *dst, const void *src, size_t n);

/// True if the AVX2 path is active on this CPU (runtime dispatch result).
bool copyRunAvx2Available();

} // namespace reloc

#endif // RELOC_COPYRUN_H
