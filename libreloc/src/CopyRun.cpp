//===- CopyRun.cpp - contiguous byte-run copy -----------------------------===//

#include "reloc/CopyRun.h"

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RELOC_HAS_AVX2_INTRINSICS 1
#endif

namespace reloc {

void copyRunScalar(void *dst, const void *src, size_t n) {
  std::memcpy(dst, src, n);
}

#ifdef RELOC_HAS_AVX2_INTRINSICS
__attribute__((target("avx2"))) static void
copyRunAvx2(void *dstV, const void *srcV, size_t n) {
  auto *dst = static_cast<uint8_t *>(dstV);
  const auto *src = static_cast<const uint8_t *>(srcV);
  // Scalar prologue: advance dst to a 32-byte boundary (or copy all if
  // n is small). Alignment shortfall is a downgrade, never an error.
  size_t prologue = (32 - (reinterpret_cast<uintptr_t>(dst) & 31)) & 31;
  if (prologue > n)
    prologue = n;
  std::memcpy(dst, src, prologue);
  size_t i = prologue;
  // 32-byte body: unaligned load (src alignment unknown), aligned store.
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
    _mm256_store_si256(reinterpret_cast<__m256i *>(dst + i), v);
  }
  // Scalar tail.
  if (i < n)
    std::memcpy(dst + i, src + i, n - i);
}
#endif

namespace {
bool detectAvx2() {
#ifdef RELOC_HAS_AVX2_INTRINSICS
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}
const bool kAvx2 = detectAvx2();
} // namespace

bool copyRunAvx2Available() { return kAvx2; }

void copyRun(void *dst, const void *src, size_t n) {
#ifdef RELOC_HAS_AVX2_INTRINSICS
  if (kAvx2) {
    copyRunAvx2(dst, src, n);
    return;
  }
#endif
  copyRunScalar(dst, src, n);
}

} // namespace reloc
