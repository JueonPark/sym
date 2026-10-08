//===- Transpose.cpp - tiled CPU transpose --------------------------------===//

#include "Transpose.h"

#include "reloc/Bind.h"
#include "reloc/CopyRun.h"
#include "reloc/Execute.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RELOC_TRANSPOSE_AVX2 1
#endif

namespace reloc {
namespace detail {
namespace {

// Cache blocks contain several register tiles, reusing neighboring source
// cache lines before advancing through the full matrix. No shape-specific JIT
// or global ISA flags are needed; dimensions and leading strides stay dynamic.
constexpr int64_t kBlock = 32;
constexpr int64_t kBytes = 4;

#ifdef RELOC_TRANSPOSE_AVX2
constexpr int64_t kTile = 8;
__attribute__((target("avx2"), always_inline)) inline void
transpose8x8(const uint8_t *src, uint8_t *dst, int64_t srcStride,
             int64_t dstStride) {
  __m256i input[8], pairs[8], quads[8];
  for (int i = 0; i < 8; ++i)
    input[i] = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i * srcStride * kBytes));
  for (int i = 0; i < 8; i += 2) {
    pairs[i] = _mm256_unpacklo_epi32(input[i], input[i + 1]);
    pairs[i + 1] = _mm256_unpackhi_epi32(input[i], input[i + 1]);
  }
  for (int i = 0; i < 8; i += 4) {
    quads[i] = _mm256_unpacklo_epi64(pairs[i], pairs[i + 2]);
    quads[i + 1] = _mm256_unpackhi_epi64(pairs[i], pairs[i + 2]);
    quads[i + 2] = _mm256_unpacklo_epi64(pairs[i + 1], pairs[i + 3]);
    quads[i + 3] = _mm256_unpackhi_epi64(pairs[i + 1], pairs[i + 3]);
  }
  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + i * dstStride * kBytes),
        _mm256_permute2x128_si256(quads[i], quads[i + 4], 0x20));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + (i + 4) * dstStride * kBytes),
        _mm256_permute2x128_si256(quads[i], quads[i + 4], 0x31));
  }
}

__attribute__((target("avx2"))) void transpose32Avx2(const uint8_t *src,
                                                     uint8_t *dst, int64_t rows,
                                                     int64_t columns,
                                                     int64_t srcStride) {
  for (int64_t rb = 0; rb < rows; rb += kBlock) {
    const int64_t re = rb + std::min(kBlock, rows - rb);
    for (int64_t cb = 0; cb < columns; cb += kBlock) {
      const int64_t ce = cb + std::min(kBlock, columns - cb);
      int64_t r = rb;
      for (; re - r >= kTile; r += kTile) {
        int64_t c = cb;
        for (; ce - c >= kTile; c += kTile)
          transpose8x8(src + (c * srcStride + r) * kBytes,
                       dst + (r * columns + c) * kBytes, srcStride, columns);
        // Column tail: a full set of destination rows, fewer than 8 columns.
        for (; c < ce; ++c)
          for (int64_t rr = r; rr < r + kTile; ++rr)
            std::memcpy(dst + (rr * columns + c) * kBytes,
                        src + (c * srcStride + rr) * kBytes, kBytes);
      }
      // Row tail, including worker/chunk boundaries inside a register tile.
      for (; r < re; ++r)
        for (int64_t c = cb; c < ce; ++c)
          std::memcpy(dst + (r * columns + c) * kBytes,
                      src + (c * srcStride + r) * kBytes, kBytes);
    }
  }
}
#endif

// The kernels below repeat the blocking above for a last-dim torch.stack,
// reading source row c (destination column c) through the inputs' pointer
// table instead of a row pitch. They stay separate so the single-source
// kernels keep their codegen. The row functor is taken by value: through a
// reference every byte store to dst may alias it, so the tail loops would
// reload its fields per element.
struct TableRows {
  const uint8_t *const *rows;
  int64_t offset; // bytes from each row's element 0 to this window
  const uint8_t *operator()(int64_t c) const { return rows[c] + offset; }
};

#ifdef RELOC_TRANSPOSE_AVX2
template <class Rows>
__attribute__((target("avx2"), always_inline)) inline void
transpose8x8Rows(const Rows rowAt, int64_t c, int64_t r, uint8_t *dst,
                 int64_t dstStride) {
  __m256i input[8], pairs[8], quads[8];
  for (int i = 0; i < 8; ++i)
    input[i] = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(rowAt(c + i) + r * kBytes));
  for (int i = 0; i < 8; i += 2) {
    pairs[i] = _mm256_unpacklo_epi32(input[i], input[i + 1]);
    pairs[i + 1] = _mm256_unpackhi_epi32(input[i], input[i + 1]);
  }
  for (int i = 0; i < 8; i += 4) {
    quads[i] = _mm256_unpacklo_epi64(pairs[i], pairs[i + 2]);
    quads[i + 1] = _mm256_unpackhi_epi64(pairs[i], pairs[i + 2]);
    quads[i + 2] = _mm256_unpacklo_epi64(pairs[i + 1], pairs[i + 3]);
    quads[i + 3] = _mm256_unpackhi_epi64(pairs[i + 1], pairs[i + 3]);
  }
  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + i * dstStride * kBytes),
        _mm256_permute2x128_si256(quads[i], quads[i + 4], 0x20));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + (i + 4) * dstStride * kBytes),
        _mm256_permute2x128_si256(quads[i], quads[i + 4], 0x31));
  }
}

template <class Rows>
__attribute__((target("avx2"))) void
transpose32Avx2Rows(const Rows rowAt, uint8_t *dst, int64_t rows,
                    int64_t columns) {
  for (int64_t rb = 0; rb < rows; rb += kBlock) {
    const int64_t re = rb + std::min(kBlock, rows - rb);
    for (int64_t cb = 0; cb < columns; cb += kBlock) {
      const int64_t ce = cb + std::min(kBlock, columns - cb);
      int64_t r = rb;
      for (; re - r >= kTile; r += kTile) {
        int64_t c = cb;
        for (; ce - c >= kTile; c += kTile)
          transpose8x8Rows(rowAt, c, r, dst + (r * columns + c) * kBytes,
                           columns);
        // Column tail: a full set of destination rows, fewer than 8 columns.
        for (; c < ce; ++c)
          for (int64_t rr = r; rr < r + kTile; ++rr)
            std::memcpy(dst + (rr * columns + c) * kBytes,
                        rowAt(c) + rr * kBytes, kBytes);
      }
      // Row tail, including worker/chunk boundaries inside a register tile.
      for (; r < re; ++r)
        for (int64_t c = cb; c < ce; ++c)
          std::memcpy(dst + (r * columns + c) * kBytes, rowAt(c) + r * kBytes,
                      kBytes);
    }
  }
}
#endif

template <class Rows>
void transpose32ScalarRows(const Rows rowAt, uint8_t *dst, int64_t rows,
                           int64_t columns) {
  for (int64_t rb = 0; rb < rows; rb += kBlock) {
    const int64_t re = rb + std::min(kBlock, rows - rb);
    for (int64_t cb = 0; cb < columns; cb += kBlock) {
      const int64_t ce = cb + std::min(kBlock, columns - cb);
      for (int64_t r = rb; r < re; ++r)
        for (int64_t c = cb; c < ce; ++c)
          // Constant width folds to an unaligned load/store without a tiny
          // runtime-sized memcpy call, and preserves every bit of any dtype.
          std::memcpy(dst + (r * columns + c) * kBytes, rowAt(c) + r * kBytes,
                      kBytes);
    }
  }
}

} // namespace

void transpose32Scalar(const uint8_t *src, uint8_t *dst, int64_t rows,
                       int64_t columns, int64_t srcStride) {
  for (int64_t rb = 0; rb < rows; rb += kBlock) {
    const int64_t re = rb + std::min(kBlock, rows - rb);
    for (int64_t cb = 0; cb < columns; cb += kBlock) {
      const int64_t ce = cb + std::min(kBlock, columns - cb);
      for (int64_t r = rb; r < re; ++r)
        for (int64_t c = cb; c < ce; ++c)
          // Constant width folds to an unaligned load/store without a tiny
          // runtime-sized memcpy call, and preserves every bit of any dtype.
          std::memcpy(dst + (r * columns + c) * kBytes,
                      src + (c * srcStride + r) * kBytes, kBytes);
    }
  }
}

bool tryGatherTranspose32(const BoundPlan &bound, const uint8_t *src,
                          uint8_t *dst, int64_t outerBegin, int64_t outerEnd) {
  if (bound.typed || bound.elementSize != kBytes || !bound.padRegions.empty() ||
      bound.extents.size() != 2 || bound.srcStrides.size() != 2 ||
      bound.dstStrides.size() != 2 || bound.extents[0] <= 0 ||
      bound.extents[1] <= 0 || bound.srcStrides[0] != 1 ||
      bound.srcStrides[1] != bound.extents[0] ||
      bound.dstStrides[0] != bound.extents[1] || bound.dstStrides[1] != 1)
    return false;
  if (outerBegin >= outerEnd)
    return true;

  const int64_t rows = outerEnd - outerBegin;
  const int64_t columns = bound.extents[1];
  // Normalize to real addresses in this window before entering the kernel.
  // In the pipeline dst may be a rebased element-zero address; only these
  // requested rows belong to the actual staging allocation.
  src += outerBegin * kBytes;
  dst += outerBegin * columns * kBytes;
  // A singleton matrix dimension makes this a contiguous copy. Keep the
  // existing copyRun throughput instead of visiting scalar tile tails.
  if (bound.extents[0] == 1 || columns == 1) {
    copyRun(dst, src, static_cast<size_t>(rows * columns * kBytes));
    return true;
  }
#ifdef RELOC_TRANSPOSE_AVX2
  if (copyRunAvx2Available()) {
    transpose32Avx2(src, dst, rows, columns, bound.srcStrides[1]);
    return true;
  }
#endif
  transpose32Scalar(src, dst, rows, columns, bound.srcStrides[1]);
  return true;
}

bool tryGatherTranspose32Stacked(const BoundPlan &bound,
                                 const StackedSource &source, uint8_t *dst,
                                 int64_t outerBegin, int64_t outerEnd) {
  // Exactly the [N, Z] -> [Z, N] transpose of the logical source: each source
  // row is one whole input, so the row table replaces the row pitch.
  if (bound.typed || bound.elementSize != kBytes || !bound.padRegions.empty() ||
      bound.extents.size() != 2 || bound.srcStrides.size() != 2 ||
      bound.dstStrides.size() != 2 ||
      bound.extents[0] != source.segmentElements ||
      bound.extents[1] != source.count || bound.srcStrides[0] != 1 ||
      bound.srcStrides[1] != source.segmentElements ||
      bound.dstStrides[0] != bound.extents[1] || bound.dstStrides[1] != 1)
    return false;
  if (outerBegin >= outerEnd)
    return true;
  const int64_t rows = outerEnd - outerBegin;
  const int64_t columns = bound.extents[1];
  dst += outerBegin * columns * kBytes;
  const TableRows rowAt{source.bases, outerBegin * kBytes};
  // One input is one contiguous source row. (Unlike the strided kernel, Z == 1
  // with several inputs is not contiguous: rows live in separate buffers.)
  if (columns == 1) {
    copyRun(dst, rowAt(0), static_cast<size_t>(rows * kBytes));
    return true;
  }
#ifdef RELOC_TRANSPOSE_AVX2
  if (copyRunAvx2Available()) {
    transpose32Avx2Rows(rowAt, dst, rows, columns);
    return true;
  }
#endif
  transpose32ScalarRows(rowAt, dst, rows, columns);
  return true;
}

} // namespace detail
} // namespace reloc
