# Reloc Plan Wire Format v0

Binary encoding of a `#reloc.plan` attribute for the compiler → runtime
handoff. The decoder (`libreloc`, P2) never links MLIR: everything needed to
interpret a plan is defined here. The compiler-side encoder is
`mlir::reloc::encodePlan` (`sym/dialect/reloc/IR/RelocSerialization.h`).

Design rules for v0: fixed-width little-endian integers only, length-prefixed
strings, every section always present in a fixed order, no varints, no
compression, no optional sections. Inputs the format cannot represent are
encoder errors, never silent truncation.

## Primitives

| name | encoding |
|------|----------|
| `u8` | 1 byte |
| `u32` | 4 bytes, little-endian, unsigned |
| `i64` | 8 bytes, little-endian, two's complement |
| `str` | `u32` byte length, then that many UTF-8 bytes (no terminator) |

## Expressions (`expr`)

A symbolic scalar is a postfix opcode stream evaluated on a stack machine.
Encoding: `u32 op_count`, then `op_count` operations. A valid stream leaves
exactly one value on the stack. Binary operators pop the right operand, then
the left, and push the result.

| opcode | name | inline operand | meaning |
|--------|------|----------------|---------|
| `0x00` | PUSH_SYM | `u32` symbol index | push the value of symbol table entry *i* |
| `0x01` | PUSH_CONST | `i64` value | push a constant |
| `0x02` | ADD | — | push `l + r` |
| `0x03` | SUB | — | push `l - r` |
| `0x04` | MUL | — | push `l * r` |
| `0x05` | FLOORDIV | — | push `l floordiv r` (rounds toward −∞) |
| `0x06` | MOD | — | push `l mod r` (floor modulo; result has the divisor's sign) |
| `0x07` | PUSH_DIM | `u32` dim index | push destination-axis coordinate *d_i* — valid **only** inside inverse-map expressions |

FLOORDIV/MOD semantics match MLIR affine `floordiv`/`mod` and sym's pinned
`div`/`mod` semantics.

## Element types (`type`)

`u8 kind` + `u32 bitwidth`:

| kind | meaning | valid bitwidths |
|------|---------|-----------------|
| `0` | IEEE float | 16, 32, 64 |
| `1` | bfloat | 16 |
| `2` | signless integer | 1..64 |
| `3` | index | 64 (by convention) |

Anything else (complex types, non-signless integers, exotic floats, integers
wider than 64 bits) is unrepresentable in v0.

## Tensor descriptor (`tensor_desc`)

```
u32 rank
rank × expr        extents
u32 stride_count   (0 = canonical row-major, else == rank)
stride_count × expr strides
expr               offset
type               element type
```

## Typed value (`typed_value`, pad fill)

`type` + `u64 raw` — the value's bit pattern zero-extended to 64 bits
(IEEE/bfloat bit pattern for floats, two's-complement bits for integers),
little-endian.

## Plan layout (fixed section order)

```
0. magic            4 bytes ASCII "RPLN"
1. version          u32 = 0
2. symbol table     u32 count, count × str
3. src              tensor_desc
4. dst              tensor_desc
5. perm             u32 count, count × u32     (dst axis k <- src-view axis perm[k])
6. axes             u32 count, per axis:
                      str name, expr extent, expr src_stride, expr dst_stride
7. pad_fill         u32 count, per entry:
                      u32 dst_axis, expr lo, expr hi, typed_value fill
                      (lo/hi are leading/trailing pad WIDTHS, tensor.pad-style)
8. divisibility     u32 count, per entry: expr, i64 divisor
9. alignment        u32 count, per entry: u32 axis, i64 bytes
                      (axis < the axis count; enforced by the attribute
                      verifier, so the u32 narrowing is lossless)
10. contiguity      u32 count, count × u8 (0 or 1; count is 0 or the axis count)
11. flags           u8 no_copy, u8 runtime_pad_check   (0 or 1)
12. inverse         u32 num_dims, u32 num_results, num_results × expr
                      (square on the axes space: num_dims == num_results ==
                       axis count; PUSH_DIM allowed, PUSH_SYM not allowed in v0)
```

## Symbol table ordering

Symbol indices are assigned by **first use in encoding order** — the encoder
walks sections 3..8 in the layout order above (within a descriptor: extents,
strides, offset; within an axis: extent, src_stride, dst_stride; within a pad:
lo, hi; then each divisibility entry's expr in order), appending each
previously-unseen symbol name. This makes the byte stream a pure function of
the attribute.

## Worked example

The 1-D identity plan
`#reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0],
axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}],
inverse = affine_map<(d0) -> (d0)>>` encodes as (hex, annotated):

```
52 50 4c 4e                                  magic "RPLN"                            (4 bytes)
00 00 00 00                                  version 0                               (4 bytes)
00 00 00 00                                  symbol table: 0 entries                 (4 bytes)
01 00 00 00                                  src: rank 1                             (4 bytes)
  01 00 00 00 01 08 00 00 00 00 00 00 00     extent[0]: 1 op, PUSH_CONST 8           (13 bytes)
00 00 00 00                                  src: stride_count 0 (row-major)         (4 bytes)
  01 00 00 00 01 00 00 00 00 00 00 00 00     offset: 1 op, PUSH_CONST 0              (13 bytes)
02 20 00 00 00                               elem: kind 2 (int), bitwidth 32         (5 bytes)
01 00 00 00                                  dst: rank 1                             (4 bytes)
  01 00 00 00 01 08 00 00 00 00 00 00 00     extent[0]: 1 op, PUSH_CONST 8           (13 bytes)
00 00 00 00                                  dst: stride_count 0 (row-major)         (4 bytes)
  01 00 00 00 01 00 00 00 00 00 00 00 00     offset: 1 op, PUSH_CONST 0              (13 bytes)
02 20 00 00 00                               elem: kind 2 (int), bitwidth 32         (5 bytes)
01 00 00 00 00 00 00 00                      perm: count 1, [0]                      (8 bytes)
01 00 00 00                                  axes: count 1                           (4 bytes)
  01 00 00 00 78                             name "x" (str: len 1, "x")              (5 bytes)
  01 00 00 00 01 08 00 00 00 00 00 00 00     extent: 1 op, PUSH_CONST 8              (13 bytes)
  01 00 00 00 01 01 00 00 00 00 00 00 00     src_stride: 1 op, PUSH_CONST 1          (13 bytes)
  01 00 00 00 01 01 00 00 00 00 00 00 00     dst_stride: 1 op, PUSH_CONST 1          (13 bytes)
00 00 00 00                                  pad_fill: count 0                       (4 bytes)
00 00 00 00                                  divisibility: count 0                   (4 bytes)
00 00 00 00                                  alignment: count 0                      (4 bytes)
00 00 00 00                                  contiguity: count 0                     (4 bytes)
00 00                                        flags: no_copy 0, runtime_pad_check 0   (2 bytes)
01 00 00 00 01 00 00 00                      inverse: num_dims 1, num_results 1      (8 bytes)
  01 00 00 00 07 00 00 00 00                 result[0]: 1 op, PUSH_DIM 0             (9 bytes)
```

Section totals: magic 4 + version 4 + symbol table 4 + src tensor_desc 39
(4 rank + 13 extent + 4 stride_count + 13 offset + 5 elem type) + dst
tensor_desc 39 + perm 8 + axes 48 (4 count + 1 × (5 name + 13 extent + 13
src_stride + 13 dst_stride)) + pad_fill 4 + divisibility 4 + alignment 4 +
contiguity 4 + flags 2 + inverse 17 (4 num_dims + 4 num_results + 9 result
expr).

Total: 4+4+4+39+39+8+48+4+4+4+4+2+17 = **181 bytes**.

## Versioning

The `version` field is bumped on any layout change; v0 decoders must reject
other versions (decoder-side enforcement lands with P2).

v0 carries no per-section byte lengths, so decoders parse sections in full to
advance; any layout change (including additions) bumps the version, and v0
decoders must reject unknown versions rather than attempt partial reads.
