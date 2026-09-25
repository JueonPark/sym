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

| Version | Content | Encoder | Decoder |
| --- | --- | --- | --- |
| `0` | layout-only `#reloc.plan` (this document above); frozen, byte-identical goldens | `encodePlan` | `decodePlan` (v0 only) |
| `1` | typed `#reloc.typed_plan` (C3, issue [#143](https://github.com/JueonPark/sym/issues/143); below) | `encodeTypedPlan` | `decodeTypedPlan` (v1 only) |

`decodePlan` never accepts v1 and `decodeTypedPlan` never accepts v0: a
runtime built before v1 rejects a v1 blob with `unsupported wire format
version` at byte offset 4, and a v1 consumer asks `peekWireVersion` (or reads
bytes 4..8) before choosing the decoder. Nothing in v0 is reinterpreted: the
layout-only `elementSize` and `totalBytes` keep their meaning, and typed
footprints are separate fields of the typed bound result.

# Reloc Plan Wire Format v1 (typed plans)

Binary encoding of a `#reloc.typed_plan` ([reloc-typed-folding.md](reloc-typed-folding.md)):
the logical source and result descriptors, ONE folded layout plan (the v0
body verbatim, with fused pad fills in the result dtype), the ordered value
stages with their parameters and channel maps, and the original fills with
their entry stages. Same primitives, `expr`, `type`, `tensor_desc` and
`typed_value` as v0; the same design rules (fixed-width little-endian, fixed
section order, no optional sections, counts checked against the remaining
byte budget before any allocation).

## New primitives

```
stage_type   type, u8 signedness             (0 signless, 1 signed, 2 unsigned)
param        u8 kind                         (0 none, 1 inline, 2 binding)
             kind 1: u8 rank (0 or 1), type element, u32 count, count × u64 raw
                     (rank 0 => count 1; raw = the element's bit pattern
                      zero-extended to 64 bits, like typed_value)
             kind 2: str name, u8 rank (0 or 1), rank × expr extents, type element
```

Signedness is semantic: storage stays signless `i8` in the descriptors, and
the quantize output / dequantize input `stage_type` says `signed`. A `param`
of kind `binding` is the stable runtime identity of a parameter: its name,
its declared rank/extents (sym expressions over the plan symbols) and its
element type. No address, device or framework object is ever encoded.

## Expressions in v1

The v0 opcode set is unchanged. A third context, the **channel** context,
allows both `PUSH_SYM` (plan symbols) and `PUSH_DIM` (a coordinate of the
logical **result** descriptor, index `< result rank`). Plan-context and
inverse-context rules are as in v0.

## Plan layout (fixed section order)

```
0. magic            4 bytes ASCII "RPLN"
1. version          u32 = 1
2. symbol table     u32 count, count × str         (first use over sections 3..7)
3. source           tensor_desc                    (logical source; its type is the source dtype)
4. result           tensor_desc                    (logical result; logical rank; its type is the result dtype)
5. layout           the v0 body: v0 sections 3..12 verbatim
                      src/dst types == source/result types; pad_fill values are
                      the FUSED fills in the result dtype
6. stages           u32 count (>= 1), per stage:
                      u8 transform        0 cast, 1 quantize, 2 dequantize
                      u8 policy           0 ieee_rne, 1 exact, 2 symmetric_rne, 3 affine
                      stage_type input
                      stage_type output
                      u32 rank, rank × expr shape   (logical operand shape)
                      param scale
                      param zero_point
                      i64 axis            (-1 = per tensor, else a channel axis of `shape`)
                      u8 has_channel      (0 or 1)
                      [u32 num_dims, expr channel]  (present iff has_channel;
                                                     channel context; num_dims == result rank)
7. fills            u32 count, per entry:
                      u32 dst_axis, u32 stage, typed_value original
```

## Decoder-enforced invariants

Every rule below fails decoding (before any allocation beyond the checked
counts) with the byte offset of the violated item:

- transform/policy pairs and types follow [reloc-typed-semantics.md](reloc-typed-semantics.md):
  cast `f32 (signless) -> f16 (signless)` under `ieee_rne` or `f16 -> f32`
  under `exact`; quantize `f32 -> int 8 (signed)` under `symmetric_rne`;
  dequantize `int 8 (signed) -> f32` under `affine`;
- stage 0 consumes the source type, stage `k+1` consumes what stage `k`
  produces, the last stage produces the result type; the layout's `src`/`dst`
  types equal the source/result types and its `src` rank equals the source
  rank;
- a cast carries no parameters (both `param` kinds `none`, axis `-1`, no
  channel); quantize/dequantize carry a scale (`kind != none`) whose element
  type is `f32`; a zero point is an integer (`inline`: any width ≤ 64,
  `binding`: `int 32`); an absent zero point means the constant 0;
- axis `-1`: both parameters rank 0 and no channel; axis in `[0, rank)`:
  scale rank 1, zero point rank 0 or 1, channel present with
  `num_dims == result rank`; a rank-1 inline parameter whose axis extent is a
  constant stream must have exactly that many values (symbolic extents are
  bind-time guards);
- inline scale values are finite and strictly positive; inline zero points
  lie in `[-128, 127]` and are 0 under `symmetric_rne`, which also rejects a
  zero-point binding;
- a binding name is unique within a stage; a name reused by another stage
  must declare the same rank, element type and extents (otherwise
  "conflicting parameter declaration");
- every fill's `dst_axis` names exactly one layout `pad_fill` entry and
  every layout pad has exactly one fill; `stage <= stage count`; the original
  fill's type equals the type entering that stage (the source type at stage
  0); the layout's fused fill type equals the result type; folding the
  original through the later stages with the C1 reference arithmetic
  (casts; quantize/dequantize only with inline per-tensor parameters — any
  other parameter after the entry point is rejected) reproduces the fused
  fill bit for bit (NaN matches NaN);
- the layout body obeys every v0 rule, and no bytes follow section 7.

A decoded typed plan is a validated *representation*. It does not certify
that any kernel can execute it: binding (`bindTyped`) adds the symbol and
parameter guards and the footprints, and execution capability is R3's.

## Symbol table ordering (v1)

First use in encoding order over sections 3..7: source descriptor, result
descriptor, the layout body (as v0 orders it), then each stage's shape,
scale binding extents, zero-point binding extents and channel expression,
then nothing (fills carry no expressions). Channel maps use `PUSH_SYM` with
these indices; the attribute's own `symbols` list only fixes affine symbol
positions and is not encoded separately.
