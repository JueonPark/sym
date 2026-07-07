# Design: A1 — `reloc` dialect scaffolding + build integration

- **Issue**: [#16](https://github.com/JueonPark/sym/issues/16) (sub-issue of tracking [#14](https://github.com/JueonPark/sym/issues/14), P1a)
- **Date**: 2026-07-07
- **Status**: Approved for planning

## Goal

An empty-but-registered `reloc` MLIR dialect that builds against the existing
tree and round-trips through `sym-opt`. It becomes the shared dialect shell that
later sub-issues build on: A2 (`#reloc.plan` attribute) and A5 (`reloc.*` ops)
are parallel and share only this shell.

This sub-issue is **pure scaffolding**. No attributes, types, ops, verifiers,
transfer functions, passes, or serialization — those belong to later P1a
sub-issues (A2–A5) and P1b/P2.

## Design principles

- **Mirror `sym` exactly.** Match the existing directory layout, file naming,
  CMake idioms, and code style of `sym/dialect/sym/`. No restructuring of `sym`.
- **Minimal ("bare") dialect.** Only the `Reloc_Dialect` TableGen def and its
  generated decls/defs. No `useDefault{Type,Attribute}PrinterParser` hooks and
  no `Reloc_Type`/`Reloc_Attr` base classes yet — those land in A2 alongside the
  first attribute, so each PR adds exactly what it needs.
- **Surgical integration.** The dialect registration lands in the existing
  `sym-opt` driver; no separate `reloc-opt` tool.

## New files

```
sym/dialect/reloc/
├── CMakeLists.txt                 # add_subdirectory(IR)
└── IR/
    ├── CMakeLists.txt             # dialect TableGen target + MLIRRelocDialect library
    ├── RelocDialect.td            # def Reloc_Dialect
    ├── RelocDialect.h             # namespace mlir::reloc; includes RelocDialect.h.inc
    └── RelocDialect.cpp           # RelocDialect::initialize() {}  (empty)
test/dialect/reloc/
└── smoke.mlir                     # module {} round-trip through sym-opt
```

### `RelocDialect.td`

Dialect definition only, mirroring `SymDialect.td` but trimmed:

```tablegen
#ifndef RELOC_DIALECT_TD
#define RELOC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Reloc_Dialect : Dialect {
  let name = "reloc";
  let cppNamespace = "::mlir::reloc";
  let summary = "A dialect for symbolic relocation / layout-transform plans";
  let description = [{
    The Reloc dialect provides a serializable, symbolic representation of
    folded layout-transform chains (RelocationPlan IR) and the intermediate
    op set that a downstream folding pass consumes.
  }];
}

#endif // RELOC_DIALECT_TD
```

No printer/parser hooks and no `Reloc_Type`/`Reloc_Attr` base classes at this
stage (added in A2 with the first attribute).

### `RelocDialect.h`

Mirrors `SymDialect.h` structure, trimmed to the bare dialect:

- MLIR IR includes (`Dialect.h`, `OpDefinition.h`, `OpImplementation.h`, etc.).
- `namespace mlir { namespace reloc { } }` (empty for now — no enum helpers yet).
- `#include "RelocDialect.h.inc"`.
- **No** attr/type/op `.inc` includes — none are generated yet.

### `RelocDialect.cpp`

```cpp
#include "RelocDialect.h"
// ... minimal includes matching sym ...
#include "RelocDialect.cpp.inc"

void RelocDialect::initialize() {}
```

Empty `initialize()` — nothing to register yet.

## Build wiring

### `sym/dialect/reloc/IR/CMakeLists.txt`

Follows `sym/dialect/sym/IR/CMakeLists.txt`, but only the dialect TableGen pair
(no types/attrs/ops/interfaces):

```cmake
set(LLVM_TARGET_DEFINITIONS RelocDialect.td)
mlir_tablegen(RelocDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RelocDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RelocDialectIncGen)

add_mlir_dialect_library(MLIRRelocDialect
  RelocDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR

  DEPENDS
  RelocDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(MLIRRelocDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
```

### `sym/dialect/reloc/CMakeLists.txt`

```cmake
add_subdirectory(IR)
```

### `sym/dialect/CMakeLists.txt` (edit existing)

```cmake
add_subdirectory(sym)
add_subdirectory(reloc)
```

## Driver integration (edits to existing files)

### `sym/tools/SymOptMain.cpp`

- `#include "RelocDialect.h"`
- `registry.insert<mlir::reloc::RelocDialect>();`

### `sym/tools/CMakeLists.txt`

- Add reloc's IR source + binary dirs to `sym-opt`'s `target_include_directories`.
- Add `MLIRRelocDialect` to `target_link_libraries`.

> The issue says the "only edit to existing files" is `SymOptMain.cpp`; linking
> the new library unavoidably also requires `tools/CMakeLists.txt`. Both are
> treated as required integration edits (confirmed with the author).

## Test

`lit` auto-discovers `.mlir` files recursively under `test/` (see
`test/lit.cfg.py`: `suffixes = ['.mlir']`, `test_source_root = dirname`), so no
CMake change is needed to register the test.

`test/dialect/reloc/smoke.mlir`:

```mlir
// RUN: sym-opt %s | sym-opt | FileCheck %s
// CHECK: module
module {}
```

Because `reloc` defines no ops yet, an empty `module {}` that survives a
round-trip through `sym-opt` with the dialect registered is the meaningful smoke
test. (Registration itself is asserted separately via `--show-dialects`.)

## Docs

- Add a `reloc` entry to the README project-structure section.
- Record the toolchain pin: the build doc targets **LLVM/MLIR 21.1.8**; the repo
  currently pins commit `2078da43e25a4623cab2d0d60decddf709aaea28` in
  `build_tools/llvm_version.txt`. Note the discrepancy in build docs.

## Acceptance criteria

- `sym-opt --show-dialects` lists `reloc`.
- `test/dialect/reloc/smoke.mlir` passes `lit`.
- Existing `sym` tests still pass unmodified (`check-sym` green before and after).

## Out of scope (later sub-issues)

- Plan attribute `#reloc.plan` and sym↔affine bridge → A2.
- Plan verifier → A3.
- Wire-format serialization → A4.
- `reloc.transpose` / `reloc.reshape` / `reloc.pad` ops → A5.
- Any transfer function, fold pass, canonicalization, runtime decoder, cost
  model → P1b / P2 / P3.

## Estimate

~200 LOC, ~0.5 day.
