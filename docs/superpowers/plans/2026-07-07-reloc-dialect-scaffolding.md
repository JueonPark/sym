# reloc Dialect Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an empty-but-registered `reloc` MLIR dialect that builds in the existing tree and round-trips through `sym-opt`.

**Architecture:** Mirror the `sym` dialect layout exactly under `sym/dialect/reloc/`. A bare `Reloc_Dialect` TableGen definition generates dialect decls/defs; a new `MLIRRelocDialect` CMake library builds them; the dialect is registered into the existing `sym-opt` driver. No attributes, types, ops, verifiers, or passes — those land in later P1a sub-issues (A2–A5).

**Tech Stack:** C++17, MLIR/LLVM (TableGen, `add_mlir_dialect_library`), CMake + Ninja, `lit` + `FileCheck`.

## Global Constraints

- **Toolchain target:** LLVM/MLIR **21.1.8** per the external build doc §1.1. The repo currently pins commit `2078da43e25a4623cab2d0d60decddf709aaea28` in `build_tools/llvm_version.txt` — note this discrepancy in build docs; do not change the pin.
- **Mirror `sym` exactly:** match directory layout, file naming, CMake idioms, and code style of `sym/dialect/sym/`. No restructuring of `sym`.
- **Bare dialect only:** no `useDefault{Type,Attribute}PrinterParser` hooks, no `Reloc_Type`/`Reloc_Attr` base classes. Those arrive in A2.
- **Namespace:** `::mlir::reloc`. Dialect mnemonic: `reloc`.
- **Style:** conform to `.clang-format` / `.clang-tidy` in repo root.
- **Build dir:** `build/sym` (configured with `MLIR_DIR=/home/jueonpark/sym/build/llvm-project/build/lib/cmake/mlir`). Build/test via `ninja -C build/sym <target>`.
- **Commits:** conventional style matching repo history (`update(reloc): ...`), ending with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- **Existing `sym` tests must stay green** (`check-sym`) before and after.

---

### Task 1: `reloc` dialect scaffolding, build integration, and driver registration

Registers an empty `reloc` dialect and makes `sym-opt --show-dialects` list it. Driven by a `lit` test whose registration check fails until the dialect is wired in.

**Files:**
- Create: `test/dialect/reloc/smoke.mlir`
- Create: `sym/dialect/reloc/IR/RelocDialect.td`
- Create: `sym/dialect/reloc/IR/RelocDialect.h`
- Create: `sym/dialect/reloc/IR/RelocDialect.cpp`
- Create: `sym/dialect/reloc/IR/CMakeLists.txt`
- Create: `sym/dialect/reloc/CMakeLists.txt`
- Modify: `sym/dialect/CMakeLists.txt` (add `add_subdirectory(reloc)`)
- Modify: `sym/tools/SymOptMain.cpp` (include + register reloc)
- Modify: `sym/tools/CMakeLists.txt` (include dirs + link `MLIRRelocDialect`)

**Interfaces:**
- Consumes: nothing from earlier tasks (this is the first task).
- Produces: `mlir::reloc::RelocDialect` — a registrable MLIR dialect class (generated from `RelocDialect.td`), declared via `RelocDialect.h`, linked as the `MLIRRelocDialect` CMake library. `RelocDialect::initialize()` is a defined no-op. Later tasks (A2/A5) extend `initialize()` and add TableGen targets in `sym/dialect/reloc/IR/CMakeLists.txt`.

- [ ] **Step 1: Write the failing test**

Create `test/dialect/reloc/smoke.mlir`:

```mlir
// RUN: sym-opt %s | sym-opt | FileCheck %s
// RUN: sym-opt --show-dialects | FileCheck %s --check-prefix=DIALECTS

// CHECK: module
// DIALECTS: reloc

module {}
```

The first RUN line checks an empty module round-trips through `sym-opt`. The second checks that `reloc` appears in the registered-dialect list — this is what fails until the dialect is wired in.

- [ ] **Step 2: Run the test to verify it fails**

Run: `ninja -C build/sym check-sym`
Expected: FAIL on `test/dialect/reloc/smoke.mlir` — the `DIALECTS` check cannot find `reloc` in the `--show-dialects` output (`sym-opt` builds unchanged; the round-trip `CHECK` passes, the `DIALECTS` check fails). All other `sym` tests still pass.

- [ ] **Step 3: Create the dialect TableGen definition**

Create `sym/dialect/reloc/IR/RelocDialect.td`:

```tablegen
// RelocDialect.td - Main dialect definition for Reloc
#ifndef RELOC_DIALECT_TD
#define RELOC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Reloc Dialect Definition
//===----------------------------------------------------------------------===//

def Reloc_Dialect : Dialect {
  let name = "reloc";
  let cppNamespace = "::mlir::reloc";
  let summary = "A dialect for symbolic relocation / layout-transform plans";
  let description = [{
    The Reloc dialect provides a serializable, symbolic representation of
    folded layout-transform chains (RelocationPlan IR) and the intermediate
    op set that a downstream folding pass consumes. This is the dialect shell;
    attributes, ops, verifiers, and serialization are added in later
    sub-issues.
  }];
}

#endif // RELOC_DIALECT_TD
```

- [ ] **Step 4: Create the dialect header**

Create `sym/dialect/reloc/IR/RelocDialect.h`:

```cpp
#ifndef RELOC_DIALECT_H
#define RELOC_DIALECT_H

#include "mlir/IR/Dialect.h"

// Include TableGen outputs for Dialect
#include "RelocDialect.h.inc"

#endif // RELOC_DIALECT_H
```

- [ ] **Step 5: Create the dialect implementation**

Create `sym/dialect/reloc/IR/RelocDialect.cpp`:

```cpp
#include "RelocDialect.h"

using namespace mlir;
using namespace mlir::reloc;

// Include generated dialect definitions
#include "RelocDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Reloc Dialect Initialization
//===----------------------------------------------------------------------===//

void RelocDialect::initialize() {
  // No types, attributes, or operations registered yet (added in A2/A5).
}
```

- [ ] **Step 6: Create the IR CMake file**

Create `sym/dialect/reloc/IR/CMakeLists.txt`:

```cmake
# TableGen for Dialect
set(LLVM_TARGET_DEFINITIONS RelocDialect.td)
mlir_tablegen(RelocDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RelocDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RelocDialectIncGen)

# Combined TableGen target for IR
add_custom_target(RelocIRIncGen DEPENDS
  RelocDialectIncGen
)

# Dialect Library
add_mlir_dialect_library(MLIRRelocDialect
  RelocDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR

  DEPENDS
  RelocIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Add include path for generated files
target_include_directories(MLIRRelocDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
```

- [ ] **Step 7: Create the dialect-level CMake file**

Create `sym/dialect/reloc/CMakeLists.txt`:

```cmake
add_subdirectory(IR)
```

- [ ] **Step 8: Hook reloc into the dialect build**

Modify `sym/dialect/CMakeLists.txt` from:

```cmake
add_subdirectory(sym)
```

to:

```cmake
add_subdirectory(sym)
add_subdirectory(reloc)
```

- [ ] **Step 9: Register reloc in the `sym-opt` driver**

Modify `sym/tools/SymOptMain.cpp`. Add the include next to the existing `#include "SymDialect.h"`:

```cpp
#include "RelocDialect.h"
```

Add the registration next to the existing `registry.insert<mlir::sym::SymDialect>();`:

```cpp
  // Register our Reloc dialect
  registry.insert<mlir::reloc::RelocDialect>();
```

- [ ] **Step 10: Link and include reloc in the `sym-opt` CMake target**

Modify `sym/tools/CMakeLists.txt`. In the `target_include_directories(sym-opt PRIVATE ...)` block, add reloc's source and binary IR dirs:

```cmake
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/IR
```

In the `target_link_libraries(sym-opt PRIVATE ...)` block, add the dialect library (next to `MLIRSymDialect`):

```cmake
  MLIRRelocDialect
```

- [ ] **Step 11: Run the test to verify it passes**

Run: `ninja -C build/sym check-sym`
Expected: PASS — all tests green, including `test/dialect/reloc/smoke.mlir` (both the `CHECK` round-trip and the `DIALECTS: reloc` check). Ninja auto-reconfigures CMake because `sym/dialect/CMakeLists.txt` changed, picking up the new subdirectory, library, and TableGen targets.

- [ ] **Step 12: Manually confirm the acceptance criterion**

Run: `./build/sym/sym/tools/sym-opt --show-dialects`
Expected: the comma-separated `Available Dialects:` line includes `reloc`.

- [ ] **Step 13: Commit**

```bash
git add sym/dialect/reloc/ test/dialect/reloc/ \
        sym/dialect/CMakeLists.txt sym/tools/SymOptMain.cpp sym/tools/CMakeLists.txt
git commit -m "$(cat <<'EOF'
update(reloc): Introduce reloc dialect scaffolding (#16)

Add an empty-but-registered reloc dialect mirroring the sym dialect
layout, wired into the build and registered in the sym-opt driver.
Attributes, ops, verifier, and serialization land in later P1a sub-issues.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Documentation — project structure entry and toolchain note

Records the new dialect in the README and documents the LLVM/MLIR version pin.

**Files:**
- Modify: `README.md` (project-structure section)

**Interfaces:**
- Consumes: the `reloc` dialect and file layout produced by Task 1.
- Produces: nothing consumed by later tasks (documentation only).

- [ ] **Step 1: Add the reloc dialect to the README project-structure tree**

In `README.md`, locate the `## Project Structure` code block. After the `sym/dialect/sym/` subtree (before the closing ``` of the tree, following the `Transforms/` block), add:

```
│   └── dialect/reloc/          # RelocationPlan IR dialect (P1a, scaffolding)
│       └── IR/
│           ├── RelocDialect.td # Dialect TableGen definition
│           └── RelocDialect.h/cpp # Dialect implementation
```

- [ ] **Step 2: Add a toolchain-version note to the README build section**

In `README.md`, under `### Prerequisites` (in the `## Building` section), after the existing `- LLVM/MLIR (built from source or pre-built)` bullet, add:

```markdown
  - Target version: **LLVM/MLIR 21.1.8**. The repo pins commit
    `2078da43e25a4623cab2d0d60decddf709aaea28` in `build_tools/llvm_version.txt`;
    this is the version currently used by the project.
```

- [ ] **Step 3: Verify the README renders and mentions reloc**

Run: `grep -n "reloc" README.md`
Expected: at least the two additions above appear (the structure tree entry and, if present, any other mention). Visually confirm the Project Structure code fence is still balanced (opening and closing ``` intact).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(reloc): Document reloc dialect and LLVM toolchain pin (#16)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Dialect files mirroring `sym` layout → Task 1, Steps 3–5. ✓
- CMake wiring (`add_mlir_dialect`, TableGen targets, hook into `sym/dialect/CMakeLists.txt`) → Task 1, Steps 6–8. ✓
- Register `reloc` in `SymOptMain.cpp` (+ unavoidable `tools/CMakeLists.txt`) → Task 1, Steps 9–10. ✓
- `test/dialect/reloc/` + smoke test → Task 1, Step 1. ✓
- Toolchain pin in build docs → Task 2, Step 2. ✓
- Acceptance: `--show-dialects` lists `reloc` → Task 1, Step 12; smoke test passes `lit` → Task 1, Step 11; `check-sym` green → Task 1, Steps 2 & 11. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete file content or exact edit. The only in-code comment ("added in A2/A5") is intentional scope documentation, not a deferred step. ✓

**Type consistency:** `mlir::reloc::RelocDialect`, `RelocDialect::initialize()`, library `MLIRRelocDialect`, TableGen targets `RelocDialectIncGen`/`RelocIRIncGen`, and generated `RelocDialect.{h,cpp}.inc` names are used consistently across Steps 3–10. ✓
