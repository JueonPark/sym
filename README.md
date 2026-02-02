# Sym

A symbolic, algebraic type inference engine built on MLIR.

## Overview

Sym is an MLIR dialect that enables symbolic shape tracking and inference for tensor operations. It provides:

- **Symbolic Tensor Types**: Tensors with symbolic dimension expressions (e.g., `!sym.tensor<[s0, s1], f32>`)
- **Symbolic Expressions**: Rich expression language including symbols, constants, and binary operations
- **Automatic Simplification**: Algebraic simplification of symbolic expressions (e.g., `x + 0 → x`, `x * 1 → x`)
- **Broadcasting Unification**: NumPy-style broadcasting semantics for symbolic shapes
- **Dialect-Agnostic Interface**: Attach symbolic shape inference to any MLIR dialect

## Key Features

### Symbolic Types

```mlir
// Symbolic tensor with symbolic dimensions s0 and s1
!sym.tensor<[s0, s1], f32>

// Mixed symbolic and concrete dimensions
!sym.tensor<[s0, 64, s1], i32>

// Symbolic expressions as dimensions
!sym.tensor<[s0 * s1, s0 + 1], f64>
```

### Symbolic Expressions

- **Symbols**: Named symbolic values (e.g., `s0`, `batch_size`)
- **Constants**: Integer constants (e.g., `1`, `64`, `256`)
- **Binary Operations**: `add`, `sub`, `mul`, `div`, `mod`

### Operations

| Operation | Description |
|-----------|-------------|
| `sym.constant` | Create a symbolic tensor constant |
| `sym.change_type` | Convert a tensor to a symbolic tensor type |

### Passes

| Pass | Description |
|------|-------------|
| `--symbolic-shape-inference` | Propagate symbolic shapes through operations |

## Project Structure

```
sym/
├── CMakeLists.txt              # Root CMake configuration
├── sym/
│   ├── CMakeLists.txt          # Dialect library CMake
│   └── dialect/sym/
│       ├── IR/                 # Dialect definitions
│       │   ├── SymDialect.td   # Dialect TableGen definition
│       │   ├── SymTypes.td     # Type definitions
│       │   ├── SymAttrs.td     # Attribute definitions
│       │   ├── SymOps.td       # Operation definitions
│       │   ├── SymInterfaces.td # Interface definitions
│       │   ├── SymDialect.h/cpp # Dialect implementation
│       │   ├── SymUtils.h/cpp  # UnificationSolver utility
│       │   └── SymExtensions.cpp # External models for arith ops
│       └── Transforms/         # Optimization passes
│           ├── SymPasses.td    # Pass TableGen definitions
│           ├── SymPasses.h     # Pass declarations
│           └── SymbolicShapeInference.cpp # Shape inference pass
├── tools/
│   └── SymOptMain.cpp          # sym-opt driver tool
├── test/                       # LIT tests
└── build_tools/                # Build utilities
```

## Building

### Prerequisites

- CMake 3.20+
- Ninja
- LLVM/MLIR (built from source or pre-built)

### Build Sym with LLVM/MLIR altogether

```bash
./build_tools/build_mlir.sh
```

### Build Sym with external MLIR

```bash
mkdir -p build/sym && cd build/sym
cmake -G Ninja ../.. \
  -DMLIR_DIR={YOUR_MLIR_PATH} \
  -DLLVM_EXTERNAL_LIT={YOUR_LLVM-LIT_PATH}
ninja
```

### Run Tests

```bash
ninja check-sym
```

## Usage

### Command Line

```bash
# Parse and verify symbolic types
./build/sym/sym/tools/sym-opt test/dialect/sym/types.mlir

# Run symbolic shape inference
./build/sym/sym/tools/sym-opt --symbolic-shape-inference input.mlir
```

### MLIR Examples

```mlir
// Define a function with symbolic tensor types
func.func @matmul(%a: !sym.tensor<[s0, s1], f32>, 
                  %b: !sym.tensor<[s1, s2], f32>) 
    -> !sym.tensor<[s0, s2], f32> {
  // Operations here...
}

// Use sym.change_type to convert standard tensors
func.func @convert(%input: tensor<4x8xf32>) -> !sym.tensor<[s0, s1], f32> {
  %result = sym.change_type %input : tensor<4x8xf32> -> !sym.tensor<[s0, s1], f32>
  return %result : !sym.tensor<[s0, s1], f32>
}
```

### C++ API

```cpp
#include "sym/dialect/sym/IR/SymDialect.h"
#include "sym/dialect/sym/IR/SymUtils.h"

// Create symbolic types
auto s0 = SymbolExprAttr::get(ctx, "s0");
auto s1 = SymbolExprAttr::get(ctx, "s1");
auto tensorType = SymbolicTensorType::get(ctx, {s0, s1}, FloatType::getF32(ctx));

// Unify shapes with broadcasting
SmallVector<Attribute> result;
auto diag = UnificationSolver::unify(shape1, shape2, result, loc);
if (failed(diag)) {
  // Handle unification failure
}
```

## Supported External Operations

The `SymbolicShapeOpInterface` is attached to the following `arith` dialect operations:

**Integer Operations**: `addi`, `subi`, `muli`, `divsi`, `divui`, `remsi`, `remui`, `andi`, `ori`, `xori`, `shli`, `shrsi`, `shrui`, `maxsi`, `maxui`, `minsi`, `minui`, `cmpi`

**Floating-Point Operations**: `addf`, `subf`, `mulf`, `divf`, `remf`, `maximumf`, `minimumf`, `maxnumf`, `minnumf`, `cmpf`

**Conversion Operations**: `extsi`, `extui`, `extf`, `trunci`, `truncf`, `sitofp`, `uitofp`, `fptosi`, `fptoui`

## License

See [LICENSE](LICENSE) for details.
