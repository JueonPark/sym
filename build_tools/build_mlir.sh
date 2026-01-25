#!/bin/bash
set -e

# Configuration
LLVM_COMMIT=$(cat build_tools/llvm_version.txt)
BUILD_DIR="build"
LLVM_BUILD_DIR="${BUILD_DIR}/llvm-project"

echo "--- Setting up MLIR (Hash: $LLVM_COMMIT) ---"

# 1. Fetch LLVM (Only if not already correct)
if [ ! -d "$LLVM_BUILD_DIR" ]; then
    echo "Cloning LLVM..."
    git clone https://github.com/llvm/llvm-project.git "$LLVM_BUILD_DIR/src"
    cd "$LLVM_BUILD_DIR/src"
    git checkout "$LLVM_COMMIT"
    cd ../../..
fi

# 2. Build LLVM (The Heavy Step)
echo "Building LLVM..."
mkdir -p "$LLVM_BUILD_DIR/build"
cmake -G Ninja -S "$LLVM_BUILD_DIR/src/llvm" -B "$LLVM_BUILD_DIR/build" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RTTI=ON

ninja -C "$LLVM_BUILD_DIR/build" check-mlir

# 3. Build Your Project (Sym)
echo "Building Sym..."
mkdir -p "$BUILD_DIR/sym"
cmake -G Ninja -S . -B "$BUILD_DIR/sym" \
    -DMLIR_DIR="$PWD/$LLVM_BUILD_DIR/build/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="$PWD/$LLVM_BUILD_DIR/build/bin/llvm-lit"

ninja -C "$BUILD_DIR/sym"