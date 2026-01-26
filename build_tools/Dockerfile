# Multi-stage build for LLVM/MLIR
# Stage 1: Build LLVM/MLIR
FROM ubuntu:22.04 AS builder

ARG LLVM_COMMIT

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone and checkout specific commit
RUN git clone --depth 1 https://github.com/llvm/llvm-project.git src && \
    cd src && \
    git fetch --depth 1 origin ${LLVM_COMMIT} && \
    git checkout ${LLVM_COMMIT}

# Build LLVM/MLIR
RUN cmake -G Ninja -S src/llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=ON \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF

RUN ninja -C build -j$(nproc) && \
    ninja -C build FileCheck count not && \
    ninja -C build install && \
    cp build/bin/FileCheck build/bin/count build/bin/not /opt/llvm/bin/

# Stage 2: Minimal runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir lit

# Copy only the installed LLVM/MLIR artifacts
COPY --from=builder /opt/llvm /opt/llvm

# Set environment variables
ENV MLIR_DIR=/opt/llvm/lib/cmake/mlir
ENV LLVM_DIR=/opt/llvm/lib/cmake/llvm
ENV PATH="/opt/llvm/bin:${PATH}"

WORKDIR /workspace
