cmake_minimum_required(VERSION 3.20)
project(sym)

# 1. Find the package (The script ensures this works)
find_package(MLIR REQUIRED CONFIG)

# 2. Load MLIR Utils
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 3. Add Subdirectories
add_subdirectory(sym)

# 4. Add Testing Support
add_subdirectory(test)