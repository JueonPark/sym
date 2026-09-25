cmake_minimum_required(VERSION 3.20)
project(sym)

option(SYM_BUILD_TESTS "Build tests and generate conformance corpora" ON)
option(SYM_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SYM_BUILD_EXAMPLES "Build installed artifact consumer" ON)
option(SYM_BUILD_PYTHON "Build optional Python bindings" ON)
set(SYM_SOURCE_REVISION "" CACHE STRING "Release source revision override")

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
#
# libreloc is added here (rather than in section 3) because its test
# subdirectory calls add_test(): CMake only registers add_test() into
# CTestTestfile.cmake for directories where testing was already enabled
# when add_subdirectory() processed them, so it must come after
# enable_testing() and before test/ (whose SYM_TEST_DEPENDS references
# the libreloc-test target).
if(SYM_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(libreloc)
if(SYM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(SYM_BUILD_TESTS)
  add_subdirectory(unittest)
  add_subdirectory(test)
endif()
include(cmake/SymInstall.cmake)
