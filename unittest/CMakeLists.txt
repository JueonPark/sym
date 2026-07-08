# The C++ unit tests are mandatory. LLVM build trees export the llvm_gtest
# targets unconditionally; installed LLVM packages export them when built
# with -DLLVM_INSTALL_GTEST=ON (the CI image does — see
# build_tools/Dockerfile). Fail at configure time with a clear message
# rather than at compile time with a missing gtest/gtest.h.
if(NOT TARGET llvm_gtest)
  message(FATAL_ERROR
    "llvm_gtest target not found: the LLVM package at MLIR_DIR/LLVM_DIR "
    "must export gtest. Build LLVM with -DLLVM_INSTALL_GTEST=ON, or point "
    "MLIR_DIR/LLVM_DIR at an LLVM build tree.")
endif()

add_subdirectory(reloc)
