# Plain add_executable by intent: only the LIBRARY is bound by the
# MLIR-free contract; the test binary may use llvm_gtest (which links
# LLVMSupport) as shared test infrastructure.
add_executable(libreloc-test
  VersionTest.cpp
)

target_link_libraries(libreloc-test PRIVATE
  reloc_runtime
  llvm_gtest
  llvm_gtest_main
)

add_test(NAME libreloc-test COMMAND libreloc-test)

# The issue #41 acceptance check: no MLIR/LLVM symbols — defined or
# undefined — in the runtime's dynamic symbol table. Runs locally via
# ctest and in CI's existing ctest step (plus an explicit CI step for
# visibility).
add_test(NAME reloc-runtime-mlir-free
         COMMAND bash -c "set -e; nm -D $<TARGET_FILE:reloc_runtime> >/dev/null; ! nm -D $<TARGET_FILE:reloc_runtime> | grep -qiE 'mlir|llvm'")
# The first nm invocation makes an nm failure (missing/corrupt file) FAIL
# the test instead of vanishing into grep's empty input (a bare
# `! nm | grep` would false-PASS when nm errors).
