# Plain add_executable by intent: only the LIBRARY is bound by the
# MLIR-free contract; the test binary may use llvm_gtest (which links
# LLVMSupport) as shared test infrastructure.
add_executable(libreloc-test
  VersionTest.cpp
  DecodeTest.cpp
  EvalTest.cpp
  BindTest.cpp
  ExecuteTest.cpp
  BackendTest.cpp
  PoolTest.cpp
  GatherPoolTest.cpp
  ChunkScheduleTest.cpp
  PipelineTest.cpp
  CudaPipelineTest.cpp
  QuantTest.cpp
)

target_link_libraries(libreloc-test PRIVATE
  reloc_runtime
  llvm_gtest
  llvm_gtest_main
)

if(RELOC_ENABLE_CUDA)
  target_link_libraries(libreloc-test PRIVATE CUDA::cudart)
endif()

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

# nm cannot see header-only MLIR/LLVM usage (inline code exports no
# symbols), and the repo's global include paths make such includes
# compile silently — guard the contract at the source level too.
# grep's exit status must be exactly 1 (no match): 0 means a violating
# include was found, and 2 means the scan itself failed (e.g. a scanned
# directory was renamed away) — a bare `! grep` would negate 2 into a
# false PASS and silently disable this guard forever.
add_test(NAME reloc-runtime-no-mlir-includes
         COMMAND bash -c "grep -rE '#include *[\"<](mlir|llvm)[/\"]' ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../cuda ${CMAKE_CURRENT_SOURCE_DIR}/../python ${CMAKE_CURRENT_SOURCE_DIR}/../quant; test \$? -eq 1")
