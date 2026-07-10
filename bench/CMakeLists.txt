# bench/: the E4 measurement-protocol skeleton (protocol.h, issue #47) and
# its consumers. CPU targets always build and run under ctest (CI); the
# PoC GPU driver is gated on RELOC_ENABLE_CUDA like libreloc's backend.
# Plain add_executable by intent (same rationale as libreloc/test): these
# tools may use llvm_gtest, but bench code includes no MLIR/LLVM headers.
add_executable(bench-protocol-test ProtocolTest.cpp)
target_include_directories(bench-protocol-test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-protocol-test PRIVATE cxx_std_17)
target_link_libraries(bench-protocol-test PRIVATE llvm_gtest llvm_gtest_main)
add_test(NAME bench-protocol-test COMMAND bench-protocol-test)

add_executable(bench-bind-cost bind_cost.cpp)
target_include_directories(bench-bind-cost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-bind-cost PRIVATE cxx_std_17)
target_link_libraries(bench-bind-cost PRIVATE reloc_runtime)
# Smoke-run under ctest: proves bench/protocol.h is reused by a second
# micro-measurement (issue #47 acceptance) on every CI run. Tiny counts --
# this is a does-it-work gate, not a measurement.
add_test(NAME bench-bind-cost-smoke
         COMMAND bench-bind-cost --n 4096 --json - --warmup 2 --iters 5 --reruns 2)
