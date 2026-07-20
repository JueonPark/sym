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

add_executable(bench-gather-bw gather_bw.cpp)
target_include_directories(bench-gather-bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-gather-bw PRIVATE cxx_std_17)
target_link_libraries(bench-gather-bw PRIVATE reloc_runtime)
# Smoke-run under ctest (same rationale as bench-bind-cost-smoke): proves the
# D1 gather-BW driver runs end-to-end -- including its parallel-vs-executeH2D
# byte-exactness gate -- on every CI run. Tiny counts; not a measurement.
add_test(NAME bench-gather-bw-smoke
         COMMAND bench-gather-bw --n 256 --threads 2 --json - --warmup 1
                 --iters 3 --reruns 2)

add_executable(bench-quant-bw quant_bw.cpp)
target_include_directories(bench-quant-bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-quant-bw PRIVATE cxx_std_17)
target_link_libraries(bench-quant-bw PRIVATE reloc_runtime)
# Smoke-run under ctest (same rationale as the other bench smokes): proves
# every R0.1 kernel runs end-to-end -- including the byte-exactness gate
# against the serial scalar path -- on every CI run, at whatever SIMD tier
# the CI host supports (variant=auto). Tiny counts; not a measurement.
add_test(NAME bench-quant-bw-smoke
         COMMAND bench-quant-bw --kernel all --n 256 --threads 2
                 --variant auto --json - --warmup 1 --iters 3 --reruns 2)

# R0.3 (issue #76): the rtrack pipeline & measurement harness. The CPU-side
# pieces (plan builders, chunk math, stats, CSV, workload table) run under
# ctest on every CI run; the pipeline driver below is GPU-only and
# CUDA-gated like bench-poc-transpose.
add_executable(bench-rtrack-test rtrack/RtrackTest.cpp)
target_include_directories(bench-rtrack-test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-rtrack-test PRIVATE cxx_std_17)
target_link_libraries(bench-rtrack-test PRIVATE reloc_runtime llvm_gtest
  llvm_gtest_main)
add_test(NAME bench-rtrack-test COMMAND bench-rtrack-test)

if(RELOC_ENABLE_CUDA)
  # The C7 PoC driver. GPU-only, never in CI (mirrors libreloc's CUDA
  # gating); run locally on the desktop per docs/poc-reproduction.md.
  # enable_language(CUDA) and find_package(CUDAToolkit) are re-issued here
  # (libreloc/CMakeLists.txt already called both) because CMake scopes
  # per-language build rules and imported targets to the directory that
  # created them and its descendants, not to sibling directories:
  # libreloc/ and bench/ are siblings under the root CMakeLists.txt, so
  # neither is visible here without this. Both calls are safe to repeat
  # (enable_language no-ops after first detection; FindCUDAToolkit.cmake
  # guards each add_library() with if(NOT TARGET ...)).
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(bench-poc-transpose poc_transpose.cu)
  target_include_directories(bench-poc-transpose
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(bench-poc-transpose PRIVATE cxx_std_17)
  target_link_libraries(bench-poc-transpose PRIVATE reloc_runtime
    CUDA::cudart)

  # R0.3 pipeline driver (issue #76). GPU-only, never in CI; run locally
  # per bench/rtrack/README.md.
  add_executable(bench-rtrack rtrack/rtrack_bench.cu)
  target_include_directories(bench-rtrack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(bench-rtrack PRIVATE cxx_std_17)
  target_link_libraries(bench-rtrack PRIVATE reloc_runtime CUDA::cudart)
endif()
