cmake_minimum_required(VERSION 3.20)
project(padded_h2d_campaign LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(RELOC_ENABLE_CUDA ON CACHE BOOL "")
find_package(CUDAToolkit REQUIRED)
if(SYM_BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
  add_library(llvm_gtest ALIAS GTest::gtest)
  add_library(llvm_gtest_main ALIAS GTest::gtest_main)
  enable_testing()
endif()
add_subdirectory("${RUNTIME_ROOT}/libreloc" libreloc)

add_executable(transfer transfer.cpp)
target_link_libraries(transfer PRIVATE reloc_runtime CUDA::cudart)

# Build a separate diagnostic executable with timers around the actual fill
# call. The primary executable uses the regular shared runtime.
set(pipeline "${RUNTIME_ROOT}/libreloc/src/Pipeline.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${pipeline}")
file(READ "${pipeline}" source)
set(call "fillStagingWindow(bound, staging, c.bytes);")
string(FIND "${source}" "${call}" position)
if(position EQUAL -1)
  message(FATAL_ERROR "Staging instrumentation point missing")
endif()
string(REPLACE "${call}" "auto startFill = std::chrono::steady_clock::now();
    ${call}
    phaseNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - startFill).count();
    ++phaseCalls;" instrumented "${source}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/phases.cpp"
     "#include <chrono>\nlong long phaseNs = 0, phaseCalls = 0;\n${instrumented}")
add_executable(phases transfer.cpp "${CMAKE_CURRENT_BINARY_DIR}/phases.cpp")
target_include_directories(phases PRIVATE "${RUNTIME_ROOT}/libreloc/src")
target_compile_definitions(phases PRIVATE MEASURE_PHASES)
target_link_libraries(phases PRIVATE reloc_runtime CUDA::cudart)
