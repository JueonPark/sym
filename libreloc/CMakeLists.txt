# libreloc: the standalone plan-execution runtime (P2, issue #40).
#
# MLIR-free by contract: this target must never appear in the same
# link line as MLIR/LLVM libraries. The root CMake's global
# include_directories(${MLIR_INCLUDE_DIRS}) pollution is include-path
# only and tolerated for v0 (issue #41); the acceptance check — the
# reloc-runtime-mlir-free CTest test and the CI assertion step — is on
# the link interface of the built shared object.
option(RELOC_ENABLE_CUDA "Build libreloc's CUDA backend (requires the CUDA toolkit)" OFF)

add_library(reloc_runtime SHARED
  src/Version.cpp
  src/Decode.cpp
  src/Bind.cpp
  src/Execute.cpp
  src/CopyRun.cpp
  src/HostBackend.cpp
  src/PinnedBufferPool.cpp
  src/ChunkSchedule.cpp
  src/GatherPool.cpp
  src/Pipeline.cpp
  quant/Quant.cpp
)

target_compile_features(reloc_runtime PUBLIC cxx_std_17)

target_include_directories(reloc_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# R0.1 SIMD variants (issue #74): per-TU ISA flags + runtime dispatch via
# __builtin_cpu_supports, so the shared object runs on any x86-64 host
# (the 2080 Ti box's CPU may be AVX2-only). Non-x86 builds get scalar only.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_sources(reloc_runtime PRIVATE
    quant/QuantAVX2.cpp
    quant/QuantAVX512.cpp
  )
  set_source_files_properties(quant/QuantAVX2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(quant/QuantAVX512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw")
  target_compile_definitions(reloc_runtime PRIVATE RELOC_QUANT_HAVE_X86_SIMD=1)
endif()

if(RELOC_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(reloc_runtime PUBLIC RELOC_ENABLE_CUDA=1)
  target_sources(reloc_runtime PRIVATE cuda/CudaBackend.cu)
  set_source_files_properties(cuda/CudaBackend.cu PROPERTIES LANGUAGE CUDA)
  target_link_libraries(reloc_runtime PRIVATE CUDA::cudart)
endif()

add_subdirectory(test)
add_subdirectory(python)
