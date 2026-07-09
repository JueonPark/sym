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
)

target_compile_features(reloc_runtime PUBLIC cxx_std_17)

target_include_directories(reloc_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(RELOC_ENABLE_CUDA)
  # #C5 adds cuda/*.cu translation units here; until then the option only
  # wires the dependency and the feature macro.
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(reloc_runtime PUBLIC RELOC_ENABLE_CUDA=1)
  target_link_libraries(reloc_runtime PRIVATE CUDA::cudart)
endif()

add_subdirectory(test)
