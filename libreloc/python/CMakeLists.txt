# Pure observation is available even when pybind11/runtime bindings are absent.
foreach(package pyreloc reloc_torch sym_reloc)
  file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${package}/*.py)
  foreach(src ${sources})
    get_filename_component(fname ${src} NAME)
    configure_file(${src} ${CMAKE_BINARY_DIR}/python/${package}/${fname} COPYONLY)
  endforeach()
endforeach()
set(PYBIND11_FINDPYTHON ON)
# pyreloc: the pybind11 extension (issue #46). Optional by design -- a
# build without pybind11 (e.g. a bare local checkout) skips it with a
# notice and everything else still builds; CI installs pybind11 and its
# test job runs an explicit `python3 -c "import pyreloc"` gate before
# pytest, so a missing module fails CI loudly instead of converting the
# suite (which importorskips pyreloc) into silent skips.
if(SKBUILD)
  find_package(pybind11 CONFIG REQUIRED)
else()
  find_package(pybind11 CONFIG QUIET)
endif()
if(NOT pybind11_FOUND)
  message(STATUS
    "pyreloc: pybind11 not found -- Python bindings disabled "
    "(pip install pybind11 and configure with -Dpybind11_DIR=...)")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

pybind11_add_module(pyreloc_ext MODULE PyReloc.cpp PyTransfer.cpp PyPrefold.cpp PyDispatch.cpp)
target_link_libraries(pyreloc_ext PRIVATE reloc_runtime)
set_target_properties(pyreloc_ext PROPERTIES
  OUTPUT_NAME _pyreloc
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyreloc)

# Optional libtorch-backed request builders for reloc_torch.fast. Off by
# default: it binds the module to the exact Torch the build found, so wheels
# and Torch-free installs never carry it; reloc_torch.fast falls back to its
# pure-Python descriptor path when the module is absent.
option(RELOC_ENABLE_TORCH_EXT "Build _pyreloc_torch against the configured Torch" OFF)
if(RELOC_ENABLE_TORCH_EXT)
  # Locate the Torch package of the configured interpreter directly: its
  # TorchConfig.cmake insists on a matching CUDA toolkit scan that this
  # metadata-only module does not need. Only CPU-side tensor metadata is read.
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE RELOC_TORCH_ROOT RESULT_VARIABLE torch_status)
  if(NOT torch_status EQUAL 0 OR NOT EXISTS "${RELOC_TORCH_ROOT}/include/torch/csrc/autograd/python_variable.h")
    message(FATAL_ERROR "RELOC_ENABLE_TORCH_EXT: torch is not importable from ${Python_EXECUTABLE}")
  endif()
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
    OUTPUT_VARIABLE RELOC_TORCH_CXX11_ABI)
  pybind11_add_module(pyreloc_torch_ext MODULE PyTorchViews.cpp)
  # Torch ships its own pybind11; the helper must see the project's pybind11
  # first so its types live in the same registry as _pyreloc's.
  target_include_directories(pyreloc_torch_ext BEFORE PRIVATE ${pybind11_INCLUDE_DIRS})
  target_include_directories(pyreloc_torch_ext SYSTEM PRIVATE
    "${RELOC_TORCH_ROOT}/include" "${RELOC_TORCH_ROOT}/include/torch/csrc/api/include")
  target_link_directories(pyreloc_torch_ext PRIVATE "${RELOC_TORCH_ROOT}/lib")
  target_link_libraries(pyreloc_torch_ext PRIVATE reloc_runtime torch torch_cpu c10 torch_python)
  if(RELOC_ENABLE_CUDA)
    target_link_libraries(pyreloc_torch_ext PRIVATE CUDA::cudart)
  endif()
  target_compile_definitions(pyreloc_torch_ext PRIVATE _GLIBCXX_USE_CXX11_ABI=${RELOC_TORCH_CXX11_ABI})
  target_compile_features(pyreloc_torch_ext PRIVATE cxx_std_20) # ATen headers require C++20
  set_target_properties(pyreloc_torch_ext PROPERTIES
    OUTPUT_NAME _pyreloc_torch
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyreloc
    BUILD_RPATH "${RELOC_TORCH_ROOT}/lib"
    INSTALL_RPATH "${RELOC_TORCH_ROOT}/lib")
endif()

if(SYM_BUILD_TESTS)
  pybind11_add_module(typed_host_test MODULE tests/TypedHostHarness.cpp)
  target_link_libraries(typed_host_test PRIVATE reloc_runtime)
  set_target_properties(typed_host_test PROPERTIES
    OUTPUT_NAME _typed_host_test
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
  # Fault injection is available only in the build tree, never in installed wheels.
  pybind11_add_module(pyreloc_transfer_test MODULE tests/TransferResourceHarness.cpp)
  target_link_libraries(pyreloc_transfer_test PRIVATE reloc_runtime)
  set_target_properties(pyreloc_transfer_test PROPERTIES
    OUTPUT_NAME _reloc_transfer_test
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
  add_test(NAME pyreloc-transfer-harness-import
    COMMAND ${Python_EXECUTABLE} -c "import _reloc_transfer_test")
  set_tests_properties(pyreloc-transfer-harness-import PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endif()

# Local CUDA timing/fault qualification; excluded from wheels/install targets.
if(SYM_BUILD_TESTS AND RELOC_ENABLE_CUDA AND UNIX AND NOT APPLE)
  pybind11_add_module(cuda_kernel_test MODULE tests/CudaKernelHarness.cpp)
  target_link_libraries(cuda_kernel_test PRIVATE reloc_runtime CUDA::cudart)
  set_target_properties(cuda_kernel_test PROPERTIES
    OUTPUT_NAME _cuda_kernel_test
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
  add_library(typed_dispatch_faults SHARED tests/DispatchFaultShim.cpp)
  target_include_directories(typed_dispatch_faults PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(typed_dispatch_faults PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(typed_dispatch_faults PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
endif()
