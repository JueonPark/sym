# Native SDK uses bin/lib; wheels contain the same tools in a private package.
include(CMakePackageConfigHelpers)
if(SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  if(NOT TARGET pyreloc_ext OR NOT TARGET reloc-run-artifact)
    message(FATAL_ERROR "Sym wheels require Python bindings and examples")
  endif()
  if(RELOC_ENABLE_CUDA)
    set(sym_variant cu126)
  else()
    set(sym_variant cpu)
  endif()
  if(NOT SKBUILD_PROJECT_VERSION_FULL MATCHES "[+]${sym_variant}$")
    message(FATAL_ERROR "Wheel version must match CUDA build variant (${sym_variant})")
  endif()
  if(NOT Python_VERSION VERSION_EQUAL "3.14.7" OR NOT Python_SOABI STREQUAL "cpython-314-x86_64-linux-gnu")
    message(FATAL_ERROR "Sym wheels require regular-GIL CPython 3.14.7 on Linux x86_64")
  endif()
  if(SYM_SOURCE_REVISION)
    set(SYM_EXPORT_REVISION "${SYM_SOURCE_REVISION}")
  else()
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE SYM_EXPORT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  file(READ ${PROJECT_SOURCE_DIR}/build_tools/llvm_version.txt sym_llvm_revision)
  string(STRIP "${sym_llvm_revision}" sym_llvm_revision)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/build_info.json.in
    ${PROJECT_BINARY_DIR}/_build_info.json @ONLY)
  install(FILES ${PROJECT_BINARY_DIR}/_build_info.json DESTINATION sym_reloc COMPONENT Python)
  set(sym_bin sym_reloc/_native/bin)
  set(sym_lib sym_reloc/_native/lib)
else()
  set(sym_bin bin)
  set(sym_lib lib)
endif()
set_target_properties(reloc_runtime PROPERTIES
  EXPORT_NAME runtime INSTALL_RPATH "$ORIGIN")
install(TARGETS reloc_runtime EXPORT SymRelocTargets
  LIBRARY DESTINATION ${sym_lib} COMPONENT Runtime)
foreach(tool sym-opt sym-reloc-export reloc-run-artifact)
  if(TARGET ${tool})
    set_target_properties(${tool} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    install(TARGETS ${tool} RUNTIME DESTINATION ${sym_bin} COMPONENT Tools)
  endif()
endforeach()
if(SKBUILD)
  set_target_properties(pyreloc_ext PROPERTIES
    INSTALL_RPATH "$ORIGIN/../sym_reloc/_native/lib")
  install(TARGETS pyreloc_ext LIBRARY DESTINATION pyreloc COMPONENT Python)
else()
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/libreloc/include/reloc
    DESTINATION include COMPONENT Development)
  install(EXPORT SymRelocTargets NAMESPACE SymReloc::
    DESTINATION lib/cmake/SymReloc COMPONENT Development)
  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/SymRelocConfig.cmake.in
    ${PROJECT_BINARY_DIR}/SymRelocConfig.cmake INSTALL_DESTINATION lib/cmake/SymReloc)
  install(FILES ${PROJECT_BINARY_DIR}/SymRelocConfig.cmake
    DESTINATION lib/cmake/SymReloc COMPONENT Development)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/libreloc/examples/recipes/
    DESTINATION share/sym/recipes COMPONENT Runtime)
endif()
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE
  DESTINATION ${sym_lib}/licenses/sym COMPONENT Runtime)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/llvm
  ${PROJECT_SOURCE_DIR}/third_party/pybind11
  DESTINATION ${sym_lib}/licenses COMPONENT Runtime)
