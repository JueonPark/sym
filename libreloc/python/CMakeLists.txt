# Pure observation is available even when pybind11/runtime bindings are absent.
foreach(package pyreloc reloc_torch)
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
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS
    "pyreloc: pybind11 not found -- Python bindings disabled "
    "(pip install pybind11 and configure with -Dpybind11_DIR=...)")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

pybind11_add_module(pyreloc_ext MODULE PyReloc.cpp PyTransfer.cpp PyPrefold.cpp)
target_link_libraries(pyreloc_ext PRIVATE reloc_runtime)
set_target_properties(pyreloc_ext PROPERTIES
  OUTPUT_NAME _pyreloc
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyreloc)
