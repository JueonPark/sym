get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sym-opt
  SymOptMain.cpp
  PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(sym-opt)

target_include_directories(sym-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/IR
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/Transforms
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/Transforms
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/IR
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/Transforms
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/Transforms
)

target_link_libraries(sym-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRSymDialect
  MLIRRelocDialect
  MLIRSymTransforms
  MLIRRelocTransforms
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(sym-opt)

add_llvm_executable(sym-reloc-export RelocExportMain.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(sym-reloc-export)
get_target_property(sym_tool_includes sym-opt INCLUDE_DIRECTORIES)
target_include_directories(sym-reloc-export PRIVATE ${sym_tool_includes})
if(SYM_SOURCE_REVISION)
  set(SYM_EXPORT_REVISION "${SYM_SOURCE_REVISION}")
else()
execute_process(COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE SYM_EXPORT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
endif()
if(NOT SYM_EXPORT_REVISION)
  set(SYM_EXPORT_REVISION "source-archive")
endif()
set(SYM_EXPORT_REVISION "${SYM_EXPORT_REVISION}" PARENT_SCOPE)
target_compile_definitions(sym-reloc-export PRIVATE
  SYM_EXPORT_BUILD_ID="${SYM_EXPORT_REVISION}/${CMAKE_BUILD_TYPE}")
target_link_libraries(sym-reloc-export PRIVATE
  MLIRSymDialect MLIRRelocDialect MLIRRelocTransforms
  MLIRFuncDialect MLIRArithDialect MLIRParser MLIRPass MLIRIR MLIRSupport)
mlir_check_all_link_libraries(sym-reloc-export)
