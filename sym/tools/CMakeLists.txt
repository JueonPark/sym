get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sym-opt
  SymOptMain.cpp
)

llvm_update_compile_flags(sym-opt)

target_include_directories(sym-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/IR
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/Transforms
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/Transforms
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/IR
)

target_link_libraries(sym-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRSymDialect
  MLIRRelocDialect
  MLIRSymTransforms
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(sym-opt)
