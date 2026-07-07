# TableGen for Dialect
set(LLVM_TARGET_DEFINITIONS RelocDialect.td)
mlir_tablegen(RelocDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RelocDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RelocDialectIncGen)

# Combined TableGen target for IR
add_custom_target(RelocIRIncGen DEPENDS
  RelocDialectIncGen
)

# Dialect Library
add_mlir_dialect_library(MLIRRelocDialect
  RelocDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR

  DEPENDS
  RelocIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Add include path for generated files
target_include_directories(MLIRRelocDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
