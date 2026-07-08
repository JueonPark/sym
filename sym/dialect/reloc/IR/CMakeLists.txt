# TableGen for Dialect
set(LLVM_TARGET_DEFINITIONS RelocDialect.td)
mlir_tablegen(RelocDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RelocDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RelocDialectIncGen)

# TableGen for Attributes
set(LLVM_TARGET_DEFINITIONS RelocAttrs.td)
mlir_tablegen(RelocAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(RelocAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(RelocAttributesIncGen)

# TableGen for Operations
set(LLVM_TARGET_DEFINITIONS RelocOps.td)
mlir_tablegen(RelocOps.h.inc -gen-op-decls)
mlir_tablegen(RelocOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RelocOpsIncGen)

# Combined TableGen target for IR
add_custom_target(RelocIRIncGen DEPENDS
  RelocDialectIncGen
  RelocAttributesIncGen
  RelocOpsIncGen
)

# Dialect Library
add_mlir_dialect_library(MLIRRelocDialect
  RelocAttributes.cpp
  RelocDialect.cpp
  RelocOps.cpp
  RelocSerialization.cpp
  RelocUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR

  DEPENDS
  RelocIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSymDialect
)

# Add include path for generated files
target_include_directories(MLIRRelocDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
