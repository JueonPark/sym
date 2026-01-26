# TableGen for Dialect
set(LLVM_TARGET_DEFINITIONS SymDialect.td)
mlir_tablegen(SymDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SymDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(SymDialectIncGen)

# TableGen for Types
set(LLVM_TARGET_DEFINITIONS SymTypes.td)
mlir_tablegen(SymTypes.h.inc -gen-typedef-decls)
mlir_tablegen(SymTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(SymTypesIncGen)

# TableGen for Attributes
set(LLVM_TARGET_DEFINITIONS SymAttrs.td)
mlir_tablegen(SymAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SymAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(SymAttributesIncGen)

# TableGen for Operations
set(LLVM_TARGET_DEFINITIONS SymOps.td)
mlir_tablegen(SymOps.h.inc -gen-op-decls)
mlir_tablegen(SymOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(SymOpsIncGen)

# Combined TableGen target
add_custom_target(SymIncGen DEPENDS 
  SymDialectIncGen 
  SymTypesIncGen 
  SymAttributesIncGen
  SymOpsIncGen
)

# Dialect Library
add_mlir_dialect_library(MLIRSymDialect
  SymDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym

  DEPENDS
  SymIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Add include path for generated files
target_include_directories(MLIRSymDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
