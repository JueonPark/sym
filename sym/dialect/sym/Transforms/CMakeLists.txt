# TableGen for Passes
set(LLVM_TARGET_DEFINITIONS SymPasses.td)
mlir_tablegen(SymPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(SymPassesIncGen)

# Transforms Library
add_mlir_library(MLIRSymTransforms
  SymbolicShapeInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/Transforms
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/IR

  DEPENDS
  SymPassesIncGen
  SymIRIncGen

  LINK_LIBS PUBLIC
  MLIRSymDialect
  MLIRFuncDialect
  MLIRPass
)

# Add include path for generated files and IR headers
target_include_directories(MLIRSymTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/IR
)
