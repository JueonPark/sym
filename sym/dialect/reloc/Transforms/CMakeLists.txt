# TableGen for Passes
set(LLVM_TARGET_DEFINITIONS RelocPasses.td)
mlir_tablegen(RelocPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(RelocPassesIncGen)

# Transforms Library
add_mlir_library(MLIRRelocTransforms
  PlanBuilder.cpp
  TestRelocUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/Transforms
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR

  DEPENDS
  RelocPassesIncGen
  RelocIRIncGen

  LINK_LIBS PUBLIC
  MLIRRelocDialect
  MLIRSymDialect
  MLIRPass
)

# Add include path for generated files and IR headers
target_include_directories(MLIRRelocTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/IR
)
