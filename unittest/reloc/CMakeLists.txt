add_llvm_executable(reloc-plan-builder-test
  PlanBuilderTest.cpp
)

llvm_update_compile_flags(reloc-plan-builder-test)

target_include_directories(reloc-plan-builder-test PRIVATE
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/IR
  ${PROJECT_SOURCE_DIR}/sym/dialect/reloc/Transforms
  ${PROJECT_BINARY_DIR}/sym/dialect/reloc/Transforms
  ${PROJECT_SOURCE_DIR}/sym/dialect/sym/IR
  ${PROJECT_BINARY_DIR}/sym/dialect/sym/IR
)

target_link_libraries(reloc-plan-builder-test PRIVATE
  MLIRRelocTransforms
  MLIRRelocDialect
  MLIRSymDialect
  MLIRIR
  MLIRSupport
  llvm_gtest
  llvm_gtest_main
)
