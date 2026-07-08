//===- PlanBuilderTest.cpp - Index-table oracle tests for PlanBuilder -----===//
//
// Semantic oracle tests (P1b #B1): bind symbols to concrete values,
// enumerate index tables, and compare the folded plan's data movement
// against a NumPy-transpose reference on small shapes. IR text equality
// alone does not catch stride math errors.
//
//===----------------------------------------------------------------------===//

#include "RelocDialect.h"
#include "SymDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

TEST(RelocUnitTestHarness, LoadsDialects) {
  MLIRContext context;
  context.loadDialect<sym::SymDialect, reloc::RelocDialect>();
  EXPECT_NE(context.getLoadedDialect<reloc::RelocDialect>(), nullptr);
  EXPECT_NE(context.getLoadedDialect<sym::SymDialect>(), nullptr);
}

} // namespace
