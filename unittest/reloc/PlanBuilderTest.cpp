//===- PlanBuilderTest.cpp - Index-table oracle tests for PlanBuilder -----===//
//
// Semantic oracle tests (P1b #B1): bind symbols to concrete values,
// enumerate index tables, and compare the folded plan's data movement
// against a NumPy-transpose reference on small shapes. IR text equality
// alone does not catch stride math errors.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocDialect.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "gtest/gtest.h"
#include <numeric>
#include <vector>

using namespace mlir;

namespace {

TEST(RelocUnitTestHarness, LoadsDialects) {
  MLIRContext context;
  context.loadDialect<sym::SymDialect, reloc::RelocDialect>();
  EXPECT_NE(context.getLoadedDialect<reloc::RelocDialect>(), nullptr);
  EXPECT_NE(context.getLoadedDialect<sym::SymDialect>(), nullptr);
}

//===----------------------------------------------------------------------===//
// Reference tensors (NumPy semantics, row-major)
//===----------------------------------------------------------------------===//

struct Tensor {
  SmallVector<int64_t> shape;
  std::vector<int64_t> data; // row-major
};

static int64_t product(ArrayRef<int64_t> values) {
  return std::accumulate(values.begin(), values.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t running = 1;
  for (int64_t k = static_cast<int64_t>(shape.size()) - 1; k >= 0; --k) {
    strides[k] = running;
    running *= shape[k];
  }
  return strides;
}

static Tensor iota(ArrayRef<int64_t> shape) {
  Tensor t;
  t.shape.assign(shape.begin(), shape.end());
  t.data.resize(product(shape));
  std::iota(t.data.begin(), t.data.end(), 0);
  return t;
}

/// NumPy-transpose reference: result dim k = operand dim perm[k];
/// result[i] = operand[j] where j[perm[k]] = i[k].
static Tensor transposeRef(const Tensor &t, ArrayRef<int64_t> perm) {
  Tensor result;
  for (int64_t source : perm)
    result.shape.push_back(t.shape[source]);
  result.data.resize(t.data.size());
  SmallVector<int64_t> oldStrides = rowMajorStrides(t.shape);
  SmallVector<int64_t> newStrides = rowMajorStrides(result.shape);
  SmallVector<int64_t> index(perm.size(), 0);
  for (size_t n = 0; n < t.data.size(); ++n) {
    int64_t newOff = 0, oldOff = 0;
    for (size_t k = 0; k < perm.size(); ++k) {
      newOff += index[k] * newStrides[k];
      oldOff += index[k] * oldStrides[perm[k]];
    }
    result.data[newOff] = t.data[oldOff];
    for (int64_t k = static_cast<int64_t>(perm.size()) - 1; k >= 0; --k) {
      if (++index[k] < result.shape[k])
        break;
      index[k] = 0;
    }
  }
  return result;
}

/// NumPy-reshape reference: a reshape of row-major data reinterprets the
/// same flat buffer under a new shape.
static Tensor reshapeRef(const Tensor &t, ArrayRef<int64_t> shape) {
  Tensor result;
  result.shape.assign(shape.begin(), shape.end());
  result.data = t.data;
  return result;
}

//===----------------------------------------------------------------------===//
// Plan evaluation (the index-table side of the oracle)
//===----------------------------------------------------------------------===//

static int64_t floorDiv(int64_t lhs, int64_t rhs) {
  int64_t quotient = lhs / rhs;
  return (lhs % rhs != 0 && (lhs < 0) != (rhs < 0)) ? quotient - 1 : quotient;
}

/// Evaluate a sym expression under concrete symbol bindings. Div/Mod use
/// floor semantics (sym's pinned semantics; matches affine floordiv/mod).
static int64_t evalExpr(Attribute expr,
                        const llvm::StringMap<int64_t> &bindings) {
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(expr))
    return constant.getValue();
  if (auto symbol = dyn_cast<sym::SymbolExprAttr>(expr)) {
    auto it = bindings.find(symbol.getName());
    EXPECT_TRUE(it != bindings.end())
        << "unbound symbol " << symbol.getName().str();
    return it == bindings.end() ? 0 : it->second;
  }
  auto binary = cast<sym::BinaryExprAttr>(expr);
  int64_t lhs = evalExpr(binary.getLhs(), bindings);
  int64_t rhs = evalExpr(binary.getRhs(), bindings);
  switch (binary.getOpcode()) {
  case sym::SymbolicExprOp::Add:
    return lhs + rhs;
  case sym::SymbolicExprOp::Sub:
    return lhs - rhs;
  case sym::SymbolicExprOp::Mul:
    return lhs * rhs;
  case sym::SymbolicExprOp::Div:
    return floorDiv(lhs, rhs);
  case sym::SymbolicExprOp::Mod:
    return lhs - floorDiv(lhs, rhs) * rhs;
  }
  llvm_unreachable("unknown SymbolicExprOp");
}

/// Apply `plan` to an iota source buffer: for every dst index over the axis
/// extents, dst[sum(i_k * dst_stride_k)] = src[src_offset +
/// sum(i_k * src_stride_k)]. src is iota, so each moved value IS its source
/// linear offset. Returns the dst buffer.
static std::vector<int64_t>
applyPlan(reloc::PlanAttr plan, const llvm::StringMap<int64_t> &bindings) {
  ArrayRef<reloc::AxisInfoAttr> axes = plan.getAxes();
  int64_t rank = static_cast<int64_t>(axes.size());
  SmallVector<int64_t> extents, srcStrides, dstStrides;
  for (reloc::AxisInfoAttr axis : axes) {
    extents.push_back(evalExpr(axis.getExtent(), bindings));
    srcStrides.push_back(evalExpr(axis.getSrcStride(), bindings));
    dstStrides.push_back(evalExpr(axis.getDstStride(), bindings));
  }
  int64_t srcOffset = evalExpr(plan.getSrc().getOffset(), bindings);
  std::vector<int64_t> dst(product(extents), -1);
  SmallVector<int64_t> index(rank, 0);
  for (size_t n = 0; n < dst.size(); ++n) {
    int64_t srcOff = srcOffset, dstOff = 0;
    for (int64_t k = 0; k < rank; ++k) {
      srcOff += index[k] * srcStrides[k];
      dstOff += index[k] * dstStrides[k];
    }
    dst[dstOff] = srcOff;
    for (int64_t k = rank - 1; k >= 0; --k) {
      if (++index[k] < extents[k])
        break;
      index[k] = 0;
    }
  }
  return dst;
}

//===----------------------------------------------------------------------===//
// Fixture
//===----------------------------------------------------------------------===//

class PlanBuilderTest : public ::testing::Test {
protected:
  PlanBuilderTest() {
    context.loadDialect<sym::SymDialect, reloc::RelocDialect>();
  }

  Attribute dim(int64_t value) {
    return sym::ConstantExprAttr::get(&context, value);
  }
  Attribute dim(StringRef name) {
    return sym::SymbolExprAttr::get(&context, name);
  }
  Attribute div(Attribute lhs, int64_t rhs) {
    return sym::getSimplifiedBinaryExpr(&context, sym::SymbolicExprOp::Div, lhs,
                                        dim(rhs));
  }
  Attribute mul(int64_t lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(&context, sym::SymbolicExprOp::Mul,
                                        dim(lhs), rhs);
  }
  sym::SymbolicTensorType makeType(ArrayRef<Attribute> shape) {
    return sym::SymbolicTensorType::get(&context, shape,
                                        Float32Type::get(&context));
  }
  reloc::PlanAttr finalize(const reloc::PlanBuilder &builder) {
    return builder.finalize(UnknownLoc::get(&context));
  }

  /// inverse o forward == id on the axes space.
  static void expectInverseInvertsForward(reloc::PlanAttr plan) {
    AffineMap forward = AffineMap::getPermutationMap(
        plan.getPerm().asArrayRef(), plan.getContext());
    EXPECT_TRUE(plan.getInverse().getValue().compose(forward).isIdentity());
  }

  MLIRContext context;
  llvm::StringMap<int64_t> noBindings;
};

TEST_F(PlanBuilderTest, IdentityPlanFinalizes) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4), dim(2)}));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({0, 1, 2}));
  EXPECT_TRUE(plan.getInverse().getValue().isIdentity());
  EXPECT_FALSE(plan.getNoCopy()); // no_copy detection is #B5, not #B1
  EXPECT_TRUE(plan.getPadFill().empty());
  EXPECT_TRUE(plan.getDivisibility().empty());
  // finalize emits per-axis contiguity: srcStride == 1 proven (B2).
  EXPECT_EQ(plan.getContiguity().asArrayRef(),
            ArrayRef<bool>({false, false, true}));
  expectInverseInvertsForward(plan);
  EXPECT_EQ(applyPlan(plan, noBindings), iota({6, 4, 2}).data);
}

TEST_F(PlanBuilderTest, SingleTransposeMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4), dim(2)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {2, 0, 1})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({2, 0, 1}));
  expectInverseInvertsForward(plan);
  EXPECT_EQ(applyPlan(plan, noBindings),
            transposeRef(iota({6, 4, 2}), {2, 0, 1}).data);
}

TEST_F(PlanBuilderTest, ComposedTransposesMatchOracle) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4), dim(2)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {2, 0, 1})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {0, 2, 1})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  // Composition: perm[k] <- old_perm[op_perm[k]] = [2, 1, 0].
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({2, 1, 0}));
  expectInverseInvertsForward(plan);
  Tensor expected =
      transposeRef(transposeRef(iota({6, 4, 2}), {2, 0, 1}), {0, 2, 1});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, InversePairComposesToIdentity) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4), dim(2)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {2, 0, 1})));
  ASSERT_TRUE(succeeded(
      reloc::foldTranspose(builder, {1, 2, 0}))); // inverse of [2, 0, 1]
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  // Identity elision: the composed perm IS the identity permutation, and
  // src/dst strides realign; #B5 turns this into no_copy.
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({0, 1, 2}));
  EXPECT_TRUE(plan.getInverse().getValue().isIdentity());
  EXPECT_TRUE(reloc::isPureView(plan));
  EXPECT_FALSE(plan.getNoCopy()); // still false until #B5
  EXPECT_EQ(applyPlan(plan, noBindings), iota({6, 4, 2}).data);
}

TEST_F(PlanBuilderTest, SymbolicExtentsPreservedVerbatim) {
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({dim(6), n, dim(2)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0, 2})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  // The symbol rides through the fold untouched (same uniqued attribute).
  EXPECT_EQ(plan.getAxes()[0].getExtent(), n);
  EXPECT_EQ(plan.getDst().getExtents()[0], n);
  expectInverseInvertsForward(plan);
  llvm::StringMap<int64_t> bindings;
  bindings["N"] = 4;
  EXPECT_EQ(applyPlan(plan, bindings),
            transposeRef(iota({6, 4, 2}), {1, 0, 2}).data);
}

TEST_F(PlanBuilderTest, StaticSplitMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(4096)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(64), dim(64)})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({0, 1}));
  EXPECT_TRUE(plan.getDivisibility().empty());
  EXPECT_EQ(plan.getAxes()[0].getSrcStride(), dim(64));
  EXPECT_EQ(plan.getAxes()[1].getSrcStride(), dim(1));
  // A contiguous reshape moves no data: dst equals the flat iota buffer.
  EXPECT_EQ(applyPlan(plan, noBindings), iota({4096}).data);
}

TEST_F(PlanBuilderTest, ThreeWaySplitMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(24)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(2), dim(3), dim(4)})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getAxes()[0].getSrcStride(), dim(12));
  EXPECT_EQ(plan.getAxes()[1].getSrcStride(), dim(4));
  EXPECT_EQ(plan.getAxes()[2].getSrcStride(), dim(1));
  EXPECT_EQ(applyPlan(plan, noBindings), iota({24}).data);
}

TEST_F(PlanBuilderTest, StaticMergeMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(8), dim(128)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(1024)})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getAxes().size(), 1u);
  EXPECT_EQ(plan.getAxes()[0].getSrcStride(), dim(1));
  EXPECT_EQ(applyPlan(plan, noBindings), iota({1024}).data);
}

TEST_F(PlanBuilderTest, RegroupMatchesOracle) {
  // m:n group: [4, 6] -> [2, 12] is merge-then-split; a trailing transpose
  // forces real data movement through the regrouped strides.
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(2), dim(12)})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected = transposeRef(reshapeRef(iota({4, 6}), {2, 12}), {1, 0});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, ReshapeThenTransposeMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(4096)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(64), dim(64)})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected = transposeRef(reshapeRef(iota({4096}), {64, 64}), {1, 0});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, TrailingUnitDimsFold) {
  reloc::PlanBuilder dropOne(makeType({dim(6), dim(1)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(dropOne, {dim(6)})));
  reloc::PlanAttr dropped = finalize(dropOne);
  ASSERT_TRUE(dropped);
  EXPECT_EQ(applyPlan(dropped, noBindings), iota({6}).data);

  reloc::PlanBuilder addOne(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(addOne, {dim(6), dim(1)})));
  reloc::PlanAttr added = finalize(addOne);
  ASSERT_TRUE(added);
  EXPECT_EQ(applyPlan(added, noBindings), iota({6}).data);
}

TEST_F(PlanBuilderTest, NonContiguousMergeBails) {
  // Transposed data is not mergeable: outer src stride 1 != 6 * 4.
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(24)})));
  // Bail leaves the builder untouched: the transpose-only plan still
  // finalizes and its oracle still holds.
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan.getPerm().asArrayRef(), ArrayRef<int64_t>({1, 0}));
  EXPECT_EQ(applyPlan(plan, noBindings),
            transposeRef(iota({4, 6}), {1, 0}).data);
}

TEST_F(PlanBuilderTest, ElementCountMismatchBails) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(4)})));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(4), dim(2)})));
  EXPECT_EQ(builder.axes.size(), 1u);
}

TEST_F(PlanBuilderTest, SymbolicSplitEmitsDivisibility) {
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({n, dim(4)}));
  ASSERT_TRUE(
      succeeded(reloc::foldReshape(builder, {div(n, 8), dim(8), dim(4)})));
  ASSERT_EQ(builder.divisibility.size(), 1u);
  EXPECT_EQ(builder.divisibility[0].getExpr(), n);
  EXPECT_EQ(builder.divisibility[0].getDivisor(), 8);
  // Exercise the plan with a binding where the divisibility holds.
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {2, 0, 1})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  llvm::StringMap<int64_t> bindings;
  bindings["N"] = 48;
  Tensor expected =
      transposeRef(reshapeRef(iota({48, 4}), {6, 8, 4}), {2, 0, 1});
  EXPECT_EQ(applyPlan(plan, bindings), expected.data);
}

TEST_F(PlanBuilderTest, SymbolicMergeWhenContiguityProvable) {
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({n, dim(4)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {mul(4, n)})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_TRUE(plan.getDivisibility().empty());
  EXPECT_EQ(plan.getContiguity().asArrayRef(), ArrayRef<bool>({true}));
  llvm::StringMap<int64_t> bindings;
  bindings["N"] = 6;
  EXPECT_EQ(applyPlan(plan, bindings), iota({24}).data);
}

TEST_F(PlanBuilderTest, TwoSymbolicSplitEntriesBail) {
  reloc::PlanBuilder builder(makeType({dim("N")}));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim("M"), dim("K")})));
  EXPECT_EQ(builder.axes.size(), 1u);
}

TEST_F(PlanBuilderTest, UnprovableSymbolicMergeBails) {
  Attribute n = dim("N"), m = dim("M");
  reloc::PlanBuilder builder(makeType({n, m}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  EXPECT_TRUE(failed(reloc::foldReshape(
      builder, {sym::getSimplifiedBinaryExpr(&context, sym::SymbolicExprOp::Mul,
                                             m, n)})));
  EXPECT_EQ(builder.axes.size(), 2u);
}

TEST_F(PlanBuilderTest, DivisibilityDeduplicated) {
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({n, n}));
  ASSERT_TRUE(succeeded(
      reloc::foldReshape(builder, {div(n, 64), dim(64), div(n, 64), dim(64)})));
  EXPECT_EQ(builder.divisibility.size(), 1u);
}

TEST_F(PlanBuilderTest, ReferenceChainMatchesBuildDocConstraintSet) {
  // Build doc §2.1 acceptance: [N, N] -> blocked 4D view + transpose;
  // constraint set must be exactly {divisible(N, 64),
  // contiguous = [false, false, false, true], no_copy = false}.
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({n, n}));
  ASSERT_TRUE(succeeded(
      reloc::foldReshape(builder, {div(n, 64), dim(64), div(n, 64), dim(64)})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {2, 0, 1, 3})));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan.getDivisibility().size(), 1u);
  EXPECT_EQ(plan.getDivisibility()[0].getExpr(), n);
  EXPECT_EQ(plan.getDivisibility()[0].getDivisor(), 64);
  EXPECT_EQ(plan.getContiguity().asArrayRef(),
            ArrayRef<bool>({false, false, false, true}));
  EXPECT_FALSE(plan.getNoCopy());
  EXPECT_TRUE(plan.getPadFill().empty());
  llvm::StringMap<int64_t> bindings;
  bindings["N"] = 128;
  Tensor expected =
      transposeRef(reshapeRef(iota({128, 128}), {2, 64, 2, 64}), {2, 0, 1, 3});
  EXPECT_EQ(applyPlan(plan, bindings), expected.data);
}

} // namespace
