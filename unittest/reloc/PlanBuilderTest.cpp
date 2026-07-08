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

} // namespace
