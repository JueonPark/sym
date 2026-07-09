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
#include "RelocSerialization.h"
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

/// NumPy-pad reference: grows `axis` by `lo` leading and `hi` trailing
/// cells that keep the -1 sentinel (the oracle checks pad GEOMETRY; fill
/// values are compared at the attribute level).
static Tensor padRef(const Tensor &t, int64_t axis, int64_t lo, int64_t hi) {
  Tensor result;
  result.shape.assign(t.shape.begin(), t.shape.end());
  result.shape[axis] += lo + hi;
  result.data.assign(product(result.shape), -1);
  SmallVector<int64_t> oldStrides = rowMajorStrides(t.shape);
  SmallVector<int64_t> newStrides = rowMajorStrides(result.shape);
  SmallVector<int64_t> index(t.shape.size(), 0);
  for (size_t n = 0; n < t.data.size(); ++n) {
    int64_t oldOff = 0, newOff = 0;
    for (size_t k = 0; k < t.shape.size(); ++k) {
      int64_t shifted = index[k] + (static_cast<int64_t>(k) == axis ? lo : 0);
      oldOff += index[k] * oldStrides[k];
      newOff += shifted * newStrides[k];
    }
    result.data[newOff] = t.data[oldOff];
    for (int64_t k = static_cast<int64_t>(t.shape.size()) - 1; k >= 0; --k) {
      if (++index[k] < t.shape[k])
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
  // Pads: valid data lands at index + lo on padded axes; dst cells outside
  // the valid region keep the -1 sentinel. Without pads this reduces to
  // the previous behavior (dst extents == axis extents, lo == 0).
  SmallVector<int64_t> lo(rank, 0);
  for (reloc::PadFillAttr pad : plan.getPadFill())
    lo[pad.getDstAxis()] = evalExpr(pad.getLo(), bindings);
  SmallVector<int64_t> paddedExtents;
  for (Attribute extent : plan.getDst().getExtents())
    paddedExtents.push_back(evalExpr(extent, bindings));
  std::vector<int64_t> dst(product(paddedExtents), -1);
  SmallVector<int64_t> index(rank, 0);
  const size_t total = static_cast<size_t>(product(extents));
  for (size_t n = 0; n < total; ++n) {
    int64_t srcOff = srcOffset, dstOff = 0;
    for (int64_t k = 0; k < rank; ++k) {
      srcOff += index[k] * srcStrides[k];
      dstOff += (index[k] + lo[k]) * dstStrides[k];
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
  TypedAttr fill(double value) {
    return FloatAttr::get(Float32Type::get(&context), value);
  }
  Attribute alignUpPad(Attribute extent, int64_t align) {
    // ((extent + align - 1) floordiv align) * align - extent
    Attribute sum = sym::getSimplifiedBinaryExpr(
        &context, sym::SymbolicExprOp::Add, extent, dim(align - 1));
    Attribute quotient = sym::getSimplifiedBinaryExpr(
        &context, sym::SymbolicExprOp::Div, sum, dim(align));
    Attribute rounded = sym::getSimplifiedBinaryExpr(
        &context, sym::SymbolicExprOp::Mul, quotient, dim(align));
    return sym::getSimplifiedBinaryExpr(&context, sym::SymbolicExprOp::Sub,
                                        rounded, extent);
  }
  sym::SymbolicTensorType makeType(ArrayRef<Attribute> shape) {
    return sym::SymbolicTensorType::get(&context, shape,
                                        Float32Type::get(&context));
  }
  reloc::PlanAttr finalize(const reloc::PlanBuilder &builder) {
    return builder.finalize(UnknownLoc::get(&context));
  }
  reloc::PlanAttr canonical(reloc::PlanAttr raw) {
    reloc::PlanAttr canon =
        reloc::canonicalizePlan(raw, UnknownLoc::get(&context));
    EXPECT_TRUE(canon);
    return canon;
  }
  std::vector<uint8_t> encoded(reloc::PlanAttr plan) {
    FailureOr<std::vector<uint8_t>> bytes =
        reloc::encodePlan(plan, UnknownLoc::get(&context));
    EXPECT_TRUE(succeeded(bytes));
    return succeeded(bytes) ? *bytes : std::vector<uint8_t>();
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

TEST_F(PlanBuilderTest, DivisibilityDeduplicatedAcrossFolds) {
  // The same divisibility fact can be re-derived by a later, independent
  // fold on the same builder; it must dedup against plan.divisibility (not
  // just within a single fold's own emissions).
  Attribute n = dim("N");
  reloc::PlanBuilder builder(makeType({n, n}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {div(n, 64), dim(64), n})));
  EXPECT_EQ(builder.divisibility.size(), 1u);
  // The first two entries are 1:1 keeps of the axes just split off above;
  // the trailing `n` splits again, emitting divisible(N, 64) a second time.
  ASSERT_TRUE(succeeded(
      reloc::foldReshape(builder, {div(n, 64), dim(64), div(n, 64), dim(64)})));
  EXPECT_EQ(builder.divisibility.size(), 1u);
}

TEST_F(PlanBuilderTest, NonPositiveExtentBails) {
  // The frozen P1a reshape verifier accepts [24] -> [-8, -3] (matching
  // element count); foldReshape must still bail rather than fold negative
  // extents/strides into the plan.
  reloc::PlanBuilder builder(makeType({dim(24)}));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(-8), dim(-3)})));
  EXPECT_EQ(builder.axes.size(), 1u);
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
  EXPECT_TRUE(plan.getAlignment().empty());
  EXPECT_FALSE(plan.getRuntimePadCheck());
  llvm::StringMap<int64_t> bindings;
  bindings["N"] = 128;
  Tensor expected =
      transposeRef(reshapeRef(iota({128, 128}), {2, 64, 2, 64}), {2, 0, 1, 3});
  EXPECT_EQ(applyPlan(plan, bindings), expected.data);
}

TEST_F(PlanBuilderTest, PadMatchesOracle) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(1), fill(1.0))));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan.getPadFill().size(), 1u);
  EXPECT_EQ(plan.getPadFill()[0].getDstAxis(), 0);
  EXPECT_EQ(plan.getPadFill()[0].getValue(), fill(1.0));
  EXPECT_FALSE(plan.getRuntimePadCheck()); // static pad range is Proven
  EXPECT_EQ(plan.getAxes()[0].getExtent(), dim(6)); // valid extent kept
  EXPECT_EQ(applyPlan(plan, noBindings), padRef(iota({6}), 0, 1, 1).data);
}

TEST_F(PlanBuilderTest, RepeatedSameValuePadsMerge) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(0), fill(0.0))));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(2), dim(3), fill(0.0))));
  ASSERT_EQ(builder.pads.size(), 1u); // merged, not stacked
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  // Second pad wraps the first: total lo = 3, hi = 3.
  Tensor expected = padRef(padRef(iota({6}), 0, 1, 0), 0, 2, 3);
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, DifferentFillValueSecondPadBails) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(0), fill(0.0))));
  EXPECT_TRUE(failed(reloc::foldPad(builder, 0, dim(0), dim(1), fill(1.0))));
  EXPECT_EQ(builder.pads.size(), 1u); // untouched by the bail
}

TEST_F(PlanBuilderTest, ZeroPadIsNoOp) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(0), dim(0), fill(0.0))));
  EXPECT_TRUE(builder.pads.empty());
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_TRUE(plan.getPadFill().empty());
  EXPECT_EQ(applyPlan(plan, noBindings), iota({6}).data);
}

TEST_F(PlanBuilderTest, NegativePadWidthBails) {
  reloc::PlanBuilder builder(makeType({dim(6)}));
  EXPECT_TRUE(failed(reloc::foldPad(builder, 0, dim(-1), dim(0), fill(0.0))));
  EXPECT_TRUE(failed(reloc::foldPad(builder, 5, dim(1), dim(0), fill(0.0))));
  EXPECT_TRUE(builder.pads.empty());
}

TEST_F(PlanBuilderTest, AlignmentPadFoldsAndBinds) {
  // Issue #23 acceptance: hi = align_up(N, A) - N for A in {64, 128} on a
  // non-multiple extent; the pad range is symbolically Unknown, so the
  // plan must carry runtime_pad_check.
  for (int64_t align : {int64_t(64), int64_t(128)}) {
    Attribute n = dim("N");
    reloc::PlanBuilder builder(makeType({n}));
    ASSERT_TRUE(succeeded(
        reloc::foldPad(builder, 0, dim(0), alignUpPad(n, align), fill(0.0))));
    reloc::PlanAttr plan = finalize(builder);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan.getRuntimePadCheck());
    llvm::StringMap<int64_t> bindings;
    bindings["N"] = 150; // align_up: 192 (A=64) / 256 (A=128)
    int64_t hi = align == 64 ? 42 : 106;
    EXPECT_EQ(applyPlan(plan, bindings), padRef(iota({150}), 0, 0, hi).data);
  }
}

TEST_F(PlanBuilderTest, PadThenTransposeRenumbersPad) {
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 1, dim(0), dim(2), fill(0.0))));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  ASSERT_EQ(builder.pads.size(), 1u);
  EXPECT_EQ(builder.pads[0].axis, 0); // padded axis moved to dst position 0
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected = transposeRef(padRef(iota({4, 6}), 1, 0, 2), {1, 0});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, PadThenSplitOnPaddedAxisBails) {
  // Issue #15 design decision 3: the valid region is not expressible
  // per-axis after splitting a padded axis.
  reloc::PlanBuilder builder(makeType({dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(1), fill(0.0))));
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(2), dim(4)})));
  // Bail leaves the builder untouched: the pad-only plan still holds.
  EXPECT_EQ(builder.axes.size(), 1u);
  EXPECT_EQ(builder.pads.size(), 1u);
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  EXPECT_EQ(applyPlan(plan, noBindings), padRef(iota({6}), 0, 1, 1).data);
}

TEST_F(PlanBuilderTest, PadThenMergeWithPaddedAxisBails) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(1), fill(0.0))));
  // Padded view is [8, 4]; merging to [32] would fold the pad into the
  // merged axis -> bail.
  EXPECT_TRUE(failed(reloc::foldReshape(builder, {dim(32)})));
  EXPECT_EQ(builder.axes.size(), 2u);
}

TEST_F(PlanBuilderTest, PadSurvivesReshapeKeepAndRemaps) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(1), fill(0.0))));
  // Padded view is [8, 4]; the reshape keeps the padded axis 1:1 (target
  // entry 8 == padded extent) and splits the unpadded 4 into (2, 2).
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(8), dim(2), dim(2)})));
  ASSERT_EQ(builder.pads.size(), 1u);
  EXPECT_EQ(builder.pads[0].axis, 0);
  EXPECT_EQ(builder.axes[0].extent, dim(6)); // valid extent preserved
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected = reshapeRef(padRef(iota({6, 4}), 0, 1, 1), {8, 2, 2});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, PadRemapsWhenPrecedingAxesMerge) {
  // Merge (2, 3) -> 6 before a padded 1:1-kept axis: the pad's index must
  // remap from 2 to 1.
  reloc::PlanBuilder builder(makeType({dim(2), dim(3), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 2, dim(1), dim(1), fill(0.0))));
  ASSERT_TRUE(succeeded(reloc::foldReshape(builder, {dim(6), dim(8)})));
  ASSERT_EQ(builder.pads.size(), 1u);
  EXPECT_EQ(builder.pads[0].axis, 1);
  EXPECT_EQ(builder.axes[1].extent, dim(6)); // valid extent preserved
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected = reshapeRef(padRef(iota({2, 3, 6}), 2, 1, 1), {6, 8});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, TwoPadsOnDifferentAxesRenumberThroughTranspose) {
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(0), fill(0.0))));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 1, dim(0), dim(2), fill(0.0))));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  ASSERT_EQ(builder.pads.size(), 2u);
  // The axis-0 pad (lo 1) moved to dst position 1; the axis-1 pad (hi 2)
  // moved to dst position 0.
  const reloc::PlanPad *padOnZero = builder.findPad(0);
  const reloc::PlanPad *padOnOne = builder.findPad(1);
  ASSERT_TRUE(padOnZero && padOnOne);
  EXPECT_EQ(padOnZero->hi, dim(2));
  EXPECT_EQ(padOnOne->lo, dim(1));
  reloc::PlanAttr plan = finalize(builder);
  ASSERT_TRUE(plan);
  Tensor expected =
      transposeRef(padRef(padRef(iota({4, 6}), 0, 1, 0), 1, 0, 2), {1, 0});
  EXPECT_EQ(applyPlan(plan, noBindings), expected.data);
}

TEST_F(PlanBuilderTest, CanonicalizeMergesAdjacentAxes) {
  // Identity plan over [8, 128]: both sides contiguous, view-adjacent.
  reloc::PlanBuilder builder(makeType({dim(8), dim(128)}));
  reloc::PlanAttr raw = finalize(builder);
  ASSERT_TRUE(raw);
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(raw, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  ASSERT_EQ(canon.getAxes().size(), 1u);
  EXPECT_EQ(canon.getAxes()[0].getName(), "d0");
  EXPECT_EQ(canon.getAxes()[0].getExtent(), dim(1024));
  EXPECT_EQ(canon.getAxes()[0].getSrcStride(), dim(1));
  EXPECT_EQ(canon.getDst().getExtents().size(), 1u); // dst rank collapses
  EXPECT_TRUE(canon.getNoCopy()); // pure view after the merge
  // Semantics preserved: same flat buffer.
  EXPECT_EQ(applyPlan(canon, noBindings), applyPlan(raw, noBindings));
}

TEST_F(PlanBuilderTest, CanonicalizeRespectsViewAdjacency) {
  // Transposed [4, 6]: dst axes are adjacent but map to view axes in the
  // wrong order (perm [1, 0]) — no merge; no_copy recomputed to false.
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  reloc::PlanAttr raw = finalize(builder);
  ASSERT_TRUE(raw);
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(raw, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  EXPECT_EQ(canon.getAxes().size(), 2u);
  EXPECT_FALSE(canon.getNoCopy());
  EXPECT_EQ(applyPlan(canon, noBindings), applyPlan(raw, noBindings));
}

TEST_F(PlanBuilderTest, CanonicalizeMergeCascadesToFixpoint) {
  // Identity over [6, 4, 2] fully collapses to one axis of 48.
  reloc::PlanBuilder builder(makeType({dim(6), dim(4), dim(2)}));
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(finalize(builder), UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  ASSERT_EQ(canon.getAxes().size(), 1u);
  EXPECT_EQ(canon.getAxes()[0].getExtent(), dim(48));
  EXPECT_TRUE(canon.getNoCopy());
}

TEST_F(PlanBuilderTest, CanonicalizePadBlocksMergeAndKeepsNoCopyUnset) {
  reloc::PlanBuilder builder(makeType({dim(6), dim(4)}));
  ASSERT_TRUE(succeeded(reloc::foldPad(builder, 0, dim(1), dim(1), fill(0.0))));
  reloc::PlanAttr raw = finalize(builder);
  ASSERT_TRUE(raw);
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(raw, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  EXPECT_EQ(canon.getAxes().size(), 2u); // padded axis never merges
  ASSERT_EQ(canon.getPadFill().size(), 1u);
  EXPECT_EQ(canon.getPadFill()[0].getDstAxis(), 0);
  EXPECT_FALSE(canon.getNoCopy());
  EXPECT_EQ(applyPlan(canon, noBindings), applyPlan(raw, noBindings));
}

TEST_F(PlanBuilderTest, CanonicalizeClearsWrongNoCopy) {
  // Hand-assemble a plan claiming no_copy on a non-pure view (the plan
  // verifier does not police the flag; canonicalization must).
  reloc::PlanBuilder builder(makeType({dim(4), dim(6)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(builder, {1, 0})));
  reloc::PlanAttr raw = finalize(builder);
  ASSERT_TRUE(raw);
  reloc::PlanAttr lying = reloc::PlanAttr::get(
      &context, raw.getSrc(), raw.getDst(), raw.getPerm(), raw.getAxes(),
      raw.getPadFill(), raw.getDivisibility(), raw.getAlignment(),
      raw.getContiguity(), /*noCopy=*/true, raw.getRuntimePadCheck(),
      raw.getInverse());
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(lying, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  EXPECT_FALSE(canon.getNoCopy());
}

TEST_F(PlanBuilderTest, CanonicalizeConstantFolds) {
  // Bypass parse-time simplification with a raw BinaryExprAttr (4 + 4).
  Attribute rawSum = sym::BinaryExprAttr::get(
      &context, sym::SymbolicExprOp::Add, dim(4), dim(4));
  reloc::PlanBuilder builder(makeType({rawSum}));
  reloc::PlanAttr raw = finalize(builder);
  ASSERT_TRUE(raw);
  EXPECT_EQ(raw.getAxes()[0].getExtent(), rawSum); // builder kept it raw
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(raw, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  EXPECT_EQ(canon.getAxes()[0].getExtent(), dim(8));
}

TEST_F(PlanBuilderTest, CanonicalizeIsIdempotent) {
  reloc::PlanBuilder builder(makeType({dim(8), dim(128)}));
  reloc::PlanAttr once =
      reloc::canonicalizePlan(finalize(builder), UnknownLoc::get(&context));
  ASSERT_TRUE(once);
  // Attributes are uniqued: a truly idempotent canonicalization returns
  // the identical attribute.
  EXPECT_EQ(reloc::canonicalizePlan(once, UnknownLoc::get(&context)), once);
}

TEST_F(PlanBuilderTest, CanonicalizeSafeSubsetForNonFoldNormalPlans) {
  // Explicit dst strides: no axis merging, but no_copy is still
  // recomputed and expressions still fold. Build a pure-view plan with
  // explicit (row-major) dst strides and a stale no_copy=false.
  MLIRContext *ctx = &context;
  Attribute zero = dim(0);
  auto desc = [&](ArrayRef<Attribute> extents, ArrayRef<Attribute> strides) {
    return reloc::TensorDescAttr::get(ctx, extents, strides, zero,
                                      Float32Type::get(ctx));
  };
  SmallVector<reloc::AxisInfoAttr> axes = {
      reloc::AxisInfoAttr::get(ctx, "o", dim(8), dim(128), dim(128)),
      reloc::AxisInfoAttr::get(ctx, "i", dim(128), dim(1), dim(1))};
  auto plan = reloc::PlanAttr::get(
      ctx, desc({dim(8), dim(128)}, {}),
      desc({dim(8), dim(128)}, {dim(128), dim(1)}),
      DenseI64ArrayAttr::get(ctx, {0, 1}), axes, {}, {}, {},
      DenseBoolArrayAttr::get(ctx, ArrayRef<bool>()), /*noCopy=*/false,
      /*runtimePadCheck=*/false,
      AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(2, ctx)));
  reloc::PlanAttr canon =
      reloc::canonicalizePlan(plan, UnknownLoc::get(&context));
  ASSERT_TRUE(canon);
  EXPECT_EQ(canon.getAxes().size(), 2u); // no merging in the safe subset
  EXPECT_TRUE(canon.getNoCopy());        // but no_copy is recomputed
}

TEST_F(PlanBuilderTest, ConfluenceBlockedTransposeVsDirectTranspose) {
  // Path A: [128,128] -> reshape [2,64,2,64] -> transpose [2,3,0,1]
  // Path B: [128,128] -> transpose [1,0]
  // The same logical transform; A's four axes must merge pairwise back to
  // B's two.
  Attribute d2 = dim(2), d64 = dim(64);
  reloc::PlanBuilder a(makeType({dim(128), dim(128)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(a, {d2, d64, d2, d64})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(a, {2, 3, 0, 1})));
  reloc::PlanAttr rawA = finalize(a);
  ASSERT_TRUE(rawA);

  reloc::PlanBuilder b(makeType({dim(128), dim(128)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(b, {1, 0})));
  reloc::PlanAttr rawB = finalize(b);
  ASSERT_TRUE(rawB);

  reloc::PlanAttr canonA = canonical(rawA), canonB = canonical(rawB);
  EXPECT_EQ(canonA, canonB);                   // uniqued attrs: identical
  EXPECT_EQ(encoded(canonA), encoded(canonB)); // byte-identical (#A4)
  EXPECT_EQ(applyPlan(canonA, noBindings), applyPlan(rawA, noBindings));
  EXPECT_EQ(applyPlan(canonB, noBindings), applyPlan(rawB, noBindings));
}

TEST_F(PlanBuilderTest, ConfluenceInversePairVsIdentity) {
  // Path A: transpose [2,0,1] then its inverse [1,2,0]; Path B: no folds.
  // Both canonicalize to the fully-merged single-axis pure view.
  reloc::PlanBuilder a(makeType({dim(6), dim(4), dim(2)}));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(a, {2, 0, 1})));
  ASSERT_TRUE(succeeded(reloc::foldTranspose(a, {1, 2, 0})));
  reloc::PlanBuilder b(makeType({dim(6), dim(4), dim(2)}));
  reloc::PlanAttr canonA = canonical(finalize(a));
  reloc::PlanAttr canonB = canonical(finalize(b));
  EXPECT_EQ(canonA, canonB);
  EXPECT_EQ(encoded(canonA), encoded(canonB));
  ASSERT_EQ(canonA.getAxes().size(), 1u);
  EXPECT_TRUE(canonA.getNoCopy()); // inverse pair eliminated -> pure view
  EXPECT_EQ(applyPlan(canonA, noBindings), iota({6, 4, 2}).data);
}

TEST_F(PlanBuilderTest, ConfluenceSplitPathsToSameShape) {
  // Path A: [24] -> reshape [2,3,4] -> reshape [6,4]; Path B: [24] ->
  // reshape [6,4] directly.
  reloc::PlanBuilder a(makeType({dim(24)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(a, {dim(2), dim(3), dim(4)})));
  ASSERT_TRUE(succeeded(reloc::foldReshape(a, {dim(6), dim(4)})));
  reloc::PlanBuilder b(makeType({dim(24)}));
  ASSERT_TRUE(succeeded(reloc::foldReshape(b, {dim(6), dim(4)})));
  reloc::PlanAttr canonA = canonical(finalize(a));
  reloc::PlanAttr canonB = canonical(finalize(b));
  EXPECT_EQ(canonA, canonB);
  EXPECT_EQ(encoded(canonA), encoded(canonB));
  EXPECT_EQ(applyPlan(canonA, noBindings), applyPlan(finalize(b), noBindings));
}

} // namespace
