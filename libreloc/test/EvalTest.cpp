//===- EvalTest.cpp - expr-stream stack-machine evaluator tests -----------===//

#include "reloc/Bind.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <string>

namespace {

using reloc::evalExpr;
using reloc::ExprOp;
using reloc::ExprStream;
using reloc::ExprToken;
using reloc::SymbolValues;

int64_t evalOk(const ExprStream &stream, const SymbolValues &symbols) {
  int64_t out = 0;
  std::string error;
  EXPECT_TRUE(evalExpr(stream, symbols, out, error)) << error;
  return out;
}

std::string evalErr(const ExprStream &stream, const SymbolValues &symbols) {
  int64_t out = 0;
  std::string error;
  EXPECT_FALSE(evalExpr(stream, symbols, out, error));
  return error;
}

ExprToken konst(int64_t v) { return {ExprOp::PushConst, v}; }
ExprToken sym(int64_t i) { return {ExprOp::PushSym, i}; }
ExprToken op(ExprOp o) { return {o, 0}; }

TEST(Eval, Constant) { EXPECT_EQ(evalOk({konst(42)}, {}), 42); }

TEST(Eval, SymbolResolvesByIndex) {
  EXPECT_EQ(evalOk({sym(0)}, {7}), 7);
  EXPECT_EQ(evalOk({sym(1)}, {7, 9}), 9);
}

TEST(Eval, BinaryArithmetic) {
  EXPECT_EQ(evalOk({sym(0), konst(64), op(ExprOp::Mul)}, {512}), 32768);
  EXPECT_EQ(evalOk({konst(10), konst(3), op(ExprOp::Sub)}, {}), 7);
  EXPECT_EQ(evalOk({konst(10), konst(3), op(ExprOp::Add)}, {}), 13);
}

TEST(Eval, FloorDivRoundsTowardNegInfinity) {
  EXPECT_EQ(evalOk({konst(7), konst(2), op(ExprOp::FloorDiv)}, {}), 3);
  EXPECT_EQ(evalOk({konst(-7), konst(2), op(ExprOp::FloorDiv)}, {}), -4);
  EXPECT_EQ(evalOk({konst(7), konst(-2), op(ExprOp::FloorDiv)}, {}), -4);
  EXPECT_EQ(evalOk({konst(-7), konst(-2), op(ExprOp::FloorDiv)}, {}), 3);
  // The alignment idiom: (N + 63) floordiv 64 for N = 1000 -> 16.
  EXPECT_EQ(evalOk({sym(0), konst(63), op(ExprOp::Add), konst(64),
                    op(ExprOp::FloorDiv)},
                   {1000}),
            16);
}

TEST(Eval, ModTakesDivisorSign) {
  EXPECT_EQ(evalOk({konst(7), konst(3), op(ExprOp::Mod)}, {}), 1);
  EXPECT_EQ(evalOk({konst(-7), konst(3), op(ExprOp::Mod)}, {}), 2);
  EXPECT_EQ(evalOk({konst(7), konst(-3), op(ExprOp::Mod)}, {}), -2);
}

TEST(Eval, DivideByZeroIsError) {
  EXPECT_NE(
      evalErr({konst(1), konst(0), op(ExprOp::FloorDiv)}, {}).find("zero"),
      std::string::npos);
  EXPECT_NE(evalErr({konst(1), konst(0), op(ExprOp::Mod)}, {}).find("zero"),
            std::string::npos);
}

TEST(Eval, OverflowIsError) {
  const int64_t max = std::numeric_limits<int64_t>::max();
  EXPECT_NE(
      evalErr({konst(max), konst(1), op(ExprOp::Add)}, {}).find("overflow"),
      std::string::npos);
  EXPECT_NE(
      evalErr({konst(max), konst(2), op(ExprOp::Mul)}, {}).find("overflow"),
      std::string::npos);
  const int64_t min = std::numeric_limits<int64_t>::min();
  EXPECT_NE(evalErr({konst(min), konst(-1), op(ExprOp::FloorDiv)}, {})
                .find("overflow"),
            std::string::npos);
  // Sub overflow: INT64_MIN - 1 underflows.
  EXPECT_NE(
      evalErr({konst(min), konst(1), op(ExprOp::Sub)}, {}).find("overflow"),
      std::string::npos);
  // Mod's reconstruction overflow: floordiv(INT64_MAX, -3) succeeds (q =
  // -3074457345618258603), but reconstructing the remainder via q*b
  // overflows (q*-3 = 9223372036854775809 > INT64_MAX). This exercises the
  // mod-specific overflow guard, distinct from floordiv's own overflow path.
  EXPECT_NE(
      evalErr({konst(max), konst(-3), op(ExprOp::Mod)}, {}).find("overflow"),
      std::string::npos);
}

TEST(Eval, PushDimIsErrorInPlanContext) {
  EXPECT_NE(evalErr({{ExprOp::PushDim, 0}}, {}).find("PUSH_DIM"),
            std::string::npos);
}

} // namespace
