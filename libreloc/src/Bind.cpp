//===- Bind.cpp - symbol evaluation and BoundPlan construction ------------===//

#include "reloc/Bind.h"

#include <cstdint>
#include <limits>

namespace reloc {
namespace {

bool floorDiv(int64_t a, int64_t b, int64_t &out, std::string &error) {
  if (b == 0)
    return (error = "division by zero"), false;
  if (a == std::numeric_limits<int64_t>::min() && b == -1)
    return (error = "integer overflow in floordiv"), false;
  int64_t q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0)))
    --q;
  out = q;
  return true;
}

} // namespace

bool evalExpr(const ExprStream &stream, const SymbolValues &symbols,
              int64_t &out, std::string &error) {
  std::vector<int64_t> stack;
  stack.reserve(stream.size());
  for (const ExprToken &token : stream) {
    switch (token.op) {
    case ExprOp::PushSym:
      // Index range guaranteed by the decoder; guard defensively anyway.
      if (token.value < 0 || static_cast<size_t>(token.value) >= symbols.size())
        return (error = "symbol index out of range"), false;
      stack.push_back(symbols[token.value]);
      break;
    case ExprOp::PushConst:
      stack.push_back(token.value);
      break;
    case ExprOp::PushDim:
      return (error = "PUSH_DIM is not valid in a plan expression"), false;
    case ExprOp::Add:
    case ExprOp::Sub:
    case ExprOp::Mul:
    case ExprOp::FloorDiv:
    case ExprOp::Mod: {
      int64_t b = stack.back();
      stack.pop_back();
      int64_t a = stack.back();
      stack.pop_back();
      int64_t r = 0;
      switch (token.op) {
      case ExprOp::Add:
        if (__builtin_add_overflow(a, b, &r))
          return (error = "integer overflow in add"), false;
        break;
      case ExprOp::Sub:
        if (__builtin_sub_overflow(a, b, &r))
          return (error = "integer overflow in sub"), false;
        break;
      case ExprOp::Mul:
        if (__builtin_mul_overflow(a, b, &r))
          return (error = "integer overflow in mul"), false;
        break;
      case ExprOp::FloorDiv:
        if (!floorDiv(a, b, r, error))
          return false;
        break;
      case ExprOp::Mod: {
        int64_t q = 0;
        if (!floorDiv(a, b, q, error))
          return false;
        int64_t prod = 0;
        if (__builtin_mul_overflow(q, b, &prod) ||
            __builtin_sub_overflow(a, prod, &r))
          return (error = "integer overflow in mod"), false;
        break;
      }
      default:
        break;
      }
      stack.push_back(r);
      break;
    }
    }
  }
  if (stack.size() != 1)
    return (error = "expression did not evaluate to a single value"), false;
  out = stack.back();
  return true;
}

BindResult bind(const RelocationPlan &, const SymbolMap &, Strategy) {
  return BindError{"bind not implemented"}; // Task 2
}

} // namespace reloc
