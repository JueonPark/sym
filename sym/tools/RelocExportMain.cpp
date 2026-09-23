//===- RelocExportMain.cpp - Supported layout-plan export
//------------------===//

#include "PlanBuilder.h"
#include "RelocDialect.h"
#include "RelocPasses.h"
#include "RelocSerialization.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace json = llvm::json;

namespace {
llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::Required,
                                 llvm::cl::desc("INPUT"));
llvm::cl::opt<std::string>
    output("output", llvm::cl::Required,
           llvm::cl::desc("Plan output (must not exist)"));
llvm::cl::opt<std::string>
    manifest("manifest", llvm::cl::Required,
             llvm::cl::desc("JSON output (must not exist)"));

// Output paths are exclusively created, never truncated. Only files owned by
// this invocation are cleaned up, including when writing the second file fails.
class Outputs {
public:
  ~Outputs() {
    if (ownsPlan && !keepPlan)
      llvm::sys::fs::remove(output);
    if (ownsManifest && !keepManifest)
      llvm::sys::fs::remove(manifest);
  }
  bool preflight() {
    if (output == manifest || llvm::sys::fs::exists(output) ||
        llvm::sys::fs::exists(manifest)) {
      llvm::errs() << "output paths must be distinct and must not exist\n";
      return false;
    }
    return true;
  }
  bool write(StringRef path, StringRef data, bool &owned) {
    std::error_code ec;
    llvm::raw_fd_ostream stream(path, ec, llvm::sys::fs::CD_CreateNew,
                                llvm::sys::fs::FA_Write,
                                llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "cannot create " << path << ": " << ec.message() << '\n';
      return false;
    }
    owned = true;
    stream << data;
    stream.close();
    if (stream.has_error()) {
      llvm::errs() << "cannot write " << path << ": "
                   << stream.error().message() << '\n';
      stream.clear_error();
      return false;
    }
    return true;
  }
  bool publish(json::Object meta, StringRef blob = {}) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << llvm::formatv("{0:2}\n", json::Value(std::move(meta)));
    if (!blob.empty() && !write(output, blob, ownsPlan))
      return false;
    if (!write(manifest, text, ownsManifest))
      return false;
    keepPlan = ownsPlan;
    keepManifest = true;
    return true;
  }

private:
  bool ownsPlan = false, ownsManifest = false;
  bool keepPlan = false, keepManifest = false;
};

json::Object baseManifest(StringRef status) {
  return json::Object{
      {"schema_version", 1},
      {"wire_version", 0},
      {"status", status},
      {"compiler", json::Object{{"name", "sym-reloc-export"},
                                {"interface_version", 1},
                                {"llvm_version", LLVM_VERSION_STRING},
                                {"build_identity", SYM_EXPORT_BUILD_ID}}}};
}

// Validate before serialization: the public vocabulary only permits constant,
// positive divisors. Constants already use signed i64 in the sym dialect.
bool supportedExpr(Attribute expr) {
  if (isa<sym::ConstantExprAttr, sym::SymbolExprAttr>(expr))
    return true;
  auto binary = dyn_cast<sym::BinaryExprAttr>(expr);
  if (!binary || !supportedExpr(binary.getLhs()) ||
      !supportedExpr(binary.getRhs()))
    return false;
  if (binary.getOpcode() == sym::SymbolicExprOp::Div ||
      binary.getOpcode() == sym::SymbolicExprOp::Mod) {
    auto divisor = dyn_cast<sym::ConstantExprAttr>(binary.getRhs());
    return divisor && divisor.getValue() > 0;
  }
  return true;
}

json::Value expression(Attribute expr) {
  if (auto c = dyn_cast<sym::ConstantExprAttr>(expr))
    return json::Array{"const", c.getValue()};
  if (auto s = dyn_cast<sym::SymbolExprAttr>(expr))
    return json::Array{"symbol", s.getName()};
  auto b = cast<sym::BinaryExprAttr>(expr);
  auto lhs = expression(b.getLhs());
  auto rhs = expression(b.getRhs());
  switch (b.getOpcode()) {
  case sym::SymbolicExprOp::Add:
    return json::Array{"add", std::move(lhs), std::move(rhs)};
  case sym::SymbolicExprOp::Sub:
    return json::Array{
        "add", std::move(lhs),
        json::Array{"mul", json::Array{"const", -1}, std::move(rhs)}};
  case sym::SymbolicExprOp::Mul:
    return json::Array{"mul", std::move(lhs), std::move(rhs)};
  case sym::SymbolicExprOp::Div:
  case sym::SymbolicExprOp::Mod:
    return json::Array{
        b.getOpcode() == sym::SymbolicExprOp::Div ? "floordiv" : "mod",
        std::move(lhs), cast<sym::ConstantExprAttr>(b.getRhs()).getValue()};
  }
  llvm_unreachable("unknown sym expression opcode");
}

StringRef dtype(Type type) {
  if (type.isF32())
    return "float32";
  if (type.isF16())
    return "float16";
  if (type.isSignlessInteger(8))
    return "int8";
  return {};
}

json::Object descriptor(sym::SymbolicTensorType type) {
  json::Array shape, strides;
  // Build dense strides using JSON, avoiding signed overflow from multiplying
  // static extents in host arithmetic. Elide the terminal multiplication by 1.
  for (Attribute extent : type.getShape())
    shape.push_back(expression(extent));
  for (size_t i = 0; i < type.getShape().size(); ++i) {
    json::Value stride = json::Array{"const", 1};
    bool first = true;
    for (size_t j = type.getShape().size(); j > i + 1; --j) {
      auto extent = expression(type.getShape()[j - 1]);
      if (first)
        stride = std::move(extent);
      else
        stride = json::Array{"mul", std::move(extent), std::move(stride)};
      first = false;
    }
    strides.push_back(std::move(stride));
  }
  return json::Object{{"shape", std::move(shape)},
                      {"strides", std::move(strides)},
                      {"offset", json::Array{"const", 0}},
                      {"dtype", dtype(type.getElementType())}};
}

std::string digest(StringRef bytes) {
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(hash, true);
}
} // namespace

int main(int argc, char **argv) {
  if (!llvm::cl::ParseCommandLineOptions(argc, argv, "Layout plan exporter\n",
                                         &llvm::errs()))
    return 1;
  Outputs outputs;
  if (!outputs.preflight())
    return 1;
  auto unsupported = [&](StringRef reason, StringRef detail) {
    auto meta = baseManifest("unsupported");
    meta["reason"] = reason;
    meta["detail"] = detail;
    return outputs.publish(std::move(meta)) ? 2 : 1;
  };
  auto buffer = llvm::MemoryBuffer::getFile(input);
  if (!buffer) {
    llvm::errs() << "cannot read " << input << ": "
                 << buffer.getError().message() << '\n';
    return 1;
  }
  std::string inputHash = digest((*buffer)->getBuffer());
  DialectRegistry registry;
  registry.insert<sym::SymDialect, reloc::RelocDialect, func::FuncDialect,
                  arith::ArithDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module || failed(verify(*module)))
    return 1;
  SmallVector<func::FuncOp> functions;
  for (Operation &op : module->getBody()->getOperations()) {
    if (auto function = dyn_cast<func::FuncOp>(op))
      functions.push_back(function);
    else
      return unsupported("unsupported_operation",
                         "module may contain only one function");
  }
  if (functions.size() != 1)
    return unsupported("invalid_function_count",
                       "expected exactly one function");
  func::FuncOp function = functions.front();
  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()) ||
      function.getNumArguments() != 1 || function.getNumResults() != 1)
    return unsupported("unsupported_signature",
                       "expected one block, one input and one result");
  auto source =
      dyn_cast<sym::SymbolicTensorType>(function.getArgument(0).getType());
  auto destination =
      dyn_cast<sym::SymbolicTensorType>(function.getResultTypes()[0]);
  if (!source || !destination)
    return unsupported("unsupported_signature",
                       "input and result must be symbolic tensors");
  // Validate every intermediate too, before PlanBuilder can assert on rank 0.
  auto validateType = [&](Type type) -> StringRef {
    auto tensor = dyn_cast<sym::SymbolicTensorType>(type);
    if (!tensor || tensor.getShape().empty())
      return "unsupported_descriptor";
    if (dtype(tensor.getElementType()).empty())
      return "unsupported_dtype";
    for (Attribute extent : tensor.getShape()) {
      if (!supportedExpr(extent))
        return "unsupported_expression";
      if (auto c = dyn_cast<sym::ConstantExprAttr>(extent);
          c && c.getValue() <= 0)
        return "unsupported_descriptor";
    }
    return {};
  };
  for (Type type : {Type(source), Type(destination)})
    if (auto reason = validateType(type); !reason.empty())
      return unsupported(reason, "requires positive rank, positive static "
                                 "extents and supported dtype/expressions");
  Value previous = function.getArgument(0);
  size_t chainCount = 0;
  auto &block = function.front();
  for (Operation &op : block.without_terminator()) {
    if (isa<reloc::PlanResultOp, reloc::TypedPlanResultOp>(op))
      return unsupported("prefolded_input",
                         "input must contain original reloc chain operations");
    // C1 defines the typed value transforms and their semantics; the wire v0
    // artifact and this interface stay layout-only until C2/C3 supply the
    // typed representation and encoder (docs/reloc-typed-semantics.md).
    if (reloc::isTypedValueTransformOp(&op))
      return unsupported("typed_unsupported",
                         "typed value transforms (reloc.cast, reloc.quantize, "
                         "reloc.dequantize) have no typed artifact "
                         "representation or encoder yet");
    if (!reloc::isFoldableChainOp(&op))
      return unsupported(
          "unsupported_operation",
          "function contains an operation outside the layout chain");
    if (op.hasAttr("reloc.fallback"))
      return unsupported("fold_unsupported", "chain has a fallback marker");
    if (op.getOperand(0) != previous || !previous.hasOneUse())
      return unsupported(
          "disconnected_chain",
          "every chain value must have exactly its next chain use");
    if (auto reason = validateType(op.getResult(0).getType()); !reason.empty())
      return unsupported(reason, "unsupported intermediate tensor descriptor");
    if (auto pad = dyn_cast<reloc::PadOp>(op))
      if (!supportedExpr(pad.getLo()) || !supportedExpr(pad.getHi()))
        return unsupported("unsupported_expression",
                           "pad widths need supported expressions");
    previous = op.getResult(0);
    ++chainCount;
  }
  if (!chainCount)
    return unsupported("empty_chain", "expected at least one reloc operation");
  auto ret = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != previous ||
      !previous.hasOneUse())
    return unsupported("disconnected_chain",
                       "return must consume exactly the final chain result");
  auto logicalSource = descriptor(source);
  auto logicalDestination = descriptor(destination);
  PassManager passes(&context);
  passes.addPass(reloc::createRelocFoldPass());
  if (failed(passes.run(*module)) || failed(verify(*module)))
    return 1;
  reloc::PlanResultOp result;
  size_t planCount = 0;
  bool residual = false;
  module->walk([&](Operation *op) {
    if (auto plan = dyn_cast<reloc::PlanResultOp>(op)) {
      result = plan;
      ++planCount;
    }
    // A typed plan (C2) is not a v0 artifact: never present it as success.
    if (op->hasAttr("reloc.fallback") || reloc::isFoldableChainOp(op) ||
        isa<reloc::TypedPlanResultOp>(op))
      residual = true;
  });
  if (residual || planCount != 1)
    return unsupported("fold_unsupported",
                       "reloc-fold did not produce exactly one complete plan");
  json::Array divisibility;
  for (auto constraint : result.getPlan().getDivisibility()) {
    if (!supportedExpr(constraint.getExpr()) || constraint.getDivisor() <= 0)
      return unsupported("unsupported_expression",
                         "compiler constraint is outside manifest vocabulary");
    divisibility.push_back(
        json::Object{{"expr", expression(constraint.getExpr())},
                     {"divisor", constraint.getDivisor()}});
  }
  std::vector<std::string> symbolNames;
  auto encoded =
      reloc::encodePlan(result.getPlan(), result.getLoc(), &symbolNames);
  if (failed(encoded))
    return 1;
  StringRef blob(reinterpret_cast<const char *>(encoded->data()),
                 encoded->size());
  json::Array symbols;
  for (const auto &name : symbolNames)
    symbols.push_back(name);
  auto meta = baseManifest("ok");
  meta["plan_count"] = 1;
  meta["symbols"] = std::move(symbols);
  meta["logical_source"] = std::move(logicalSource);
  meta["logical_destination"] = std::move(logicalDestination);
  meta["constraints"] = json::Object{{"divisibility", std::move(divisibility)}};
  meta["plan_sha256"] = digest(blob);
  meta["input_sha256"] = inputHash;
  return outputs.publish(std::move(meta), blob) ? 0 : 1;
}
