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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
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

#include <optional>

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
// C3 (issue #143): typed plans are opt-in. The layout-only interface
// (schema 1, wire v0) never encodes a typed value transform, so an old
// consumer that does not know the flag keeps receiving exactly what it did.
llvm::cl::opt<bool> typedPlans(
    "typed",
    llvm::cl::desc("Admit typed value transforms (reloc.cast, reloc.quantize, "
                   "reloc.dequantize). A chain containing one exports as wire "
                   "format v1 with manifest schema 2; a layout-only chain "
                   "exports as wire format v0 with manifest schema 1 either "
                   "way, byte for byte."));

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

// Schema 1 / wire v0 is the layout-only manifest (R1); schema 2 / wire v1
// describes a typed plan (C3). Unsupported manifests describe no plan and
// always use schema 1 so that every consumer of either interface reads them.
json::Object baseManifest(StringRef status, int64_t schemaVersion = 1,
                          int64_t wireVersion = 0) {
  return json::Object{
      {"schema_version", schemaVersion},
      {"wire_version", wireVersion},
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

// Stage parameters may also be i32 (zero points); tensors may not.
StringRef parameterDtype(Type type) {
  if (StringRef name = dtype(type); !name.empty())
    return name;
  if (type.isSignlessInteger(32))
    return "int32";
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

//===----------------------------------------------------------------------===//
// Schema 2 (typed plans, C3): stages, fills and parameter declarations.
// Scalar bits are lowercase hex strings of the exact bit pattern (never
// decimal text), like the pad fills of the wire format itself.
//===----------------------------------------------------------------------===//

std::string hexBits(const llvm::APInt &bits) {
  return llvm::utohexstr(bits.getZExtValue(), /*LowerCase=*/true);
}

json::Value parameter(Attribute attr) {
  if (!attr)
    return nullptr;
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    json::Array shape, bits;
    for (int64_t extent : dense.getType().getShape())
      shape.push_back(extent);
    if (isa<FloatType>(dense.getElementType())) {
      for (const llvm::APFloat &value : dense.getValues<llvm::APFloat>())
        bits.push_back(hexBits(value.bitcastToAPInt()));
    } else {
      for (const llvm::APInt &value : dense.getValues<llvm::APInt>())
        bits.push_back(hexBits(value));
    }
    return json::Object{{"kind", "inline"},
                        {"dtype", parameterDtype(dense.getElementType())},
                        {"shape", std::move(shape)},
                        {"bits", std::move(bits)}};
  }
  auto binding = cast<reloc::ParamBindingAttr>(attr);
  json::Array extents;
  for (Attribute extent : binding.getExtents())
    extents.push_back(expression(extent));
  return json::Object{{"kind", "binding"},
                      {"name", binding.getName()},
                      {"dtype", parameterDtype(binding.getElementType())},
                      {"extents", std::move(extents)}};
}

// Channel maps are affine over logical result coordinates (["dim", i]) and
// plan symbols; their divisors may be symbolic, so floordiv/mod take two
// channel expressions (a vocabulary of its own, see docs/reloc-export.md).
std::optional<json::Value> channelExpression(AffineExpr expr,
                                             ArrayAttr symbols) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return json::Array{"const", cast<AffineConstantExpr>(expr).getValue()};
  case AffineExprKind::DimId:
    return json::Array{
        "dim", static_cast<int64_t>(cast<AffineDimExpr>(expr).getPosition())};
  case AffineExprKind::SymbolId: {
    unsigned position = cast<AffineSymbolExpr>(expr).getPosition();
    if (position >= symbols.size())
      return std::nullopt;
    return json::Array{"symbol",
                       cast<StringAttr>(symbols[position]).getValue()};
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::Mod: {
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto lhs = channelExpression(binary.getLHS(), symbols);
    auto rhs = channelExpression(binary.getRHS(), symbols);
    if (!lhs || !rhs)
      return std::nullopt;
    const char *tag = expr.getKind() == AffineExprKind::Add        ? "add"
                      : expr.getKind() == AffineExprKind::Mul      ? "mul"
                      : expr.getKind() == AffineExprKind::FloorDiv ? "floordiv"
                                                                   : "mod";
    return json::Array{tag, std::move(*lhs), std::move(*rhs)};
  }
  case AffineExprKind::CeilDiv:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<json::Value> stageJson(reloc::ValueStageAttr stage,
                                     ArrayAttr symbols) {
  json::Array shape;
  for (Attribute extent : stage.getShape())
    shape.push_back(expression(extent));
  json::Object out{
      {"transform", reloc::stringifyValueTransform(stage.getTransform())},
      {"policy", reloc::stringifyNumericPolicy(stage.getPolicy())},
      {"input_dtype", dtype(stage.getInputType())},
      {"output_dtype", dtype(stage.getOutputType())},
      {"shape", std::move(shape)},
      {"scale", parameter(stage.getScale())},
      {"zero_point", parameter(stage.getZeroPoint())},
      {"axis", stage.getAxis()},
      {"channel", nullptr}};
  if (AffineMap channel = stage.getChannel()) {
    if (channel.getNumResults() != 1)
      return std::nullopt;
    auto expr = channelExpression(channel.getResult(0), symbols);
    if (!expr)
      return std::nullopt;
    out["channel"] =
        json::Object{{"dims", static_cast<int64_t>(channel.getNumDims())},
                     {"expr", std::move(*expr)}};
  }
  return json::Value(std::move(out));
}

json::Value fillJson(reloc::TypedFillAttr fill) {
  llvm::APInt bits;
  if (auto value = dyn_cast<FloatAttr>(fill.getValue()))
    bits = value.getValue().bitcastToAPInt();
  else
    bits = cast<IntegerAttr>(fill.getValue()).getValue();
  return json::Object{{"dst_axis", fill.getDstAxis()},
                      {"stage", fill.getStage()},
                      {"dtype", dtype(fill.getValue().getType())},
                      {"bits", hexBits(bits)}};
}

// Runtime parameter declarations in first-declaration order, one per name
// (the plan verifier guarantees consistent redeclarations).
json::Array parameterDeclarations(reloc::TypedPlanAttr plan) {
  json::Array out;
  SmallVector<StringRef> seen;
  for (reloc::ValueStageAttr stage : plan.getStages()) {
    for (Attribute attr : {stage.getScale(), stage.getZeroPoint()}) {
      auto binding = dyn_cast_or_null<reloc::ParamBindingAttr>(attr);
      if (!binding || llvm::is_contained(seen, binding.getName()))
        continue;
      seen.push_back(binding.getName());
      json::Array extents;
      for (Attribute extent : binding.getExtents())
        extents.push_back(expression(extent));
      out.push_back(
          json::Object{{"name", binding.getName()},
                       {"dtype", parameterDtype(binding.getElementType())},
                       {"extents", std::move(extents)}});
    }
  }
  return out;
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
  bool typedChain = false;
  auto &block = function.front();
  for (Operation &op : block.without_terminator()) {
    if (isa<reloc::PlanResultOp, reloc::TypedPlanResultOp>(op))
      return unsupported("prefolded_input",
                         "input must contain original reloc chain operations");
    // C1 defines the typed value transforms; C2 folds them into
    // #reloc.typed_plan; C3 encodes that as wire format v1 behind --typed.
    // The layout-only interface (schema 1, wire v0) never encodes them.
    if (reloc::isTypedValueTransformOp(&op)) {
      if (!typedPlans)
        return unsupported("typed_unsupported",
                           "typed value transforms (reloc.cast, "
                           "reloc.quantize, reloc.dequantize) need --typed; "
                           "the layout-only interface (schema 1, wire format "
                           "v0) never encodes them");
      typedChain = true;
      // Binding extents enter the manifest and the wire: same vocabulary.
      for (NamedAttribute attr : op.getAttrs())
        if (auto binding = dyn_cast<reloc::ParamBindingAttr>(attr.getValue()))
          for (Attribute extent : binding.getExtents())
            if (!supportedExpr(extent))
              return unsupported("unsupported_expression",
                                 "parameter binding extents need supported "
                                 "expressions");
    }
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
  reloc::TypedPlanResultOp typedResult;
  size_t planCount = 0;
  bool residual = false;
  module->walk([&](Operation *op) {
    if (auto plan = dyn_cast<reloc::PlanResultOp>(op)) {
      result = plan;
      ++planCount;
    } else if (auto plan = dyn_cast<reloc::TypedPlanResultOp>(op)) {
      // A typed plan (C2) is never a v0 artifact: without --typed it is a
      // residual, never a success.
      if (typedPlans) {
        typedResult = plan;
        ++planCount;
      } else {
        residual = true;
      }
    }
    if (op->hasAttr("reloc.fallback") || reloc::isFoldableChainOp(op))
      residual = true;
  });
  if (residual || planCount != 1 || (typedResult != nullptr) != typedChain)
    return unsupported("fold_unsupported",
                       "reloc-fold did not produce exactly one complete plan");
  reloc::PlanAttr layout =
      typedResult ? typedResult.getPlan().getLayout() : result.getPlan();
  json::Array divisibility;
  for (auto constraint : layout.getDivisibility()) {
    if (!supportedExpr(constraint.getExpr()) || constraint.getDivisor() <= 0)
      return unsupported("unsupported_expression",
                         "compiler constraint is outside manifest vocabulary");
    divisibility.push_back(
        json::Object{{"expr", expression(constraint.getExpr())},
                     {"divisor", constraint.getDivisor()}});
  }
  json::Array stages, fills;
  if (typedResult) {
    reloc::TypedPlanAttr plan = typedResult.getPlan();
    for (reloc::ValueStageAttr stage : plan.getStages()) {
      auto entry = stageJson(stage, plan.getSymbols());
      if (!entry)
        return unsupported("unsupported_expression",
                           "channel map is outside manifest vocabulary");
      stages.push_back(std::move(*entry));
    }
    for (reloc::TypedFillAttr fill : plan.getFills())
      fills.push_back(fillJson(fill));
  }
  std::vector<std::string> symbolNames;
  FailureOr<std::vector<uint8_t>> encoded =
      typedResult
          ? reloc::encodeTypedPlan(typedResult.getPlan(), typedResult.getLoc(),
                                   &symbolNames)
          : reloc::encodePlan(result.getPlan(), result.getLoc(), &symbolNames);
  if (failed(encoded))
    return 1;
  StringRef blob(reinterpret_cast<const char *>(encoded->data()),
                 encoded->size());
  json::Array symbols;
  for (const auto &name : symbolNames)
    symbols.push_back(name);
  auto meta = typedResult ? baseManifest("ok", 2, 1) : baseManifest("ok");
  meta["plan_count"] = 1;
  meta["symbols"] = std::move(symbols);
  meta["logical_source"] = std::move(logicalSource);
  meta["logical_destination"] = std::move(logicalDestination);
  meta["constraints"] = json::Object{{"divisibility", std::move(divisibility)}};
  if (typedResult) {
    meta["stages"] = std::move(stages);
    meta["fills"] = std::move(fills);
    meta["parameters"] = parameterDeclarations(typedResult.getPlan());
  }
  meta["plan_sha256"] = digest(blob);
  meta["input_sha256"] = inputHash;
  return outputs.publish(std::move(meta), blob) ? 0 : 1;
}
