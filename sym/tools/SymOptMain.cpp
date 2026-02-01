//===- SymOptMain.cpp - sym-opt driver ------------------------------------===//
//
// Main entry point for sym-opt tool.
//
//===----------------------------------------------------------------------===//

#include "SymDialect.h"
#include "SymPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all MLIR core dialects
  mlir::registerAllDialects(registry);

  // Register all MLIR core passes
  mlir::registerAllPasses();

  // Register Sym dialect passes
  mlir::sym::registerSymPasses();

  // Register external models for arith ops
  mlir::sym::registerSymbolicShapeOpInterfaceExternalModels(registry);

  // Register our Sym dialect
  registry.insert<mlir::sym::SymDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Sym optimizer driver\n", registry));
}