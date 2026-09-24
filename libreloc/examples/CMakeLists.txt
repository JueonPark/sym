# R4 (issue #148): the standalone public C++ consumer of compiler-generated
# plan artifacts. Links only the MLIR/LLVM/Torch-free runtime (and the CUDA
# runtime when RELOC_ENABLE_CUDA is on); never the compiler.
add_executable(reloc-run-artifact RunArtifact.cpp)
target_link_libraries(reloc-run-artifact PRIVATE reloc_runtime)
if(RELOC_ENABLE_CUDA)
  target_link_libraries(reloc-run-artifact PRIVATE CUDA::cudart)
endif()

# Same contract as the runtime: the example's own dynamic dependencies carry
# no MLIR/LLVM library.
add_test(NAME reloc-run-artifact-mlir-free
         COMMAND bash -c "set -e; readelf -d $<TARGET_FILE:reloc-run-artifact> >/dev/null; ! readelf -d $<TARGET_FILE:reloc-run-artifact> | grep -qiE 'NEEDED.*(mlir|llvm)'")
