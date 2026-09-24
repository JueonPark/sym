//===- RunArtifact.cpp - standalone consumer of a compiled plan artifact --===//
//
// R4 (issue #148): the smallest public C++ path from compiler-generated wire
// bytes to relocated bytes. It links only the MLIR/LLVM/Torch-free runtime
// (plus the CUDA runtime when built with RELOC_ENABLE_CUDA) and uses only
// public interfaces: decodePlan -> bind -> validateTransfer ->
// executeTransfer through a CopyBackend.
//
//   reloc-run-artifact PLAN --symbols NAME=VALUE,... --input INPUT
//                      --output OUTPUT --direction host|h2d|d2h
//
// INPUT holds the dense row-major source elements of the plan's source
// descriptor (raw little-endian scalars); OUTPUT receives the dense
// destination (bound.totalBytes). `host` runs the forward relocation host
// to host through HostBackend. `h2d` stages nothing: the host source is
// transferred into a device buffer the example owns, then copied back only
// to write OUTPUT (reported as verification_readback_bytes, separate from the
// requested transfer). `d2h` first uploads INPUT to a device buffer (the
// documented logical source), then runs the requested forward device-to-host
// relocation into the host OUTPUT. A one-line JSON report goes to stdout;
// every failure is a diagnostic on stderr and a nonzero exit. Typed (wire v1)
// artifacts are refused with a pointer to the Python dispatch path.
//
//===----------------------------------------------------------------------===//

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/HostBackend.h"
#include "reloc/Transfer.h"
#include "reloc/Version.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

int usage() {
  std::cerr << "usage: reloc-run-artifact PLAN --symbols NAME=VALUE,... "
               "--input INPUT --output OUTPUT --direction host|h2d|d2h\n";
  return 2;
}

int fail(const std::string &message) {
  std::cerr << "reloc-run-artifact: " << message << '\n';
  return 1;
}

bool readFile(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  out.assign(std::istreambuf_iterator<char>(in), {});
  return true;
}

bool writeFile(const std::string &path, const void *data, size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char *>(data),
            static_cast<std::streamsize>(bytes));
  return static_cast<bool>(out);
}

bool parseSymbols(const std::string &text, reloc::SymbolMap &out,
                  std::string &error) {
  size_t start = 0;
  while (start < text.size()) {
    size_t end = text.find(',', start);
    if (end == std::string::npos)
      end = text.size();
    std::string item = text.substr(start, end - start);
    size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size()) {
      error = "malformed symbol binding '" + item + "' (expected NAME=VALUE)";
      return false;
    }
    char *tail = nullptr;
    long long value = std::strtoll(item.c_str() + eq + 1, &tail, 10);
    if (*tail != '\0') {
      error = "symbol value is not an integer in '" + item + "'";
      return false;
    }
    std::string name = item.substr(0, eq);
    if (out.count(name)) {
      error = "symbol '" + name + "' bound twice";
      return false;
    }
    out[name] = value;
    start = end + 1;
  }
  return true;
}

reloc::BufferView denseView(uintptr_t base, size_t bytes,
                            std::vector<int64_t> extents, uint32_t width,
                            reloc::MemoryKind kind, int device) {
  reloc::BufferView view;
  view.base = base;
  view.capacityBytes = bytes;
  view.offsetBytes = 0;
  view.strides.assign(extents.size(), 1);
  for (size_t k = extents.size(); k-- > 1;)
    view.strides[k - 1] = view.strides[k] * extents[k];
  view.extents = std::move(extents);
  view.elementSize = width;
  view.kind = kind;
  view.device = device;
  return view;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage();
  std::string planPath = argv[1], symbols, input, output, direction;
  bool haveSymbols = false;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (i + 1 >= argc)
      return usage();
    std::string value = argv[++i];
    if (arg == "--symbols") {
      symbols = value;
      haveSymbols = true;
    } else if (arg == "--input") {
      input = value;
    } else if (arg == "--output") {
      output = value;
    } else if (arg == "--direction") {
      direction = value;
    } else {
      return usage();
    }
  }
  if (input.empty() || output.empty() ||
      (direction != "host" && direction != "h2d" && direction != "d2h"))
    return usage();

  std::vector<uint8_t> blob;
  if (!readFile(planPath, blob))
    return fail("cannot read plan " + planPath);
  std::optional<uint32_t> version =
      reloc::peekWireVersion(blob.data(), blob.size());
  if (version && *version == reloc::kTypedWireFormatVersion)
    return fail(
        "typed (wire v1) artifact: run it through pyreloc.prepare_dispatch "
        "or reloc_torch.dispatch (docs/runtime-dispatch.md); this example "
        "executes layout-only (wire v0) plans");
  auto decoded = reloc::decodePlan(blob.data(), blob.size());
  if (auto *error = std::get_if<reloc::DecodeError>(&decoded))
    return fail("decode error at byte offset " + std::to_string(error->offset) +
                ": " + error->message);
  const auto &plan = std::get<reloc::RelocationPlan>(decoded);

  reloc::SymbolMap symbolMap;
  std::string error;
  if (haveSymbols && !parseSymbols(symbols, symbolMap, error))
    return fail(error);
  auto bound = reloc::bind(plan, symbolMap);
  if (auto *bindError = std::get_if<reloc::BindError>(&bound))
    return fail("bind error: " + bindError->message);
  const auto &b = std::get<reloc::BoundPlan>(bound);

  // The dense logical source: the plan's source descriptor under these symbols.
  reloc::SymbolValues values;
  for (const std::string &name : plan.symbols)
    values.push_back(symbolMap.at(name));
  std::vector<int64_t> sourceExtents;
  int64_t sourceElements = 1;
  for (const reloc::ExprStream &extent : plan.src.extents) {
    int64_t v = 0;
    if (!reloc::evalExpr(extent, values, v, error))
      return fail("source extent: " + error);
    sourceExtents.push_back(v);
    sourceElements *= v;
  }
  const uint32_t width = b.elementSize;
  const size_t sourceBytes = static_cast<size_t>(sourceElements) * width;
  const size_t destinationBytes = static_cast<size_t>(b.totalBytes);

  std::vector<uint8_t> source;
  if (!readFile(input, source))
    return fail("cannot read input " + input);
  if (source.size() != sourceBytes)
    return fail("input holds " + std::to_string(source.size()) +
                " bytes but the plan's source descriptor needs " +
                std::to_string(sourceBytes));
  std::vector<uint8_t> destination(destinationBytes, 0);
  const std::vector<int64_t> destinationExtents = {
      static_cast<int64_t>(destinationBytes / width)};

  size_t requested = 0, readback = 0;
  reloc::TransferOptions options;
  if (direction == "host") {
    auto validated = reloc::validateTransfer(
        b,
        denseView(reinterpret_cast<uintptr_t>(source.data()), sourceBytes,
                  sourceExtents, width, reloc::MemoryKind::Host, -1),
        denseView(reinterpret_cast<uintptr_t>(destination.data()),
                  destinationBytes, destinationExtents, width,
                  reloc::MemoryKind::Host, -1),
        reloc::TransferDirection::HostToDevice);
    if (auto *e = std::get_if<reloc::TransferError>(&validated))
      return fail(e->code + ": " + e->message);
    reloc::HostBackend backend(2);
    auto request = std::get<reloc::TransferRequest>(std::move(validated));
    if (auto e = reloc::executeTransfer(request, backend, options))
      return fail(e->code + ": " + e->message);
    requested = destinationBytes;
  } else {
#ifdef RELOC_ENABLE_CUDA
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
      return fail("no CUDA device is available");
    const bool h2d = direction == "h2d";
    void *deviceBuffer = nullptr;
    const size_t deviceBytes = h2d ? destinationBytes : sourceBytes;
    if (cudaMalloc(&deviceBuffer, deviceBytes) != cudaSuccess)
      return fail("cudaMalloc failed");
    struct Free {
      void *p;
      ~Free() { cudaFree(p); }
    } guard{deviceBuffer};
    if (!h2d && cudaMemcpy(deviceBuffer, source.data(), sourceBytes,
                           cudaMemcpyHostToDevice) != cudaSuccess)
      return fail("staging the logical source on the device failed");
    auto srcView =
        h2d ? denseView(reinterpret_cast<uintptr_t>(source.data()), sourceBytes,
                        sourceExtents, width, reloc::MemoryKind::Host, -1)
            : denseView(reinterpret_cast<uintptr_t>(deviceBuffer), sourceBytes,
                        sourceExtents, width, reloc::MemoryKind::Cuda, device);
    auto dstView =
        h2d ? denseView(reinterpret_cast<uintptr_t>(deviceBuffer),
                        destinationBytes, destinationExtents, width,
                        reloc::MemoryKind::Cuda, device)
            : denseView(reinterpret_cast<uintptr_t>(destination.data()),
                        destinationBytes, destinationExtents, width,
                        reloc::MemoryKind::Host, -1);
    auto validated =
        reloc::validateTransfer(b, srcView, dstView,
                                h2d ? reloc::TransferDirection::HostToDevice
                                    : reloc::TransferDirection::DeviceToHost);
    if (auto *e = std::get_if<reloc::TransferError>(&validated))
      return fail(e->code + ": " + e->message);
    reloc::CudaBackend backend(2, device);
    auto request = std::get<reloc::TransferRequest>(std::move(validated));
    if (auto e = reloc::executeTransfer(request, backend, options))
      return fail(e->code + ": " + e->message);
    requested = h2d ? destinationBytes : sourceBytes;
    if (h2d) {
      // Verification only: the requested transfer already completed.
      if (cudaMemcpy(destination.data(), deviceBuffer, destinationBytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail("verification readback failed");
      readback = destinationBytes;
    }
#else
    return fail("direction '" + direction +
                "' needs a runtime built with RELOC_ENABLE_CUDA");
#endif
  }
  if (!writeFile(output, destination.data(), destinationBytes))
    return fail("cannot write output " + output);
  std::printf("{\"direction\": \"%s\", \"source_bytes\": %zu, "
              "\"destination_bytes\": %zu, \"requested_transfer_bytes\": %zu, "
              "\"verification_readback_bytes\": %zu, \"symbols\": %zu}\n",
              direction.c_str(), sourceBytes, destinationBytes, requested,
              readback, plan.symbols.size());
  return 0;
}
