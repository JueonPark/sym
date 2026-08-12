//===- PyReloc.cpp - pybind11 bindings for libreloc -----------------------===//
//
// The Python surface of P2 (issue #46): load_plan/bind plus the host and
// CUDA executors. Pointers cross this boundary as integers (issue #40
// design decision 2); torch/numpy mapping lives in pure Python
// (pyreloc/torch_interop.py) so libtorch is never linked. Decode/bind
// failures surface as pyreloc.DecodeError / pyreloc.BindError carrying the
// C++ diagnostic string.
//
//===----------------------------------------------------------------------===//

#include "reloc/Bind.h"
#include "reloc/CostModel.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/Pipeline.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

namespace py = pybind11;

namespace {

struct DecodeException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

reloc::RelocationPlan loadPlan(const py::bytes &data) {
  std::string buf = data;
  auto result = reloc::decodePlan(reinterpret_cast<const uint8_t *>(buf.data()),
                                  buf.size());
  if (auto *err = std::get_if<reloc::DecodeError>(&result))
    throw DecodeException("decode error at byte offset " +
                          std::to_string(err->offset) + ": " + err->message);
  return std::get<reloc::RelocationPlan>(std::move(result));
}

uint32_t planElementSize(const reloc::RelocationPlan &plan) {
  return plan.dst.elementType.bitwidth / 8;
}

struct BindException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

const char *strategyName(reloc::Strategy s) {
  switch (s) {
  case reloc::Strategy::Auto:
    return "auto";
  case reloc::Strategy::ViewNoCopy:
    return "view_no_copy";
  case reloc::Strategy::SingleThreadSimd:
    return "single_thread_simd";
  case reloc::Strategy::MultiThreadTiled:
    return "multi_thread_tiled";
  case reloc::Strategy::ChunkedPipeline:
    return "chunked_pipeline";
  }
  return "unknown";
}

reloc::Strategy parseStrategy(const std::string &name) {
  for (reloc::Strategy s :
       {reloc::Strategy::Auto, reloc::Strategy::ViewNoCopy,
        reloc::Strategy::SingleThreadSimd, reloc::Strategy::MultiThreadTiled,
        reloc::Strategy::ChunkedPipeline})
    if (name == strategyName(s))
      return s;
  throw py::value_error("unknown strategy '" + name +
                        "' (expected auto/view_no_copy/single_thread_simd/"
                        "multi_thread_tiled/chunked_pipeline)");
}

// Byte span a source buffer must cover: max element offset reachable via
// srcStrides over the valid index space, plus one element.
size_t minSrcBytes(const reloc::BoundPlan &b) {
  int64_t maxOff = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    maxOff += (b.extents[k] - 1) * b.srcStrides[k];
  return static_cast<size_t>(maxOff + 1) * b.elementSize;
}

int64_t validElements(const reloc::BoundPlan &b) {
  int64_t n = 1;
  for (int64_t e : b.extents)
    n *= e;
  return n;
}

// Validate a (ptr, nbytes) pair against a required byte count. Runs with
// the GIL held (before any release), so py::value_error is safe.
void checkBuffer(const char *what, uintptr_t ptr, size_t nbytes,
                 size_t required) {
  if (ptr == 0)
    throw py::value_error(std::string(what) + " pointer is null");
  if (nbytes < required)
    throw py::value_error(std::string(what) +
                          " buffer too small: " + std::to_string(nbytes) +
                          " bytes, plan requires " + std::to_string(required));
}

// Shared validation for the gather-parallelism kwargs. Runs with the GIL
// held (before any release), so py::value_error is safe. gather_pool wins
// over gather_threads when both are given.
void checkGatherArgs(int gatherThreads,
                     const std::shared_ptr<reloc::GatherPool> &pool) {
  if (gatherThreads < 0)
    throw py::value_error("gather_threads must be >= 0 (0 = all cores)");
  if (pool && pool->closed())
    throw py::value_error("gather_pool is closed");
}

reloc::BoundPlan bindPlan(const reloc::RelocationPlan &plan,
                          const std::map<std::string, int64_t> &symbols,
                          const std::string &strategy,
                          const reloc::costmodel::CostModel *model,
                          double wireRatio, int k, int64_t nReuse) {
  auto result = reloc::bind(plan, symbols, parseStrategy(strategy), model,
                            wireRatio, k, nReuse);
  if (auto *err = std::get_if<reloc::BindError>(&result))
    throw BindException(err->message);
  return std::get<reloc::BoundPlan>(std::move(result));
}

void relocateHost(const reloc::BoundPlan &b, uintptr_t srcPtr, size_t srcBytes,
                  uintptr_t dstPtr, size_t dstBytes, int gatherThreads,
                  std::shared_ptr<reloc::GatherPool> gatherPool) {
  checkBuffer("src", srcPtr, srcBytes, minSrcBytes(b));
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *src = reinterpret_cast<const void *>(srcPtr);
  void *dst = reinterpret_cast<void *>(dstPtr);
  py::gil_scoped_release release;
  switch (b.strategy) {
  case reloc::Strategy::MultiThreadTiled:
    reloc::executeH2DThreaded(b, src, dst);
    break;
  case reloc::Strategy::ChunkedPipeline: {
    reloc::HostBackend backend(2);
    if (gatherPool)
      reloc::executeH2DPipelined(b, src, dst, backend, /*nBuffers=*/2,
                                 /*chunkSizeOverride=*/0, *gatherPool);
    else
      reloc::executeH2DPipelined(b, src, dst, backend, /*nBuffers=*/2,
                                 /*chunkSizeOverride=*/0,
                                 static_cast<unsigned>(gatherThreads));
    break;
  }
  default:
    // Auto / ViewNoCopy / SingleThreadSimd all materialize via the
    // single-thread copy (a no_copy view has nothing to publish across a
    // language boundary that handed us a destination buffer).
    reloc::executeH2D(b, src, dst);
    break;
  }
}

void relocateInverseHost(const reloc::BoundPlan &b, uintptr_t dstPtr,
                         size_t dstBytes, uintptr_t srcPtr, size_t srcBytes) {
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkBuffer("src(out)", srcPtr, srcBytes, minSrcBytes(b));
  const void *dst = reinterpret_cast<const void *>(dstPtr);
  void *src = reinterpret_cast<void *>(srcPtr);
  py::gil_scoped_release release;
  reloc::executeD2H(b, dst, src);
}

void h2dCuda(const reloc::BoundPlan &b, uintptr_t srcPtr, size_t srcBytes,
             uintptr_t dstPtr, size_t dstBytes, int nBuffers, int nStreams,
             int gatherThreads, std::shared_ptr<reloc::GatherPool> gatherPool) {
#ifdef RELOC_ENABLE_CUDA
  checkBuffer("src", srcPtr, srcBytes, minSrcBytes(b));
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *src = reinterpret_cast<const void *>(srcPtr);
  void *dst = reinterpret_cast<void *>(dstPtr);
  py::gil_scoped_release release;
  reloc::CudaBackend backend(nStreams);
  if (gatherPool)
    reloc::executeH2DPipelined(b, src, dst, backend, nBuffers,
                               /*chunkSizeOverride=*/0, *gatherPool);
  else
    reloc::executeH2DPipelined(b, src, dst, backend, nBuffers,
                               /*chunkSizeOverride=*/0,
                               static_cast<unsigned>(gatherThreads));
#else
  (void)b, (void)srcPtr, (void)srcBytes, (void)dstPtr, (void)dstBytes;
  (void)nBuffers, (void)nStreams, (void)gatherThreads, (void)gatherPool;
  throw std::runtime_error("pyreloc was built without RELOC_ENABLE_CUDA");
#endif
}

void d2hCuda(const reloc::BoundPlan &b, uintptr_t dstPtr, size_t dstBytes,
             uintptr_t srcPtr, size_t srcBytes, int nBuffers, int nStreams,
             int gatherThreads, std::shared_ptr<reloc::GatherPool> gatherPool) {
#ifdef RELOC_ENABLE_CUDA
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkBuffer("src(out)", srcPtr, srcBytes, minSrcBytes(b));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *dst = reinterpret_cast<const void *>(dstPtr);
  void *src = reinterpret_cast<void *>(srcPtr);
  py::gil_scoped_release release;
  reloc::CudaBackend backend(nStreams);
  if (gatherPool)
    reloc::executeD2HPipelined(b, dst, src, backend, nBuffers,
                               /*chunkSizeOverride=*/0, *gatherPool);
  else
    reloc::executeD2HPipelined(b, dst, src, backend, nBuffers,
                               /*chunkSizeOverride=*/0,
                               static_cast<unsigned>(gatherThreads));
#else
  (void)b, (void)dstPtr, (void)dstBytes, (void)srcPtr, (void)srcBytes;
  (void)nBuffers, (void)nStreams, (void)gatherThreads, (void)gatherPool;
  throw std::runtime_error("pyreloc was built without RELOC_ENABLE_CUDA");
#endif
}

} // namespace

PYBIND11_MODULE(_pyreloc, m) {
  m.doc() = "libreloc Python bindings (issue #46). Buffers are passed as "
            "(pointer, nbytes) integer pairs -- see pyreloc.torch_interop.";

  py::register_exception<DecodeException>(m, "DecodeError");

  py::class_<reloc::RelocationPlan>(m, "PlanHandle")
      .def_property_readonly(
          "symbols", [](const reloc::RelocationPlan &p) { return p.symbols; })
      .def_property_readonly(
          "num_axes",
          [](const reloc::RelocationPlan &p) { return p.axes.size(); })
      .def_property_readonly("element_size", &planElementSize)
      .def_property_readonly(
          "no_copy", [](const reloc::RelocationPlan &p) { return p.noCopy; })
      .def("__repr__", [](const reloc::RelocationPlan &p) {
        std::ostringstream os;
        os << "PlanHandle(symbols=[";
        for (size_t i = 0; i < p.symbols.size(); ++i)
          os << (i ? ", " : "") << "'" << p.symbols[i] << "'";
        os << "], axes=" << p.axes.size()
           << ", element_size=" << planElementSize(p)
           << ", no_copy=" << (p.noCopy ? "True" : "False") << ")";
        return os.str();
      });

  m.def("load_plan", &loadPlan, py::arg("data"),
        "Decode a wire-format-v0 plan blob. Raises DecodeError with the "
        "byte offset and diagnostic on invalid input.");

  py::register_exception<BindException>(m, "BindError");

  py::class_<reloc::BoundPlan>(m, "BoundPlan")
      .def_property_readonly(
          "extents", [](const reloc::BoundPlan &b) { return b.extents; })
      .def_property_readonly(
          "src_strides", [](const reloc::BoundPlan &b) { return b.srcStrides; })
      .def_property_readonly(
          "dst_strides", [](const reloc::BoundPlan &b) { return b.dstStrides; })
      .def_property_readonly(
          "element_size",
          [](const reloc::BoundPlan &b) { return b.elementSize; })
      .def_property_readonly(
          "total_bytes", [](const reloc::BoundPlan &b) { return b.totalBytes; })
      .def_property_readonly("no_copy",
                             [](const reloc::BoundPlan &b) { return b.noCopy; })
      .def_property_readonly(
          "strategy",
          [](const reloc::BoundPlan &b) { return strategyName(b.strategy); })
      .def_property_readonly("valid_elements", &validElements)
      .def_property_readonly("min_src_bytes", &minSrcBytes)
      .def_property_readonly(
          "decision",
          [](const reloc::BoundPlan &b) -> py::object {
            if (!b.decision)
              return py::none();
            const reloc::costmodel::MethodDecision &d = *b.decision;
            py::dict out;
            out["method"] = std::string(reloc::costmodel::methodName(d.method));
            out["t_a_ms"] = d.tAMs;
            out["t_b_ms"] = d.tBMs;
            out["threshold_bytes"] = d.thresholdBytes;
            out["pattern"] =
                std::string(reloc::costmodel::patternName(d.pattern));
            out["b_placement"] =
                std::string(reloc::costmodel::placementName(d.bPlacement));
            out["k"] = d.k;
            out["n_reuse"] = d.nReuse;
            return std::move(out);
          },
          "Cost-model decision populated when bind() was given a model and "
          "the calibration had the needed pattern/r keys; None otherwise "
          "(bind-time pricing is t8 + Overlapped -- Bind.cpp step 8).")
      .def("__repr__", [](const reloc::BoundPlan &b) {
        std::ostringstream os;
        os << "BoundPlan(extents=[";
        for (size_t i = 0; i < b.extents.size(); ++i)
          os << (i ? ", " : "") << b.extents[i];
        os << "], total_bytes=" << b.totalBytes << ", strategy='"
           << strategyName(b.strategy) << "')";
        return os.str();
      });

  namespace cmns = reloc::costmodel;
  py::class_<cmns::CostModel>(m, "Calibration",
                              "Parsed costmodel calibration (issue #97).")
      .def_property_readonly("machine", &cmns::CostModel::machine);

  m.def(
      "load_calibration",
      [](const std::string &path) {
        auto r = cmns::CostModel::load(path);
        if (auto *err = std::get_if<std::string>(&r))
          throw py::value_error(*err);
        return std::get<cmns::CostModel>(r);
      },
      py::arg("path"),
      "Load a costmodel calibration (.cal). Raises ValueError with the "
      "parser diagnostic on invalid input.");

  m.def("bind", &bindPlan, py::arg("plan"), py::arg("symbols"),
        py::arg("strategy") = "auto", py::kw_only(),
        py::arg("model") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("wire_ratio") = 1.0, py::arg("k") = 1,
        py::arg("n_reuse") = -1,
        "Bind a plan against {symbol: value}. Raises BindError on symbol "
        "mismatch or violated correctness constraints. With `model`, "
        "populates BoundPlan.decision (bind-time t8/Overlapped pricing); "
        "decision stays None if the calibration lacks the needed keys.");

  m.def(
      "predict",
      [](const cmns::CostModel &cal, const std::string &pattern,
         int64_t srcBytes, double r, int threads, int k, int64_t nReuse,
         bool broadcast, const std::string &bPlacement) {
        cmns::Pattern p;
        if (pattern == "contiguous")
          p = cmns::Pattern::Contiguous;
        else if (pattern == "blocked")
          p = cmns::Pattern::Blocked;
        else if (pattern == "single_element")
          p = cmns::Pattern::SingleElement;
        else if (pattern == "tiled")
          p = cmns::Pattern::Tiled;
        else
          throw py::value_error("unknown pattern '" + pattern + "'");
        cmns::BPlacement bp;
        if (bPlacement == "overlapped")
          bp = cmns::BPlacement::Overlapped;
        else if (bPlacement == "serial")
          bp = cmns::BPlacement::Serial;
        else
          throw py::value_error("unknown b_placement '" + bPlacement +
                                "' (expected 'overlapped' or 'serial')");
        auto d = cmns::decide(cal, p, srcBytes, r, threads, k, nReuse,
                              broadcast, bp);
        if (!d)
          throw py::value_error("calibration lacks keys for pattern '" +
                                pattern + "' at r=" + std::to_string(r) + " t" +
                                std::to_string(threads) + " k" +
                                std::to_string(k));
        py::dict out;
        out["method"] = std::string(cmns::methodName(d->method));
        out["t_a_ms"] = d->tAMs;
        out["t_b_ms"] = d->tBMs;
        out["threshold_bytes"] = d->thresholdBytes;
        out["pattern"] = std::string(cmns::patternName(d->pattern));
        out["b_placement"] = std::string(cmns::placementName(d->bPlacement));
        return out;
      },
      py::arg("calibration"), py::kw_only(), py::arg("pattern"),
      py::arg("src_bytes"), py::arg("r"), py::arg("threads") = 8,
      py::arg("k") = 1, py::arg("n_reuse") = -1, py::arg("broadcast") = false,
      py::arg("b_placement") = "overlapped",
      "Run the C++ cost model. Returns {method, t_a_ms, t_b_ms, "
      "threshold_bytes, pattern, b_placement}.");

  py::class_<reloc::GatherPool, std::shared_ptr<reloc::GatherPool>>(
      m, "GatherPool",
      "Persistent gather/scatter worker pool (issue #65). threads == 0 "
      "resolves to the hardware thread count; the pool owns threads-1 OS "
      "workers (the calling thread is the last worker of each dispatch). "
      "close() joins every worker deterministically -- also usable as a "
      "context manager. Dispatches and close() are serialized internally, "
      "so a pool shared between Python threads is safe -- parallelized "
      "calls simply run one at a time. A closed pool cannot be passed to "
      "relocate/h2d/d2h.")
      .def(py::init([](int threads) {
             if (threads < 0)
               throw py::value_error("threads must be >= 0 (0 = all cores)");
             return std::make_shared<reloc::GatherPool>(
                 static_cast<unsigned>(threads));
           }),
           py::arg("threads") = 0)
      .def_property_readonly("threads", &reloc::GatherPool::threadCount)
      .def_property_readonly("closed", &reloc::GatherPool::closed)
      .def("close", &reloc::GatherPool::close,
           py::call_guard<py::gil_scoped_release>(),
           "Join all workers. Idempotent.")
      .def("__enter__",
           [](const std::shared_ptr<reloc::GatherPool> &p) { return p; })
      .def(
          "__exit__",
          [](reloc::GatherPool &p, const py::object &, const py::object &,
             const py::object &) {
            p.close();
            return false;
          },
          py::call_guard<py::gil_scoped_release>());

  m.def("relocate", &relocateHost, py::arg("bound"), py::arg("src_ptr"),
        py::arg("src_nbytes"), py::arg("dst_ptr"), py::arg("dst_nbytes"),
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        "Host relocation (CPU strategies): src -> dst-layout buffer. "
        "gather_threads / gather_pool parallelize the chunked_pipeline "
        "strategy's per-chunk gather (0 = all cores; a given pool wins).");
  m.def("relocate_inverse", &relocateInverseHost, py::arg("bound"),
        py::arg("dst_ptr"), py::arg("dst_nbytes"), py::arg("src_ptr"),
        py::arg("src_nbytes"),
        "Host inverse: reconstruct the source from a dst-layout buffer.");
  m.def("h2d", &h2dCuda, py::arg("bound"), py::arg("src_ptr"),
        py::arg("src_nbytes"), py::arg("dst_ptr"), py::arg("dst_nbytes"),
        py::arg("n_buffers") = 4, py::arg("n_streams") = 2,
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        "Pinned/stream pipeline H2D (CUDA builds; dst_ptr is a device "
        "pointer). gather_threads/gather_pool parallelize per-chunk gather. "
        "Raises RuntimeError without RELOC_ENABLE_CUDA.");
  m.def("d2h", &d2hCuda, py::arg("bound"), py::arg("dst_ptr"),
        py::arg("dst_nbytes"), py::arg("src_ptr"), py::arg("src_nbytes"),
        py::arg("n_buffers") = 4, py::arg("n_streams") = 2,
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        "Pinned/stream pipeline D2H inverse (CUDA builds; dst_ptr is a "
        "device pointer). gather_threads/gather_pool parallelize per-chunk "
        "scatter. Raises RuntimeError without RELOC_ENABLE_CUDA.");

#ifdef RELOC_ENABLE_CUDA
  m.attr("cuda_enabled") = true;
#else
  m.attr("cuda_enabled") = false;
#endif
}
