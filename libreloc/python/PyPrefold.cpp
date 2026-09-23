//===- PyPrefold.cpp - owned validated prefold bindings -------------------===//

#include "PyPrefold.h"

#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/Prefold.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

/// Raised as pyreloc.PrefoldError.
struct PrefoldException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Member declaration order is the destruction contract: the artifact (last
/// declared) is destroyed first and frees its staging through the backend,
/// then the pool joins its workers, then the backend goes away.
struct OwnedPrefold {
  reloc::HostBackend backend;
  reloc::GatherPool pool;
  reloc::prefold::PrefoldArtifact artifact;
  bool closed = false;

  explicit OwnedPrefold(unsigned gatherThreads)
      : backend(1), pool(gatherThreads) {}

  ~OwnedPrefold() { close(); }

  bool valid() const { return !closed && artifact.valid(); }
  int64_t nbytes() const { return artifact.bytes(); }

  void close() {
    if (closed)
      return;
    closed = true;
    artifact = reloc::prefold::PrefoldArtifact(); // frees staging first
    pool.close();
  }

  void copyTo(uintptr_t dst, size_t dstBytes) {
    if (!valid())
      throw PrefoldException("prefold handle is closed or invalid");
    if (dst == 0)
      throw py::value_error("destination pointer is null");
    if (dstBytes < static_cast<size_t>(artifact.bytes()))
      throw py::value_error(
          "destination buffer too small: " + std::to_string(dstBytes) +
          " bytes, artifact has " + std::to_string(artifact.bytes()));
    const void *src = artifact.data();
    const size_t bytes = static_cast<size_t>(artifact.bytes());
    py::gil_scoped_release release;
    std::memcpy(reinterpret_cast<void *>(dst), src, bytes);
  }
};

reloc::prefold::OutputSpec parseSpec(const std::string &spec) {
  if (spec == "s8_quant_pack")
    return reloc::prefold::OutputSpec::S8QuantPack;
  if (spec == "s8_gather_quant")
    return reloc::prefold::OutputSpec::S8GatherQuant;
  throw py::value_error("output_spec must be 's8_quant_pack' or "
                        "'s8_gather_quant', got '" +
                        spec + "'");
}

bool mulOk(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_mul_overflow(a, b, &out);
}

bool addOk(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_add_overflow(a, b, &out);
}

std::shared_ptr<OwnedPrefold> prefoldS8(const reloc::BoundPlan &bound,
                                        uintptr_t srcPtr, size_t srcBytes,
                                        uintptr_t scalesPtr, size_t scalesBytes,
                                        const std::string &outputSpec,
                                        int gatherThreads) {
  const reloc::prefold::OutputSpec spec = parseSpec(outputSpec);
  if (gatherThreads < 0)
    throw py::value_error("gather_threads must be >= 0 (0 = all cores)");
  if (srcPtr == 0 || scalesPtr == 0)
    throw PrefoldException(
        "source and inverse-scale pointers must be non-null");
  if (bound.elementSize != 4)
    throw PrefoldException("prefold_s8 requires a float32 (4-byte) plan");
  if (bound.extents.size() < 2)
    throw PrefoldException("prefold_s8 requires a plan of rank >= 2 (channel "
                           "axis plus inner data)");
  if (!bound.padRegions.empty())
    throw PrefoldException("prefold_s8 does not support padded plans");
  if (bound.extents.size() != bound.srcStrides.size() ||
      bound.extents.size() != bound.dstStrides.size())
    throw PrefoldException("bound plan has inconsistent axes");
  // Packed destination: dstStrides[k] == prod(extents[k+1..]) for every k.
  int64_t packed = 1;
  for (size_t k = bound.extents.size(); k-- > 0;) {
    if (bound.extents[k] < 1)
      throw PrefoldException("plan extents must be >= 1");
    if (bound.dstStrides[k] != packed)
      throw PrefoldException(
          "prefold_s8 requires a packed row-major destination");
    if (!mulOk(packed, bound.extents[k], packed))
      throw PrefoldException("plan element count overflows");
  }
  if (packed != bound.totalBytes / bound.elementSize)
    throw PrefoldException("plan totalBytes disagrees with its extents");
  if (spec == reloc::prefold::OutputSpec::S8QuantPack &&
      bound.srcStrides != bound.dstStrides)
    throw PrefoldException(
        "s8_quant_pack requires an identity (contiguous) plan");
  // Source footprint: max element offset the plan reads, plus one.
  int64_t maxRead = 0;
  for (size_t k = 0; k < bound.extents.size(); ++k) {
    if (bound.srcStrides[k] < 0)
      throw PrefoldException("plan source strides must be non-negative");
    int64_t reach = 0;
    if (!mulOk(bound.extents[k] - 1, bound.srcStrides[k], reach) ||
        !addOk(maxRead, reach, maxRead))
      throw PrefoldException("plan source reach overflows");
  }
  int64_t needed = 0;
  if (!mulOk(maxRead + 1, static_cast<int64_t>(bound.elementSize), needed))
    throw PrefoldException("plan source footprint overflows");
  if (static_cast<size_t>(needed) > srcBytes)
    throw PrefoldException(
        "source buffer too small: " + std::to_string(srcBytes) +
        " bytes, plan reads " + std::to_string(needed));
  // Exactly one positive finite inverse scale per channel (outer axis).
  int64_t scaleBytes = 0;
  if (!mulOk(bound.extents[0], 4, scaleBytes))
    throw PrefoldException("scale count overflows");
  if (static_cast<size_t>(scaleBytes) != scalesBytes)
    throw PrefoldException("inverse scales must hold exactly " +
                           std::to_string(bound.extents[0]) +
                           " float32 values (" + std::to_string(scaleBytes) +
                           " bytes), got " + std::to_string(scalesBytes));
  const auto *scales = reinterpret_cast<const float *>(scalesPtr);
  for (int64_t c = 0; c < bound.extents[0]; ++c)
    if (!(scales[c] > 0.0f) || !std::isfinite(scales[c]))
      throw PrefoldException("inverse scale " + std::to_string(c) +
                             " must be positive and finite");

  auto owner =
      std::make_shared<OwnedPrefold>(static_cast<unsigned>(gatherThreads));
  {
    py::gil_scoped_release release;
    owner->artifact = reloc::prefold::prefoldArtifact(
        bound, reinterpret_cast<const float *>(srcPtr), spec, scales,
        owner->backend, owner->pool);
  }
  if (!owner->artifact.valid())
    throw PrefoldException("prefold failed: the runtime rejected the plan or "
                           "could not allocate the artifact");
  return owner;
}

} // namespace

void registerPrefoldBindings(py::module_ &m) {
  py::register_exception<PrefoldException>(m, "PrefoldError");

  py::class_<OwnedPrefold, std::shared_ptr<OwnedPrefold>>(
      m, "PrefoldHandle",
      "Owned load-time prefolded int8 image (T4 Task 3). nbytes is the image "
      "size; copy_to copies it into a caller buffer while the owner stays "
      "alive; close releases the artifact, then the gather pool, then the "
      "backend. Also a context manager.")
      .def_property_readonly("nbytes", &OwnedPrefold::nbytes)
      .def_property_readonly("closed",
                             [](const OwnedPrefold &o) { return o.closed; })
      .def("copy_to", &OwnedPrefold::copyTo, py::arg("dst_ptr"),
           py::arg("dst_bytes"))
      .def("close", &OwnedPrefold::close, "Idempotent.")
      .def("__enter__",
           [](const std::shared_ptr<OwnedPrefold> &o) { return o; })
      .def("__exit__", [](OwnedPrefold &o, const py::object &,
                          const py::object &, const py::object &) {
        o.close();
        return false;
      });

  m.def("prefold_s8", &prefoldS8, py::arg("bound"), py::arg("src_ptr"),
        py::arg("src_bytes"), py::arg("inv_scales_ptr"),
        py::arg("inv_scales_bytes"), py::kw_only(), py::arg("output_spec"),
        py::arg("gather_threads") = 1,
        "Fold a float32 source through `bound` into an owned int8 image using "
        "the existing prefolder. output_spec is 's8_quant_pack' (identity "
        "plan, contiguous per-channel quantize) or 's8_gather_quant' (fused "
        "strided gather + per-channel quantize). Inverse scales: exactly one "
        "positive finite float32 per outer (channel) extent. Every "
        "precondition is validated before any kernel runs; failures raise "
        "PrefoldError.");
}
