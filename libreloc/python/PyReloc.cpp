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

#include "reloc/Decode.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
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
}
