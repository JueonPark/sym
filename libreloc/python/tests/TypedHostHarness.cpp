// Build-only host-kernel timing; no Python, allocation, binding or DMA in
// samples. Transform loads/stores are part of the measured kernel.
#include "reloc/GatherPool.h"
#include "reloc/TypedExecute.h"
#include <chrono>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_typed_host_test, m) {
  py::module_::import("pyreloc");
  m.def(
      "measure",
      [](const reloc::TypedBoundPlan &bound, py::array source,
         py::array destination, unsigned threads, unsigned samples) {
        auto prepared = reloc::typed::prepareProgram(bound);
        if (auto *error = std::get_if<reloc::typed::ExecutionError>(&prepared))
          throw py::value_error(error->message);
        const auto &p = std::get<reloc::typed::Program>(prepared);
        const auto stages = static_cast<uint32_t>(p.stages.size());
        if (!threads || !samples || samples > 10000 ||
            !(source.flags() & py::array::c_style) ||
            !(destination.flags() & py::array::c_style) ||
            !destination.writeable() ||
            source.itemsize() != reloc::typed::widthAt(p, 0) ||
            destination.itemsize() != reloc::typed::widthAt(p, stages) ||
            source.nbytes() != reloc::typed::bytesAt(p, 0, false) ||
            destination.nbytes() != reloc::typed::bytesAt(p, stages, true))
          throw py::value_error("invalid dense benchmark buffers/counts");
        const void *src = source.data();
        void *dst = destination.mutable_data();
        std::vector<double> times;
        {
          py::gil_scoped_release release;
          reloc::GatherPool pool(threads);
          auto execute = [&] {
            if (auto error =
                    reloc::typed::executeHost(p, 0, stages, src, dst, &pool))
              throw std::runtime_error(error->message);
          };
          for (unsigned i = 0; i < 3; ++i)
            execute();
          for (unsigned i = 0; i < samples; ++i) {
            const auto start = std::chrono::steady_clock::now();
            execute();
            times.push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count());
          }
        }
        return times;
      },
      py::arg("bound"), py::arg("source"), py::arg("destination"),
      py::arg("threads") = 1, py::arg("samples") = 30);
}
